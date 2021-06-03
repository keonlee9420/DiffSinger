import os
import json
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from functools import partial
from inspect import isfunction

from .modules import Denoiser
from utils.tools import mel_normalize, mel_denormalize, get_noise_schedule_list


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, preprocess_config, model_config, train_config):
        super().__init__()
        self.denoise_fn = Denoiser(preprocess_config, model_config)
        self.mel_bins = preprocess_config['preprocessing']['mel']['n_mel_channels']

        betas = np.array(
            get_noise_schedule_list(model_config["denoiser"]["noise_schedule"])
        )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = train_config['loss']['loss_type']

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            self.mel_stats = stats["mel"]

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool):
        noise_pred = self.denoise_fn(x, t, cond)
        epsilon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            epsilon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=epsilon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def interpolate(self, x1, x2, t, cond, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
        x = x[:, 0].transpose(1, 2)
        return self.denorm_spec(x)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        noised_mel = self.q_sample(x_start=x_start, t=t, noise=noise)
        epsilon = self.denoise_fn(noised_mel, t, cond)

        if self.loss_type == 'l1':
            if mask is not None:
                mask = mask.unsqueeze(-1).transpose(1, 2)
                loss = (noise - epsilon).abs().squeeze(1).masked_fill(mask, 0.0).mean()
            else:
                print('are you sure w/o mask?')
                loss = (noise - epsilon).abs().mean()

        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, epsilon)
        else:
            raise NotImplementedError()
        noised_mel, epsilon = noised_mel.squeeze().transpose(1, 2), epsilon.squeeze().transpose(1, 2)
        return noised_mel, epsilon, loss

    @torch.no_grad()
    def sampling(self):
        b, *_, device = *self.cond.shape, self.cond.device
        t = self.num_timesteps
        shape = (self.cond.shape[0], 1, self.mel_bins, self.cond.shape[2])
        x = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), self.cond)
        x = x[:, 0].transpose(1, 2)
        output = self.denorm_spec(x)
        return output

    def forward(self, mel, cond, mel_mask):
        b, *_, device = *cond.shape, cond.device
        output=epsilon = None
        loss=t = torch.tensor([0.], device=device, requires_grad=False)
        self.cond = cond.transpose(1, 2)
        if mel is None:
            output = self.sampling()
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = mel
            x = self.norm_spec(x)
            x = x.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
            output, epsilon, loss = self.p_losses(x, t, self.cond, mask=mel_mask)
        return output, epsilon, loss, t

    def norm_spec(self, x):
        return mel_normalize(x, *self.mel_stats)

    def denorm_spec(self, x):
        return mel_denormalize(x, *self.mel_stats)

    def out2mel(self, x):
        return x
