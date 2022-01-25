import os
import json
import yaml

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
from tqdm import tqdm


matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def to_device(data, device):
    if len(data) == 18:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device) if mels is not None else mels
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).long().to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        cwt_specs = torch.from_numpy(cwt_specs).float().to(device) if cwt_specs is not None else cwt_specs
        f0_means = torch.from_numpy(f0_means).float().to(device) if f0_means is not None else f0_means
        f0_stds = torch.from_numpy(f0_stds).float().to(device) if f0_stds is not None else f0_stds
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        mel2phs = torch.from_numpy(mel2phs).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, lr=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/noise_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(args, targets, predictions, vocoder, model_config, preprocess_config, diffusion):

    basename = targets[0][0]
    src_len = predictions[10][0].item()
    mel_len = predictions[11][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    duration = targets[11][0, :src_len].int().detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch_prediction = predictions[4][0, :src_len].detach().cpu().numpy()
        pitch_prediction = expand(pitch_prediction, duration)
        pitch_target = targets[9][0, :src_len].detach().cpu().numpy()
        pitch_target = expand(pitch_target, duration)
    else:
        pitch_prediction = predictions[4][0, :mel_len].detach().cpu().numpy()
        pitch_target = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy_prediction = predictions[5][0, :src_len].detach().cpu().numpy()
        energy_prediction = expand(energy_prediction, duration)
        energy_target = targets[10][0, :src_len].detach().cpu().numpy()
        energy_target = expand(energy_target, duration)
    else:
        energy_prediction = predictions[5][0, :mel_len].detach().cpu().numpy()
        energy_target = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    if args.model == "aux":
        mel_prediction = predictions[0][0, :mel_len].float().detach().transpose(0, 1)
    else:
        # diffusion_step = predictions[3][0].item()
        # noisy_mels = predictions[0][0, :mel_len].detach().transpose(0, 1)
        # noise_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
        mel_prediction = diffusion.sampling()[0, :mel_len].detach().transpose(0, 1)
        diffusion.aux_mel = None

    fig = plot_mel(
        [
            # (noisy_mels.cpu().numpy(), pitch_target, energy_target),
            # (noise_prediction.cpu().numpy(), None, None),
            (mel_prediction.cpu().numpy(), pitch_prediction, energy_prediction),
            (mel_target.cpu().numpy(), pitch_target, energy_target),
        ],
        stats,
        # [f"Diffused Spectrogram at {diffusion_step}-step", f"Epsilon Prediction from {diffusion_step}-step", "Sampled Spectrogram", "Ground-Truth Spectrogram"],
        ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    teacher_forced_tag = "_teacher_forced" if args.teacher_forced else ""
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[10][i].item()
        mel_len = predictions[11][i].item()
        mel_prediction = predictions[0][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[7][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[4][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[4][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[5][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[5][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}{}.png".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.png".format(basename, teacher_forced_tag))
        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(fig_save_dir)
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[0].transpose(1, 2)
    lengths = predictions[11] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}{}.wav".format(basename, args.speaker_id, teacher_forced_tag)\
                if multi_speaker and args.mode == "single" else "{}{}.wav".format(basename, teacher_forced_tag)),
            sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean
    plt.tight_layout()

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        if pitch is not None:
            pitch = pitch * pitch_std + pitch_mean
            ax1 = add_axis(fig, axes[i][0])
            ax1.plot(pitch, color="tomato", linewidth=.7)
            ax1.set_xlim(0, mel.shape[1])
            ax1.set_ylim(0, pitch_max)
            ax1.set_ylabel("F0", color="tomato")
            ax1.tick_params(
                labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
            )

        if energy is not None:
            ax2 = add_axis(fig, axes[i][0])
            ax2.plot(energy, color="darkviolet", linewidth=.7)
            ax2.set_xlim(0, mel.shape[1])
            ax2.set_ylim(energy_min, energy_max)
            ax2.set_ylabel("Energy", color="darkviolet")
            ax2.yaxis.set_label_position("right")
            ax2.tick_params(
                labelsize="x-small",
                colors="darkviolet",
                bottom=False,
                labelbottom=False,
                left=False,
                labelleft=False,
                right=True,
                labelright=True,
            )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def get_noise_schedule_list(schedule_mode, timesteps, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError
    return schedule_list


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode='constant', value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur
