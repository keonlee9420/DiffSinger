import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.model import get_model
from utils.tools import get_configs_of, to_device
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize(model, step, configs, loader):
    preprocess_config, model_config, train_config = configs

    num_timesteps = int(model_config["denoiser"]["timesteps"])
    n = 0
    kld_T = 0
    kld_ts = [0] * (num_timesteps+1)

    for batchs in tqdm(loader):
        for batch in batchs:
            n += 1
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                target_mel = batch[6]
                teacher_forced_mel = model(*(batch[2:]))[0][0]
                kld_T += model.diffusion.expected_kld_T(teacher_forced_mel)

                for t in range(1, num_timesteps+1):
                    kld_t = model.diffusion.expected_kld_t(teacher_forced_mel, target_mel, t)
                    kld_ts[t] += kld_t

    kld_T = kld_T / n
    kld_ts = [kld_t / n for kld_t in kld_ts[1:]]

    K = 1
    for kld_t in kld_ts:
        if kld_t <= kld_T:
            break
        K += 1

    print("\nPredicted Boundary K is", K)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="aux",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    args.model = "aux"
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"_{}".format("shallow")
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"_{}".format("shallow")
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"_{}".format("aux")

    # Log Configuration
    print("\n==================================== Prediction Configuration ====================================")
    print(' ---> Total Batch Size:', int(train_config["optimizer"]["batch_size"]))
    print(' ---> Path of ckpt:', train_config["path"]["ckpt_path"])
    print("================================================================================================")

    # Get model
    model = get_model(args, configs, device, train=False)

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    synthesize(model, args.restore_step, configs, loader)
