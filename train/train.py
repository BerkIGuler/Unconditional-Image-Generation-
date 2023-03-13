"""training setup for ddpm training"""
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from datasets import load_dataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import os

import train_config
from train_helpers import train_loop

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def transform(examples):
    images = [preprocess(image) for image in examples["image"]]
    return {"images": images}


if __name__ == "__main__":

    train_config.parse_args()
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)

    gpu_id = train_config.cfg["GPU_ID"]
    device = torch.device(f"cuda:{gpu_id}") if \
        torch.cuda.is_available() else torch.device("cpu")

    # training folder
    results_folder = os.path.join(parent_dir, train_config.cfg["DATASET_NAME"])
    os.makedirs(results_folder, exist_ok=True)
    save_args_txt_path = os.path.join(results_folder, "args.txt")
    with open(save_args_txt_path, "w") as fout:
        fout.write(str(train_config.cfg))

    preprocess = transforms.Compose(
        [
            transforms.Resize((train_config.cfg["IMAGE_SIZE"], train_config.cfg["IMAGE_SIZE"])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # calculated for breast_cancer dataset train set
            transforms.Normalize([0.3248, 0.3248, 0.3248], [0.2201, 0.2201, 0.2201]),
        ]
    )
    # for each class
    for dr in os.listdir(train_config.cfg["DATASET_PATH"]):
        class_folder = os.path.join(train_config.cfg["DATASET_PATH"], dr)
        class_results_folder = os.path.join(results_folder, dr)
        os.makedirs(class_results_folder, exist_ok=True)
        output_dir = class_results_folder
        dataset = load_dataset("imagefolder", data_dir=class_folder)

        dataset = dataset["train"]
        dataset.set_transform(transform)

        train_dataloader = DataLoader(
            dataset, batch_size=train_config.cfg["TRAIN_BATCH_SIZE"], shuffle=True)

        if train_config.cfg["PRETRAINED_MODEL"] == "None":

            # initialize a UNET with random weights
            model = UNet2DModel(
                # the target image resolution
                sample_size=train_config.cfg["IMAGE_SIZE"],
                # the number of input channels, 3 for RGB images
                in_channels=3,
                # the number of output channels
                out_channels=3,
                # how many ResNet layers to use per UNet block
                layers_per_block=2,
                # the number of output channes for each UNet block
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    # a regular ResNet downsampling block
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    # a ResNet downsampling block with spatial self-attention
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    # a regular ResNet upsampling block
                    "UpBlock2D",
                    # a ResNet upsampling block with spatial self-attention
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D"
                ),
            )
        else:
            model = UNet2DModel.from_pretrained(train_config.cfg["PRETRAINED_MODEL"])
        model.to(device)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=train_config.cfg["TRAIN_TIMESTEPS"]
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_config.cfg["LEARNING_RATE"]
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=train_config.cfg["LR_WARMUP_STEPS"],
            num_training_steps=(train_config.cfg["TRAINING_STEPS"]),
        )
        train_loop(
            output_dir, model, noise_scheduler,
            optimizer, train_dataloader, lr_scheduler
        )
