import os
from diffusers import DiffusionPipeline, DDIMScheduler
import torch

import sample_config
from logger import get_global_logger
import logging


sample_config.parse_args()
sample_counter = 0
gpu_id = sample_config.cfg["GPU_ID"]
device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else "cpu"
logger = get_global_logger(__name__, log_level=logging.INFO)


def sample(pipe):
    imgs = pipe(batch_size=5).images
    return imgs


def save_batch_to_disk(save_folder, ims, class_name, target_sample_size):
    global sample_counter
    for im in ims:
        im_name = os.path.join(
            save_folder, f"{class_name}_ddpm_{sample_counter}.jpg")
        im.save(im_name)
        sample_counter += 1
        if target_sample_size >= len(os.listdir(save_folder)):
            return


def sample_for_new_class(class_name, pipe):
    global sample_counter
    logger.info(f"starting sampling for class {class_name}")

    parent_dir = os.path.dirname(sample_config.cfg["DATASET_PATH"])
    original_class_folder = os.path.join(sample_config.cfg["DATASET_PATH"], "train", class_name)
    dataset_name = os.path.basename(sample_config.cfg["DATASET_PATH"])
    new_dataset_name = dataset_name + "_ddpm" + str(sample_config.cfg["GTO_RATIO"])
    sampled_class_path = os.path.join(parent_dir, new_dataset_name, class_name)
    os.makedirs(sampled_class_path, exist_ok=True)
    initial_folder_size = len(os.listdir(original_class_folder))
    target_sample_size = int(initial_folder_size * sample_config.cfg["GTO_RATIO"])

    logger.info(f"initial folder size {initial_folder_size} and sample size is {target_sample_size}")
    while len(os.listdir(sampled_class_path)) < target_sample_size:
        ims = sample(pipe)
        save_batch_to_disk(sampled_class_path, ims, class_name, target_sample_size)

    logger.info(f"sampling for {class_name} done")


def load_pipeline(class_name, device):
    model_path = os.path.join(
        sample_config.cfg["MODELS_PATH"], class_name,
        "models", sample_config.cfg["CHECKPOINT_STEP"])
    pipe = DiffusionPipeline.from_pretrained(model_path)
    if sample_config.cfg["SCHEDULER"] == "ddim":
        scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler = scheduler
    pipe.to(device)
    return pipe


def sample_dataset():
    global sample_counter
    train_path = os.path.join(sample_config.cfg["DATASET_PATH"], "train")
    for class_name in os.listdir(train_path):
        sample_counter = 0
        pipeline = load_pipeline(class_name, device)
        sample_for_new_class(class_name, pipeline)
