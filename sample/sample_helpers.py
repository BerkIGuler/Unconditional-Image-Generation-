import os
from diffusers import DiffusionPipeline

import sample_config


sample_counter = 0


def sample(pipe):
    imgs = pipe(batch_size=sample_config.cfg["BATCH_SIZE"]).images
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
    folder = os.path.join(sample_config.cfg["SAVE_DIR"], class_name)
    initial_folder_size = len(os.listdir(folder))
    target_sample_size = int(initial_folder_size * (1 + sample_config.cfg["GTO_RATIO"]))
    while len(os.listdir(folder)) < target_sample_size:
        ims = sample(pipe)
        save_batch_to_disk(folder, ims, class_name, target_sample_size)


def load_pipeline(class_name, device):
    model_path = os.path.join(sample_config.cfg["MODELS_DIR"], class_name)
    pipe = DiffusionPipeline.from_pretrained(model_path)
    pipe.to(device)
    return pipe


def sample_dataset(dataset_folder, device):
    global sample_counter
    for class_name in os.listdir(dataset_folder):
        sample_counter = 0
        pipeline = load_pipeline(class_name, device)
        sample_for_new_class(class_name, pipeline)
