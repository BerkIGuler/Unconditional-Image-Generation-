from PIL import Image
import torch
import os
import torch.nn.functional as f
from accelerate import Accelerator
# from accelerate.utils import LoggerType
from tqdm.auto import tqdm
from diffusers import DDPMPipeline

import train_config


def train_loop(out_dir, mdl, noise_sch, optim, train_loader, lr_sch):

    class_name = os.path.basename(out_dir)
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=train_config.cfg["MIXED_PRECISION"],
        gradient_accumulation_steps=train_config.cfg["GRADIENT_ACCUMULATION_STEPS"],
        logging_dir=os.path.join(out_dir, "logs"),
        log_with="all"
    )
    device = accelerator.device

    if accelerator.is_main_process:
        accelerator.init_trackers(f"ddpm_training for {class_name} class")
        
    # Prepare everything
    mdl, optim, train_loader, lr_sch = accelerator.prepare(
        mdl, optim, train_loader, lr_sch,
    )

    global_step = 0

    # Now you train the model
    while global_step < train_config.cfg["TRAINING_STEPS"]:
        epoch = global_step // len(train_loader)
        progress_bar = tqdm(
            total=len(train_loader),
            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch: {epoch}")

        for step, batch in enumerate(train_loader):
            clean_images = batch['images']
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(device)

            # Sample a random timestep for each image
            time_steps = torch.randint(
                0, noise_sch.num_train_timesteps,
                (train_config.cfg["TRAIN_BATCH_SIZE"],),
                device=device).long()

            # Add noise to the clean images at each timestep
            noisy_images = noise_sch.add_noise(
                clean_images, noise, time_steps).to(device)

            with accelerator.accumulate(mdl):
                # Predict the noise residual
                noise_pred = mdl(noisy_images, time_steps, return_dict=False)[0]
                loss = f.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                # clips grad vector to MAX_GRAD_NORM
                accelerator.clip_grad_norm_(mdl.parameters(), train_config.cfg["MAX_GRAD_NORM"])
                optim.step()
                lr_sch.step()
                optim.zero_grad()

            progress_bar.update(1)
            logs = {
                "train batch loss": loss.detach().item(),
                "lr": lr_sch.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch sample some demo images with and save the model
            if (global_step % train_config.cfg["SAVE_IMAGE_STEPS"] == 0 or
                    global_step == train_config.cfg["TRAINING_STEPS"]) and \
                    accelerator.is_main_process:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(mdl),
                    scheduler=noise_sch)
                evaluate(out_dir, epoch, pipeline)
                pipeline.save_pretrained(out_dir)

            if global_step == train_config.cfg["TRAINING_STEPS"]:
                break

    accelerator.end_training()


# helper functions
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(out_dir, epoch, pipeline):
    # Sample some images from random noise
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=train_config.cfg["EVAL_BATCH_SIZE"],
        generator=torch.Generator(device="cuda:0")
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(out_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
