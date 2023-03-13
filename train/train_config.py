"""Config for ddpm training"""
import argparse

# dict to store configuration details
cfg = dict()

# fixed args
cfg["MAX_GRAD_NORM"] = 1.0


def parse_args():
    """Modifies args global var

    parses command line arguments from where this function is called
    global cfg can be accessed from other modules without extra parsing

    :return: None
    """
    global cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="data folder with class sub-folders")

    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help="dataset name")

    parser.add_argument("--im_size",
                        type=int,
                        required=True,
                        help="size of generated images of shape (size, size, 3)")

    parser.add_argument("--tbs",
                        type=int,
                        required=True,
                        help="train batch size")

    parser.add_argument("--ebs",
                        type=int,
                        required=True,
                        help="eval batch size")

    parser.add_argument("--training_steps",
                        type=int,
                        default=0,
                        help="number of training_steps")

    parser.add_argument("--lr",
                        type=float,
                        required=True,
                        help="learning rate")

    parser.add_argument("--lr_warmup_steps",
                        type=int,
                        required=True,
                        help="learning rate warmup steps")

    parser.add_argument("--save_image_steps",
                        type=int,
                        required=True,
                        help="samples and saves sample images every arg steps")

    parser.add_argument("--save_model_steps",
                        type=int,
                        required=True,
                        help="every arg steps save the model")

    parser.add_argument("--train_timesteps",
                        type=int,
                        required=True,
                        help="Number of noising steps during training")

    parser.add_argument("--pretrained_model",
                        type=str,
                        default="None",
                        help="Pretrained model name from hf, if None train from scratch")

    parser.add_argument("--gpu_id",
                        type=int,
                        required=True,
                        help="gpu_id")

    args = parser.parse_args()

    cfg["GPU_ID"] = args.gpu_id
    cfg["DATASET_PATH"] = args.dataset_path
    cfg["DATASET_NAME"] = args.dataset_name
    cfg["IMAGE_SIZE"] = args.im_size
    cfg["TRAIN_BATCH_SIZE"] = args.tbs
    cfg["EVAL_BATCH_SIZE"] = args.ebs
    cfg["TRAINING_STEPS"] = args.training_steps
    cfg["LEARNING_RATE"] = args.lr
    cfg["LR_WARMUP_STEPS"] = args.lr_warmup_steps
    cfg["SAVE_IMAGE_STEPS"] = args.save_image_steps
    cfg["SAVE_MODEL_STEPS"] = args.save_model_steps
    cfg["TRAIN_TIMESTEPS"] = args.train_timesteps
    cfg["PRETRAINED_MODEL"] = args.pretrained_model
