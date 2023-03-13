import argparse


# dict to store configuration details
cfg = dict()

# default argument
cfg["NUM_STEPS"] = 1000


def parse_args():
    """Modifies args global var

    parses command line arguments from where this function is called
    global cfg can be accessed from other modules without extra parsing

    :return: None
    """
    global cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs",
                        type=int,
                        required=True,
                        help="sampling batch size")
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="sampling batch size")
    parser.add_argument("--gpu_id",
                        type=str,
                        required=True,
                        help="CUDA supported gpu id")
    parser.add_argument("--gen_to_orig_ratio",
                        type=float,
                        required=True,
                        help="ratio of generated images to original images")
    parser.add_argument("--checkpoint_name",
                        type=str,
                        required=True,
                        help="which checkpoint name")
    parser.add_argument("--checkpoint_step",
                        type=str,
                        required=True,
                        help="which checkpoint to load for each class")
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        choices=["ddpm", "ddim"],
                        help="which sch to use during sampling")
    args = parser.parse_args()

    cfg["BATCH_SIZE"] = args.bs
    cfg["DATASET_PATH"] = args.dataset_path
    cfg["GPU_ID"] = args.gpu_id
    cfg["GTO_RATIO"] = args.gen_to_orig_ratio
    cfg["MODELS_PATH"] = "/auto/k2/bguler/scripts/DDAN/" + \
                         "src/DDPM/hf_ddpm/" + args.checkpoint_name
    cfg["CHECKPOINT_STEP"] = args.checkpoint_step
    cfg["SCHEDULER"] = args.scheduler
