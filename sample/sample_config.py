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
    parser.add_argument("--save_dir",
                        type=str,
                        required=True,
                        help="saved images' directory")
    parser.add_argument("--dataset_name",
                        type=str,
                        required=True,
                        help="dataset name")
    parser.add_argument("--gpu_id",
                        type=str,
                        required=True,
                        help="CUDA supported gpu id")
    parser.add_argument("--gen_to_orig_ratio",
                        type=str,
                        required=True,
                        help="ratio of generated images to original images")
    args = parser.parse_args()

    cfg["BATCH_SIZE"] = args.bs
    cfg["DATASET_NAME"] = args.dataset_name
    cfg["GPU_ID"] = args.gpu_id
    cfg["SAVE_DIR"] = args.save_dir
    cfg["GTO_RATIO"] = args.gen_to_orig_ratio
    cfg["MODELS_DIR"] = "/auto/k2/bguler/scripts/DDAN/" + \
                        f"chest_xray/src/DDPM/hf_ddpm/{args.dataset_name}"
