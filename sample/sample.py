import torch

from sample_helpers import sample_dataset
import sample_config


if __name__ == "__main__":
    gpu_id = sample_config.cfg["GPU_ID"]
    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else "cpu"
    sample_dataset(sample_config.cfg["SAVE_DIR"], device)
