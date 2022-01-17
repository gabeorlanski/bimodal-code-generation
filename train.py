import logging
import sys

import hydra
import torch
from omegaconf import DictConfig, open_dict
from pathlib import Path
import numpy as np
import os
import random
import shutil

from src.training import train_model
from src.config import setup_tracking_env_from_cfg

logger = logging.getLogger(__name__)

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


@hydra.main(config_path="conf", config_name="train_config")
def run(cfg: DictConfig):
    # For some reason it does not clear the log files. So it needs to be done
    # manually.
    for f in Path.cwd().glob("*.log"):
        with f.open("w"):
            pass

    logger.info("Starting Train")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    setup_tracking_env_from_cfg(cfg)

    if "training" not in cfg:
        raise KeyError("Missing 'training' key in config.")

    seed = cfg["seed"]
    numpy_seed = cfg["numpy_seed"]
    torch_seed = cfg["pytorch_seed"]
    logger.info(f"Seed={seed}")
    logger.info(f"NumPy Seed={numpy_seed}")
    logger.info(f"Torch Seed={torch_seed}")

    with open_dict(cfg):
        cfg.training.local_rank=int(os.environ.get('LOCAL_RANK',-1))

    random.seed(cfg["seed"])
    np.random.seed(cfg["numpy_seed"])
    torch.manual_seed(torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    train_model(cfg)
    logger.info("Training Is Done")


if __name__ == "__main__":
    run()
