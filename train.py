import logging
import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import os
import random

from src.data import load_task_from_cfg
from src.training import train_model
from src.tracking import setup_tracking_env_from_cfg

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
    random.seed(cfg["seed"])
    np.random.seed(cfg["numpy_seed"])
    torch.manual_seed(torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = train_model(cfg, PROJECT_ROOT.joinpath(cfg["data_path"]))

    logger.info(f"Saving best model to {Path.cwd()}")
    torch.save(model.state_dict(), Path.cwd().joinpath("best_model.bin"))
    logger.info("Training Finished")


if __name__ == "__main__":
    run()
