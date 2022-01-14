import logging
import hydra
import torch
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import os
import random

from src.training import train_lm
from src.tracking import setup_tracking_env_from_cfg

logger = logging.getLogger(__name__)

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


@hydra.main(config_path="conf", config_name="train_lm_config")
def run(cfg: DictConfig):
    # For some reason it does not clear the log files. So it needs to be done
    # manually.
    for f in Path.cwd().glob("*.log"):
        with f.open("w"):
            pass

    logger.info("Starting Train")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if cfg.model_type.name != 'causal_lm':
        logger.error("'train_lm' requires using 'model_type' of 'causal_lm'.")
        raise ValueError("Invalid model type.")

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

    model = train_lm(cfg)

    logger.info(f"Saving best model to {Path.cwd()}")
    torch.save(model.state_dict(), Path.cwd().joinpath("best_model.bin"))
    logger.info("Training Finished")


if __name__ == "__main__":
    run()
