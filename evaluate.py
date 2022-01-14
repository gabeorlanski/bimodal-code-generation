import logging
import hydra
import yaml
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM

from src.config import get_device_from_cfg, load_model_from_cfg, merge_configs
from src.evaluation import evaluate_model

logger = logging.getLogger(__name__)

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


@hydra.main(config_path="conf", config_name="eval_config")
def run(cfg: DictConfig):
    # For some reason it does not clear the log files. So it needs to be done
    # manually.
    for f in Path.cwd().glob("*.log"):
        with f.open("w"):
            pass

    model_path = PROJECT_ROOT.joinpath(cfg["model_path"])
    logger.info(f"Reading config from '{model_path}'")
    train_cfg = OmegaConf.create(
        yaml.load(
            model_path.joinpath(".hydra", "config.yaml").open("r", encoding="utf-8"),
            yaml.Loader,
        )
    )
    logger.info(f"Evaluating '{train_cfg['name']}' on '{cfg['task']['name']}'")

    merged_cfg = merge_configs(cfg, train_cfg)
    model = load_model_from_cfg(merged_cfg, get_device_from_cfg(merged_cfg))
    evaluate_model(cfg, train_cfg=train_cfg, model=model)

    logger.info("Finished Evaluation")


if __name__ == "__main__":
    run()
