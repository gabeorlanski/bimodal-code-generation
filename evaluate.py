import logging
import hydra
import yaml
from hydra.core.config_store import ConfigStore
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM

from src.config import setup_config_store
from src.common.config import get_device_from_cfg
from src.data import Task, Preprocessor, Postprocessor
from src.training import train_model
from src.evaluation import evaluate_model

logger = logging.getLogger(__name__)

cs = setup_config_store()

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

    logger.info(f"Loading model from '{model_path.joinpath('best_model.bin')}'")
    model = AutoModelForSeq2SeqLM.from_pretrained(train_cfg["model"])
    model.load_state_dict(torch.load(model_path.joinpath("best_model.bin")))
    model = model.to(get_device_from_cfg(train_cfg))
    evaluate_model(cfg, train_cfg=train_cfg, model=model)

    logger.info("Finished Evaluation")


if __name__ == "__main__":
    run()
