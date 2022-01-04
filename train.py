import logging
import hydra
from hydra.core.config_store import ConfigStore
import torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from functools import partial
from transformers import AutoTokenizer
from dataclasses import asdict
import numpy as np
import os
import random

from src.config import setup_config_store
from src.data import DatasetReader, Preprocessor, Postprocessor
from src.training import train_model

logger = logging.getLogger(__name__)

cs = setup_config_store()

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


@hydra.main(config_path='conf', config_name="train_config")
def run(cfg: DictConfig):
    # For some reason it does not clear the log files. So it needs to be done
    # manually.
    for f in Path.cwd().glob('*.log'):
        with f.open('w'):
            pass

    logger.info("Starting Train")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    seed = cfg['seed']
    numpy_seed = cfg['numpy_seed']
    torch_seed = cfg['pytorch_seed']
    logger.info(f"Seed={seed}")
    logger.info(f"NumPy Seed={numpy_seed}")
    logger.info(f"Torch Seed={torch_seed}")
    random.seed(cfg['seed'])
    np.random.seed(cfg['numpy_seed'])
    torch.manual_seed(torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    # Load the preprocessors and postprocessors specified
    logger.debug("Loading processors")
    preprocessors = []
    postprocessors = []

    if cfg.get('preprocessors') is not None:
        logger.debug("preprocessors found")
        preprocessors = [
            partial(Preprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg['preprocessors'].items()
        ]
    logger.info(f"{len(preprocessors)} preprocessors found")

    if cfg.get('postprocessors') is not None:
        logger.debug("postprocessors found")
        postprocessors = [
            partial(Postprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg['postprocessors'].items()
        ]
    logger.info(f"{len(postprocessors)} postprocessors found")

    logger.info(f"Loading tokenizer for '{cfg['model_name']}'")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    # Load the dataset reader
    logger.info(f"Initializing reader registered to name '{cfg['dataset']['name']}'")
    reader_cfg = asdict(OmegaConf.to_object(cfg['dataset']))

    # Dont want these for kwargs of the reader
    reader_cfg.pop('train_path')
    reader_cfg.pop('validation_path')

    reader = DatasetReader.by_name(reader_cfg.pop('name'))
    reader = reader(
        tokenizer=tokenizer,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        **reader_cfg
    )

    train_model(PROJECT_ROOT.joinpath(cfg['data_path']), cfg, reader)

    logger.info("Training Finished")


if __name__ == '__main__':
    run()
