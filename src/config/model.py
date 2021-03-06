"""
Config related util functions.
"""
from pathlib import Path
from typing import Callable, Tuple

import torch
import logging
from transformers import (
    AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig, PreTrainedModel
)
from omegaconf import DictConfig

from src.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "get_device_from_cfg",
    "load_model_from_cfg"
]

MODEL_TYPE_TO_CLS = {
    "seq2seq": AutoModelForSeq2SeqLM,
    "lm"     : AutoModelForCausalLM
}


def get_device_from_cfg(cfg: DictConfig) -> torch.device:
    """
    Get the torch device from a config. Assumes that there is a ``device``
    key at the top level.

    Args:
        cfg: The config.

    Returns:
        The torch device.
    """

    # Cast to a string to guarantee that it will be one type rather than mixed
    # ints and strings.
    device_str = str(cfg.get("device", "cpu"))
    if device_str == "cpu" or device_str == "-1":
        return torch.device("cpu")
    else:
        return torch.device(f'cuda{":" + device_str if device_str != "cuda" else ""}')


def load_model_from_cfg(
        cfg: DictConfig,
        model_path: Path = None
) -> Tuple[Callable, PreTrainedModel]:
    logger.info(f"Loading model '{cfg['model']}' of type "
                f"'{cfg['objective']}'")
    model_cls = MODEL_TYPE_TO_CLS[cfg['objective']]

    logger.info(f"Loading '{cfg['model']}' from HuggingFace'")
    if cfg['is_checkpoint']:
        if model_path is None:
            logger.info('USING MODEL PATH')
            model_path = PROJECT_ROOT.joinpath(cfg["model_path"])

        logger.info(f"Loading checkpoint from {model_path}")
        model = model_cls.from_pretrained(str(model_path.resolve().absolute()))
    else:
        logger.info('NOT USING CHECKPOINT')
        if cfg.get('from_scratch', False):
            logger.info(f"INITIALIZING MODEL FROM SCRATCH")
            model = model_cls.from_config(AutoConfig.from_pretrained(cfg['model']))
        else:
            model = model_cls.from_pretrained(cfg['model'])

    if 'generation' in cfg:
        for k, v in cfg.generation.items():
            if hasattr(model.config, k):
                setattr(model.config, k, v)
    return model_cls, model
