"""
Config related util functions.
"""
import torch
import logging
from transformers import (AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig, PreTrainedModel)
from omegaconf import DictConfig

from src.common import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "get_device_from_cfg",
    "load_model_from_cfg"
]

MODEL_TYPE_TO_CLS = {
    "seq2seq"  : AutoModelForSeq2SeqLM,
    "lm": AutoModelForCausalLM
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


def load_model_from_cfg(cfg: DictConfig) -> PreTrainedModel:
    logger.info(f"Loading model '{cfg['model']}' of type "
                f"'{cfg['objective']}'")
    model_cls = MODEL_TYPE_TO_CLS[cfg['objective']]

    logger.info(f"Loading '{cfg['model']}' from HuggingFace'")
    model = model_cls.from_pretrained(cfg['model'])
    if cfg['is_checkpoint']:
        logger.info(f"Loading checkpoint 'best_model.bin` from '{cfg['model_path']}'")
        model_path = PROJECT_ROOT.joinpath(cfg["model_path"])
        model.load_state_dict(torch.load(model_path.joinpath("best_model.bin")))

    return model
