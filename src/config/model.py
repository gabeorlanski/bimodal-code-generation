"""
Config related util functions.
"""
from typing import List, Dict, Tuple, Callable
import torch
from copy import deepcopy
import logging
from functools import partial
from transformers import AutoTokenizer
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

__all__ = [
    "get_device_from_cfg"
]


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
