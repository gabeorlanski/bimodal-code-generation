"""
Data Utility functions for dealing with configs
"""
import logging
from functools import partial
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, List, Callable
from transformers import AutoTokenizer

from src.data.processor import Preprocessor, Postprocessor
from src.data.task import Task

logger = logging.getLogger(__name__)

__all__ = [
    "load_task_from_cfg",
    "load_processors_from_cfg"
]


def load_processors_from_cfg(cfg: DictConfig) -> Tuple[List[Callable], List[Callable]]:
    """
    Create the pre- and post- processors from a given config.

    Args:
        cfg (DictConfig): The config to use.

    Returns:
        Tuple[List[Callable], List[Callable]]: The created preprocessors and
            postprocessors.
    """
    logger.debug("Loading processors")
    preprocessors = []
    postprocessors = []

    if cfg.get("preprocessors") is not None:
        logger.debug("Preprocessors found")
        preprocessors = [
            partial(Preprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["preprocessors"].items()
        ]
    logger.info(f"{len(preprocessors)} preprocessors found")

    if cfg.get("postprocessors") is not None:
        logger.debug("Postprocessors found")
        postprocessors = [
            partial(Postprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["postprocessors"].items()
        ]
    logger.info(f"{len(postprocessors)} postprocessors found")
    return preprocessors, postprocessors


def load_task_from_cfg(cfg: DictConfig) -> Task:
    """
    Create a Task from a cfg

    Args:
        cfg (DictConfig): The config to use.
        tokenizer (PreTrainedTokenizer): The tokenizer to be passed to the task.

    Returns:
        Task: The created task object.
    """
    logger.info(f"Initializing task registered to name '{cfg['task']['name']}'")
    task_cls = Task.by_name(cfg["task"]["name"])
    cfg_dict = OmegaConf.to_object(cfg["task"])
    preprocessors, postprocessors = load_processors_from_cfg(cfg)
    return task_cls(
        tokenizer=AutoTokenizer.from_pretrained(cfg['model']),
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        **cfg_dict.get('args', {}),
    )
