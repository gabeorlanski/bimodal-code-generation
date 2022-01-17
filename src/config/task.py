"""
Config related util functions.
"""
from typing import List, Dict, Tuple, Callable
import logging
from functools import partial
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf

from tio import Task, Metric, Preprocessor, Postprocessor

logger = logging.getLogger(__name__)
__all__ = [
    "load_processors_from_cfg",
    "load_task_from_cfg"
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
    logger.info("Loading processors")

    def _create_processors(processor_cls, processor_list):

        return [
            partial(processor_cls.by_name(name), **func_kwargs)
            for name, func_kwargs in map(lambda d: next(iter(d.items())), processor_list)
        ]

    preprocessors = _create_processors(Preprocessor, cfg.get('preprocessors', []))
    postprocessors = _create_processors(Preprocessor, cfg.get('postprocessors', []))

    model_type_preprocessors = cfg.get('model_type', {}).get('preprocessors', [])
    model_type_postprocessors = cfg.get('model_type', {}).get('postprocessors', [])
    if model_type_preprocessors:
        logger.info(
            f"Found {len(model_type_preprocessors)} preprocessors specific to the model type"
        )
    if model_type_postprocessors:
        logger.info(
            f"Found {len(model_type_postprocessors)} postprocessors specific to the model type"
        )

    preprocessors.extend(_create_processors(Preprocessor, model_type_preprocessors or []))
    postprocessors = _create_processors(
        Postprocessor,
        model_type_postprocessors or []) + postprocessors

    logger.info(f"{len(preprocessors)} total preprocessors")
    logger.info(f"{len(postprocessors)} total postprocessors")

    return preprocessors, postprocessors


def load_task_from_cfg(
        cfg: DictConfig
) -> Task:
    """
    Create a Task from a cfg

    Args:
        cfg (DictConfig): The config to use.

    Returns:
        Task: The created task object.
    """
    logger.info(f"Initializing task registered to name '{cfg['task']['name']}'")
    preprocessors, postprocessors = load_processors_from_cfg(cfg)
    logger.info(f"Metrics are {cfg.get('metrics', [])}")
    metrics = [Metric.by_name(metric) for metric in cfg.get('metrics', [])]

    return Task.get_task(
        name=cfg["task"]["name"],
        tokenizer=AutoTokenizer.from_pretrained(cfg['model'],use_fast=False),
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_fns=metrics,
        additional_kwargs=cfg.get('task_kwargs', {}),
    )
