"""
Config related util functions.
"""
import os
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

    preprocessor_list = cfg.get('preprocessors', [])
    postprocessor_list = cfg.get('postprocessors', [])

    task_preprocessors = cfg.task.get('preprocessors', [])
    task_postprocessors = cfg.task.get('postprocessors', [])
    logger.info(f"{len(task_preprocessors)} task preprocessors")
    logger.info(f"{len(task_postprocessors)} task postprocessors")

    model_type_preprocessors = cfg.get('model_type', {}).get('preprocessors', [])
    model_type_postprocessors = cfg.get('model_type', {}).get('postprocessors', [])
    logger.info(
        f"Found {len(model_type_preprocessors)} preprocessors specific to the model type"
    )
    logger.info(
        f"Found {len(model_type_postprocessors)} postprocessors specific to the model type"
    )
    if cfg.task.get('override_processors', False):
        logger.warning("Overriding processors with task processors.")
        preprocessor_list = task_preprocessors
        postprocessor_list = task_postprocessors
    else:
        logger.info('Using all processors')
        preprocessor_list = preprocessor_list + task_preprocessors + model_type_preprocessors
        postprocessor_list = postprocessor_list + task_postprocessors + model_type_postprocessors

    preprocessors = _create_processors(Preprocessor, preprocessor_list)
    postprocessors = _create_processors(Postprocessor, postprocessor_list)

    logger.info(f"{len(preprocessors)} total preprocessors")
    logger.info(f"{len(postprocessors)} total postprocessors")

    return preprocessors, postprocessors


def load_task_from_cfg(
        cfg: DictConfig,
        tokenizer_kwargs=None
) -> Task:
    """
    Create a Task from a cfg

    Args:
        cfg (DictConfig): The config to use.

    Returns:
        Task: The created task object.
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    logger.info(f"Initializing task registered to name '{cfg['task']['name']}'")
    preprocessors, postprocessors = load_processors_from_cfg(cfg)
    logger.info(f"Metrics are {list(cfg.get('metrics', []))}")
    metrics = []
    for metric in cfg.get('metrics'):
        if isinstance(metric, dict):
            metric_name, metric_dict = list(metric.items())
        else:
            metric_name = metric
            metric_dict = {}
        metrics.append(Metric.from_dict(metric_name, metric_dict))

    return Task.get_task(
        name=cfg["task"]["name"],
        tokenizer=AutoTokenizer.from_pretrained(
            cfg['model'],
            use_fast=os.environ.get('DISABLE_FAST_TOK', 'false') != 'true',
            **tokenizer_kwargs
        ),
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_fns=metrics,
        additional_kwargs=cfg.get('task_kwargs', {}),
    )
