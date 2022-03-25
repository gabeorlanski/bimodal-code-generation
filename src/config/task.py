"""
Config related util functions.
"""
import inspect
import os
from copy import deepcopy
from typing import List, Dict, Tuple, Callable
import logging
from functools import partial

import yaml
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from jinja2 import Environment
from tio import Task, Metric, Preprocessor, Postprocessor
from src.common import PROJECT_ROOT

logger = logging.getLogger(__name__)
__all__ = [
    "load_processors_from_cfg",
    "load_task_from_cfg",
    "load_tokenizer_from_cfg",
    "get_prompts_from_cfg",
]

PROMPT_REGISTRY = {}


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

    preprocessor_list = list(cfg.get('preprocessors', []))
    postprocessor_list = list(cfg.get('postprocessors', []))

    task_preprocessors = list(cfg.task.get('preprocessors', []))
    task_postprocessors = list(cfg.task.get('postprocessors', []))
    logger.info(f"{len(task_preprocessors)} task preprocessors")
    logger.info(f"{len(task_postprocessors)} task postprocessors")

    model_type_preprocessors = list(cfg.get('model_type', {}).get('preprocessors', []))
    model_type_postprocessors = list(cfg.get('model_type', {}).get('postprocessors', []))
    logger.info(
        f"Found {len(model_type_preprocessors)} preprocessors specific to the model type"
    )
    logger.info(
        f"Found {len(model_type_postprocessors)} postprocessors specific to the model type"
    )
    if cfg.task.get('override_preprocessors', False):
        logger.warning("Overriding preprocessors with task processors.")
        preprocessor_list = task_preprocessors
    else:
        logger.info('Using all preprocessors')
        preprocessor_list = preprocessor_list + task_preprocessors + model_type_preprocessors

    if cfg.task.get('override_postprocessors', False):
        logger.warning("Overriding postprocessors with task postprocessors.")
        postprocessor_list = task_postprocessors
    else:
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
        tokenizer_kwargs (Dict): Keyword arguments for the tokenizer.

    Returns:
        Task: The created task object.
    """
    logger.info(f"Initializing task registered to name '{cfg['task']['name']}'")
    preprocessors, postprocessors = load_processors_from_cfg(cfg)
    metrics = []
    metrics_to_create = []
    if 'metrics' in cfg:
        metrics_to_create = set(OmegaConf.to_object(cfg.get('metrics')))
    if 'metrics' in cfg.task:
        metrics_to_create.update(OmegaConf.to_object(cfg.task.get('metrics')))
    logger.info(f"Metrics are {list(cfg.get('metrics', []))}")

    for metric in metrics_to_create:
        if isinstance(metric, dict):
            metric_name, metric_dict = list(metric.items())
        else:
            metric_name = metric
            metric_dict = {}
        metrics.append(Metric.from_dict(metric_name, metric_dict))

    task_sig = set(inspect.signature(Task).parameters)
    cls_sig = set(inspect.signature(Task.by_name(cfg['task']['name'])).parameters)
    additional_kwargs = OmegaConf.to_object(cfg.task.get('params')) if 'params' in cfg.task else {}
    additional_kwargs.update({
        k: v for k, v in cfg.task.items()
        if k in cls_sig.difference(task_sig)
    })

    return Task.get_task(
        name=cfg["task"]["name"],
        tokenizer=load_tokenizer_from_cfg(cfg, tokenizer_kwargs),
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_fns=metrics,
        split_mapping=cfg.task.get('split_mapping', {}),
        additional_kwargs=additional_kwargs
    )


def load_tokenizer_from_cfg(cfg, tokenizer_kwargs=None):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    return AutoTokenizer.from_pretrained(
        cfg['model'],
        use_fast=os.environ.get('DISABLE_FAST_TOK', 'false') != 'true',
        **tokenizer_kwargs
    )


def apply_prompts(input_kwargs, prompts) -> str:
    inputs = deepcopy(input_kwargs)

    if 'input_sequence' not in inputs:
        raise KeyError("Missing key 'input_sequence'")

    for prompt in prompts:
        if set(inputs) & set(prompt['flags']):
            raise KeyError(f"Found conflict input keys {set(inputs) & set(prompt['flags'])} "
                           f"with prompt {prompt['path']}")

        prompt_kwargs = {**inputs, **prompt['params']}
        prompt_kwargs.update(prompt['flags'])
        inputs['input_sequence'] = PROMPT_REGISTRY[prompt['name']].render(**prompt_kwargs)

    return inputs['input_sequence']


def get_prompts_from_cfg(cfg, jinja_env: Environment) -> Callable:
    if 'prompts' not in cfg or not cfg.prompts or cfg.prompts.get('disable_prompts', False):
        return lambda input_kwargs: input_kwargs['input_sequence']

    if 'file' not in cfg.prompts:
        raise KeyError('Missing yaml prompt file')

    prompts_raw = PROJECT_ROOT.joinpath(cfg.prompts.file)
    logger.info(f"Loading prompt file from {prompts_raw}")
    prompts_raw = yaml.load(
        prompts_raw.read_text(),
        yaml.Loader
    )

    if 'pipe' not in cfg.prompts:
        raise KeyError("Missing 'pipe' key in prompts")

    prompts = []
    global_params = OmegaConf.to_object(cfg.prompts['params']) if 'params' in cfg.prompts else {}
    global_flags = OmegaConf.to_object(cfg.prompts['flags']) if 'flags' in cfg.prompts else {}
    logger.info(f"Found {len(cfg.prompts)} prompt{'s' if len(cfg.prompts) > 1 else ''}")
    for prompt in cfg.prompts.pipe:
        if not isinstance(prompt, str):
            prompt_name = prompt['name']
        else:
            prompt_name = prompt
        logger.debug(f"Loading prompt from {prompt_name}")
        if prompt_name not in prompts_raw:
            raise KeyError(f"No prompt with name {prompt_name}")
        prompt_params = deepcopy(prompts_raw[prompt_name].get('params', {}))
        prompt_params.update(global_params)
        prompt_flags = deepcopy(prompts_raw[prompt_name].get('flags', {}))
        prompt_flags.update(global_flags)
        PROMPT_REGISTRY[prompt_name] = jinja_env.from_string(prompts_raw[prompt_name]['template'])
        prompts.append({
            "name"  : prompt_name,
            "params": prompt_params,
            "flags" : prompt_flags
        })

    return partial(apply_prompts, prompts=prompts)
