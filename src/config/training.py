import collections
import json
import logging
import math
import sys
from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from dataclasses import dataclass
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from src.common import PROJECT_ROOT
from src.common.util import get_world_size

__all__ = [
    "get_training_args_from_cfg",
    "TrainingArguments",
    "get_steps_from_training_args",
    "get_lr_scheduler"
]

logger = logging.getLogger(__name__)


@dataclass()
class TrainingArguments(Seq2SeqTrainingArguments):
    lr_power: float = 2.0
    end_lr: float = 1e-16
    num_cycles: int = 1
    use_8bit_adam: bool = False


def get_training_args_from_cfg(cfg: DictConfig) -> TrainingArguments:
    """
    Get the training arguments to create a HuggingFace training arguments
    objects from the passed in config.

    Special Keys:
        ``batch_size``: will be used for the keys ``per_device_train_batch_size``
        and ``per_device_eval_batch_size``

    Args:
        cfg (DictConfig): The OmegaConf config.

    Returns:
        TrainingArguments: The processed training arguments.
    """

    training_args = OmegaConf.to_object(cfg["training"])

    if 'deepspeed' in training_args:
        training_args['deepspeed'] = json.loads(
            PROJECT_ROOT.joinpath(training_args['deepspeed']).read_text())


    if 'report_to' not in training_args:
        training_args['report_to']=["none"]
    batch_size = training_args.pop("batch_size", None)
    if batch_size:
        if 'per_device_train_batch_size' not in training_args:
            training_args["per_device_train_batch_size"] = batch_size
        if 'per_device_eval_batch_size' not in training_args:
            training_args["per_device_eval_batch_size"] = batch_size
    return TrainingArguments(**training_args)


def get_steps_from_training_args(
        train_args: TrainingArguments, train_data
) -> Tuple[int, int]:
    train_dataset_is_sized = isinstance(train_data, collections.abc.Sized)
    effective_batch_size = (
            train_args.train_batch_size
            * train_args.gradient_accumulation_steps
            * train_args.world_size
    )
    if train_dataset_is_sized and train_args.max_steps < 0:
        num_update_steps_per_epoch = len(train_data) // effective_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        total_steps = int(num_update_steps_per_epoch * train_args.num_train_epochs)
    else:
        total_steps = train_args.max_steps

    if train_args.warmup_steps > 0:
        warmup_steps = train_args.warmup_steps
    else:
        warmup_steps = max(train_args.warmup_ratio, 0) * total_steps

    return int(total_steps), int(warmup_steps)


def get_lr_scheduler(train_args: TrainingArguments, optimizer,
                     total_steps, warmup_steps):
    scheduler_name = train_args.lr_scheduler_type.name.lower()
    logger.debug(f"Looking for scheduler {scheduler_name}")
    scheduler = TYPE_TO_SCHEDULER_FUNCTION[train_args.lr_scheduler_type]
    if scheduler_name == "polynomial":
        logger.info(
            f"Using polynomial with end_lr={train_args.end_lr} and power={train_args.lr_power}"
        )
        return scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            lr_end=train_args.end_lr,
            power=train_args.lr_power
        )
    elif 'cosine' in scheduler_name:
        if scheduler_name == 'cosine':
            num_cycles = 0.5
        else:
            num_cycles = train_args.num_cycles
        logger.info(f"Using cosine with num_cylces={num_cycles}")
        return scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles
        )

    # If not found return a linear scheduler
    logger.info(f"Using {scheduler_name} with default arguments")
    return scheduler(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
