import logging
import math
from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from dataclasses import dataclass
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from src.common.util import get_world_size

__all__ = [
    "get_training_args_from_cfg",
    "TrainingArguments",
    "get_steps_from_training_args",
    "get_lr_scheduler"
]

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    lr_power = 2.0
    end_lr = 1e-16
    num_cycles = 1


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

    batch_size = training_args.pop("batch_size", None)
    if batch_size:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size
    return TrainingArguments(**training_args)


def get_steps_from_training_args(
        train_args: TrainingArguments, train_data
) -> Tuple[int, int]:
    if train_args.max_steps > 0:
        total_steps = train_args.max_steps
    else:
        # Have to account for when distributed, batch size is n_gpu*batch size.
        effective_batch_size = train_args.per_device_train_batch_size
        if get_world_size() > 0:
            effective_batch_size *= get_world_size()

        steps_per_epoch = math.ceil(
            len(train_data) / effective_batch_size
        )
        steps_per_epoch = math.ceil(steps_per_epoch / train_args.gradient_accumulation_steps)
        total_steps = int(steps_per_epoch * train_args.num_train_epochs)


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
