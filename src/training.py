from pathlib import Path

import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import logging
from omegaconf import OmegaConf, DictConfig

from tio import Task, load_task_from_cfg

from src.common.config import get_device_from_cfg
from src.trainer import CustomTrainer

logger = logging.getLogger(__name__)


def get_training_args_from_config(cfg: DictConfig) -> Seq2SeqTrainingArguments:
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
    logger.info("Training Arguments:")
    for k, v in training_args.items():
        logger.info(f"\t{k:<30} = {v}")

    batch_size = training_args.pop("batch_size", None)
    if batch_size:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size
    return Seq2SeqTrainingArguments(**training_args)


# TODO(gabeorlanski): Add in parallel support
def train_model(cfg: DictConfig, data_path: Path):
    """
    Train a model with a given reader.

    Args:
        cfg (DictConfig): The config.
        data_path (Path): Path to the data folder

    Returns:
        The best model.
    """
    task: Task = load_task_from_cfg(cfg)
    tokenizer = task.tokenizer

    train_path = data_path.joinpath(cfg["task"]["paths"]["train"])
    logger.info(f"Reading training data is from '{train_path}'")
    train_data = task.get_split("train", num_procs=cfg.get('num_proc', 1), set_format="torch")

    validation_path = data_path.joinpath(cfg["task"]["paths"]["validation"])
    logger.info(f"Reading training data is from '{validation_path}'")
    validation_data = task.get_split(
        "validation", num_procs=cfg.get('num_proc', 1), set_format="torch"
    )

    logger.info(f"{len(train_data)} training samples")
    logger.info(f"{len(validation_data)} validation samples")

    device = get_device_from_cfg(cfg)
    logger.info(f"Using device {device}")

    logger.debug("Loading Model")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]).to(device)

    logger.debug("Initializing trainer")
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id,
    )
    trainer = CustomTrainer(
        cfg=cfg,
        model=model,
        args=get_training_args_from_config(cfg),
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collator,
        compute_metrics=lambda preds: task.evaluate(
            *task.postprocess_np(
                preds.predictions, preds.label_ids
            )
        )
    )
    trainer.train()
    return model
