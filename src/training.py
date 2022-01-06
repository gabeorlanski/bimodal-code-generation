from pathlib import Path
from typing import Dict

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import logging
import torch
from omegaconf import OmegaConf, DictConfig

from src.data import DatasetReader
from src.common.config import get_device_from_cfg
from src.evaluation.evaluator import Evaluator

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

    training_args = OmegaConf.to_object(cfg['training'])
    logger.info("Training Arguments:")
    for k, v in training_args.items():
        logger.info(f"\t{k:<24}={v}")

    batch_size = training_args.pop('batch_size', None)
    if batch_size:
        training_args['per_device_train_batch_size'] = batch_size
        training_args['per_device_eval_batch_size'] = batch_size
    return Seq2SeqTrainingArguments(**training_args)


# TODO(gabeorlanski): Add in parallel support
def train_model(cfg: DictConfig, data_path: Path, reader: DatasetReader):
    """
    Train a model with a given reader.

    Args:
        cfg (DictConfig): The config.
        data_path (Path): Path to the data folder.
        reader (DatasetReader): The dataset reader to use.
    """
    train_path = data_path.joinpath(cfg['dataset']['train_path'])
    logger.info(f"Reading training data is from '{train_path}'")
    train_raw, train_data = reader.read_data(train_path, set_format="torch")

    validation_path = data_path.joinpath(cfg['dataset']['validation_path'])
    logger.info(f"Reading training data is from '{validation_path}'")
    validation_raw, validation_data = reader.read_data(validation_path, set_format="torch")

    logger.info(f"{len(train_data)} training samples")
    logger.info(f"{len(validation_data)} validation samples")

    device = get_device_from_cfg(cfg)
    logger.info(f"Using device {device}")

    logger.debug("Loading Model")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg['model']).to(device)
    evaluator = Evaluator(reader.tokenizer, cfg.get('metrics', []))

    logger.debug("Initializing trainer")
    collator = DataCollatorForSeq2Seq(
        reader.tokenizer,
        model,
        return_tensors='pt',
        label_pad_token_id=reader.tokenizer.pad_token_id
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=get_training_args_from_config(cfg),
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collator,
        compute_metrics=evaluator
    )
    trainer.train()
