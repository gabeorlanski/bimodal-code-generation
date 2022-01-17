from pathlib import Path

import logging
from omegaconf import OmegaConf, DictConfig
from transformers import DataCollatorForSeq2Seq
from transformers import AdamW, get_scheduler
import torch
from tio import Task
from src import config
from src.common import get_stats_from_list
from src.data import langauge_modeling
from src.trainer import Trainer
from src.old_trainer import CustomTrainer
from functools import partial
from datasets import set_caching_enabled, Dataset

logger = logging.getLogger(__name__)


def setup_seq2seq(cfg, task):
    logger.info("Getting train data")
    train_data = task.get_split("train", num_procs=cfg.get('num_proc', 1))
    logger.info("Getting validation data")
    validation_data = task.get_split(
        "validation", num_procs=cfg.get('num_proc', 1)
    )

    logger.info(f"{len(train_data)} training samples")
    logger.info(f"{len(validation_data)} validation samples")
    for name, ds in [('Train', train_data), ("Validation", validation_data)]:
        logger.info(f"Stats for {name} data:")
        input_lens = list(map(len, ds['input_ids']))
        target_lens = list(map(len, ds['labels']))
        logger.info(f"\t{name} Inputs:")
        for metric, value in get_stats_from_list(input_lens).items():
            logger.info(f"\t\t{metric} = {value:0.2f}")
        logger.info(f"\t{name} Labels:")
        for metric, value in get_stats_from_list(target_lens).items():
            logger.info(f"\t\t{metric} = {value:0.2f}")

    return train_data, validation_data, evaluate


def setup_lm(cfg, task):
    # Add a preprocessor to concat the inputs and labels.
    def concat(ex):
        ex['input_sequence'] = f"{ex['input_sequence']} {ex['target']}"
        ex['target'] = ex['input_sequence']
        return ex

    task.preprocessors.append(concat)

    group_texts = partial(
        langauge_modeling.group_texts,
        tokenizer=task.tokenizer,
        seq_length=cfg.data_args.seq_length
    )

    def make_split(split_name):
        split_data = task.preprocess(split_name, num_procs=cfg.get('num_proc', 1))
        split_data = Dataset.from_dict(group_texts(split_data['input_sequence']))
        return split_data

    logger.info("Getting train data")
    train_data = make_split('train')

    logger.info("Getting validation data")
    validation_data = make_split('validation')
    return train_data, validation_data, evaluate


# TODO(gabeorlanski): Add in parallel support
def train_model(cfg: DictConfig):
    """
    Train a model with a given task.

    Args:
        cfg (DictConfig): The config.

    Returns:
        The best model.
    """

    device = config.get_device_from_cfg(cfg)
    logger.info(f"Using device {device}")

    set_caching_enabled(not cfg.get('disable_cache', False))

    logger.debug("Loading Model")
    model_cls, model = config.load_model_from_cfg(cfg)

    task: Task = config.load_task_from_cfg(cfg)
    tokenizer = task.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    if cfg.objective == 'seq2seq':
        logger.info(f"Setting Up Seq2Seq Objective")
        train_data, validation_data, evaluate_fn = setup_seq2seq(
            cfg,
            task
        )
    elif cfg.objective == "lm":
        logger.info("Setting up the LM objective")
        train_data, validation_data, evaluate_fn = setup_lm(
            cfg,
            task
        )
    else:
        logger.error(f"{cfg.objective} is not a valid objective")
        raise ValueError("Invalid Objective")

    logger.debug("Initializing trainer")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
        pad_to_multiple_of=2,
        return_tensors='pt'
    )

    trainer= CustomTrainer(
        cfg,
        model=model,
        args=config.get_training_args_from_cfg(cfg),
        eval_dataset=validation_data,
        train_dataset=train_data,
        data_collator=collator,

    )
    trainer.train()

    if cfg.training.local_rank <= 0:
        logger.info(f"Saving best model to {Path().joinpath('best_model.bin')}")
        torch.save(trainer.model.state_dict(),Path().joinpath('best_model.bin'))

    return


def evaluate(args, model, data_loader, device):
    model.eval()
    losses = []
    for step, batch in enumerate(data_loader):
        local_batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(
                local_batch['input_ids'],
                labels=local_batch.get('labels', local_batch['input_ids'])
            )
        loss = outputs.loss.repeat(args.eval_batch_size)
        losses.append(loss)
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return {'loss': loss.item(), 'perplexity': perplexity.item()}
