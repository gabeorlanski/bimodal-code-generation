import math
from pathlib import Path

import logging
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import (
    DataCollatorForSeq2Seq, EvalPrediction, AdamW, get_polynomial_decay_schedule_with_warmup
)
from functools import partial
from datasets import set_caching_enabled, Dataset
import torch
from tio import Task
from src import config
from src.common import get_stats_from_list
from src.data import langauge_modeling
from src.training.trainer import CustomTrainer

logger = logging.getLogger(__name__)


def evaluate_seq2seq(eval_predictions: EvalPrediction, task: Task):
    preds, targets = eval_predictions
    return task.evaluate(
        [[p] for p in task.postprocess(preds)],
        task.postprocess(targets)
    )


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

    return train_data, validation_data, partial(evaluate_seq2seq, task=task)


def setup_lm(cfg, task):
    # Add a preprocessor to concat the inputs and labels.
    def concat(ex):
        ex['input_sequence'] = f"{ex['input_sequence']}{ex['target']}"
        ex['target'] = ex['input_sequence']
        return ex

    task.preprocessors.append(concat)

    group_texts = partial(
        langauge_modeling.group_texts,
        concat_token=task.tokenizer.eos_token_id,
        seq_length=cfg.data_args.seq_length,

    )

    def make_split(split_name):
        split_data = task.get_split(split_name, num_procs=cfg.get('num_proc', 1))
        split_data = Dataset.from_dict(group_texts(split_data['input_ids']))
        return split_data

    logger.info("Getting train data")
    train_data = make_split('train')

    logger.info("Getting validation data")
    validation_data = make_split('validation')
    return train_data, validation_data, None


def train_model(cfg: DictConfig):
    """
    Train a model with a given task.

    Args:
        cfg (DictConfig): The config.

    Returns:
        The best model.
    """

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
        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=2,
            return_tensors='pt',
            label_pad_token_id=tokenizer.pad_token_id
        )
        with open_dict(cfg):
            cfg.training.predict_with_generate = True

        for k, v in cfg.generation.items():
            if k == 'num_return_sequences':
                v_use = 1
            elif k == 'max_length':
                v_use = 512
            else:
                v_use = v
            setattr(model.config, k, v_use)
    elif cfg.objective == "lm":
        logger.info("Setting up the LM objective")

        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token = task.tokenizer.eos_token

        train_data, validation_data, evaluate_fn = setup_lm(
            cfg,
            task
        )
        model.config.eos_token_id = task.tokenizer.eos_token_id
        model.config.pad_token_id = task.tokenizer.pad_token_id
        model.config.bos_token_id = task.tokenizer.bos_token_id or task.tokenizer.eos_token

        collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=1,
            return_tensors='pt'
        )
    else:
        logger.error(f"{cfg.objective} is not a valid objective")
        raise ValueError("Invalid Objective")
    model.resize_token_embeddings(len(tokenizer))

    logger.debug("Initializing trainer")

    train_args = config.get_training_args_from_cfg(cfg)
    if train_args.local_rank <= 0:
        logger.info("Training Arguments:")
        for arg_name in sorted(train_args.to_sanitized_dict()):
            logger.info(
                f"{arg_name:>30} = {getattr(train_args, arg_name)}"
            )

    device = train_args.device
    logger.info(f"Using device {device}")
    trainer = CustomTrainer(
        cfg,
        model=model,
        args=train_args,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        train_dataset=train_data,
        data_collator=collator,
        compute_metrics=evaluate_fn,

    )
    trainer.train()

    if cfg.training.local_rank <= 0:
        logger.info(f"Saving best model to {Path().joinpath('best_model').absolute().resolve()}")
        trainer.model.save_pretrained(Path().joinpath('best_model'))

    return


def get_grouped_params(model, args, no_decay=None):
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{
        "params": params_with_wd, "weight_decay": args.weight_decay
    },
        {"params": params_without_wd, "weight_decay": 0.0}]


def evaluate_lm(args, model, data_loader, device):
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
