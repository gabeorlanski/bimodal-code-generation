import json
import math
from itertools import chain
from pathlib import Path

import logging
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import (
    DataCollatorForSeq2Seq, EvalPrediction, AdamW
)
from functools import partial
from datasets import set_caching_enabled, Dataset, load_dataset
from datasets.iterable_dataset import iterable_dataset
import torch
from tio import Task
from src import config
from src.common import get_stats_from_list, PROJECT_ROOT, set_global_logging_level
from src.data import langauge_modeling, NON_REGISTERED_TASKS
from src.training.trainer import CustomTrainer
from src.config import get_steps_from_training_args, get_lr_scheduler
from src.data.tensorize import TensorizedTask

logger = logging.getLogger(__name__)

set_global_logging_level(logging.ERROR,
                         ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])


def evaluate_seq2seq(eval_predictions: EvalPrediction, task: Task):
    preds, targets = eval_predictions
    return task.evaluate(
        [[p] for p in task.postprocess(preds)],
        task.postprocess(targets)
    )


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
    logger.info(f"{len(train_data)} total training samples")

    logger.info("Getting validation data")
    validation_data = make_split('validation')
    return train_data, validation_data, None


def setup_pretrain(cfg, tokenizer, train_args):
    concat_delim = tokenizer('\n')['input_ids']
    block_size = cfg.task.sequence_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated = []
        for input_ids, label in zip(examples['input_ids'], examples['labels']):
            concatenated.extend(input_ids + concat_delim + label + [tokenizer.eos_token_id])
        total_length = len(concatenated)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        blocks = [concatenated[i: i + block_size] for i in range(0, total_length, block_size)]

        return {
            "input_ids": blocks,
            "labels"   : blocks.copy()
        }

    # For the HF Trainer, we need the eval set to have a size, so we split up
    # the initialization so one is in streaming mode and the other is not.
    train_dataset = TensorizedTask(
        name=cfg.task.dump_name,
        data_path=PROJECT_ROOT.joinpath(cfg.task.data_path),
        objective=cfg.objective,
        tokenizer=tokenizer,
        sequence_length=cfg.task.sequence_length,
        effective_batch_size=(
                train_args.train_batch_size
                * train_args.gradient_accumulation_steps
                * train_args.world_size
        ),
        max_samples=cfg.task.get('debug_max_samples', -1),
        seed=cfg.seed,
        buffer_size=cfg.get('buffer_size', 25)
    )

    eval_dataset = load_dataset(
        'json',
        data_files={'validation': str(PROJECT_ROOT.joinpath(cfg.task.validation_path))},
        streaming=False
    )['validation']
    eval_dataset = eval_dataset.map(
        group_texts,
        batched=True,
        remove_columns=['attention_mask']
    ).shuffle(seed=cfg.seed)
    return train_dataset, eval_dataset, None


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

    if cfg.task.name in NON_REGISTERED_TASKS:
        tokenizer = config.load_tokenizer_from_cfg(cfg)
        task = None  # type: ignore
    else:
        task: Task = config.load_task_from_cfg(cfg)
        tokenizer = task.tokenizer

    train_args = config.get_training_args_from_cfg(cfg)
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
    elif cfg.objective == 'lm':
        if cfg.task.name in NON_REGISTERED_TASKS:
            logger.info(f"Setting up the SO pretrain objective")
            train_data, validation_data, evaluate_fn = setup_pretrain(
                cfg,
                tokenizer,
                train_args
            )
        else:
            logger.info("Setting up the LM objective")

            if task.tokenizer.pad_token is None:
                task.tokenizer.pad_token = task.tokenizer.eos_token

            train_data, validation_data, evaluate_fn = setup_lm(
                cfg,
                task
            )

        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token
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

    if train_args.local_rank <= 0:
        logger.info("Training Arguments:")
        for arg_name in sorted(train_args.to_sanitized_dict()):
            logger.info(
                f"{arg_name:>30} = {getattr(train_args, arg_name)}"
            )

    logger.info(f"Setting up the optimizer")
    # optimizer = AdamW(
    #     get_grouped_params(model, train_args),
    #     lr=train_args.learning_rate,
    #     betas=(train_args.adam_beta1, train_args.adam_beta2),
    #     eps=train_args.adam_epsilon,
    #     weight_decay=train_args.weight_decay
    # )

    total_steps, warmup_steps = get_steps_from_training_args(train_args, train_data)

    logger.info(f"{total_steps} total training steps and {warmup_steps} warmup")

    # lr_scheduler = get_lr_scheduler(train_args, optimizer, total_steps, warmup_steps)

    device = train_args.device
    model = model.to(device)
    logger.info(f"Using device {device}")
    logger.info(f"Model Is On {model.device}")
    trainer = CustomTrainer(
        cfg,
        model=model,
        args=train_args,
        eval_dataset=validation_data,
        tokenizer=tokenizer,
        train_dataset=train_data,
        data_collator=collator,
        compute_metrics=evaluate_fn,
        # optimizers=(optimizer, lr_scheduler)

    )
    trainer.train()

    return trainer.model


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
