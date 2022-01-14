from pathlib import Path

import logging
from omegaconf import OmegaConf, DictConfig
from transformers import DataCollatorForSeq2Seq
from transformers import AdamW, get_scheduler
from accelerate import Accelerator
import torch
from tio import Task
from src import config
from src.common import get_stats_from_list
from src.old_trainer import CustomTrainer, create_log_metric_message
from src.data.langauge_modeling import create_dataloaders
from src.trainer import Trainer, TrainingArguments
from functools import partial

logger = logging.getLogger(__name__)


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

    logger.debug("Loading Model")
    model = config.load_model_from_cfg(cfg, device)

    task: Task = config.load_task_from_cfg(cfg)
    tokenizer = task.tokenizer

    logger.info("Getting train data")
    train_data = task.get_split("train", num_procs=cfg.get('num_proc', 1), set_format="torch")
    logger.info("Getting validation data")
    validation_data = task.get_split(
        "validation", num_procs=cfg.get('num_proc', 1), set_format="torch"
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model,
        return_tensors="pt",
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=2
    )

    train_args = config.get_training_args_from_cfg(cfg)
    # I have to do this for some reason.
    model.config.max_length = train_args.generation_max_length

    logger.debug("Initializing trainer")
    trainer = CustomTrainer(
        cfg=cfg,
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collator,
        compute_metrics=lambda preds: task.evaluate(
            *task.postprocess(
                preds.predictions, preds.label_ids
            )
        )
    )
    trainer.train()
    return model


def log_metrics(step_count, metrics):
    logger.info(f"Finished step {step_count}")
    for k, v in metrics.items():
        logger.info(f"{k:>20} = {v:0.3f}")


def evaluate(args, accelerator, model, data_loader):
    model.eval()
    losses = []
    for step, batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.per_device_eval_batch_size)
        losses.append(accelerator.gather(loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()


def get_grouped_params(model, args, no_decay=None):
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight"]
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def train_lm(cfg: DictConfig):
    """
    Train a language model with a given reader.

    Args:
        cfg (DictConfig): The config.

    Returns:
        The best model.
    """
    # device = config.get_device_from_cfg(cfg)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Using device {device}")

    logger.debug("Loading Model")
    model = config.load_model_from_cfg(cfg)

    task: Task = config.load_task_from_cfg(cfg)

    # Add a preprocessor to concat the inputs and labels.
    def concat(ex):
        ex['input_sequence'] = f"{ex['input_sequence']} {ex['target']}"
        ex['target'] = ex['input_sequence']
        return ex

    task.preprocessors.append(concat)

    tokenizer = task.tokenizer

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

    logger.info("Creating the dataloaders")
    # train_args = config.get_training_args_from_cfg(cfg)
    # train_args.predict_with_generate = False

    trainer = Trainer(
        model,
        TrainingArguments(**cfg['training']),
        device,
        tokenizer,
        evaluate_fn=lambda *args, **kwargs: {'test': 1.0},
        data_loading_fn=partial(create_dataloaders, tokenizer=tokenizer, cfg=cfg)
    )
    trainer(
        train_data,
        validation_data
    )
    return trainer.model
