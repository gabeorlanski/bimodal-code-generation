import json
import math
from itertools import chain
from pathlib import Path

import logging

import numpy as np
import ujson
from omegaconf import OmegaConf, DictConfig, open_dict
from transformers import (
    DataCollatorForSeq2Seq, EvalPrediction, AdamW
)
from functools import partial
from datasets import set_caching_enabled, Dataset, load_dataset
from datasets.iterable_dataset import iterable_dataset
from torch.utils.data import DataLoader, IterableDataset
import torch
from tio import Task
from src import config
from src.common import get_stats_from_list, PROJECT_ROOT, set_global_logging_level
from src.data import langauge_modeling, NON_REGISTERED_TASKS
from src.data.stackoverflow import StackOverflowProcessor
from src.training.trainer import CustomTrainer, HFIterableWrapper
from src.config import get_steps_from_training_args, get_lr_scheduler, get_prompts_from_cfg
from src.data.tensorize import TensorizedTask
from tqdm import tqdm
from src.data import NPV
import bitsandbytes as bnb
from jinja2 import BaseLoader, Environment, StrictUndefined
from src.common import PROJECT_ROOT
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

set_global_logging_level(logging.ERROR,
                         ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined


def evaluate_seq2seq(eval_predictions: EvalPrediction, task: Task):
    preds, targets = eval_predictions
    preds = task.tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    targets = task.tokenizer.batch_decode(
        targets,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return task.evaluate(
        [[p] for p in map(task.postprocess, preds)],
        list(map(task.postprocess, targets))
    )


def evaluate_func_npv(eval_predictions: EvalPrediction, choices,
                      targets,
                      tokenizer):
    predictions = []

    tmp_targets = eval_predictions.label_ids.copy()
    tmp_targets[tmp_targets == -100] = tokenizer.eos_token_id
    tmp_targets = tokenizer.batch_decode(tmp_targets, skip_special_tokens=True)
    targets_found = [targets[t] for t in tmp_targets]

    choice_len = max(map(len, choices))
    assert all(len(c) == choice_len for c in choices)

    for label, pred in zip(eval_predictions.label_ids, eval_predictions.predictions):
        start = end = -1
        for i, t in enumerate(label):
            if t != -100 and start == -1:
                start = i
            elif start != -1 and t == -100:
                end = i
                break
        text_span = label[start:end]
        input_len = text_span.shape[0] - choice_len

        probs = []
        for i, choice in enumerate(choices):
            choice_prob = 0
            for j, t in enumerate(choice):
                choice_prob += pred[input_len + j][t]
            probs.append(choice_prob)
        predictions.append(probs)

    predictions = np.array(predictions).argmax(axis=1)
    f1 = f1_score(targets_found, predictions, average='macro')
    return {'f1': f1 * 100}


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

    if cfg.task.name == 'npv':
        def special_concat(ex):
            ex['input_sequence'] = f"{ex['input_sequence']}{ex['target']}"
            ex['label'] = ex['target']
            ex['target'] = ex['input_sequence']
            return ex

        task.preprocessors.append(special_concat)
    else:
        task.preprocessors.append(concat)

    group_texts = partial(
        langauge_modeling.raw_group_texts,
        concat_token=task.tokenizer.eos_token_id,
        seq_length=cfg.data_args.seq_length,

    )

    def make_split(split_name):
        split_data = task.get_split(
            split_name,
            num_procs=cfg.get('num_proc', 1),
            add_special_tokens=False
        )
        split_data = Dataset.from_dict(group_texts(split_data['input_ids']))
        return split_data

    logger.info("Getting train data")
    train_data = make_split('train')
    logger.info(f"{len(train_data)} total training samples")

    logger.info("Getting validation data")
    if cfg.task.name == 'npv':
        task: NPV

        raw_validation_data = task.preprocess('validation')
        validation_data = task.get_split(
            'validation',
            num_procs=cfg.get('num_proc', 1),
            add_special_tokens=False,
            do_truncate=True
        )
        validation_data = validation_data.map(
            lambda ex, idx: {'label': raw_validation_data['label'][idx], **ex},
            with_indices=True
        )
    else:
        validation_data = make_split('validation')
    return train_data, validation_data, None


def setup_hf_pretrain(cfg, tokenizer, train_args, prompt_fn):
    concat_delim = tokenizer('\n')['input_ids']
    block_size = cfg.task.sequence_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {}
        for k, v in examples.items():
            concatenated_examples[k] = []
            for ex in v:
                concatenated_examples[k].extend(ex + [tokenizer.eos_token_id])
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    raw_train_dataset = load_dataset(
        cfg.task.train.dataset,
        cfg.task.train.subset,
        split=cfg.task.train.split,
        streaming=True
    )

    def tokenize(ex, text_key):
        return {'input_ids': tokenizer(ex[text_key], add_special_tokens=False)['input_ids']}

    train_dataset = raw_train_dataset.shuffle(seed=cfg.seed, buffer_size=1000).map(
        lambda e: {'input_seq': e[cfg.task.train.text_key], 'labels': e[cfg.task.train.text_key]}
    )
    if cfg.task.train.get('max_train_samples', -1) > 0:
        logger.info(f"Taking {cfg.task.train.get('max_train_samples')} from train")
        train_dataset = train_dataset.take(cfg.task.train.get('max_train_samples'))

    raw_eval_dataset = load_dataset(
        cfg.task.validation.dataset,
        cfg.task.validation.subset,
        split=cfg.task.validation.split,
    )
    if cfg.task.validation.max_val_samples > 0:
        logger.info(f"Selecting {cfg.task.validation.max_val_samples} for debug")
        raw_eval_dataset = raw_eval_dataset.select(range(cfg.task.validation.max_val_samples))

    raw_eval_dataset = raw_eval_dataset.shuffle(seed=cfg.seed).map(
        lambda e: tokenize(e, cfg.task.validation.text_key),
        num_proc=cfg.get("num_proc", 4),
        remove_columns=raw_eval_dataset.column_names
    )
    eval_dataset = raw_eval_dataset.map(
        group_texts,
        batched=True,
        remove_columns=raw_eval_dataset.column_names,

    )
    return HFIterableWrapper(
        train_dataset,
        tokenizer=tokenizer,
        objective=cfg.objective,
        field_concat_tokens=concat_delim,
        concat_token=tokenizer.eos_token_id,
        input_fields=['input_ids'],
        sequence_length=cfg.task.sequence_length
    ), eval_dataset, None


def setup_tensorized(cfg, tokenizer, train_args, prompt_fn):
    concat_delim = tokenizer('\n')['input_ids']
    block_size = cfg.task.sequence_length

    processor = StackOverflowProcessor(
        prompt_fn=prompt_fn,
        **OmegaConf.to_object(cfg.processor.params)
    )

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

    dump_path = PROJECT_ROOT.joinpath(cfg.task.raw_dump_path)

    effective_batch_size = (
            train_args.train_batch_size
            * train_args.gradient_accumulation_steps
            * train_args.world_size
    )

    # For the HF Trainer, we need the eval set to have a size, so we split up
    # the initialization so one is in streaming mode and the other is not.
    train_dataset = TensorizedTask(
        name=cfg.task.tensorized_name,
        raw_data_name=cfg.task.raw_dump_name,
        dump_path=dump_path,
        objective=cfg.objective,
        processor=processor,
        tokenizer=tokenizer,
        max_samples_to_yield=effective_batch_size * train_args.max_steps,
        sequence_length=cfg.task.sequence_length,
        buffer_size=cfg.task.get('buffer_size', 1)
    )

    if train_args.local_rank <= 0:
        logger.info("Processor Params:")
        for k, v in cfg.processor.params.items():
            logger.info(f"\t{k:>24}={v}")

        logger.info("Tensorized Task Params:")
        for k, v in train_dataset.params.items():
            logger.info(f"\t{k:>24}={v}")

    logger.info(f"Creating Eval Dataset")
    eval_dataset = {'input_ids': [], 'labels': []}
    eval_file = dump_path.joinpath(f"{cfg.task.raw_dump_name}_val.jsonl")

    line_num = 0
    num_no_samples = 0
    total_samples = 0
    for line in eval_file.open('r'):

        sample = ujson.loads(line)
        line_num += 1
        processed = processor(sample)
        if line_num % 1000 == 0:
            logger.info(f"Read {line_num} lines for eval")
        if not processed:
            logger.debug(f"Line {line_num} with id {sample['id']} had no "
                         f"samples produced.")
            num_no_samples += 1
            continue
        for i, instance in enumerate(processed):
            eval_dataset['input_ids'].append(
                tokenizer(
                    instance['input'],
                    max_length=train_dataset.sequence_length,
                    truncation=cfg.objective != 'seq2seq'
                )['input_ids'],
            )
            eval_dataset['labels'].append(
                tokenizer(
                    instance['labels'],
                    max_length=train_dataset.sequence_length,
                    truncation=cfg.objective != 'seq2seq'
                )['input_ids']
            )
            total_samples += 1
        if line_num - num_no_samples >= 10000:
            break

    logger.warning(f"{num_no_samples}/{line_num} produced no samples.")

    eval_dataset = Dataset.from_dict(eval_dataset).shuffle(seed=cfg.seed)

    if cfg.objective == 'lm':
        eval_dataset = eval_dataset.map(
            group_texts,
            batched=True,
        )
    logger.info(f"{len(eval_dataset)} samples in the eval dataset")
    return train_dataset, eval_dataset, None


def train_model(cfg: DictConfig, train_args):
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
    prompt_fn = get_prompts_from_cfg(cfg, JINJA_ENV)

    def prompt_preprocessor(instance):
        instance['input_sequence'] = prompt_fn(instance)
        return instance

    is_non_registered_task = False
    if cfg.task.name in NON_REGISTERED_TASKS:
        is_non_registered_task = True
        tokenizer = config.load_tokenizer_from_cfg(cfg, force_fast=cfg.task.name == 'hf_pretrain')
        task = None  # type: ignore
    else:
        task: Task = config.load_task_from_cfg(cfg)
        tokenizer = task.tokenizer
        task.preprocessors.append(prompt_preprocessor)

    if cfg.objective == 'seq2seq':
        if is_non_registered_task:
            logger.info(f"Setting up the SO pretrain objective")
            if cfg.task.name == 'hf_pretrain':
                train_data, validation_data, evaluate_fn = setup_hf_pretrain(
                    cfg,
                    tokenizer,
                    train_args,
                    prompt_fn
                )
            else:
                train_data, validation_data, evaluate_fn = setup_tensorized(
                    cfg,
                    tokenizer,
                    train_args,
                    prompt_fn
                )
        else:
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

        if is_non_registered_task:

            eos_token = tokenizer.eos_token or tokenizer.bos_token
            tokenizer.eos_token = eos_token
            tokenizer.bos_token = eos_token
            tokenizer.pad_token = tokenizer.eos_token
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            model.config.bos_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = 'left'
            tokenizer.truncation_side = 'left'
            logger.info(f"Setting up the SO pretrain objective")
            if cfg.task.name == 'hf_pretrain':
                train_data, validation_data, evaluate_fn = setup_hf_pretrain(
                    cfg,
                    tokenizer,
                    train_args,
                    prompt_fn
                )
            else:

                train_data, validation_data, evaluate_fn = setup_tensorized(
                    cfg,
                    tokenizer,
                    train_args,
                    prompt_fn
                )
        else:
            eos_token = task.tokenizer.eos_token or task.tokenizer.bos_token
            task.tokenizer.eos_token = eos_token
            task.tokenizer.bos_token = eos_token
            task.tokenizer.pad_token = task.tokenizer.eos_token
            model.config.eos_token_id = task.tokenizer.eos_token_id
            model.config.pad_token_id = task.tokenizer.eos_token_id
            model.config.bos_token_id = task.tokenizer.eos_token_id
            task.tokenizer.padding_side = 'left'
            task.tokenizer.truncation_side = 'left'
            logger.info("Setting up the LM objective")

            train_data, validation_data, evaluate_fn = setup_lm(
                cfg,
                task
            )

            if cfg.task.name == 'npv':
                task: NPV
                choices_tokenized = [
                    task.tokenizer(c, add_special_tokens=False)['input_ids']
                    for c in task.choices
                ]

                # Hacky ways to get the labels in Because compute metrics in
                # huggingface is horrible
                full_labels = {
                    task.tokenizer.decode(seq): task.choices.index(l)
                    for seq, l in zip(validation_data['input_ids'], validation_data['label'])
                }
                validation_data = validation_data.remove_columns(['label', 'idx'])
                evaluate_fn = partial(
                    evaluate_func_npv,
                    choices=choices_tokenized,
                    targets=full_labels,
                    tokenizer=task.tokenizer
                )

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
    if cfg.debug:
        validation_data = validation_data.select(range(100))

    if train_args.local_rank <= 0:
        logger.info("Training Arguments:")
        for arg_name in sorted(train_args.to_sanitized_dict()):
            logger.info(
                f"{arg_name:>30} = {getattr(train_args, arg_name)}"
            )
        logger.info(
            f"Saving 100 debug samples to {Path.cwd().joinpath('debug_samples.json')}")
        with Path.cwd().joinpath('debug_eval_samples.json').open('w') as f:
            data_to_save = {}
            for i, v in enumerate(validation_data):
                if len(data_to_save) >= 25:
                    break
                try:
                    data_to_save[i] = {
                        'input_ids': tokenizer.decode(v['input_ids']),
                        'labels'   : tokenizer.decode(v['labels'])
                    }
                except Exception as e:
                    raise e

            json.dump(data_to_save, f, indent=True, sort_keys=True)

    resume_path = None
    if cfg.get('resume_from_checkpoint') is not None:
        logger.info("Resuming from last checkpoint")
        chk_path = Path('checkpoints')
        assert chk_path.exists()
        step_count = -1
        for directory in chk_path.glob('*'):
            if not directory.is_dir():
                continue
            *_, cur_step_count = directory.stem.split('-')
            cur_step_count = int(cur_step_count)
            if cur_step_count > step_count:
                step_count = cur_step_count
                resume_path = directory

    logger.info(f"Setting up the optimizer")
    total_steps, warmup_steps = get_steps_from_training_args(train_args, train_data)
    if not train_args.deepspeed:
        if train_args.use_8bit_adam:
            logger.info(f"Using Adam 8 Bit")
            optimizer = bnb.optim.AdamW8bit(
                get_grouped_params(model, train_args),
                lr=train_args.learning_rate,
                betas=(train_args.adam_beta1, train_args.adam_beta2),
                eps=train_args.adam_epsilon,
                weight_decay=train_args.weight_decay
            )
        else:

            logger.info(f"Using AdamW")
            optimizer = torch.optim.AdamW(
                get_grouped_params(model, train_args),
                lr=train_args.learning_rate,
                betas=(train_args.adam_beta1, train_args.adam_beta2),
                eps=train_args.adam_epsilon,
                weight_decay=train_args.weight_decay
            )

        lr_scheduler = get_lr_scheduler(train_args, optimizer, total_steps, warmup_steps)

        optimizer_arg = (optimizer, lr_scheduler)
    else:
        optimizer_arg = (None, None)

    logger.info(f"{total_steps} total training steps and {warmup_steps} warmup")
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
        optimizers=optimizer_arg

    )

    trainer.train(resume_path)

    with open_dict(cfg):
        cfg.best_model_checkpoint = trainer.state.best_model_checkpoint
        cfg.best_model_path = str(PROJECT_ROOT.joinpath(trainer.state.best_model_checkpoint))
    with Path('config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(trainer.cfg, resolve=True, sort_keys=True))
    return trainer.model, cfg


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
