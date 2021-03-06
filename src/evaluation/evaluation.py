import copy
import json
import math
import os
from collections import defaultdict, Counter
from datetime import datetime
from functools import partial
from itertools import chain
from pathlib import Path
import random
from typing import List, Union

import pandas as pd
import sklearn
from torch.nn import functional as F
import datasets
import numpy as np
import wandb
from datasets import set_caching_enabled, Dataset
from omegaconf import DictConfig, open_dict, OmegaConf
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList, \
    DataCollatorForSeq2Seq
import torch
import logging
from tqdm import tqdm
from src.config import (
    get_device_from_cfg, load_task_from_cfg, \
    get_run_base_name_from_cfg, initialize_run_from_cfg, get_prompts_from_cfg
)

from jinja2 import BaseLoader, Environment, StrictUndefined
from sklearn.metrics import (
    precision_recall_fscore_support, precision_score, accuracy_score, recall_score, f1_score
)

import multiprocessing as mp

from src.data import NPV

logger = logging.getLogger(__name__)

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

# Allow the python function zip()
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


class EOSStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, tokenizer):
        self.start_length = start_length
        self.eos_token = tokenizer.eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""

        return all(self.eos_token in row[self.start_length:] for row in input_ids)


def oracle(args, metric_list):
    # To get the oracle score, we need to repeat target for every prediction
    predictions, target = args
    return target, [
        m.get_oracle_best_pred(predictions, target)
        for m in metric_list
    ]


def get_amounts_to_generate(seq_per_sample, batch_size, num_generate_per_step):
    generate_steps_per_batch, remainder = divmod(seq_per_sample * batch_size, num_generate_per_step)
    has_remainder = remainder > 0

    amounts_to_generate = (
            [num_generate_per_step] * generate_steps_per_batch
            + [remainder] * has_remainder
    )
    return amounts_to_generate, generate_steps_per_batch, remainder


def generate_predictions(
        model,
        objective,
        dataset: Union[List[dict], Dataset],
        tokenizer,
        num_procs,
        num_generate_per_step,
        device,
        generation_kwargs,
        seq_per_sample,
        remove_input_ids_from_output,
        debug,
        min_batch_size
):
    logger.info("Starting Generation")

    logger.info(f"Generating {num_generate_per_step} per step and generating "
                f"{seq_per_sample} total per sample")
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    indices = []
    predictions = []
    labels = []
    batch_size = math.ceil(num_generate_per_step / seq_per_sample)
    batch_size = max(batch_size, min_batch_size)
    logger.info(f"Using batch size {batch_size}")

    amounts_to_generate, batch_gen_steps, remainder = get_amounts_to_generate(
        seq_per_sample, batch_size, num_generate_per_step
    )

    logger.debug(f"{len(amounts_to_generate)} steps per sample")

    max_length = generation_kwargs.pop('max_length', 256)
    if 'max_new_tokens' in generation_kwargs:
        max_new_tokens = generation_kwargs.pop('max_new_tokens')
        if 'max_length' not in generation_kwargs:
            max_length = max_new_tokens

    total_memory = torch.cuda.mem_get_info(device)[1]  # type: ignore
    tokenizer.padding_side = 'left' if objective == 'lm' else 'right'

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding='longest',
        pad_to_multiple_of=1,
        return_tensors='pt',
        label_pad_token_id=tokenizer.pad_token_id
    )

    def prepare_for_eval(ex, ex_idx):
        prompt = ex['input_sequence']
        if objective == 'lm' and not debug:
            prompt = tokenizer.eos_token + prompt
        input_ids = tokenizer(prompt)
        return {
            'labels': tokenizer(ex['target'])['input_ids'],
            'length': len(input_ids['input_ids']),
            'idx'   : ex_idx,
            **input_ids
        }

    logger.info(f"Preparing {len(dataset)} instances for evaluation")
    tokenized = dataset.map(
        prepare_for_eval,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=num_procs
    )

    if debug:
        tokenized = tokenized.sort('length', reverse=True)
    # else:
    #     tokenized = tokenized.sort('task_id')
    tokenized.set_format(type='torch')

    sequential_sampler = torch.utils.data.SequentialSampler(tokenized)

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_procs if num_procs != 1 else 0,
        shuffle=False,
        sampler=sequential_sampler,
    )

    model.eval()
    with torch.inference_mode():

        # Disable during debugging for my sanity.
        if not debug:
            progress_bar = tqdm(
                total=math.ceil(len(dataset) / batch_size) * len(amounts_to_generate),
                desc='Generating'
            )
        else:
            progress_bar = None
        completed = 0
        last_log = 0
        for instance in dataloader:
            local_indices = instance['idx'].tolist()
            local_inputs = instance["input_ids"].to(device)
            local_attention = instance['attention_mask'].to(device)
            input_len = instance['length'][0].item()

            generated_for_current_batch = [[] for _ in range(len(local_indices))]

            max_length_for_gen = max_length
            if objective == 'lm':
                if max_length + input_len > tokenizer.model_max_length:
                    logger.warning(
                        f"Batch {completed + 1} has more than the "
                        f"models max length of {tokenizer.model_max_length}."
                    )
                    # Subtract 4 to be safe.
                    max_length_for_gen = tokenizer.model_max_length - 4
                else:
                    max_length_for_gen = input_len + max_length
                    logger.debug(f"{max_length_for_gen=}")

            if 'stopping_criteria' in generation_kwargs:
                for sc in generation_kwargs['stopping_criteria']:
                    if hasattr(sc, 'start_length'):
                        sc.start_length = input_len
            i = 0
            while i < len(amounts_to_generate):
                num_to_generate = amounts_to_generate[i]
                generated_from_batch = model.generate(
                    input_ids=local_inputs,
                    attention_mask=local_attention,
                    max_length=max_length_for_gen,
                    num_return_sequences=num_to_generate,
                    use_cache=True,
                    **generation_kwargs
                ).cpu()

                slice_len = remove_input_ids_from_output * input_len
                ids_for_current_sample = generated_from_batch[:, slice_len:]

                if progress_bar:
                    progress_bar.update(1)
                elif (i + 1) == len(amounts_to_generate) // 2:
                    logger.info(f"Finished {i + 1}/{len(amounts_to_generate)} for {completed + 1}")

                decoded = tokenizer.batch_decode(
                    ids_for_current_sample,
                    skip_special_tokens=True
                )

                gen_per_sample = num_to_generate // batch_size
                for j in range(batch_size):
                    start = j * gen_per_sample
                    generated_for_current_batch[j].extend(decoded[start:start + gen_per_sample])
                i += 1

            assert all(map(lambda x: len(x) == seq_per_sample, generated_for_current_batch))

            completed += len(local_indices)
            pct_allocated = torch.cuda.max_memory_allocated(device) / total_memory
            if debug or math.floor(completed / len(dataset) * 10) != last_log:
                last_log = math.floor(completed / len(dataset) * 10)
                logger.info(
                    f"{pct_allocated * 100:0.2f}% GPU memory allocated"
                )

            for idx, preds in zip(local_indices, generated_for_current_batch):
                predictions.append(preds)
                labels.append(dataset[idx]['target'])
                indices.append(dataset[idx]['task_id'])
            if not progress_bar:
                logger.info(f"Finished {completed}/{len(dataset)} generations")

        if progress_bar:
            progress_bar.close()

    logger.info("Generating finished.")
    return {
        "indices"    : indices,
        "labels"     : labels,
        "predictions": predictions
    }


def evaluate_model_generation_task(
        cfg: DictConfig,
        model: PreTrainedModel
):
    """
    Evaluate a model with a reader on a file
    Args:
        cfg (DictConfig): The config to use.
        model (PreTrainedModel): The pretrained huggingface model to use.
    """
    task = load_task_from_cfg(cfg)
    logger.info(f"Reading data from '{cfg['data_path']}'")
    gen_kwargs = OmegaConf.to_object(cfg.get('generation', {}))
    if 'num_return_sequences' in gen_kwargs:
        gen_kwargs.pop('num_return_sequences')

    gen_overrides = cfg.task.get('generation', {})
    logger.info(f"Task {cfg.task.name} has {len(gen_overrides)} generation overrides")
    for k, v in gen_overrides.items():
        gen_kwargs[k] = v

    if cfg.objective == 'lm':
        task.tokenizer.pad_token = task.tokenizer.eos_token
        model.config.eos_token_id = task.tokenizer.eos_token_id
        model.config.pad_token_id = task.tokenizer.eos_token_id
        model.config.bos_token_id = task.tokenizer.eos_token_id
        cfg.evaluation.remove_input_ids = True

    if cfg.task.name == 'human_eval':
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, EOF_STRINGS, task.tokenizer)]
        )
    elif cfg.objective == 'lm':
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EOSStoppingCriteria(0, task.tokenizer)]
        )

    prompt_fn = get_prompts_from_cfg(cfg, JINJA_ENV)

    def prompt_preprocessor(instance):
        instance['input_sequence'] = prompt_fn(instance)
        return instance

    task.preprocessors.append(prompt_preprocessor)

    logger.info(f"Getting the data for split {cfg.split}")
    dataset = task.preprocess(cfg.split)
    logger.info(f"{len(dataset)} total samples found")
    debug_num_samples = cfg.get('debug_num_samples', None)
    if debug_num_samples is not None:
        logger.warning(f"DEBUG NUMBER OF SAMPLES={debug_num_samples}")
        dataset = dataset.map(
            lambda ex: {
                'len': len(task.tokenizer.encode(ex['input_sequence'])),
                **ex
            }
        )
        dataset = dataset.sort('len', reverse=False).select(list(range(debug_num_samples)))

    device = get_device_from_cfg(cfg)
    model = model.to(device)
    # model = amp.initialize(model)
    logger.info(f"Model is on {model.device}")
    logger.debug(f"{type(dataset)=}")

    generation_results = generate_predictions(
        model,
        objective=cfg.objective,
        dataset=dataset,
        num_procs=cfg.get('num_proc', 1),
        tokenizer=task.tokenizer,
        num_generate_per_step=cfg.evaluation.num_generate_per_step,
        device=device,
        generation_kwargs=gen_kwargs,
        seq_per_sample=cfg.evaluation.seq_per_sample,
        remove_input_ids_from_output=cfg.evaluation.get("remove_input_ids", False),
        debug=cfg.debug,
        min_batch_size=cfg.evaluation.get('min_batch_size', 1)
    )

    labels = list(map(task.postprocess, generation_results['labels']))
    predictions = list(
        map(lambda pl: list(map(task.postprocess, pl)), generation_results['predictions'])
    )
    indices = generation_results['indices']

    metrics = task.evaluate(predictions, labels)
    oracle_args = [[p, t] for p, t in zip(predictions, labels)]
    oracle_fn = partial(
        oracle,
        metric_list=task.metric_fns
    )

    # Copy the targets to another list to maintain order
    global_targets = []
    global_oracle = [[] for _ in task.metric_fns]
    logger.info(f"Calculating the Oracle Scores")
    with mp.Pool(cfg.get("num_workers", 1)) as pool:
        for target, best_predictions in tqdm(
                pool.imap_unordered(oracle_fn, oracle_args),
                total=len(predictions)
        ):
            global_targets.append(target)
            for i, v in enumerate(best_predictions):
                global_oracle[i].append(v)

    for m, preds_list in zip(task.metric_fns, global_oracle):
        for k, v in m(preds_list, global_targets).items():
            metrics[f"oracle_{k}"] = v  # type: ignore

    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        if isinstance(v, int):
            logger.info(f"\t{k:>20} = {v}")
        else:
            logger.info(f"\t{k:>20} = {v:0.3f}")

    serialized_predictions = []
    serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
    for serialized_dict in tqdm(serialize_generator, total=len(indices), desc="Serializing"):
        serialized_predictions.append(serialized_dict)

    return metrics, serialized_predictions


def consolidate_ensembled_preds(split, task: NPV, choice_list, ensemble_choices):
    num_predictions_made = []
    predictions = []
    oracle_preds = []
    indices = []
    targets = []
    tie_break_correct = []
    for task_id, probs in ensemble_choices.items():
        target = task.raw_processed_dict[split][task_id]['result']
        targets.append(target)
        preds = probs.argmax(dim=-1)
        choice_pred_counts = {k: 0 for k in choice_list}
        raw_counts = preds.bincount()

        num_predictions_made.append(len(preds))

        highest_score = raw_counts.max()
        if all(highest_score == v for v in raw_counts) and raw_counts.size(0) == len(choice_list):
            tie_break_probs = [[] for _ in range(len(choice_list))]
            for i, idx in enumerate(preds):
                tie_break_probs[idx].append(probs[i, idx].item())

            tie_break_probs = torch.tensor(list(map(np.mean, tie_break_probs)))
            predicted_choice = choice_list[tie_break_probs.argmax().item()]
            tie_break_correct.append(1 if predicted_choice == target else 0)
        else:
            predicted_choice = choice_list[raw_counts.argmax().item()]
        predictions.append(predicted_choice)
        indices.append(task_id)

        found_correct = False
        for i, v in enumerate(raw_counts.tolist()):
            if choice_list[i] == target and v > 0:
                found_correct = True
                oracle_preds.append(choice_list[i])
            choice_pred_counts[choice_list[i]] = v
        if not found_correct:
            oracle_preds.append(predicted_choice)
    return (num_predictions_made, tie_break_correct), predictions, oracle_preds, indices, targets


def generate_log_probs(
        batch,
        choice_list,
        choices_tokenized,
        model,
        longest_choice_len,
        device
):
    n_seqs = batch['input_ids'].size(0)
    input_size = batch['input_ids'].size(1)
    max_len = input_size + longest_choice_len
    input_ids = torch.zeros((n_seqs, max_len)).long()
    input_ids[:, :input_size] = batch['input_ids']
    attention_mask = torch.zeros((n_seqs, max_len)).long()
    attention_mask[:, :input_size] = batch['attention_mask']
    attention_mask[:, input_size:input_size + longest_choice_len] = 1
    local_predictions = defaultdict(list)
    for choice, choice_tokens in zip(choice_list, choices_tokenized):

        choice_tensor = torch.tensor([choice_tokens] * n_seqs).long()
        labels = -100 * torch.ones((n_seqs, max_len)).long()
        labels[:, input_size:input_size + len(choice_tokens)] = choice_tensor

        local_input_ids = input_ids.clone().detach()
        local_input_ids[:, input_size: input_size + choice_tensor.size(1)] = choice_tensor
        local_input_ids = local_input_ids.to(device)
        local_attention_mask = attention_mask.clone().detach()
        local_attention_mask[:, input_size: input_size + choice_tensor.size(1)] = 1
        local_attention_mask = local_attention_mask.to(device)

        # This was heavily inspired and has elements from:
        # https://github.com/peterwestuw/surface-form-competition/blob/main/utils.py
        logits = model(
            input_ids=local_input_ids,
            attention_mask=local_attention_mask,
            # labels=local_input_ids
        ).logits.cpu()
        logits = logits.contiguous()

        # Get the score for each token in the choice tokenized
        scores = torch.gather(
            logits[:, input_size:],
            -1,
            choice_tensor.unsqueeze(-1)
        ).squeeze(-1)

        # Sum up the log probs for each element
        scores = scores.sum(dim=1).tolist()
        for idx, p in zip(batch['idx'], scores):
            local_predictions[idx.item()].append(p)
    return local_predictions


def evaluate_npv_task(
        cfg: DictConfig,
        model: PreTrainedModel
):
    task: NPV = load_task_from_cfg(cfg)  # type: Ignore
    logger.info(f"Reading data from '{cfg['data_path']}'")
    debug = cfg.debug

    assert hasattr(task, 'choices')
    choice_list = list(map(str, task.choices))  # type: ignore
    logger.info(f"Getting the data for split {cfg.split}")
    dataset = task.preprocess(cfg.split)
    zero_shot_dataset = task.preprocess(f"zero_shot_{cfg.split}")
    debug_num_samples = cfg.get('debug_num_samples', None)

    eos_token = task.tokenizer.eos_token or task.tokenizer.bos_token
    task.tokenizer.eos_token = eos_token
    task.tokenizer.bos_token = eos_token
    task.tokenizer.pad_token = task.tokenizer.eos_token
    model.config.eos_token_id = task.tokenizer.eos_token_id
    model.config.pad_token_id = task.tokenizer.eos_token_id
    model.config.bos_token_id = task.tokenizer.eos_token_id
    task.tokenizer.padding_side = 'left'
    task.tokenizer.truncation_side = 'left'

    def tokenize(example, example_idx):

        # We do not pop so that we can still remove the columns later.
        out = {
            "idx": example_idx,
            **task.tokenizer(
                task.tokenizer.eos_token + example["input_sequence"],
                truncation=True,
                max_length=task.tokenizer.model_max_length - 1,
                add_special_tokens=False
            )
        }
        target_tokenized = task.tokenizer(
            example['target'],
            truncation=True,
            max_length=task.tokenizer.model_max_length - 1,
            add_special_tokens=False
        )
        out.update(
            {
                "labels": target_tokenized["input_ids"],
            }
        )
        return out

    tokenized = dataset.map(
        tokenize,
        with_indices=True,
        num_proc=cfg.num_proc,
        remove_columns=[c for c in dataset.column_names],
    )
    if task.with_zero_shot:
        logger.info(f"Subtracting Zero shot Probs are enabled")
    zero_shot_tokenized = zero_shot_dataset.map(
        tokenize,
        with_indices=True,
        num_proc=cfg.num_proc,
        remove_columns=[c for c in dataset.column_names],
    )

    choices_tokenized = [
        task.tokenizer(
            str(c),
            add_special_tokens=False
        )['input_ids'] for c in choice_list
    ]  # type:ignore
    logger.info(f"{len(dataset)} total samples found")

    # if debug_num_samples is not None or debug:

    logger.info(f"Adding length to the dataset")
    tokenized = tokenized.map(
        lambda ex: {
            'len': len(ex['input_ids']),
            **ex
        }
    )
    if debug_num_samples is None:
        logger.info("Sorting by length")
    tokenized = tokenized.sort('len', reverse=debug_num_samples is None)
    if debug_num_samples is not None:
        logger.warning(f"DEBUG NUMBER OF SAMPLES={debug_num_samples}")
        tokenized = tokenized.select(list(range(debug_num_samples)))

    device = get_device_from_cfg(cfg)
    logger.info(f"Putting model on {device}")
    model = model.to(device)
    logger.info(f"Model is on {model.device}")
    logger.debug(f"{type(dataset)=}")
    collator = DataCollatorForSeq2Seq(
        tokenizer=task.tokenizer,
        padding='longest',
        pad_to_multiple_of=1,
        return_tensors='pt',
        label_pad_token_id=task.tokenizer.pad_token_id
    )

    sequential_sampler = torch.utils.data.SequentialSampler(tokenized)
    batch_size = cfg.evaluation.num_generate_per_step
    logger.info(f"{batch_size=}")

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
        sampler=sequential_sampler,
    )

    model.eval()
    start_time = datetime.utcnow()
    with torch.inference_mode():
        if not debug:
            progress_bar = tqdm(
                total=math.ceil(len(dataset) / batch_size),
                desc='Generating'
            )
        else:
            progress_bar = None
        completed = 0
        longest_choice = max(choices_tokenized, key=lambda t: len(t))
        longest_choice_len = len(longest_choice)
        ensemble_choices = defaultdict(list)
        ensemble_preds = defaultdict(list)

        # A map to store already computed zero-shot probabilities for tasks.
        zs_probs_map = {}

        for batch in dataloader:
            n_seqs = batch['input_ids'].size(0)
            input_size = batch['input_ids'].size(1)
            to_calc_zs = {}
            batch_zs_probs = {}
            for idx in batch['idx'].tolist():
                tid = zero_shot_dataset[idx]['task_id']
                if tid in zs_probs_map:
                    batch_zs_probs[idx] = zs_probs_map[tid]
                else:
                    to_calc_zs[idx] = tid
            if to_calc_zs:
                zero_shot_batch = zero_shot_tokenized.select(list(to_calc_zs)).to_dict()
                zero_shot_batch = collator([{k: zero_shot_batch[k][i] for k in zero_shot_batch}
                                            for i in range(len(to_calc_zs))])
            else:
                zero_shot_batch = None
            local_predictions = generate_log_probs(
                batch,
                choice_list,
                choices_tokenized,
                model,
                longest_choice_len,
                device
            )

            if task.with_zero_shot:
                if len(batch_zs_probs) != batch['idx'].size(0) and to_calc_zs:
                    zs_predictions = generate_log_probs(
                        zero_shot_batch,
                        choice_list,
                        choices_tokenized,
                        model,
                        longest_choice_len,
                        device
                    )
                    for k, v in zs_predictions.items():
                        zs_probs_map[dataset[k]['task_id']] = v
                        batch_zs_probs[k] = v
                assert len(batch_zs_probs) == batch['idx'].size(0)
                for k in local_predictions:
                    for i in range(len(local_predictions[k])):
                        local_predictions[k][i] -= batch_zs_probs[k][i]
            for k, v in local_predictions.items():
                task_id = dataset[k]['task_id']
                ensemble_choices[task_id].append(v)
                ensemble_preds[task_id].append(dataset[k]['context_examples'])

            if progress_bar:
                progress_bar.update(1)
            completed += n_seqs
            if not progress_bar:
                logger.info(f"Finished {completed}/{len(dataset)} generations")
    end_time = datetime.utcnow()

    logger.info("Consolidating the ensemble choices")
    ensemble_choices = {tid: torch.tensor(v) for tid, v in ensemble_choices.items()}
    stats, predictions, oracle_preds, indices, targets = consolidate_ensembled_preds(
        cfg.split,
        task,
        choice_list,
        ensemble_choices
    )
    num_predictions_made, tie_breaks = stats
    global_pred_count = Counter(predictions)

    metrics = {
        "Oracle_f1"           : 100 * f1_score(
            targets,
            oracle_preds,
            labels=choice_list,
            average='micro'
        ),
        "accuracy"            : 100 * accuracy_score(
            targets,
            predictions
        ),
        "f1"                  : 100 * f1_score(
            targets,
            predictions,
            labels=choice_list,
            average='micro'
        ),
        "recall"              : 100 * recall_score(
            targets,
            predictions,
            average='micro',
            labels=choice_list
        ),
        "precision"           : 100 * precision_score(
            targets,
            predictions,
            average='micro',
            labels=choice_list
        ),
        'eval_seconds'        : (end_time - start_time).total_seconds(),
        'predictions_per_task': np.mean(num_predictions_made),
        'tie_breaks'          : len(tie_breaks),
        'tie_breaks_correct'  : sum(tie_breaks) / len(tie_breaks) * 100 if tie_breaks else 0.0
    }

    precision, recall, f1_arr, occurrences = precision_recall_fscore_support(
        targets, predictions, average=None, labels=choice_list
    )

    task: NPV
    for i, (p, r, f1, o) in enumerate(zip(precision, recall, f1_arr, occurrences)):
        choice = task.idx_to_choice[i]
        metrics[f'{choice}_precision'] = p * 100
        metrics[f'{choice}_recall'] = r * 100
        metrics[f'{choice}_f1'] = f1 * 100
        metrics[f'{choice}_count'] = global_pred_count.get(choice_list[i], 0)

    # Apply softmax to rescale the log probabilities (also multiply by -1 as CE
    # returns a positive number) then multiply by 100 and round to 5 decimal
    # places for sanity. Then convert back to a list for saving.

    serialized_predictions = []
    serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
    for i, serialized_dict in tqdm(enumerate(serialize_generator), total=len(indices),
                                   desc="Serializing"):
        choice_probs = {}
        all_ctx_examples = {}
        task_probs = (
                torch.softmax(ensemble_choices[serialized_dict['task_id']], dim=-1) * 100
        ).tolist()
        for j, probs in enumerate(task_probs):
            choice_probs[f"PRED_{j}"] = {c: round(probs[ci], 4) for ci, c in enumerate(choice_list)}
            all_ctx_examples[f'PRED_{j}'] = ensemble_preds[serialized_dict['task_id']][j]

        has_correct_oracle = oracle_preds[i] != predictions[i]
        serialized_predictions.append({
            'has_correct_oracle': has_correct_oracle,
            'prob'              : choice_probs,
            'context_examples'  : all_ctx_examples,
            **serialized_dict
        })

    if cfg.task.name == 'npv':
        df = pd.DataFrame.from_records(serialized_predictions)

        df['is_negation'] = ~pd.isnull(df['is_negation_of'])
        df = df[['prediction', 'target', 'is_negation', 'is_original', 'is_manual_fix', 'op']]

        def get_subset_stats(name, subset_df):
            return {
                f"{name}_accuracy" : 100 * accuracy_score(
                    subset_df['target'],
                    subset_df['prediction']
                ),
                f"{name}_recall"   : 100 * recall_score(
                    subset_df['target'],
                    subset_df['prediction'],
                    average='micro',
                    labels=choice_list
                ),
                f"{name}_precision": 100 * precision_score(
                    subset_df['target'],
                    subset_df['prediction'],
                    average='micro',
                    labels=choice_list
                ),
                f"{name}_f1"       : 100 * f1_score(
                    subset_df['target'],
                    subset_df['prediction'],
                    average='micro',
                    labels=choice_list
                )
            }

        # metrics.update(get_subset_stats('Actual',df[df['is_original']]))
        metrics.update(get_subset_stats('Negations', df[df['is_negation']]))
        metrics.update(get_subset_stats('Original', df[~df['is_negation']]))
        # metrics.update(
        #     get_subset_stats('Generated', df[is_generated_mask])
        # )
        # metrics.update(
        #     get_subset_stats('Gold', df[~is_generated_mask])
        # )

    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        if isinstance(v, int):
            logger.info(f"\t{k:>20} = {v}")
        else:
            logger.info(f"\t{k:>20} = {v:0.3f}")

    return metrics, serialized_predictions


def evaluate(
        cfg,
        model,
        out_path: Path,
        dry_run: bool,
        print_csv_metrics: List
):
    # datasets.logging.disable_progress_bar()
    seed = cfg["seed"]
    logger.debug(f"Setting the seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.debug(f"Starting eval loop")
    start_time = datetime.utcnow()

    splits_to_use = cfg.splits
    logger.info(f"Using split '{splits_to_use}' for task '{cfg.task.name}'")
    all_metrics = {}
    split_paths = []
    if cfg.get('disable_cache', False):
        datasets.disable_caching()
    else:
        datasets.enable_caching()

    if cfg.task.name in ['npv']:
        eval_fn = evaluate_npv_task
    else:
        eval_fn = evaluate_model_generation_task

    if not dry_run:
        for split in splits_to_use:
            logger.info(f"Evaluating split {split}")
            with open_dict(cfg):
                cfg.split = split
            metrics, predictions = eval_fn(
                copy.deepcopy(cfg),
                model=model
            )

            if print_csv_metrics:
                logger.info(f"Metrics for {split}:")

                print_metrics = []
                for k in print_csv_metrics:
                    if isinstance(metrics[k], float):
                        print_metrics.append(f"{metrics[k]:.3f}")
                    else:
                        print_metrics.append(str(metrics[k]))
                logger.info('\t'.join(print_metrics))
            all_metrics.update({f"{split}/{k}": v for k, v in metrics.items()})
            split_path = out_path.joinpath(f'{cfg.split}.jsonl')
            split_paths.append(split_path)
            logger.info(f"Saving predictions to '{split_path}'")
            with split_path.open("w", encoding="utf-8") as f:
                for serialized_dict in predictions:
                    f.write(json.dumps(serialized_dict) + '\n')

    end_time = datetime.utcnow() - start_time
    logger.info(f"Total time spent on evaluation: {end_time}")
    all_metrics['runtime'] = str(end_time)
    if not dry_run:
        metric_file = out_path.joinpath(f'metrics.json')
        with metric_file.open('w', encoding='utf-8') as f:
            json.dump(all_metrics, f)

    run_id = os.getenv('WANDB_RUN_ID')
    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.eval_run_name = os.getenv('WANDB_RUN_NAME')

    with out_path.joinpath(f'config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
    #####################################################################
    # TRACKING CODE TO REMOVE ON RELEASE                                #
    #####################################################################

    if (
            isinstance(cfg.tracking, (dict, DictConfig))
            and int(os.environ.get("LOCAL_RANK", "-1")) <= 0
    ):
        cfg, run = initialize_run_from_cfg(
            cfg,
            f"{cfg.group}[eval]",
            job_type='evaluate'
        )

        run.log({k: v for k, v in all_metrics.items()}, step=1)
        preds_artifact = wandb.Artifact(get_run_base_name_from_cfg(cfg, "preds"),
                                        type='predictions')

        preds_artifact.add_dir(str(out_path.resolve().absolute()))
        run.log_artifact(preds_artifact)
        run.finish()
    logger.info("Finished Evaluation")


def make_eval_cfg_from_ctx(ctx, cfg):
    with open_dict(cfg):
        if 'evaluation' not in cfg:
            logger.warning("No evaluation dict found in the config, using a "
                           "default set of values.")
            cfg.evaluation = {
                "num_generate_per_step": cfg.pop("num_generate_per_step", 200),
                "remove_input_ids"     : cfg.pop("seq_per_sample", False),
            }
            cfg.evaluation["seq_per_sample"] = cfg.pop(
                "seq_per_sample",
                cfg.evaluation["num_generate_per_step"]
            )
        else:
            # Remove those keys because they are deprecated
            for k in cfg.evaluation:
                cfg.pop(k, None)

        if ctx.obj['splits']:
            logger.debug(f"Found splits override of {ctx.obj['splits']}")
            cfg.splits = ctx.obj['splits']
        elif 'splits' not in cfg:
            logger.debug(f"Using task {cfg.task.name}'s splits")
            cfg.splits = cfg.task.eval_splits

        if ctx.obj['num_generate_per_step']:
            logger.debug(
                f"Found num_generate_per_step override of {ctx.obj['num_generate_per_step']}")
            cfg.evaluation.num_generate_per_step = ctx.obj['num_generate_per_step']

        if ctx.obj['min_batch_size']:
            cfg.evaluation.min_batch_size = ctx.obj['min_batch_size']

        if ctx.obj['sequences_per_sample']:
            logger.debug(
                f"Found sequences_per_sample override of {ctx.obj['sequences_per_sample']}")
            cfg.evaluation.seq_per_sample = ctx.obj['sequences_per_sample']
        if ctx.obj['num_workers']:
            cfg.num_proc = ctx.obj['num_workers']

    return cfg
