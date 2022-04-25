import copy
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from functools import partial
from itertools import chain
from pathlib import Path
import random
from typing import List, Union

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


def evaluate_model_classification_task(
        cfg: DictConfig,
        model: PreTrainedModel
):
    task = load_task_from_cfg(cfg)
    logger.info(f"Reading data from '{cfg['data_path']}'")
    debug = cfg.debug

    assert hasattr(task, 'choices')
    choice_list = list(map(str, task.choices))  # type: ignore
    logger.info(f"Getting the data for split {cfg.split}")
    dataset = task.preprocess(cfg.split)
    debug_num_samples = cfg.get('debug_num_samples', None)
    if cfg.objective == 'lm':
        eos_token = task.tokenizer.eos_token or task.tokenizer.bos_token
        task.tokenizer.eos_token = eos_token
        task.tokenizer.bos_token = eos_token
        task.tokenizer.pad_token = task.tokenizer.eos_token
        model.config.eos_token_id = task.tokenizer.eos_token_id
        model.config.pad_token_id = task.tokenizer.eos_token_id
        model.config.bos_token_id = task.tokenizer.eos_token_id
        task.tokenizer.padding_side = 'left'
        task.tokenizer.truncation_side = 'left'

    def tokenize(example, idx):
        # We do not pop so that we can still remove the columns later.
        out = {
            "idx": idx,
            **task.tokenizer(
                example["input_sequence"],
                truncation=True,
                max_length=task.tokenizer.model_max_length - 1
            )
        }
        target_tokenized = task.tokenizer(
            example['target'],
            truncation=True,
            max_length=task.tokenizer.model_max_length - 1
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
    logger.info(f"{len(dataset)} total samples found")

    if debug_num_samples is not None or debug:

        logger.info(f"Adding length to the dataset")
        tokenized = tokenized.map(
            lambda ex: {
                'len': len(ex['input_ids']),
                **ex
            }
        )
        logger.info(f'IN DEBUG MODE')
        if debug_num_samples is None:
            logger.info("Sorting by length")
        tokenized = tokenized.sort('len', reverse=debug_num_samples is None)
        if debug_num_samples is not None:
            logger.warning(f"DEBUG NUMBER OF SAMPLES={debug_num_samples}")
            tokenized = tokenized.select(list(range(debug_num_samples)))

    device = get_device_from_cfg(cfg)
    logger.info(f"Putting model on {device}")
    model = model.to(device)
    # model = amp.initialize(model)
    logger.info(f"Model is on {model.device}")
    logger.debug(f"{type(dataset)=}")
    collator = DataCollatorForSeq2Seq(
        tokenizer=task.tokenizer,
        padding='longest',
        pad_to_multiple_of=1,
        return_tensors='pt',
        label_pad_token_id=task.tokenizer.pad_token_id
    )

    choices_tokenized = [task.tokenizer(str(c))['input_ids'] for c in choice_list]  # type:ignore

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
    pred_probs = []
    targets = []
    indices = []

    model.eval()
    with torch.inference_mode():
        if not debug:
            progress_bar = tqdm(
                total=math.ceil(len(dataset) / batch_size),
                desc='Generating'
            )
        else:
            progress_bar = None
        completed = 0
        longest_choice = max(map(len, choices_tokenized))
        for batch in dataloader:
            n_seqs = batch['input_ids'].size(0)
            input_size = batch['input_ids'].size(1)
            max_len = input_size + longest_choice

            local_input_ids = torch.zeros((n_seqs, max_len)).long()
            local_input_ids[:, :input_size] = batch['input_ids']
            local_attention_mask = torch.zeros((n_seqs, max_len)).long()
            local_attention_mask[:, :input_size] = batch['attention_mask']

            local_predictions = defaultdict(list)
            targets.extend([dataset[i.item()]['target'] for i in batch['idx']])

            for choice, choice_tokens in zip(choice_list, choices_tokenized):

                choice_tensor = torch.tensor([choice_tokens] * n_seqs).long()
                labels = -100 * torch.ones((n_seqs, max_len)).long()
                labels[:, input_size:input_size + choice_tensor.size(1)] = choice_tensor

                local_input_ids = local_input_ids.to(device)
                local_attention_mask = local_attention_mask.to(device)

                # This was heavily inspired and has elements from:
                # https://github.com/peterwestuw/surface-form-competition/blob/main/utils.py
                logits = model(
                    input_ids=local_input_ids,
                    attention_mask=local_attention_mask,
                    # labels=local_input_ids
                ).logits.cpu()[:, :-1].contiguous()
                logit_shape = logits.shape

                logits = logits.view(-1, logit_shape[-1])

                ce_list = F.cross_entropy(
                    logits,
                    labels[:, 1:].contiguous().view(-1),
                    reduction='none'
                )
                ce_list = ce_list.view(n_seqs, max_len - 1).sum(dim=1).squeeze().tolist()
                try:
                    len(ce_list)
                except:
                    ce_list = [ce_list]
                for idx, p in zip(batch['idx'], ce_list):
                    local_predictions[idx.item()].append(p)

            for k, v in local_predictions.items():
                indices.append(dataset[k]['task_id'])
                pred_probs.append(v)

            if progress_bar:
                progress_bar.update(1)
            completed += n_seqs
            if not progress_bar:
                logger.info(f"Finished {completed}/{len(dataset)} generations")

    pred_probs = torch.tensor(pred_probs)
    pred_ints = pred_probs.argmin(dim=-1)
    predictions = list(map(lambda i: choice_list[i], pred_ints.tolist()))

    metrics = {
        "accuracy" : 100 * accuracy_score(
            targets,
            predictions
        ),
        "f1"       : 100 * f1_score(
            targets,
            predictions,
            labels=choice_list,
            average='macro'
        ),
        "recall"   : 100 * recall_score(
            targets,
            predictions,
            average='macro',
            labels=choice_list
        ),
        "precision": 100 * precision_score(
            targets,
            predictions,
            average='macro',
            labels=choice_list
        ),
    }

    precision, recall, f1_arr, occurrences = precision_recall_fscore_support(
        targets, predictions, average=None, labels=choice_list
    )

    pred_counts = pred_ints.bincount()
    for i, (p, r, f1, o) in enumerate(zip(precision, recall, f1_arr, occurrences)):
        metrics[f'{choice_list[i]}_precision'] = p * 100
        metrics[f'{choice_list[i]}_recall'] = r * 100
        metrics[f'{choice_list[i]}_f1'] = f1 * 100
        metrics[f'{choice_list[i]}_count'] = pred_counts[i].item()

    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        if isinstance(v, int):
            logger.info(f"\t{k:>20} = {v}")
        else:
            logger.info(f"\t{k:>20} = {v:0.3f}")

    # Apply softmax to rescale the log probabilities (also multiply by -1 as CE
    # returns a positive number) then multiply by 100 and round to 5 decimal
    # places for sanity. Then convert back to a list for saving.
    pred_probs = (torch.softmax(pred_probs * -1, dim=-1) * 100).tolist()

    serialized_predictions = []
    serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
    for i, serialized_dict in tqdm(enumerate(serialize_generator), total=len(indices),
                                   desc="Serializing"):
        choice_probs = {c: round(pred_probs[i][j], 5) for j, c in enumerate(choice_list)}
        serialized_predictions.append({'prob': choice_probs, **serialized_dict})

    return metrics, serialized_predictions


def evaluate(
        cfg,
        model,
        out_path: Path,
        dry_run: bool
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
        eval_fn = evaluate_model_classification_task
    else:
        eval_fn = evaluate_model_classification_task

    if not dry_run:
        for split in splits_to_use:
            logger.info(f"Evaluating split {split}")
            with open_dict(cfg):
                cfg.split = split
            metrics, predictions = eval_fn(
                copy.deepcopy(cfg),
                model=model
            )

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
