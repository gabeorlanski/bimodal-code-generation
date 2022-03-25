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
from apex import amp
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


def oracle(args, metric_list):
    # To get the oracle score, we need to repeat target for every prediction
    predictions, target = args
    return target, [
        m.get_oracle_best_pred(predictions, target)
        for m in metric_list
    ]


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
        debug
):
    logger.info("Starting Generation")

    logger.info(f"Using batch size of {num_generate_per_step} and generating "
                f"{seq_per_sample} per sample")

    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    indices = []
    predictions = []
    labels = []
    batch_size = math.ceil(num_generate_per_step / seq_per_sample)
    logger.info(f"Using batch size {batch_size}")

    generate_steps_per_batch, remainder = divmod(seq_per_sample * batch_size, num_generate_per_step)
    has_remainder = remainder > 0

    amounts_to_generate = (
            [num_generate_per_step] * generate_steps_per_batch
            + [remainder] * has_remainder
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
    ).sort('length', reverse=True)
    tokenized.set_format(type='torch')

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_procs if num_procs != 1 else 0
    )

    model.eval()
    with torch.inference_mode():

        # Disable during debugging for my sanity.
        if not debug:
            progress_bar = tqdm(
                total=math.ceil(len(dataset) / batch_size) * amounts_to_generate,
                desc='Generating'
            )
        else:
            progress_bar = None
        completed = 0
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

            for i, num_to_generate in enumerate(amounts_to_generate):
                generated_from_batch = model.generate(
                    input_ids=local_inputs,
                    attention_mask=local_attention,
                    max_length=max_length_for_gen,
                    num_return_sequences=num_to_generate,
                    **generation_kwargs
                )

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

            pct_allocated = torch.cuda.max_memory_allocated(device) / total_memory
            logger.debug(
                f"{pct_allocated * 100:0.2f}% GPU memory allocated"
            )

            assert all(map(lambda x: len(x) == seq_per_sample, generated_for_current_batch))
            for idx, preds in zip(local_indices, generated_for_current_batch):
                predictions.append(preds)
                labels.append(dataset[idx]['target'])
                indices.append(idx)
            completed += len(local_indices)
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


def evaluate_model(
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
        debug=cfg.debug
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
        logger.info(f"\t{k:>20} = {v:0.3f}")

    serialized_predictions = []
    serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
    for serialized_dict in tqdm(serialize_generator, total=len(indices), desc="Serializing"):
        serialized_predictions.append(serialized_dict)

    return metrics, serialized_predictions


def evaluate(
        cfg,
        model,
        out_path: Path,
        dry_run: bool
):
    datasets.set_progress_bar_enabled(False)
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

    set_caching_enabled(not cfg.get('disable_cache', False))

    if not dry_run:
        for split in splits_to_use:
            logger.info(f"Evaluating split {split}")
            with open_dict(cfg):
                cfg.split = split
            metrics, predictions = evaluate_model(
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

        if dry_run and out_path.joinpath('eval_metrics.json').exists():
            all_metrics = json.loads(out_path.joinpath('eval_metrics.json').read_text('utf-8'))
            print(all_metrics)
        run.log({f"eval/{k}": v for k, v in all_metrics.items()}, step=1)
        preds_artifact = wandb.Artifact(get_run_base_name_from_cfg(cfg, "preds"),
                                        type='predictions')

        preds_artifact.add_dir(str(out_path.resolve().absolute()))
        preds_artifact.add_file(
            str(out_path.joinpath(f'eval_config.yaml').resolve().absolute()))
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

        if ctx.obj['sequences_per_sample']:
            logger.debug(
                f"Found sequences_per_sample override of {ctx.obj['sequences_per_sample']}")
            cfg.evaluation.seq_per_sample = ctx.obj['sequences_per_sample']
        if ctx.obj['num_workers']:
            cfg.num_proc = ctx.obj['num_workers']

    return cfg
