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
from transformers import PreTrainedModel, StoppingCriteria, StoppingCriteriaList
import torch
import logging
from tqdm import tqdm
from src.config import (
    get_device_from_cfg, load_task_from_cfg, \
    get_run_base_name_from_cfg, initialize_run_from_cfg
)
from apex import amp
import multiprocessing as mp

logger = logging.getLogger(__name__)

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


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
    generate_steps_per_sample, remainder = divmod(seq_per_sample, num_generate_per_step)
    has_remainder = remainder > 0

    amounts_to_generate = [num_generate_per_step] * generate_steps_per_sample + [
        remainder] * has_remainder

    logger.debug(f"{len(amounts_to_generate)} steps per sample")

    max_length = generation_kwargs.pop('max_length', 256)
    if 'max_new_tokens' in generation_kwargs:
        max_new_tokens = generation_kwargs.pop('max_new_tokens')
        if 'max_length' not in generation_kwargs:
            max_length = max_new_tokens

    total_memory = torch.cuda.mem_get_info(device)[1]  # type: ignore
    tokenizer.padding_side = 'left' if objective == 'lm' else 'right'

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

    dataloader = torch.utils.data.DataLoader(tokenized, batch_size=1)

    model.eval()
    with torch.inference_mode():

        # Disable during debugging for my sanity.
        if not debug:
            progress_bar = tqdm(total=seq_per_sample * len(dataset), desc='Generating')
        else:
            progress_bar = None

        for instance in dataloader:
            idx = instance['idx'].item()
            generated_for_current_sample = []
            local_inputs = instance["input_ids"].to(device)
            local_attention = instance['attention_mask'].to(device)
            input_len = instance['length'].item()

            max_length_for_gen = max_length
            if objective == 'lm':
                if max_length + input_len > tokenizer.model_max_length:
                    logger.warning(
                        f"Sample {idx} has more than the "
                        f"models max length of {tokenizer.model_max_length}."
                    )
                    # Subtract 4 to be safe.
                    max_length_for_gen = tokenizer.model_max_length - 4
                else:
                    max_length_for_gen = input_len + max_length

            if 'stopping_criteria' in generation_kwargs:
                for sc in generation_kwargs['stopping_criteria']:
                    if hasattr(sc, 'start_length'):
                        sc.start_length = input_len

            for num_to_generate in amounts_to_generate:
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
                    progress_bar.update(num_to_generate)
                generated_for_current_sample.extend(tokenizer.batch_decode(
                    ids_for_current_sample,
                    skip_special_tokens=True
                ))

            pct_allocated = torch.cuda.max_memory_allocated(device) / total_memory
            logger.debug(
                f"{pct_allocated * 100:0.2f}% GPU memory allocated")
            assert len(generated_for_current_sample) == seq_per_sample
            predictions.append(generated_for_current_sample)
            labels.append(dataset[idx]['target'])
            indices.append(idx)
            if not progress_bar:
                logger.info(f"Finished {idx + 1}/{len(dataset)} generations")
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
    if cfg.objective == 'lm':
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token = task.tokenizer.eos_token
        model.config.eos_token_id = task.tokenizer.eos_token_id
        model.config.pad_token_id = task.tokenizer.pad_token_id
        model.config.bos_token_id = task.tokenizer.bos_token_id or task.tokenizer.eos_token

    if cfg.task.name == 'human_eval':
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [EndOfFunctionCriteria(0, EOF_STRINGS, task.tokenizer)]
        )

    logger.info(f"Getting the data for split {cfg.split}")
    dataset = task.preprocess(cfg.split)
    logger.info(f"{len(dataset)} total samples found")
    debug_num_samples = cfg.get('debug_num_samples', None)
    if debug_num_samples is not None:
        logger.warning(f"DEBUG NUMBER OF SAMPLES={debug_num_samples}")
        dataset = dataset.select(list(range(debug_num_samples)))

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
        num_generate_per_step=cfg.num_generate_per_step,
        device=device,
        generation_kwargs=gen_kwargs,
        seq_per_sample=cfg.seq_per_sample,
        remove_input_ids_from_output=cfg.get("remove_input_ids", False),
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
        dry_run: bool,
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

    pred_dir = Path(out_path).joinpath('predictions')
    if not pred_dir.exists():
        pred_dir.mkdir()
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
            split_path = pred_dir.joinpath(f'{cfg.split}.jsonl')
            split_paths.append(split_path)
            logger.info(f"Saving predictions to '{split_path}'")
            with split_path.open("w", encoding="utf-8") as f:
                for serialized_dict in predictions:
                    f.write(json.dumps(serialized_dict) + '\n')

    end_time = datetime.utcnow() - start_time
    logger.info(f"Total time spent on evaluation: {end_time}")
    all_metrics['runtime'] = str(end_time)
    if not dry_run:
        with out_path.joinpath('eval_metrics.json').open('w', encoding='utf-8') as f:
            json.dump(all_metrics, f)

    run_id = os.getenv('WANDB_RUN_ID')
    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.eval_run_name = os.getenv('WANDB_RUN_NAME')

    with out_path.joinpath(f'eval_config.yaml').open('w') as f:
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

        preds_artifact.add_dir(str(pred_dir.resolve().absolute()))
        preds_artifact.add_file(
            str(out_path.joinpath(f'eval_config.yaml').resolve().absolute()))
        run.log_artifact(preds_artifact)
        run.finish()
    logger.info("Finished Evaluation")
