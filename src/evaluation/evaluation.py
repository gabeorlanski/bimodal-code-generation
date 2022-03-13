import copy
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path
import random
from typing import List, Union

import datasets
import numpy as np
import wandb
from datasets import set_caching_enabled, Dataset
from omegaconf import DictConfig, open_dict, OmegaConf
from transformers import PreTrainedModel, DataCollatorForSeq2Seq, pipeline, StoppingCriteria, \
    MaxLengthCriteria, StoppingCriteriaList
import torch
import logging
from tqdm import tqdm
from src.config import get_device_from_cfg, load_task_from_cfg, get_config_for_tracking, \
    get_run_base_name_from_cfg
from apex import amp

logger = logging.getLogger(__name__)


class EOSStoppingCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, tokenizer):
        self.eos_token = tokenizer.eos_token_id

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""

        return all(self.eos_token in row[1:] for row in input_ids)


def generate_code_predictions(
        model,
        objective,
        dataset: Union[List[dict], Dataset],
        tokenizer,
        batch_size,
        device,
        generation_kwargs,
        seq_per_sample,
        remove_input_ids_from_output,
        debug
):
    logger.info("Starting Generation")
    if any(dataset[i - 1]['length'] < dataset[i]['length'] for i in range(1, len(dataset))):
        raise ValueError(f"The custom generation method only works if the "
                         f"dataset is sorted by length.")

    logger.info(f"Using batch size of {batch_size} and generating "
                f"{seq_per_sample} per sample")

    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
        [EOSStoppingCriteria(tokenizer)])
    indices = []
    predictions = []
    labels = []
    generate_steps_per_sample, remainder = divmod(seq_per_sample, batch_size)
    has_remainder = remainder > 0

    amounts_to_generate = [batch_size] * generate_steps_per_sample + [remainder] * has_remainder

    logger.debug(f"{generate_steps_per_sample} steps per sample")

    max_length = generation_kwargs.pop('max_length', 256)
    if 'max_new_tokens' in generation_kwargs:
        max_new_tokens = generation_kwargs.pop('max_new_tokens')
        if 'max_length' not in generation_kwargs:
            max_length = max_new_tokens

    total_memory = torch.cuda.mem_get_info(device)[1]
    tokenizer.padding_side = 'left'

    model.eval()
    with torch.inference_mode():

        # Disable during debugging for my sanity.
        if not debug:
            progress_bar = tqdm(total=seq_per_sample * len(dataset), desc='Generating')
        else:
            progress_bar = None

        for idx, sample in enumerate(dataset):
            generated_for_current_sample = []
            sample_tokenized = tokenizer(sample['input_sequence'], return_tensors='pt')
            local_inputs = sample_tokenized["input_ids"].to(device)
            local_attention = sample_tokenized['attention_mask'].to(device)
            input_len = dataset[idx]['length']

            max_length_for_gen = max_length
            if objective == 'lm':
                if max_length + input_len > tokenizer.model_max_length:
                    logger.warning(
                        f"Sample {sample['idx']} has more than the "
                        f"models max length of {tokenizer.model_max_length}."
                    )
                    # Subtract 4 to be safe.
                    max_length_for_gen = tokenizer.model_max_length - 4
                else:
                    max_length_for_gen = input_len + max_length

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
                f"{pct_allocated * 100:0.2f}% allocated")
            assert len(generated_for_current_sample) == seq_per_sample
            predictions.append(generated_for_current_sample)
            labels.append(sample['target'])
            indices.append(sample['idx'])
            if not progress_bar:
                logger.info(f"Finished {idx}/{len(dataset)} generations")
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

        def prepend_token(sample):
            sample['input_sequence'] = task.tokenizer.eos_token + sample['input_sequence']
            return sample

        task.preprocessors.append(prepend_token)

    logger.info(f"Getting the data for split {cfg.split}")
    dataset = task.preprocess(cfg.split)
    logger.info(f"{len(dataset)} total samples found")
    debug_num_samples = cfg.get('debug_num_samples', None)
    if debug_num_samples is not None:
        logger.warning(f"DEBUG NUMBER OF SAMPLES={debug_num_samples}")
        dataset = dataset.select(list(range(debug_num_samples)))

    logger.info("Sorting the dataset by length")

    def get_len(ex):
        ex['length'] = len(task.tokenizer.tokenize(ex['input_sequence']))
        return ex

    dataset = dataset.map(
        get_len,
        num_proc=cfg.get('num_workers', 1),
    ).sort('length', reverse=True)

    device = get_device_from_cfg(cfg)
    model = model.to(device)
    model = amp.initialize(model)
    logger.info(f"Model is on {model.device}")
    logger.info(f"{type(dataset)=}")

    generation_results = generate_code_predictions(
        model,
        objective=cfg.objective,
        dataset=dataset,
        tokenizer=task.tokenizer,
        batch_size=cfg.batch_size,
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
        run = wandb.init(
            job_type='evaluate',
            name=os.getenv('WANDB_RUN_NAME'),
            project=os.getenv('WANDB_PROJECT'),
            group=f"{cfg.group}[eval]",
            entity=os.getenv('WANDB_ENTITY'),
            config=get_config_for_tracking(cfg),
            id=run_id,
            tags=os.getenv('WANDB_RUNS_TAGS').split(','),
        )

        run.config.update(get_config_for_tracking(cfg))

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
