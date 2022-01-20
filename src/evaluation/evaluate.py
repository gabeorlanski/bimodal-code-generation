import json
import math
import os
from collections import defaultdict

import numpy as np
import wandb
from datasets import set_caching_enabled
from omegaconf import DictConfig
from transformers import PreTrainedModel, DataCollatorForSeq2Seq
import torch
import logging
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from src.config import get_device_from_cfg, merge_configs, load_task_from_cfg
from src.evaluation.util import serialize_prediction
from src.config.tracking import get_config_for_tracking

logger = logging.getLogger(__name__)


def generate_predictions(
        model,
        tokenized,
        task,
        batch_size,
        device,
        generation_kwargs,
        seq_per_sample
):
    collator = DataCollatorForSeq2Seq(
        tokenizer=task.tokenizer,
        pad_to_multiple_of=1,
        max_length=1024,
        padding="longest",
        label_pad_token_id=task.tokenizer.pad_token_id,
    )
    tokenized = tokenized.map(
        lambda ex: {'length': sum(ex['attention_mask']), **ex}
    )
    tokenized = tokenized.sort('length', reverse=True)
    tokenized = tokenized.filter(lambda ex: ex['length'] < 800)
    tokenized = tokenized.remove_columns('length')
    data_loader = torch.utils.data.DataLoader(
        tokenized,
        collate_fn=collator,
        shuffle=False,
        batch_size=batch_size,
    )

    logger.info("Starting Generation")

    logger.info(f"Using batch size of {batch_size} and generating "
                f"{seq_per_sample} per sample")
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    indices = []
    predictions = []
    labels = []
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    generate_steps_per_sample, rem = divmod(seq_per_sample, num_return_sequences)
    if rem > 0:
        logger.error(f"{seq_per_sample}/{num_return_sequences} sequences had a "
                     f"remainder of {rem}")
        raise ValueError(
            "seq_per_sample must be divisible by generation_kwargs.num_return_sequences"
        )
    logger.debug(f"{generate_steps_per_sample} steps per sample")
    num_steps_needed = generate_steps_per_sample * len(data_loader)
    logger.info(f"{num_steps_needed} total steps needed")

    model.eval()
    with torch.inference_mode():
        progress_bar = tqdm(total=num_steps_needed, desc='Generating')
        for batch in data_loader:

            generated_results = [None for _ in range(generate_steps_per_sample)]
            local_inputs = batch["input_ids"].to(device)
            local_attn = batch['attention_mask'].to(device)
            for i in range(generate_steps_per_sample):
                generated_from_batch = model.generate(
                    input_ids=local_inputs,
                    attention_mask=local_attn,
                    top_k=0,
                    **generation_kwargs
                )
                generated_results[i] = generated_from_batch.tolist()
                progress_bar.update(1)

            b = batch['input_ids'].size()[0]
            pred_tensors = [None for _ in range(seq_per_sample * b)]
            max_len = max(map(len,generated_results))
            for i, gen in enumerate(generated_results):
                for j, pred in enumerate(gen):
                    seq_idx, offset = divmod(j, num_return_sequences)
                    idx = (i * num_return_sequences) + seq_idx * seq_per_sample + offset
                    pred_tensors[idx] = pred+[task.tokenizer.pad_token_id]*(max_len-len(pred))

            preds = np.vstack(pred_tensors)
            preds[preds == 0] = task.tokenizer.pad_token_id
            postprocessed_preds, postprocessed_targets = task.postprocess(
                preds,
                batch["labels"].numpy()
            )
            for i in range(batch['input_ids'].shape[0]):
                preds = postprocessed_preds[i * seq_per_sample:(i + 1) * seq_per_sample]
                gold = postprocessed_targets[i]
                # assert len(preds) == seq_per_sample, f"{len(preds)} != {seq_per_sample}"
                # assert isinstance(gold, str)
                predictions.append(preds)
                labels.append(gold)
                indices.append(batch['idx'][i].detach().item())

        progress_bar.close()

    logger.info("Generating finished.")
    return {
        "indices"    : indices,
        "labels"     : labels,
        "predictions": predictions
    }


def distributed_generate(rank, world_size, generate_pred_kwargs):
    pass


def evaluate_model(cfg: DictConfig, model: PreTrainedModel):
    """
    Evaluate a model with a reader on a file
    Args:
        cfg (DictConfig): The config to use.
        model (PreTrainedModel): The pretrained huggingface model to use.

    """
    set_caching_enabled(not cfg.get('disable_cache', False))
    task = load_task_from_cfg(cfg)
    logger.info(f"Reading data from '{cfg['data_path']}'")

    tokenized = task.get_split(cfg['split'], overwrite_cache=True)
    logger.info(f"{len(tokenized)} total samples found")

    logger.info("Initializing the evaluator")

    if cfg.objective == 'lm':
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token = task.tokenizer.eos_token

    device = get_device_from_cfg(cfg)
    logger.info(f"Using device {device}")

    generation_results = generate_predictions(
        model.to(device),
        tokenized=tokenized,
        task=task,
        batch_size=cfg["training"].get(
            "batch_size",
            cfg['training'].get('per_device_eval_batch_size',
                                cfg['training'].get('batch_size', 1))
        ),
        device=device,
        generation_kwargs=cfg.get('generation', {}),
        seq_per_sample=cfg.get('seq_per_sample')
    )

    labels = generation_results['labels']
    predictions = generation_results['predictions']
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
