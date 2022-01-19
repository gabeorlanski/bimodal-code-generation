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
        lambda ex: {'length': len(ex['input_ids']), **ex}
    )
    tokenized = tokenized.sort('length', reverse=True)
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
    with torch.no_grad():
        progress_bar = tqdm(total=num_steps_needed, desc='Generating')
        for batch in data_loader:
            postprocessed_preds = []
            postprocessed_targets = []

            for _ in range(generate_steps_per_sample):
                generated_from_batch = model.generate(
                    input_ids=batch["input_ids"].to(device),
                    labels=batch['labels'].to(device),
                    **generation_kwargs
                )

                # We need to check how many sequences we return for each sample so
                # we can adequately collect them.
                b = batch['input_ids'].size()[0]

                generated = generated_from_batch.reshape(
                    (b, num_return_sequences, -1)).detach().cpu()
                targets = batch["labels"].detach().cpu()

                b_preds, b_targets = task.postprocess(
                    generated.numpy(),
                    targets.numpy()
                )

                if not postprocessed_targets:
                    postprocessed_preds = b_preds
                    postprocessed_targets = b_targets
                else:
                    for i, pred in enumerate(b_preds):
                        postprocessed_preds[i].extend(pred)
                progress_bar.update()

            for i in range(batch['input_ids'].shape[0]):
                preds = postprocessed_preds[i]
                gold = postprocessed_targets[i]

                if len(preds) != seq_per_sample:
                    raise Exception("??")
                assert len(preds) == seq_per_sample, f"{len(preds)} != {seq_per_sample}"
                assert isinstance(gold, str)

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
        model,
        tokenized=tokenized,
        task=task,
        batch_size=cfg["training"].get(
            "batch_size",
            cfg['training'].get('per_device_eval_batch_size', cfg['training'].get('batch_size', 1))
        ),
        device=device,
        generation_kwargs=cfg.get('generation', {}),
        seq_per_sample=cfg.get('seq_per_sample')
    )

    # Unpack the returned dict from generate predictions
    raw_data = task.preprocessed_splits[cfg['split']]
    labels = generation_results['labels']

    predictions = generation_results['predictions']
    indices = generation_results['indices']

    metrics = task.evaluate(predictions, labels)
    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.3f}")

    out_path = cfg.get('out_path', cfg['model_path'])
    out_path = Path(out_path)
    pred_path = out_path.joinpath('predictions.jsonl')
    logger.info(f"Saving predictions to {pred_path}")
    with pred_path.open("w", encoding="utf-8") as f:
        serialize_generator = task.serialize_predictions(cfg.split, indices, predictions)
        for serialized_dict in tqdm(serialize_generator, total=len(indices), desc="Saving"):
            f.write(json.dumps(serialized_dict) + '\n')

    run_id = wandb.util.generate_id()
    os.environ['RUN_ID'] = run_id
    if (
            isinstance(cfg.tracking, (dict, DictConfig))
            and int(os.environ.get("LOCAL_RANK", "-1")) <= 0
    ):
        run = wandb.init(
            job_type='evaluate',
            name=cfg.name,
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            group=cfg.group,
            config=get_config_for_tracking(cfg),
            id=run_id
        )
        run.log({f"eval/{k}": v for k, v in metrics.items()}, step=1)
        preds_artifact = wandb.Artifact(f"{cfg.group}.{cfg.name}.{cfg.task.name}",
                                        type='predictions')
        preds_artifact.add_file(str(pred_path.resolve().absolute()))
        run.log_artifact(preds_artifact)
        run.finish()

    with out_path.joinpath('eval_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(metrics, f)

    return metrics
