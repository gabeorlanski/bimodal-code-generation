import math

from datasets import set_caching_enabled
from omegaconf import DictConfig, open_dict
from transformers import PreTrainedModel, DataCollatorForSeq2Seq, PreTrainedTokenizer
import torch
from torch.nn import functional as F
import logging
from tqdm import tqdm
from pathlib import Path

from src.config import get_device_from_cfg, merge_configs, load_task_from_cfg
from src.evaluation.util import serialize_prediction

logger = logging.getLogger(__name__)


def generate_predictions(
        model,
        tokenized,
        task,
        batch_size,
        device,
        generation_kwargs,
        generate_steps
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
    tokenized = tokenized.sort('length')
    tokenized = tokenized.remove_columns('length')
    data_loader = torch.utils.data.DataLoader(
        tokenized,
        collate_fn=collator,
        shuffle=False,
        batch_size=batch_size,
    )

    logger.info("Starting Generation")
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    indices = []
    predictions = []
    labels = []
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    num_steps_needed = generate_steps * len(data_loader)
    logger.info(f"{num_steps_needed} total steps needed")

    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(total=num_steps_needed, desc='Generating')
        for batch in data_loader:
            postprocessed_preds = []
            postprocessed_targets = []

            for _ in range(generate_steps):
                generated_from_batch = model.generate(
                    input_ids=batch["input_ids"].to(device),
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

                postprocessed_preds.extend(b_preds)
                postprocessed_targets.extend(b_targets)
                progress_bar.update()

            for i in range(batch['input_ids'].shape[0]):
                preds = postprocessed_preds[i]
                gold = postprocessed_targets[i]

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


def evaluate_model(cfg: DictConfig, train_cfg: DictConfig, model: PreTrainedModel):
    """
    Evaluate a model with a reader on a file
    Args:
        cfg (DictConfig): The config to use.
        train_cfg (DictConfig): The training config.
        model (PreTrainedModel): The pretrained huggingface model to use.

    """
    # Need to add keys from training that would not show up in the evaluation
    # config.
    cfg = merge_configs(cfg, train_cfg)
    set_caching_enabled(not cfg.get('disable_cache', False))
    task = load_task_from_cfg(cfg)
    logger.info(f"Reading data from '{cfg['data_path']}'")

    tokenized = task.get_split(cfg['split'], overwrite_cache=True)
    logger.info(f"{len(tokenized)} total samples found")

    logger.info("Initializing the evaluator")

    if cfg.objective == 'lm':
        if task.tokenizer.pad_token is None:
            task.tokenizer.pad_token = task.tokenizer.eos_token

    generation_results = generate_predictions(
        model,
        tokenized=tokenized,
        task=task,
        batch_size=cfg["training"].get(
            "batch_size",
            cfg['training'].get('per_device_eval_batch_size', 1)
        ),
        device=get_device_from_cfg(cfg),
        generation_kwargs=cfg.get('generation', {}),
        generate_steps=cfg.get('generate_steps')
    )

    # Unpack the returned dict from generate predictions
    indices, predictions = [], []
    raw_data = task.preprocessed_splits[cfg['split']]
    labels = generation_results['labels']
    for idx, preds in zip(generation_results['indices'], generation_results['predictions']):
        input_sequence = raw_data[idx]["input_sequence"]
        indices.append(idx)
        if cfg.objective == "lm":
            # Remove the prompt from the predictions
            predictions.append(list(map(lambda p: p[len(input_sequence):], preds)))
        else:
            predictions.append(predictions)
    metrics = task.evaluate(predictions, labels)

    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.3f}")

    out_path = cfg.get('out_path', None)
    out_path = Path(out_path) if out_path else Path()
    out_path = out_path.joinpath('predictions.jsonl')
    logger.info(f"Saving predictions to {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for idx, preds in tqdm(zip(indices, predictions), desc="Saving"):
            f.write(serialize_prediction(
                idx=idx,
                input_sequence=raw_data[idx]["input_sequence"],
                target=raw_data[idx]["target"],
                predictions=preds

            ) + '\n')

    return metrics
