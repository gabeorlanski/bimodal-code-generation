from omegaconf import DictConfig, open_dict
from transformers import PreTrainedModel, DataCollatorForSeq2Seq, PreTrainedTokenizer
import torch
import logging
from tqdm import tqdm
from pathlib import Path

from yamrf import Task, load_task_from_cfg

from src.common import PROJECT_ROOT
from src.common.config import get_device_from_cfg
from src.evaluation.evaluator import Evaluator
from src.evaluation.util import serialize_prediction

logger = logging.getLogger(__name__)


def generate_predictions(
        model,
        tokenized,
        task,
        batch_size,
        device,
        generation_kwargs,
        postprocess_batch_count=5
):
    collator = DataCollatorForSeq2Seq(
        tokenizer=task.tokenizer,
        pad_to_multiple_of=2,
        max_length=1024,
        padding="longest",
        label_pad_token_id=task.tokenizer.pad_token_id,
    )
    tokenized = tokenized.map(
        lambda ex: {'length': len(ex['input_ids']), **ex}
    )
    tokenized = tokenized.sort('length')
    tokenized = tokenized.map(
        lambda ex: ex,
        remove_columns=['length']
    )
    data_loader = torch.utils.data.DataLoader(
        tokenized,
        collate_fn=collator,
        shuffle=False,
        batch_size=batch_size,
    )

    max_label_len = max(len(d) for d in tokenized['labels'])

    logger.info("Starting Generation")
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    indices = []
    predictions = []
    labels = []
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating"):
            generated_from_batch = model.generate(
                inputs=batch["input_ids"].to(device),
                **generation_kwargs,
            )

            # We need to check how many sequences we return for each sample so
            # we can adequately collect them.
            b = batch['input_ids'].size()[0]

            generated = generated_from_batch.reshape((b, num_return_sequences, -1)).detach().cpu()
            targets = batch["labels"].detach().cpu()

            postprocessed_preds, postprocessed_targets = task.postprocess(
                generated,
                targets
            )

            for i in range(batch['input_ids'].shape[0]):
                preds = postprocessed_preds[i]
                gold = postprocessed_targets[i]
                # Only use the first returned result for basic evaluation,
                # maybe later will be more advanced.
                predictions.append(preds)
                labels.append(gold)
                indices.append(batch['idx'][i].item())

    logger.info("Generating finished.")
    return {
        "indices"    : indices,
        "labels"     : labels,
        "predictions": predictions
    }


def merge_train_cfg_with_eval_cfg(cfg: DictConfig, train_cfg: DictConfig):
    with open_dict(cfg):
        for k, v in train_cfg.items():
            if k not in cfg:
                cfg[k] = v
    return cfg


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
    cfg = merge_train_cfg_with_eval_cfg(cfg, train_cfg)

    task = load_task_from_cfg(cfg)

    logger.info(f"Reading data from '{cfg['data_path']}'")
    tokenized = task.get_dataset(cfg['split'])
    logger.info(f"{len(tokenized)} total samples found")

    logger.info("Initializing the evaluator")

    generation_results = generate_predictions(
        model,
        tokenized,
        task,
        batch_size=cfg["training"].get("batch_size", 4),
        device=get_device_from_cfg(cfg),
        generation_kwargs=cfg.get('generation')
    )
    indices = generation_results['indices']
    predictions = generation_results['predictions']
    labels = generation_results['labels']

    metrics = task.evaluate(predictions, labels)

    # Get the full metrics suite for the predictions and the labels
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.3f}")

    logger.info(f"Saving predictions to {Path('predictions.jsonl')}")
    raw_data = task.preprocessed_splits[cfg['split']]
    with Path("predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, preds in tqdm(zip(indices, predictions), desc="Saving"):
            f.write(serialize_prediction(
                idx=idx,
                input_sequence=raw_data[idx]["input_sequence"],
                target=raw_data[idx]["target"],
                predictions=preds
            ) + '\n')

    return metrics
