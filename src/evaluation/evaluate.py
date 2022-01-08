import json
from dataclasses import asdict

from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel, AutoTokenizer, DataCollatorForSeq2Seq
import torch
import logging
from tqdm import tqdm
from pathlib import Path

from src.common import PROJECT_ROOT
from src.common.config import get_device_from_cfg
from src.evaluation.evaluator import Evaluator
from src.data import Task, load_task_from_cfg

logger = logging.getLogger(__name__)


def evaluate_model(cfg: DictConfig, train_cfg: DictConfig, model: PreTrainedModel):
    """
    Evaluate a model with a reader on a file
    Args:
        cfg (DictConfig): The config to use.
        train_cfg (DictConfig): The training config.
        model (PreTrainedModel): The pretrained huggingface model to use.

    """
    logger.info(f"Loading reader '{train_cfg['task']['name']}'")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg["model"])
    task = load_task_from_cfg(cfg, tokenizer)

    logger.info(f"Reading data from '{cfg['data_path']}'")
    raw_data, tokenized = task.read_data(
        PROJECT_ROOT.joinpath(cfg["data_path"]), set_format="torch"
    )
    logger.info(f"{len(tokenized)} total samples found")
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=2,
        max_length=1024,
        padding="longest",
        label_pad_token_id=tokenizer.pad_token_id,
    )
    data_loader = torch.utils.data.DataLoader(
        tokenized,
        collate_fn=collator,
        shuffle=False,
        batch_size=train_cfg["training"]["batch_size"],
    )

    logger.info("Starting Generation")
    generation_kwargs = cfg.get("generation", {})
    logger.info("Generation kwargs:")
    for k, v in generation_kwargs.items():
        logger.info(f"\t{k:>20} = {v}")

    generated = []
    predictions = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating"):
            generated_from_batch = model.generate(
                inputs=batch["input_ids"].to(get_device_from_cfg(train_cfg)),
                **generation_kwargs,
            )

            # We need to check how many sequences we return for each sample so
            # we can adequately collect them.
            num_return_sequences = generation_kwargs.get("num_return_sequences", 1)

            for i in range(batch["input_ids"].shape[0]):
                preds = task.tokenizer.batch_decode(
                    generated_from_batch[
                        i * num_return_sequences : (i + 1) * num_return_sequences
                    ],
                    skip_special_tokens=True,
                )

                gold = tokenizer.decode(batch["labels"][i], skip_special_tokens=True)

                # Only use the first returned result for basic evaluation,
                # maybe later will be more advanced.
                predictions.append(preds[0])
                labels.append(gold)
                generated.append((batch["idx"][i].item(), preds))

    logger.info("Initializing the evaluator")
    # We want the union of metrics from both the training config and the eval
    # config b/c some metrics do not need to be used during training.
    evaluator = Evaluator(
        task.tokenizer,
        list(set(cfg.get("metrics", [])).union(train_cfg.get("metrics", []))),
    )

    metrics = evaluator(predictions, labels)
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.2f}")

    logger.info(f"Saving predictions to {Path('predictions.jsonl')}")
    with Path("predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, preds in tqdm(generated, desc="Saving"):
            f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "input_sequence": raw_data[idx]["input_sequence"],
                        "target": raw_data[idx]["target"],
                        "predictions": preds,
                    }
                )
            )

    return metrics
