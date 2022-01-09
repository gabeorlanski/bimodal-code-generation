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
from src.evaluation.util import serialize_prediction
from src.pipelines import LoadDataStage, PredictStage

logger = logging.getLogger(__name__)


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
    cfg.training = train_cfg.training
    cfg.device = train_cfg.device

    load_data_stage = LoadDataStage.from_cfg(cfg)
    predict_stage = PredictStage.from_cfg(cfg)
    tokenizer = load_data_stage.tokenizer

    logger.info(f"Reading data from '{cfg['data_path']}'")
    raw_data, tokenized = load_data_stage(
        PROJECT_ROOT.joinpath(cfg["data_path"])
    )
    logger.info(f"{len(tokenized)} total samples found")

    logger.info("Initializing the evaluator")
    # We want the union of metrics from both the training config and the eval
    # config b/c some metrics do not need to be used during training.
    evaluator = Evaluator(
        tokenizer,
        list(set(cfg.get("metrics", [])).union(train_cfg.get("metrics", []))),
    )

    generation_results = predict_stage(model, tokenized, tokenizer)
    indices = generation_results['indices']
    predictions = generation_results['predictions']
    labels = generation_results['labels']

    # Get the full metrics suite for the predictions and the labels
    metrics = evaluator(predictions, labels)
    logger.info("Results:")
    for k, v in metrics.items():
        logger.info(f"\t{k:>20} = {v:0.2f}")

    logger.info(f"Saving predictions to {Path('predictions.jsonl')}")
    with Path("predictions.jsonl").open("w", encoding="utf-8") as f:
        for idx, preds in tqdm(zip(indices, predictions), desc="Saving"):
            f.write(serialize_prediction(
                idx=idx,
                input_sequence=raw_data[idx]["input_sequence"],
                target=raw_data[idx]["target"],
                predictions=preds
            ) + '\n')

    return metrics
