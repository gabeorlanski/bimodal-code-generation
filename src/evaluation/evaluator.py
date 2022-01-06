from typing import List

from src.evaluation.metrics import Metric
from transformers import EvalPrediction, PreTrainedTokenizer
import numpy as np


class Evaluator:
    def __init__(self, tokenizer: PreTrainedTokenizer, metrics: List[str]):
        self.metrics = [Metric.by_name(metric) for metric in metrics]
        self.tokenizer = tokenizer

    def __call__(self, raw_predictions: EvalPrediction):
        predictions, labels = raw_predictions
        predictions = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )
        labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        out = {}
        for metric in self.metrics:
            out.update(metric(predictions, labels))

        return out
