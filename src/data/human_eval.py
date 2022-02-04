"""
Code for handling the Human Eval from
https://arxiv.org/abs/2107.03374
"""
import json
from collections import defaultdict
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict
from datasets import Dataset, DatasetDict, load_dataset
from src.common import PROJECT_ROOT, PathType
from overrides import overrides
from tio import Task
import re

logger = logging.getLogger(__name__)


@Task.register("human_eval")
class HumanEval(Task):
    """
    Task for the Human Eval dataset
    """
    SPLIT_MAPPING = {
        "test": str(PROJECT_ROOT.joinpath('data', 'MBPP', 'test.jsonl')),
    }

    EXCLUDE_KEYS = []
    EVAL_SPLITS = ['test']

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None
    ):
        super(HumanEval, self).__init__(
            tokenizer=tokenizer,
            preprocessors=preprocessors,
            postprocessors=postprocessors, metric_fns=metric_fns,
            split_mapping=split_mapping
        )
        self._dataset = load_dataset('openai_humaneval')

    def _load_samples(self, split) -> Dataset:
        # Load the data into a dict where the key is the task_id
        if split not in self.SPLIT_MAPPING:
            raise KeyError("HumanEval only supports a test split.")
        return self._dataset[split]

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = sample['canonical_solution']
        sample['input_sequence'] = sample['prompt']
        return sample

    def serialize_task_features(
            self,
            idx: int,
            predictions: List,
            processed_sample: Dict
    ) -> Dict:
        return {
            'tests'  : [processed_sample['test']+'\n'+f"check({processed_sample['entry_point']})"]
        }
