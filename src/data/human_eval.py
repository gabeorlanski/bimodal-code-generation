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
EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]
FIRST_BLOCK_REGEX = re.compile("|".join(EOF_STRINGS))


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
        self.postprocessors = [first_block, *self.postprocessors]

    def _load_samples(self, split) -> Dataset:
        # Load the data into a dict where the key is the task_id
        if split not in self.SPLIT_MAPPING:
            raise KeyError("HumanEval only supports a test split.")
        return self._dataset[split]

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = sample['canonical_solution']
        sample['input_sequence'] = sample['prompt'].strip()
        return sample

    def serialize_task_features(
            self,
            idx: int,
            predictions: List,
            processed_sample: Dict
    ) -> Dict:
        return {
            'tests': [processed_sample['test'] + '\n' + f"check({processed_sample['entry_point']})"]
        }

    def serialize_predictions(
            self,
            split: str,
            indices: List,
            predictions: List[List]
    ):
        """
        Serialize a prediction to a dict.

        Args:
            split (str): The split the predictions came from.
            indices (List): The indices corresponding to the predictions.
            predictions (List[List]): The list of predictions for each sample.

        Returns:
            A generator of dicts for each sample.
        """

        processed_data = {d['task_id']: d for d in self.preprocessed_splits[split]}

        assert len(indices) == len(predictions), "Indices must be the same length as predictions"

        for task_id, preds in zip(indices, predictions):
            processed_sample = processed_data[task_id]
            tests_str = [
                processed_sample['test'] + '\n' + f"check({processed_sample['entry_point']})"
            ]

            # Have to add the prompt back to the predictions
            prompt = processed_sample['prompt'].split('"""')[0].strip()
            preds_list = [
                prompt + p
                for p in preds
            ]

            yield {
                'idx'           : processed_sample['idx'],
                'task_id'       : processed_sample['task_id'],
                'target'        : processed_sample['target'],
                'input_sequence': processed_sample['input_sequence'],
                'prediction'    : preds_list,
                'tests'         : tests_str
            }


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    first_block_value, *_ = FIRST_BLOCK_REGEX.split(string)
    return first_block_value.rstrip()
