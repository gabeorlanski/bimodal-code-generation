"""
Code for handling the mostly basic programming problems dataset from
https://arxiv.org/pdf/2108.07732.pdf
"""
import json
from collections import defaultdict
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict
from datasets import Dataset, DatasetDict
from src.common import PROJECT_ROOT

from tio.task import Task, PathType

logger = logging.getLogger(__name__)


@Task.register("mbpp")
class MBPP(Task):
    """
    Task for the Mostly Basic Programming Problems Dataset.
    """
    SPLIT_MAPPING = {
        "train"     : str(PROJECT_ROOT.joinpath('data', 'MBPP', 'train.jsonl')),
        "validation": str(PROJECT_ROOT.joinpath('data', 'MBPP', 'validation.jsonl')),
        "test"      : str(PROJECT_ROOT.joinpath('data', 'MBPP', 'test.jsonl')),
    }

    EXCLUDE_KEYS = [
        "challenge_test_list",
        "test_setup_code"
    ]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            additional_splits: Dict[str, PathType] = None
    ):
        super(MBPP, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            additional_splits=additional_splits
        )

        self._tokenizer = tokenizer
        self.dataset = None
        self.raw = None
        self._dataset_mapping = self.load_dataset_mapping()

    def load_dataset_mapping(self):
        out = {}
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in map(json.loads, Path(path).read_text('utf-8').splitlines(False)):
                for k, v in d.items():
                    if k in self.EXCLUDE_KEYS:
                        continue
                    split_dict[k].append(v)
            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def dataset_load_fn(self, split) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return self._dataset_mapping[split]

    @staticmethod
    def map_to_standard_entries(sample: Dict) -> Dict:
        sample["input_sequence"] = (
                sample["text"] + "\n" + "\n".join(sample["test_list"])
        )
        sample["target"] = sample["code"]
        return sample
