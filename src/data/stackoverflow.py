"""
Code for handling
"""
import json
from collections import defaultdict
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict
from datasets import Dataset, DatasetDict
from src.common import PROJECT_ROOT
from overrides import overrides
from tio.task import Task, PathType
import re

logger = logging.getLogger(__name__)


@Task.register("so")
class StackOverflow(Task):
    SPLIT_MAPPING = {}

    def __init__(
            self,
            dump_name: str,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType]
    ):
        super(StackOverflow, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping={k: PROJECT_ROOT.joinpath(v) for k, v in split_mapping.items()}
        )
        self.dump_name = dump_name

    def dataset_load_fn(self, split) -> Dataset:
        split_dict = defaultdict(list)
        path_to_split = Path(self.SPLIT_MAPPING[split])
        for d in map(json.loads, path_to_split.read_text('utf-8').splitlines(False)):
            for k, v in d.items():
                if k == 'answers':
                    split_dict[k].append(list(v.values()))
                else:
                    split_dict[k].append(v)
        return Dataset.from_dict(split_dict, split=split)

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample["input_sequence"] = (
                sample['title'] + '\n' + sample["body"] + '\n'
        )
        for answer in sample['answers']:
            sample['input_sequence'] += f"Answer: {answer['body']}\n"

        sample['target'] = ""
        return sample

    def serialize_task_features(
            self,
            idx: int,
            predictions: List,
            processed_sample: Dict
    ) -> Dict:
        return {}
