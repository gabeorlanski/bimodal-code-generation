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
from overrides import overrides
from tio.task import Task, PathType
import re

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
    EVAL_SPLITS = ['test', 'validation']

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None,
            add_def_to_prompt: bool = False
    ):
        super(MBPP, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping=split_mapping
        )

        self._tokenizer = tokenizer
        self.dataset = None
        self.excluded_columns_data = {}
        self._dataset_mapping = self.load_dataset_mapping()
        self.add_def_to_prompt = add_def_to_prompt
        self.find_def = re.compile(r'\ndef ')

    def load_dataset_mapping(self):
        out = {}
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in map(json.loads, Path(path).read_text('utf-8').splitlines(False)):
                self.excluded_columns_data[d['task_id']] = {}
                for k, v in d.items():
                    if k in self.EXCLUDE_KEYS:
                        self.excluded_columns_data[d['task_id']][k] = v
                        continue
                    split_dict[k].append(v)
            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def dataset_load_fn(self, split) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return self._dataset_mapping[split]

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample["input_sequence"] = (
                sample["text"] + "\r\n" + "\r\n".join(sample["test_list"]) + '\r\n'
        )

        target_code = sample['code']

        if self.add_def_to_prompt:
            def_match = self.find_def.search(target_code)
            if def_match is not None:
                sample['input_sequence'] += target_code[:def_match.regs[0][1]]
                target_code = target_code[def_match.regs[0][1]:]

        sample['target'] = target_code
        return sample

    def serialize_task_features(
            self,
            idx: int,
            predictions: List,
            processed_sample: Dict
    ) -> Dict:
        return {
            'tests'  : processed_sample['test_list'],
            'task_id': processed_sample['task_id'],
            **self.excluded_columns_data[processed_sample['task_id']]
        }

