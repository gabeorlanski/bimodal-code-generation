"""
Code for handling the CodeSearchNet task https://arxiv.org/pdf/1909.09436.pdf
"""
import json
from collections import defaultdict
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict
from datasets import Dataset, DatasetDict
from src.common import PROJECT_ROOT, PathType
from overrides import overrides
from tio import Task
import re

logger = logging.getLogger(__name__)


@Task.register("code_search_net")
class CodeSearchNet(Task):
    """
    Task for the CodeSearchNet Task
    """
    SPLIT_MAPPING = {}

    EXCLUDE_KEYS = []

    def __init__(
            self,
            languages: List[str],
            mode: str,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None,
    ):
        super(CodeSearchNet, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping=split_mapping
        )

        self.languages = languages
        self.mode = mode

        logger.info(f"CodeSearchNet task initialized in mode {self.mode} "
                    f"with languages '{', '.join(self.languages)}'")

        self.dataset = None
        self.excluded_columns_data = {}
        self._dataset_mapping = self.load_dataset_mapping()
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

    def _load_samples(self, split) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return self._dataset_mapping[split]

    def map_to_standard_entries(self, sample: Dict) -> Dict:

        tests_to_use = []
        len_tests_to_use = 0
        for t in sample['test_list']:
            if len_tests_to_use + len(t) > 850:
                logger.warning(f"{sample['task_id']} does not have full tests due to length")
                break
            tests_to_use.append(t)

        sample["input_sequence"] = (
                sample["text"] + "\r\n" + "\r\n".join(tests_to_use) + '\r\n'
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
