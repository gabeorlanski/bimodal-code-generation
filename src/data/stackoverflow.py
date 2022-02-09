"""
Code for handling
"""
import json
from collections import defaultdict
from pathlib import Path
import logging

import numpy as np
from typing import Callable, List, Dict, Iterator
from datasets import Dataset, DatasetDict, load_dataset
from src.common import PROJECT_ROOT, PathType
from tio import Task
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


@Task.register("so")
class StackOverflowTask(Task):

    def __init__(
            self,
            dump_name: str,
            tokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict,
            max_samples: int = None,
            answer_sorting: str = 'accepted',
            answers_per_sample: int = -1,
            max_val_samples: int = 250,
            seed=1
    ):
        super(StackOverflowTask, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping=split_mapping
        )
        self.dump_name = dump_name
        self.max_samples = max_samples
        if answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {answer_sorting}")

        self.answer_sorting = answer_sorting
        self.answers_per_sample = answers_per_sample
        self.max_val_samples = max_val_samples
        self.rng = np.random.default_rng(seed)

    def _load_sample(self):
        with self.data_path.open('r', encoding='utf-8') as data_file:
            for line in data_file:
                yield json.loads(line)

    def get_samples_mask(self, total, num_to_select):
        # Just made this function to mock
        sample_mask = np.zeros((total,), dtype=bool)
        sample_mask[self.rng.choice(total, (num_to_select,),replace=False)] = True
        return sample_mask

    def _load_samples(self, split: str) -> Dataset:
        path_to_split = PROJECT_ROOT.joinpath(self.SPLIT_MAPPING[split])
        total_samples = sum(1 for _ in path_to_split.open('r').readlines())
        logger.info(f"{self.dump_name} has {total_samples} total samples in {split}")
        if self.max_samples is not None or split == 'validation':
            if split == 'validation':
                samples_to_select = self.max_val_samples
            else:
                samples_to_select = self.max_samples
            logger.info(f"Uniformly selecting {samples_to_select} from {total_samples}")
            if total_samples < samples_to_select:
                sample_mask = np.ones((total_samples,), dtype=bool)
            else:
                sample_mask = self.get_samples_mask(
                    total_samples,
                    samples_to_select
                )

        else:
            logger.info("Not Doing sampling")
            sample_mask = np.ones((total_samples,), dtype=bool)

        split_dict = defaultdict(list)
        line_number = 0
        for d in tqdm(map(json.loads, path_to_split.read_text('utf-8').splitlines())):
            if not sample_mask[line_number]:
                line_number += 1
                continue
            for k, v in d.items():
                if k == "answers":
                    split_dict[k].append(list(v.values()))
                else:
                    split_dict[k].append(v)
            line_number += 1
        return Dataset.from_dict(split_dict)

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        # Do a list comprehension to eliminate the accepted answer
        accepted_answer = None
        answers = []
        for d in sample['answers']:
            if d['id'] == accepted_answer_id and self.answer_sorting == "accepted":
                accepted_answer = d
            else:
                answers.append(d)

        # Sort the answer keys
        answers = sorted(
            answers, key=lambda k: k['score'],
            reverse=not self.answer_sorting == 'ascending'
        )
        if accepted_answer is not None and self.answer_sorting == "accepted":
            answers = [accepted_answer, *answers]

        sample['input_sequence'] = f"{sample['title']}\n{sample['body']}"
        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample
        sample['input_sequence'] += "\n" + "\n".join(
            [k['body'] for k in answers[:answers_keep]]
        )
        sample['target'] = ""
        return sample

    def serialize_task_features(self, idx: int, predictions: List, processed_sample: Dict) -> Dict:
        return {}
