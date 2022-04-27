import json
import logging
import math
import random
import re
from collections import defaultdict
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from typing import List, Dict, Callable

import astor
import yaml
from datasets import Dataset, DatasetDict
from jinja2 import BaseLoader, Environment, StrictUndefined
from tio import Task
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.common import PathType, PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "NPV",
]

PROG_SPLIT = re.compile(r'(class |def )')

JINJA_ENV = Environment(loader=BaseLoader)  # type: ignore
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined

PROMPT_TO_USE = None


@Task.register("npv")
class NPV(Task):
    SPLIT_MAPPING = {
        "test": str(PROJECT_ROOT.joinpath('data', 'NPV', 'test.jsonl')),
    }

    EXCLUDE_KEYS = [
        'source_file', 'task', 'original_task_id'
    ]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None,
            prompt: str = "base",
            choices: List[str] = None,
            n_ctx_pairs: int = 0,
            ctx_true_pct: float = 0.5,
            shuffle_ctx_pairs: bool = False,
            stmt_prompt: str = "{stmt}",
            trailing_newline: bool = False,
            allow_ctx_same_input: bool = False,
            allow_ctx_same_output: bool = False
    ):
        super(NPV, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping=split_mapping
        )
        self.dataset = None
        self.excluded_columns_data = {}
        self.choices = choices or ["False", 'True']
        prompt_dict = yaml.load(PROJECT_ROOT.joinpath('templates/npv_prompts.yaml').open(),
                                yaml.Loader)

        global PROMPT_TO_USE
        PROMPT_TO_USE = JINJA_ENV.from_string(prompt_dict[prompt])
        self.include_target_in_prompt_kwargs = False
        self.num_context_pairs = n_ctx_pairs
        self.num_true_ctx_pairs = math.ceil(ctx_true_pct * self.num_context_pairs)
        self.num_false_ctx_pairs = max(0, self.num_context_pairs - self.num_true_ctx_pairs)
        self.shuffle_ctx_pairs = shuffle_ctx_pairs
        self.stmt_prompt = stmt_prompt
        self.trailing_newline = trailing_newline
        self.allow_ctx_same_input = allow_ctx_same_input
        self.allow_ctx_same_output = allow_ctx_same_output

        self._dataset_mapping = self.initialize_data()

    def initialize_data(self):
        out = {}
        no_save_keys = ['tid_by_result', 'instances', 'all_tasks']
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in tqdm(
                    map(json.loads, Path(path).read_text('utf-8').splitlines(False)),
                    desc=f"Reading '{split}'"
            ):

                all_instances = d.pop('all_tasks')
                instances_to_keep = d.pop('instances')
                # d['test_stmt'] = self.make_stmt_from_io(d['input'], d['op'], d['output'])
                true_ctx_examples_pool, false_ctx_examples_pool = self.get_true_false_examples(d)

                excluded_vals = {k: d[k] for k in self.EXCLUDE_KEYS}

                to_keep_dict = {
                    k: v for k, v in d.items() if k not in self.EXCLUDE_KEYS + no_save_keys
                }

                for task_dict in map(lambda t: all_instances[t], instances_to_keep):
                    task_to_save = deepcopy(to_keep_dict)

                    task_input = task_dict['input']
                    task_to_save['test_stmt'] = self.make_stmt_from_io(
                        task_dict['input'],
                        task_dict['op'],
                        task_dict['output']
                    )
                    task_to_save.update(task_dict)
                    true_samples = self.get_ctx_examples_from_pool(
                        self.num_true_ctx_pairs,
                        true_ctx_examples_pool,
                        all_instances,
                        task_input
                    )
                    false_samples = self.get_ctx_examples_from_pool(
                        self.num_false_ctx_pairs,
                        false_ctx_examples_pool,
                        all_instances,
                        task_input
                    )
                    context_samples = []
                    i = j = 0
                    while i < len(true_samples) or j < len(false_samples):
                        if i < len(true_samples):
                            sample = all_instances[true_samples[i]]
                            stmt = self.make_stmt_from_io(
                                sample['input'], sample['op'], sample['output'],
                                is_ctx=True
                            )
                            context_samples.append((stmt, 'True'))
                            i += 1
                        if j < len(false_samples):
                            sample = all_instances[false_samples[j]]
                            stmt = self.make_stmt_from_io(
                                sample['input'], sample['op'], sample['output'],
                                is_ctx=True
                            )
                            context_samples.append((stmt, 'False'))
                            j += 1

                    if self.shuffle_ctx_pairs:
                        random.shuffle(context_samples)
                    task_to_save['context_examples'] = context_samples

                    for k, v in task_to_save.items():
                        split_dict[k].append(v)
                    self.excluded_columns_data[task_dict['task_id']] = excluded_vals

            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def get_ctx_examples_from_pool(self, num_to_get, example_pool, all_instances, input_str):
        out = []
        pool_iter = iter(example_pool)
        while len(out) < num_to_get:
            try:
                next_input, next_example = next(pool_iter)
            except StopIteration:
                break
            if next_input == input_str and not self.allow_ctx_same_input:
                continue
            next_example_dict = all_instances[next_example]
            # if not self.allow_ctx_same_output and next_example_dict['output'] == output_str:
            #     continue
            out.append(next_example)
        return out

    @staticmethod
    def get_true_false_examples(sample):
        false_examples = []
        true_examples = []

        for input_str, tid_list in sample['tid_by_result']['True'].items():
            true_examples.extend([(input_str, tid) for tid in tid_list])
        for input_str, tid_list in sample['tid_by_result']['False'].items():
            false_examples.extend([(input_str, tid) for tid in tid_list])

        random.shuffle(true_examples)
        random.shuffle(false_examples)

        return true_examples, false_examples

    def _load_samples(self, split: str) -> Dataset:
        return self._dataset_mapping[split]

    def make_stmt_from_io(self, input_stmt, op, output_stmt, is_ctx=False):
        out = self.stmt_prompt.format(stmt=f"{input_stmt} {op} {output_stmt}")
        if self.trailing_newline and not is_ctx:
            return f"{out}\n"
        return out

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = sample['result']

        # Some are VERY long, so we need to adapt for that by removing context
        # examples until we either have no context or have under the threshold.
        # num_required_chars = len(sample['context'] + sample['code'].lstrip())
        # context_example_chars = sum(map(len, context_examples))
        # if num_required_chars + context_example_chars >= 1000:
        #     logger.warning(f"{sample['task_id']} has too many characters, "
        #                    f"removing some context examples")
        #     while num_required_chars + context_example_chars >= 1000 and context_examples:
        #         context_example_chars -= len(context_examples.pop(-1))

        prompt_kwargs = {
            "context_code"    : sample['context'],
            'context_examples': sample['context_examples'],
            'description'     : sample['description'],
            'code'            : sample['code'].lstrip(),
            'test_stmt'       : sample['test_stmt']
        }
        if self.include_target_in_prompt_kwargs:
            prompt_kwargs['target'] = sample['result']
        assert PROMPT_TO_USE is not None
        sample['input_sequence'] = PROMPT_TO_USE.render(prompt_kwargs)
        return sample

    def serialize_task_features(self, idx: int, predictions: List, processed_sample: Dict) -> Dict:
        return {
            'is_negation_of': processed_sample['is_negation_of'],
            'is_manual_fix' : processed_sample['is_manual_fix'],
            'is_original'   : processed_sample['is_original'],
            'op'            : processed_sample['op'],
            'input'         : processed_sample['input'],
            'output'        : processed_sample['output'],
            **self.excluded_columns_data[idx]
        }
