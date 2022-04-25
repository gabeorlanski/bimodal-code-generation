import json
import logging
import math
import random
import re
from collections import defaultdict
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

        self._dataset_mapping = self.initialize_data()

    def initialize_data(self):
        out = {}
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in tqdm(
                    map(json.loads, Path(path).read_text('utf-8').splitlines(False)),
                    desc=f"Reading '{split}'"
            ):

                excluded = {}

                d['test_stmt'] = self.make_stmt_from_io(d['input'], d['op'], d['output'])
                true_examples, false_examples = self.get_true_false_examples(d)

                context_examples = []
                for true_example, false_example in zip_longest(true_examples, false_examples):
                    if true_example is not None:
                        context_examples.append([self.make_stmt_from_io(
                            true_example['input'], true_example['op'], true_example['output'],
                            is_ctx=True
                        ), 'True'])
                    if false_example is not None:
                        context_examples.append([self.make_stmt_from_io(
                            false_example['input'], false_example['op'], false_example['output'],
                            is_ctx=True
                        ), 'False'])

                if self.shuffle_ctx_pairs:
                    random.shuffle(context_examples)
                d['context_examples'] = context_examples

                for k, v in d.items():
                    if k in self.EXCLUDE_KEYS:
                        excluded[k] = v
                    else:
                        split_dict[k].append(v)
                self.excluded_columns_data[d['task_id']] = excluded

            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def get_true_false_examples(self, sample):
        true_examples = random.sample(
            sample['context_io_pairs']['True'],
            k=min(self.num_true_ctx_pairs, len(sample['context_io_pairs']['True']))
        )
        false_examples = random.sample(
            sample['context_io_pairs']['False'],
            k=min(self.num_false_ctx_pairs, len(sample['context_io_pairs']['False']))
        )
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
            'op'            : processed_sample['op'],
            'input'         : processed_sample['input'],
            'output'        : processed_sample['output'],
            **self.excluded_columns_data[idx]
        }
