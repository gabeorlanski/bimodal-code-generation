import json
import logging
import math
import random
import re
from collections import defaultdict
from copy import deepcopy
from itertools import zip_longest, islice
from pathlib import Path
from typing import List, Dict, Callable

import astor
import numpy as np
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
        "test"      : str(PROJECT_ROOT.joinpath('data', 'NPV', 'test.jsonl')),
        "train"     : str(PROJECT_ROOT.joinpath('data', 'NPV', 'train.jsonl')),
        "validation": str(PROJECT_ROOT.joinpath('data', 'NPV', 'validation.jsonl')),
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
            true_ctx_examples: int = 0,
            false_ctx_examples: int = 0,
            ensemble_choices_size: int = 0,
            shuffle_ctx_pairs: bool = False,
            stmt_prompt: str = "__stmt__",
            trailing_newline: bool = False,
            allow_ctx_same_input: bool = False,
            allow_duplicate_output: bool = False,
            allow_duplicate_inputs: bool = False,
            allow_negated_ctx: bool = False,
            allow_generated_ctx: bool = False,
            enforce_no_negated: bool = False,
            ctx_pool_sorting_method: str = 'random',
            ctx_stmt_prompt: str = "__input__ __op__ __output__",

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
        self.choice_map = choices or {
            'True' : {'id': 1, 'text': 'True'},
            'False': {'id': 0, 'text': 'False'}
        }
        self.choices = [
            self.choice_map[k]['text']
            for k in sorted(
                self.choice_map,
                key=lambda _c:
                self.choice_map[_c]['id'])
        ]
        self.idx_to_choice = {
            self.choice_map[k]['id']: k for k in self.choice_map
        }
        prompt_dict = yaml.load(PROJECT_ROOT.joinpath('templates/npv_prompts.yaml').open(),
                                yaml.Loader)

        global PROMPT_TO_USE
        PROMPT_TO_USE = JINJA_ENV.from_string(prompt_dict[prompt])
        self.include_target_in_prompt_kwargs = False
        self.num_context_pairs = true_ctx_examples + false_ctx_examples
        self.num_true_ctx_pairs = true_ctx_examples
        self.num_false_ctx_pairs = false_ctx_examples
        self.shuffle_ctx_pairs = shuffle_ctx_pairs
        self.stmt_prompt = stmt_prompt
        self.trailing_newline = trailing_newline
        self.allow_ctx_same_input = allow_ctx_same_input
        self.allow_negated_ctx = allow_negated_ctx
        self.allow_generated_ctx = allow_generated_ctx
        self.enforce_no_negated = enforce_no_negated
        self.ctx_pool_sorting_method = ctx_pool_sorting_method
        self.allow_duplicate_output = allow_duplicate_output
        self.allow_duplicate_inputs = allow_duplicate_inputs
        self.ctx_stmt_prompt = ctx_stmt_prompt
        self.ensemble_choices_size = ensemble_choices_size
        assert ctx_pool_sorting_method in [
            'random',
            'output_length',
            'input_length',
            'total_length'
        ]

        self.context_samples_by_task = {}
        self.raw_processed_dict = defaultdict(dict)
        self._dataset_mapping = self.initialize_data()

    def initialize_data(self):
        out = {}
        no_save_keys = ['tid_by_result', 'instances', 'all_tasks']
        for split, path in self.SPLIT_MAPPING.items():
            num_ctx_examples = []
            ctx_example_length = []
            num_no_ctx_examples = 0
            split_dict = defaultdict(list)
            for d in tqdm(
                    map(json.loads, Path(path).read_text('utf-8').splitlines(False)),
                    desc=f"Reading '{split}'"
            ):

                # d['test_stmt'] = self.make_stmt_from_io(d['input'], d['op'], d['output'])
                true_ctx_examples_pool, false_ctx_examples_pool = self.get_true_false_examples(d)
                all_instances = d.pop('all_tasks')
                instances_to_keep = d.pop('instances')

                true_pool = [all_instances[tid] for tid in true_ctx_examples_pool]
                false_pool = [all_instances[tid] for tid in false_ctx_examples_pool]

                excluded_vals = {k: d[k] for k in self.EXCLUDE_KEYS}

                to_keep_dict = {
                    k: v for k, v in d.items() if k not in self.EXCLUDE_KEYS + no_save_keys
                }

                for task_dict in map(lambda t: all_instances[t], instances_to_keep):
                    excluded_to_save = deepcopy(excluded_vals)

                    context_examples, instances = self.mk_instance_from_task(
                        task_dict,
                        true_pool,
                        false_pool
                    )
                    have_saved_one_instance = False

                    if not context_examples:
                        num_no_ctx_examples += 1
                    for instance in instances:
                        if not have_saved_one_instance:
                            self.raw_processed_dict[split][task_dict['task_id']] = {
                                **instance, **to_keep_dict
                            }

                        for k, v in instance.items():
                            split_dict[k].append(v)
                        for k, v in to_keep_dict.items():
                            split_dict[k].append(v)
                        num_ctx_examples.append(len(instance['context_examples']))
                        ctx_example_length.append(
                            sum([len(c[0]) for c in instance['context_examples']])
                            if instance['context_examples'] else 0
                        )
                    self.context_samples_by_task[task_dict['task_id']] = context_examples
                    self.excluded_columns_data[task_dict['task_id']] = excluded_to_save

            out[split] = Dataset.from_dict(split_dict, split=split)
            logger.info(f"{np.mean(num_ctx_examples):.3f} mean context examples")
            logger.info(f"{np.mean(ctx_example_length):.3f} mean total length for ctx examples")
            logger.info(f"{num_no_ctx_examples}/{len(num_ctx_examples)} had no context examples")
        return DatasetDict(out)

    def mk_instance_from_task(self, task_dict, true_pool, false_pool):
        instance_dict = deepcopy(task_dict)
        instance_dict['test_stmt'] = self.make_stmt_from_io(
            task_dict['input'],
            task_dict['op'],
            task_dict['output']
        )

        context_examples = list(self.get_ctx_examples_from_pool(
            task_dict,
            true_pool,
            false_pool
        ))

        if self.shuffle_ctx_pairs:
            random.shuffle(context_examples)

        batch_size = self.ensemble_choices_size if self.ensemble_choices_size > 0 else len(
            context_examples
        )
        batch_size=max(1,batch_size)
        out = []
        for i in range(0, len(context_examples), batch_size):
            batch_ctx_examples = []
            for ex in context_examples[i:i+batch_size]:
                batch_ctx_examples.append((
                    self.make_stmt_from_io(ex['input'], ex['op'], ex['input'], is_ctx=True),
                    ex['result']
                ))
            out.append(
                {'context_examples': batch_ctx_examples, **instance_dict}
            )
        return context_examples, out

    def should_keep_example(
            self,
            ex,
            input_example,
            is_second_pass,
            yielded,
            yielded_outputs,
            yielded_inputs
    ):

        if ex['task_id'] in yielded:
            return False
        if ex['input'] == input_example['input'] and not self.allow_ctx_same_input:
            return False
        if ex['is_negation_of'] is not None and not self.allow_negated_ctx:
            if self.enforce_no_negated:
                return False
            if not is_second_pass:
                return False
        if ex['is_output_generated'] and not self.allow_generated_ctx:
            if not is_second_pass:
                return False
        if ex['output'] in yielded_outputs and not self.allow_duplicate_output:
            if not is_second_pass:
                return False
        if ex['input'] in yielded_inputs and not self.allow_duplicate_inputs:
            if not is_second_pass:
                return False

        return True

    def get_ctx_examples_from_pool(
            self,
            input_example,
            true_pool,
            false_pool
    ):
        yielded_outputs = set()
        yielded_inputs = set()
        yielded = set()

        true_examples = []
        false_examples = []

        for t_ex, f_ex in zip_longest(true_pool, false_pool):
            if (
                    t_ex is not None
                    and len(true_examples) < self.num_true_ctx_pairs
                    and self.should_keep_example(t_ex, input_example, False, yielded,
                                                 yielded_outputs, yielded_inputs)
            ):
                yielded.add(t_ex['task_id'])
                yielded_outputs.add(t_ex['output'])
                yielded_inputs.add(t_ex['input'])
                true_examples.append(t_ex)
            if (
                    f_ex is not None
                    and len(false_examples) < self.num_false_ctx_pairs
                    and self.should_keep_example(f_ex, input_example, False, yielded,
                                                 yielded_outputs, yielded_inputs)
            ):
                yielded.add(f_ex['task_id'])
                yielded_outputs.add(f_ex['output'])
                yielded_inputs.add(f_ex['input'])
                false_examples.append(f_ex)
        for t_ex, f_ex in zip_longest(true_pool, false_pool):
            if (
                    t_ex is not None
                    and len(true_examples) < self.num_true_ctx_pairs
                    and self.should_keep_example(t_ex, input_example, True, yielded,
                                                 yielded_outputs, yielded_inputs)
            ):
                yielded.add(t_ex['task_id'])
                true_examples.append(t_ex)
            if (
                    f_ex is not None
                    and len(false_examples) < self.num_false_ctx_pairs
                    and self.should_keep_example(f_ex, input_example, True, yielded,
                                                 yielded_outputs, yielded_inputs)
            ):
                yielded.add(f_ex['task_id'])
                false_examples.append(f_ex)
        for t_ex, f_ex in zip_longest(true_examples, false_examples):
            if t_ex:
                yield t_ex
            if f_ex:
                yield f_ex

    def get_true_false_examples(self, sample):
        false_examples = []
        true_examples = []
        all_tasks = sample['all_tasks']
        for input_str, tid_list in sample['tid_by_result']['True'].items():
            true_examples.extend(tid_list)
        for input_str, tid_list in sample['tid_by_result']['False'].items():
            false_examples.extend(tid_list)

        if self.ctx_pool_sorting_method != 'random':
            if self.ctx_pool_sorting_method == 'output_length':
                sort_fn = lambda tid: len(all_tasks[tid]['output'])
            elif self.ctx_pool_sorting_method == 'input_length':
                sort_fn = lambda tid: len(all_tasks[tid]['input'])
            elif self.ctx_pool_sorting_method == 'total_length':
                sort_fn = lambda tid: len(all_tasks[tid]['input']) + len(all_tasks[tid]['output'])
            else:
                raise ValueError(f"Unknown sorting method {self.ctx_pool_sorting_method=}")
            true_examples = list(sorted(true_examples, key=sort_fn))
            false_examples = list(sorted(false_examples, key=sort_fn))
        else:
            random.shuffle(true_examples)
            random.shuffle(false_examples)

        return true_examples, false_examples

    def _load_samples(self, split: str) -> Dataset:
        return self._dataset_mapping[split]

    def make_stmt_from_io(self, input_stmt, op, output_stmt, is_ctx=False):
        if is_ctx:
            stmt = self.ctx_stmt_prompt.replace('__input__', input_stmt)
            stmt = stmt.replace('__op__', op)
            stmt = stmt.replace('__output__', output_stmt)
        else:
            stmt = f"{input_stmt} {op} {output_stmt}"
        out = self.stmt_prompt.replace("{stmt}", stmt)
        if self.trailing_newline and not is_ctx:
            return f"{out}\n"
        return out

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = self.choice_map[str(sample['result'])]['text']

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
            'is_negation_of'     : processed_sample['is_negation_of'],
            'is_manual_fix'      : processed_sample['is_manual_fix'],
            'is_original'        : processed_sample['is_original'],
            'is_input_generated' : processed_sample['is_input_generated'],
            'is_output_generated': processed_sample['is_output_generated'],
            'op'                 : processed_sample['op'],
            'input'              : processed_sample['input'],
            'output'             : processed_sample['output'],
            'result'             : processed_sample['result'],
            'context_examples'   : self.context_samples_by_task[processed_sample['task_id']],
            **self.excluded_columns_data[processed_sample['task_id']]
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

        assert len(indices) == len(predictions), "Indices must be the same length as predictions"

        for task_id, preds in zip(indices, predictions):
            raw_sample = self.raw_processed_dict[split][task_id]
            sample = self.serialize_task_features(task_id, preds, raw_sample)
            processed_sample = self.map_to_standard_entries(raw_sample)

            yield {
                'task_id'       : processed_sample['task_id'],
                'target'        : processed_sample['target'],
                'input_sequence': processed_sample['input_sequence'],
                'prediction'    : preds,
                **sample
            }
