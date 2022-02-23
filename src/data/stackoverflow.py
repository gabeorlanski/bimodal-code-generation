"""
Code for handling
"""
import json
from collections import defaultdict
from copy import copy
from pathlib import Path
import logging

import torch
from torch.utils.data import IterableDataset
import numpy as np
from typing import Callable, List, Dict, Iterator
from datasets import Dataset, DatasetDict, load_dataset
from src.common import PROJECT_ROOT
from tio import Task
import re

from jinja2 import BaseLoader, Environment, StrictUndefined

from tqdm import tqdm
import multiprocessing as mp

logger = logging.getLogger(__name__)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)


class StackOverflowTextProcessor:
    def __init__(
            self,
            objective: str,
            answer_sorting: str = 'accepted',
            answers_per_sample: int = -1,
            repeat_question_for_each_answer: str = None,
            good_answer_cutoff: int = 3,
            bad_answer_cutoff: int = -1,
            answer_prompt: str = None,
            question_prompt: str = None,
            title_prompt: str = None
    ):
        self.objective = objective
        self.answer_sorting = answer_sorting.lower()
        if self.answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {self.answer_sorting}")

        if self.objective not in ['lm', 'seq2seq']:
            raise ValueError(f'Invalid objective: {self.objective}')

        self.repeat_question_for_each_answer = repeat_question_for_each_answer
        if self.repeat_question_for_each_answer is not None:
            if self.repeat_question_for_each_answer not in ['title', 'full']:
                raise ValueError(f"Invalid repeat mode: {self.repeat_question_for_each_answer}")
        else:
            self.repeat_question_for_each_answer = "NO_REPEAT_MODE"
        if self.objective == 'seq2seq':
            if answers_per_sample == 0:
                raise ValueError("Seq2Seq must have >0 answers per sample")
            self.repeat_question_for_each_answer = 'full'

        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.answer_prompt = answer_prompt
        self.question_prompt = question_prompt if question_prompt else '__BODY__'
        self.title_prompt = title_prompt if title_prompt else '__TITLE__'
        self.answers_per_sample = answers_per_sample
        self.lm_concat_delim = '\n'

    def make_instances_from_question(self, sample: Dict) -> List[Dict]:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        # Do a list comprehension to eliminate the accepted answer
        accepted_answer = None
        answers = []
        for d in sample['answers'].values():
            if d['id'] == accepted_answer_id and self.answer_sorting == "accepted":
                accepted_answer = d
            else:
                answers.append(d)

        # Sort the answer keys
        answers = sorted(
            answers, key=lambda k: k['score'],
            reverse=not self.answer_sorting == 'ascending'
        )

        # If there is an accepted answer and we are sorting by accepted, put the
        # accepted at the front of the list.
        if accepted_answer is not None and self.answer_sorting == "accepted":
            answers = [accepted_answer, *answers]

        title_str = self.title_prompt.replace('__TITLE__', sample['title'])
        body_str = self.question_prompt.replace("__BODY__", sample['body'])
        question_str = f"{title_str}\n{body_str}"

        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        # Add the quality information to the answer.
        out = []
        for i, answer in enumerate(answers[:answers_keep]):

            if self.answer_prompt:
                if answer['score'] >= self.good_answer_cutoff:
                    quality_str = "good"
                elif answer['score'] <= self.bad_answer_cutoff:
                    quality_str = "bad"
                else:
                    quality_str = "ok"
                answer_str = f"{self.answer_prompt.replace('__QUALITY__', quality_str)}\n{answer['body']}"
            else:
                answer_str = answer['body']

            if i > 0:
                if self.repeat_question_for_each_answer == 'title':
                    input_str = title_str
                elif self.repeat_question_for_each_answer == 'full':
                    input_str = question_str
                else:
                    input_str = ''
            else:
                input_str = question_str

            out.append({'input': input_str, 'target': answer_str})

        if self.objective == 'seq2seq' or self.repeat_question_for_each_answer != 'NO_REPEAT_MODE':
            return out

        return [{'input': question_str, 'target': '\n'.join(d['target'] for d in out)}]

    def __call__(self, samples, tokenizer):
        instances = list(map(self.make_instances_from_question, samples))
        inputs = []
        targets = []

        for instance_list in instances:
            for d in instance_list:
                if self.objective == 'seq2seq':
                    inputs.append(d['input'])
                    targets.append(d['target'])
                else:
                    inputs.append(f"{d['input']}{self.lm_concat_delim}{d['target']}")

        inputs_tokenized = tokenizer(inputs)
        if targets:
            targets_tokenized = tokenizer(targets)['input_ids']
        else:
            targets_tokenized = [[] for _ in range(len(inputs))]

        out = []
        for i, label in enumerate(targets_tokenized):
            out.append({
                'label'         : label,
                'input_ids'     : inputs_tokenized['input_ids'][i],
                'attention_mask': inputs_tokenized['attention_mask'][i]
            })
        return out


class StackOverflowTask(IterableDataset):

    def __init__(
            self,
            dump_name: str,
            data_path: str,
            tokenizer,
            max_samples: int = None,
            answer_sorting: str = 'accepted',
            answers_per_sample: int = -1,
            seed=1,
            max_steps: int = -1,
            num_proc: int = 1,
            sequence_length: int = 512,
            repeat_question_for_each_answer: str = None,
            good_answer_cutoff: int = 3,
            bad_answer_cutoff: int = -1,
            answer_prompt: str = None,
            question_prompt: str = None,
            join_answers_with_eos_token: bool = False,
            add_question_prompt: bool = False
    ):
        super(StackOverflowTask, self).__init__()
        self.dump_name = dump_name
        self.max_samples = max_samples
        if answer_sorting not in ['ascending', 'descending', 'accepted']:
            raise ValueError(f"Unknown answer sorting method: {answer_sorting}")

        self.repeat_question_for_each_answer = repeat_question_for_each_answer
        self.good_answer_cutoff = good_answer_cutoff
        self.bad_answer_cutoff = bad_answer_cutoff
        self.answer_prompt = answer_prompt if answer_prompt else None
        self.question_prompt = question_prompt if question_prompt else None
        self.add_question_prompt = add_question_prompt
        self.join_answers_with_eos_token = join_answers_with_eos_token

        self.tokenizer = tokenizer
        self.answer_sorting = answer_sorting
        self.answers_per_sample = answers_per_sample
        self.rng = np.random.default_rng(seed)
        self.num_proc = num_proc
        self.data_path = PROJECT_ROOT.joinpath(data_path)
        self.max_steps = max_steps
        self.sequence_length = sequence_length
        self.buffer_size = 16 * self.sequence_length
        self.preproc_batch_size = 32
        self.samples_seen = 0
        self.epoch = 0
        self.data = self._initialize_data()
        self.concat_token_id = self.tokenizer.eos_token_id
        self.num_samples = -1
        if self.max_steps != -1:
            self.num_samples = self.max_steps
        else:
            self.num_samples = len([1 for _ in self])
        logger.debug(f"{self.num_samples} total samples")
        self.samples_seen = 0
        self.epoch = 0

    def get_samples_mask(self, total, num_to_select):
        # Just made this function to mock
        sample_mask = np.zeros((total,), dtype=bool)
        sample_mask[self.rng.choice(total, (num_to_select,), replace=False)] = True
        return sample_mask

    def load_samples(self):
        """
        Load the samples from the dump file and uniformly randomly sample based
        on the max number of samples to use.
        """
        total_samples = sum(1 for _ in self.data_path.open('r').readlines())
        logger.info(f"{self.dump_name} has {total_samples} total samples")
        if self.max_samples is not None:
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

        samples = []
        line_number = 0
        for d in tqdm(map(json.loads, self.data_path.read_text('utf-8').splitlines()),
                      total=total_samples):
            if not sample_mask[line_number]:
                line_number += 1
                continue
            samples.append(d)
            line_number += 1
        return samples

    def get_text_from_sample(self, sample: Dict) -> str:
        """
        Get the text string from the sample.
        """
        # Set to -1 if there is no accepted answer because it is impossible.
        accepted_answer_id = sample['accepted_answer'] or "-1"

        # Do a list comprehension to eliminate the accepted answer
        accepted_answer = None
        answers = []
        for d in sample['answers'].values():
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

        if self.question_prompt:
            question_str = self.question_prompt.replace("__TITLE__", sample['title']).replace(
                "__BODY__", sample['body'])
        else:
            question_str = f"{sample['title']}\n{sample['body']}"
        if self.answers_per_sample == -1:
            answers_keep = len(answers)
        else:
            answers_keep = self.answers_per_sample

        out = question_str
        for i, answer in enumerate(answers[:answers_keep]):

            if self.answer_prompt:
                if answer['score'] >= self.good_answer_cutoff:
                    quality_str = "good"
                elif answer['score'] <= self.bad_answer_cutoff:
                    quality_str = "bad"
                else:
                    quality_str = "ok"
                answer_str = f"{self.answer_prompt.replace('__QUALITY__', quality_str)}\n{answer['body']}"
            else:
                answer_str = answer['body']

            # We only care about repeating after the first answer.
            if i > 0:
                if self.repeat_question_for_each_answer is not None:
                    if self.repeat_question_for_each_answer == "title":
                        answer_str = f"{sample['title']}\n{answer_str}"
                    elif self.repeat_question_for_each_answer == "full":
                        answer_str = f"{question_str}\n{answer_str}"
                if self.join_answers_with_eos_token:
                    out += f"{self.tokenizer.eos_token}{answer_str}"
                else:
                    out += f"\n{answer_str}"
            else:
                out += f"\n{answer_str}"

        return out

    def batch_tokenize(self, samples):
        """
        Batched function for tokenizing
        """
        sequences = list(map(self.get_text_from_sample, samples))
        return self.tokenizer(sequences, truncation=False, add_special_tokens=False)['input_ids']

    def __iter__(self):
        logger.debug(f"StackOverflowTask Iter was called.")
        data_iterator = iter(self.data)
        more_examples = True
        num_samples_yielded = 0
        buffer = []
        while more_examples:
            if self.num_samples != -1 and num_samples_yielded >= self.num_samples:
                break
            while len(buffer) < self.buffer_size:
                try:
                    buffer.extend([self.concat_token_id] + next(data_iterator))
                    self.samples_seen += 1
                except StopIteration:
                    if self.max_steps != -1:
                        data_iterator = iter(self.data)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            overflow = []
            for i in range(0, len(buffer), self.sequence_length):
                input_ids = buffer[i: i + self.sequence_length]
                input_len = len(input_ids)
                attention_mask = [1] * input_len
                if len(input_ids) == self.sequence_length:
                    num_samples_yielded += 1
                    yield {
                        'input_ids'     : torch.tensor(input_ids),
                        'attention_mask': torch.tensor(attention_mask),
                        'labels'        : torch.tensor(input_ids)
                    }
                else:
                    overflow.extend(copy(input_ids))
                if self.num_samples != -1 and num_samples_yielded >= self.num_samples:
                    break
            del buffer
            buffer = overflow

    def __len__(self):
        return self.num_samples

    def _initialize_data(self):
        samples = self.load_samples()
        batches = [samples[i:i + self.preproc_batch_size]
                   for i in range(0, len(samples), self.preproc_batch_size)]
        logger.info(
            f"Preprocessing {len(batches)} batches for {len(samples)} for dump {self.dump_name}"
        )

        logger.info(f"Using {self.num_proc} processes")
        with mp.Pool(self.num_proc) as pool:
            results = list(tqdm(
                pool.imap(self.batch_tokenize, batches),
                total=len(batches),
                desc='Tokenizing')
            )

        results = [tokens for batch in results for tokens in batch]

        return results


if __name__ == '__main__':
    from pathlib import Path
    from transformers import AutoTokenizer
    from src.common.log_util.util import set_global_logging_level

    set_global_logging_level(logging.ERROR,
                             ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])

    x = StackOverflowTask('exceptions', 'data/stackoverflow/exceptions.jsonl',
                          AutoTokenizer.from_pretrained('gpt2'), num_proc=4)
    print(len(x))
