"""
Code for handling
"""
import json
from collections import defaultdict
from pathlib import Path
import logging
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizerBase
from typing import Callable, List, Dict, Iterator
from datasets import Dataset, DatasetDict, load_dataset
from src.common import PROJECT_ROOT, PathType
from tio import Task
import re

logger = logging.getLogger(__name__)


class StackOverflowTask(IterableDataset):

    def __init__(
            self,
            dump_name: str,
            data_path: str,
            tokenizer: PreTrainedTokenizerBase,
            infinite=False,
            sequence_length: int = 512
    ):
        super(StackOverflowTask, self).__init__()
        self.tokenizer = tokenizer
        self.data_path = PROJECT_ROOT.joinpath(data_path)
        self.dump_name = dump_name
        self.infinite = infinite
        self.buffer_size = 16
        self.concat_token_id = self.tokenizer.eos_token_id
        self.sequence_length = sequence_length
        self.epoch = 0
        self.samples_seen = 0

    def _load_sample(self):
        with self.data_path.open('r', encoding='utf-8') as data_file:
            for line in data_file:
                yield json.loads(line)

    def _get_content_from_sample(self, sample_dict: Dict) -> str:
        out = f"{sample_dict['title']}\n{sample_dict['body']}"
        answer_string = '\n'.join(v['body'] for v in sample_dict['answers'].values())
        if answer_string:
            out += "\n" + answer_string
        return out

    def __iter__(self) -> Iterator[T_co]:
        sample_stream = self._load_sample()
        more_examples = True
        while more_examples:
            buffer = []
            while True:
                if len(buffer) >= self.buffer_size:
                    break
                try:
                    buffer.append(self._get_content_from_sample(next(sample_stream)))
                    self.samples_seen += 1
                except StopIteration:
                    if self.infinite:
                        sample_stream = self._load_sample()
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.sequence_length):
                input_ids = all_token_ids[i: i + self.sequence_length]
                input_len = len(input_ids)
                attention_mask = [1] * input_len
                if len(input_ids) == self.sequence_length:
                    yield {
                        'input_ids'     : torch.tensor(input_ids),
                        'attention_mask': torch.tensor(attention_mask),
                        'labels'        : torch.tensor(input_ids)
                    }
