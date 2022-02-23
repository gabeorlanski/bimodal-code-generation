import json
import logging
import pickle
import random
import threading
from copy import deepcopy, copy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Callable
import multiprocessing as mp

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.data.parse_so.util import log_process
from tqdm import tqdm
from src.common.file_util import human_readable_size

logger = logging.getLogger(__name__)
__all__ = [
    "TensorizedDataset",
    "tensorize",
    "TensorizedTask"
]


@dataclass
class TensorizedDataset:
    name: str
    input_token_count: int = 0
    target_token_count: int = 0
    instances: List[Dict] = field(default_factory=list)

    def add_instances(self, instance_list):
        for instance in instance_list:
            self.input_token_count += len(instance['input_ids'])
            self.target_token_count += len(instance['label'])
            self.instances.append(instance)

    @property
    def total_tokens(self):
        return self.input_token_count + self.target_token_count


def batch_process(batch, processor, tokenizer):
    return processor(batch, tokenizer)


class TensorizedTask(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences: Number of token sequences to keep in buffer.
            chars_per_token: Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
            self,
            data_path,
            objective,
            tokenizer: PreTrainedTokenizer,
            infinite=False,
            sequence_length=1024,
            max_instances=-1,
    ):
        self.objective = objective
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.sequence_length = sequence_length
        self.infinite = infinite
        self.tokenizer = tokenizer
        self.buffer_size = 1024 * self.sequence_length
        self.lm_concat_delim = self.tokenizer.encode('\n')
        self.tensorized_dataset, self.length = self._load_samples(data_path, max_instances)
        self.tensorized_dataset: TensorizedDataset
        logger.info(f"{self.length} total samples")
        self.samples_yielded = 0

    def _load_samples(self, data_path, max_instances):
        logger.info(f"Loading tensorized dataset from {data_path}")
        tensorized_dataset: TensorizedDataset = pickle.load(data_path.open('rb'))

        if max_instances != -1:
            return tensorized_dataset, max_instances

        if self.objective == 'lm':
            return tensorized_dataset, (
                    tensorized_dataset.total_tokens
                    + ((len(self.lm_concat_delim) + 1)
                       * len(tensorized_dataset.instances))
            ) // self.sequence_length
        return tensorized_dataset, len(tensorized_dataset.instances)

    def __iter__(self):
        buffer = []
        data_iter = iter(self.tensorized_dataset.instances)
        more_examples = True
        total_yielded = 0
        while more_examples and total_yielded < self.length:
            while len(buffer) < self.buffer_size:
                try:
                    current_instance = next(data_iter)
                    buffer.extend(
                        current_instance['label']
                        + self.lm_concat_delim
                        + current_instance["input_ids"]
                        + [self.tokenizer.eos_token_id]
                    )
                    self.samples_yielded += 1
                except StopIteration:
                    if self.infinite:
                        data_iter = iter(self.dataset)
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
                    total_yielded += 1
                    yield {
                        'input_ids': torch.tensor(input_ids),
                        # 'attention_mask': torch.tensor(attention_mask),
                        'labels'   : torch.tensor(input_ids),
                    }
                else:
                    overflow.extend(copy(input_ids))
                if total_yielded >= self.length:
                    more_examples = False
                    break
            del buffer
            buffer = overflow

    def __len__(self):
        return self.length


def tensorize(
        raw_data_path: Path,
        out_path: Path,
        num_workers: int,
        model_name: str,
        data_processor,
        batch_size
):
    logger.info(f"Tensorizing {raw_data_path}")
    # Setup the queues

    logger.info(f"Reading {raw_data_path}")
    lines = 0
    buffer = []
    batches = []
    batches_found = 0
    last_logged_batch = 0
    for line in map(json.loads, raw_data_path.open('r')):
        lines += 1
        buffer.append(line)
        if len(buffer) == batch_size:
            batches.append(deepcopy(buffer))
            del buffer
            buffer = []
            batches_found += 1

        if lines % 10000 == 0:
            logger.info(f"Read {lines} lines")
        if batches_found != last_logged_batch and batches_found % 1000 == 0:
            logger.info(f"Found {batches_found} batches")
            last_logged_batch = batches_found

    logger.info(f"Read {lines} lines")
    logger.info(f"Yielded {batches_found} batches")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    map_fn = partial(
        batch_process,
        processor=data_processor,
        tokenizer=tokenizer
    )

    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(map_fn, batches),
            total=len(batches),
            desc='Tokenizing')
        )

    tensorized_data = TensorizedDataset(out_path.stem)
    for processed_batch in results:
        tensorized_data.add_instances(processed_batch)

    logger.info(f"{tensorized_data.total_tokens:e} total tokens found")
    logger.info(f"{tensorized_data.input_token_count:e} input tokens found")
    logger.info(f"{tensorized_data.target_token_count:e} target tokens found")

    with out_path.open('wb') as f:
        pickle.dump(tensorized_data, f)

    logger.info(f"Size of {tensorized_data.name} is {human_readable_size(out_path.stat().st_size)}")
