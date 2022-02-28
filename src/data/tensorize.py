import json
import logging
import os
import pickle
import random
import threading
from copy import deepcopy, copy
from dataclasses import dataclass, field, asdict
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
import psutil

logger = logging.getLogger(__name__)
__all__ = [
    "TensorizedDatasetCFG",
    "tensorize",
    "TensorizedTask"
]


@dataclass
class TensorizedDatasetCFG:
    name: str
    input_token_count: int = 0
    target_token_count: int = 0
    num_instances: int = 0

    def add_instance(self, instance):
        self.num_instances += 1
        self.input_token_count += len(instance['input_ids'])
        self.target_token_count += len(instance['label'])

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
            name,
            data_path,
            objective,
            tokenizer: PreTrainedTokenizer,
            infinite=False,
            sequence_length=1024,
            max_instances=-1,
            local_rank: int = -1,
            effective_batch_size: int = 16
    ):
        self.objective = objective
        self.data_path = data_path
        self.name = name
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.sequence_length = sequence_length
        self.infinite = infinite
        self.tokenizer = tokenizer
        self.buffer_size = effective_batch_size * self.sequence_length * 25
        self.lm_concat_delim = self.tokenizer.encode('\n')
        self.max_instances = max_instances
        self.tensorized_cfg, self.length = self._load_samples(data_path, max_instances)
        self.tensorized_cfg: TensorizedDatasetCFG
        logger.info(f"{self.length} total samples with a buffer of {self.buffer_size}")
        self.samples_seen = 0
        self.tokens_seen = 0
        self.epoch = 0
        logger.debug(f"{max_instances=} while length is {self.length}")

    def _load_samples(self, data_path, max_instances):
        logger.info(f"Loading tensorized dataset from {data_path}")

        tensorized_cfg = TensorizedDatasetCFG(**json.loads(
            data_path.joinpath(f"{self.name}.cfg.json").read_text()
        ))

        logger.info(f"{tensorized_cfg.total_tokens:e} total tokens found")
        logger.info(f"{tensorized_cfg.input_token_count:e} input tokens found")
        logger.info(f"{tensorized_cfg.target_token_count:e} target tokens found")
        logger.info(f"{tensorized_cfg.num_instances:e} instances found")
        if max_instances != -1:
            return tensorized_cfg, max_instances

        if self.objective == 'lm':
            return tensorized_cfg, (
                    tensorized_cfg.total_tokens
                    + ((len(self.lm_concat_delim) + 1)
                       * tensorized_cfg.num_instances)
            ) // self.sequence_length
        return tensorized_cfg, tensorized_cfg.num_instances

    def get_samples(self):
        for line in self.data_path.joinpath(f"{self.name}.jsonl").open():
            yield json.loads(line)

    def __iter__(self):
        buffer = []
        data_iter = iter(self.get_samples())
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
                    self.samples_seen += 1
                except StopIteration:
                    if self.infinite or (self.max_instances != -1 and total_yielded < self.length):
                        data_iter = iter(self.get_samples())
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            overflow = []
            for i in range(0, len(buffer), self.sequence_length):
                input_ids = buffer[i: i + self.sequence_length]
                if len(input_ids) == self.sequence_length:
                    total_yielded += 1
                    self.tokens_seen += self.sequence_length
                    yield {
                        'input_ids'     : torch.tensor(input_ids),
                        'attention_mask': torch.tensor([1] * len(input_ids)),
                        'labels'        : torch.tensor(input_ids),
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
        output_name: str,
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

        if lines % 50000 == 0:
            logger.info(f"Read {lines} lines. ")

        if batches_found != last_logged_batch and batches_found % 1000 == 0:
            logger.info(f"Found {batches_found} batches")
            last_logged_batch = batches_found
            logger.info(f"RAM Used={psutil.virtual_memory()[2]}%")
            logger.info(f"CPU Used={psutil.getloadavg()[-1] / os.cpu_count() * 100:0.2f}%")

    logger.info(f"Read {lines} lines")
    logger.info(f"Yielded {batches_found} batches")

    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
    map_fn = partial(
        batch_process,
        processor=data_processor,
        tokenizer=tokenizer
    )

    tensorized_data = TensorizedDatasetCFG(out_path.stem)
    out_file_path = out_path.joinpath(f"{output_name}.jsonl")
    with out_file_path.open('w') as out_file:
        with mp.Pool(num_workers) as pool:
            for result in tqdm(pool.imap(map_fn, batches), total=len(batches), desc='Tokenizing'):
                for instance in result:
                    out_file.write(json.dumps(instance) + '\n')
                    tensorized_data.add_instance(instance)

    logger.info(f"{tensorized_data.total_tokens:e} total tokens found")
    logger.info(f"{tensorized_data.input_token_count:e} input tokens found")
    logger.info(f"{tensorized_data.target_token_count:e} target tokens found")
    logger.info(f"{tensorized_data.num_instances:e} instances found")

    logger.info(f"Saved to {out_file_path}")

    logger.info(f"Size of {tensorized_data.name} is {human_readable_size(out_path.stat().st_size)}")
