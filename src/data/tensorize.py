import json
import logging
import math
import os
from collections import Counter
from copy import deepcopy, copy
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from tqdm import tqdm
from src.common.file_util import human_readable_size
import psutil
import ujson

from src.data.stackoverflow import StackOverflowProcessor

logger = logging.getLogger(__name__)
__all__ = [
    "TensorizedDatasetInfo",
    "get_dataset_info_with_processor",
    "TensorizedTask"
]


@dataclass
class TensorizedDatasetInfo:
    name: str
    token_counts: Dict = field(default_factory=Counter)
    num_instances: int = 0

    def add_instance(self, instance):
        self.num_instances += 1
        for c, v in instance.items():
            self.token_counts[c] += v

    @property
    def total_tokens(self):
        return sum(self.token_counts.values())

    def to_dict(self):
        return {
            "name"         : self.name,
            "token_counts" : dict(self.token_counts),
            "num_instances": self.num_instances
        }


def batch_process(batch, processor, tokenizer):
    return processor(batch, tokenizer)


PLACEHOLDER_STR = "_PLACEHOLDER_"


class TensorizedTask(IterableDataset):

    def __init__(
            self,
            name,
            raw_data_name,
            dump_path,
            objective,
            processor,
            tokenizer: PreTrainedTokenizer,
            max_samples_to_yield,
            sequence_length=1024,
            buffer_size=1,
    ):
        self.name = name
        self.data_file_path = dump_path.joinpath(f'{raw_data_name}.jsonl')
        self.objective = objective
        if self.objective not in ['lm', 'seq2seq']:
            raise ValueError(f"Unsupported Objective {self.objective}")
        self.processor = processor
        self.task_name = None
        self.infinite = True
        if isinstance(self.processor, StackOverflowProcessor):
            self.task_name = "SO"
        self.tokenizer_name = tokenizer.name_or_path
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.sequence_length = sequence_length

        self.buffer_size = int(buffer_size * 1000)
        self.lm_concat_delim = tokenizer.encode('\n')
        self.max_samples_to_yield = max_samples_to_yield
        logger.debug(f"Reading lines from {self.data_file_path}")
        self.max_num_lines_in_file = sum(1 for _ in self.data_file_path.open())
        self.placeholder_token_count = len(
            tokenizer.tokenize(PLACEHOLDER_STR, add_special_tokens=False)
        )
        logger.info(
            f"{self.max_num_lines_in_file} total lines in {self.data_file_path} "
            f"with a buffer of {self.buffer_size}"
        )

    def _get_length(self, data_path):
        logger.info(f"Loading tensorized dataset from {data_path}")

        tensorized_cfg = TensorizedDatasetInfo(**json.loads(
            data_path.joinpath(f"{self.name}.cfg.json").read_text()
        ))

        logger.info(f"{tensorized_cfg.total_tokens:e} total tokens found")
        logger.info(f"{tensorized_cfg.num_instances:e} instances found")

        if self.objective == 'seq2seq':
            return tensorized_cfg.num_instances

        logger.info(f"Reading token counts from {data_path.joinpath(self.name + '.jsonl')}")

        total_tokens = tensorized_cfg.total_tokens + tensorized_cfg.num_instances * (
                len(self.lm_concat_delim) + 1
        )
        length = total_tokens // self.sequence_length
        logger.info(f"{length} total elements to return")
        return length

    def __iter__(self):
        data_iter = iter(self.data_file_path.open())
        ds_epoch = 0
        more_examples = True
        total_yielded = 0
        lines_seen = 0
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            slices_per_worker = self.buffer_size
            worker_id = 0
            max_yielded_per_worker = self.max_samples_to_yield
        else:
            max_yielded_per_worker = self.max_samples_to_yield // worker_info.num_workers
            slices_per_worker = int(math.ceil(self.buffer_size / worker_info.num_workers))
            worker_id = worker_info.id

        while more_examples and total_yielded < max_yielded_per_worker:
            # Read the file and add the lines to the line buffer. This buffer
            # will then be split by each process.
            processed = []
            logger.debug(f"Starting Read on line {lines_seen}")
            while len(processed) < self.buffer_size:
                try:
                    line = ujson.loads(next(data_iter))
                    if len(line) == 0:
                        if worker_id == 0:
                            logger.warning(
                                f"Line {lines_seen + 1} with id {line['id']} "
                                f"had no samples after processing."
                            )
                    else:
                        processed.extend(self.processor.make_instances_from_question(line))
                    lines_seen += 1
                    ds_epoch = lines_seen / self.max_num_lines_in_file
                except StopIteration:
                    if self.infinite:
                        logger.info(f"New Dataset Epoch")
                        data_iter = iter(self.data_file_path.open())
                    else:
                        more_examples = False
                        break

            if worker_info is None:
                start = 0
                end = self.buffer_size
            else:
                start = worker_id * slices_per_worker
                end = min(self.buffer_size, start + slices_per_worker)
            logger.debug(f"{worker_id=} On dataset epoch {ds_epoch}")
            processed_inputs, processed_labels = [], []

            for line in lines[start:end]:
                processed_inputs.append(line['input'])
                processed_labels.append(line['labels'])

            inputs_tokenized = tokenizer(
                processed_inputs,
                max_length=self.sequence_length,
                truncation=True
            )['input_ids']
            labels_tokenized = tokenizer(
                processed_labels,
                max_length=self.sequence_length,
                truncation=True
            )['input_ids']

            buffer = []
            if self.objective == 'lm':
                for input_ids, labels in zip(inputs_tokenized, labels_tokenized):
                    buffer.extend(
                        input_ids
                        + self.lm_concat_delim
                        + labels
                        + [self.concat_token_id]
                    )
                for i in range(0, len(buffer), self.sequence_length):
                    token_start = i
                    token_end = i + self.sequence_length
                    input_ids = buffer[token_start:token_end]
                    if len(input_ids) == self.sequence_length:
                        total_yielded += 1
                        yield {
                            'input_ids': torch.tensor(input_ids),
                            # 'attention_mask': torch.tensor([1] * len(input_ids)),
                            'labels'   : torch.tensor(input_ids),
                        }
                    if total_yielded >= max_yielded_per_worker:
                        logger.info(f"{worker_id} finished")
                        break
            else:
                for input_ids, labels in zip(inputs_tokenized, labels_tokenized):
                    total_yielded += 1
                    yield {
                        "input_ids": input_ids,
                        "labels"   : labels
                    }
                    if total_yielded >= max_yielded_per_worker:
                        logger.info(f"{worker_id} finished")
                        break
            # Memory Management
            del buffer

    @property
    def params(self):
        return dict(
            name=self.name,
            data_file_path=self.data_file_path,
            objective=self.objective,
            concat_token_id=self.concat_token_id,
            sequence_length=self.sequence_length,
            tokenizer=self.tokenizer_name,
            buffer_size=self.buffer_size,
            lm_concat_delim=self.lm_concat_delim,
            # length=self.length,
            infinite=self.infinite
        )


def get_token_counts_stackoverflow(sample, processor, tokenizer):
    processed = processor(sample)
    return {
        "input_ids": len(tokenizer.tokenize(
            processed['input'],
            add_special_tokens=False)
        ),
        "labels"   : len(tokenizer.tokenize(
            processed['labels'],
            add_special_tokens=False)
        )
    }


def get_token_counts_callable(sample, processor, tokenizer):
    out = {}
    token_counts = {}
    for k, v in processor(sample).items():
        if isinstance(k, str):
            out[k] = len(tokenizer.encode(v, add_special_tokens=False))
            token_counts[k] = out[k]

        else:
            out[k] = v
    return token_counts


def batch_get_tokens(batch, processor, tokenizer):
    if isinstance(processor, StackOverflowProcessor):
        out = []
        processed = processor(batch)
        for input_ids, labels in zip(processed['inputs'], processed['labels']):
            out.append({
                "inputs": len(tokenizer.tokenize(input_ids)),
                "labels": len(tokenizer.tokenize(labels)),
            })
        return out
    elif callable(processor):

        return list(map(
            lambda b: processor(b, processor, tokenizer),
            batch
        ))

    raise ValueError(f"Unsupported Task")


def get_dataset_info_with_processor(
        raw_data_path: Path,
        output_name: str,
        num_workers: int,
        model_name: str,
        processor,
        batch_size,
        debug_max_samples
):
    logger.info(f"Tensorizing {raw_data_path}")
    # Setup the queues
    max_batches_in_memory = 50000
    more_examples = True
    logger.info(f"Reading {raw_data_path}")
    lines = 0
    buffer = []
    batches = []
    batches_found = 0
    last_logged_batch = 0
    raw_lines_iter = iter(raw_data_path.open('r').readlines())

    tensorized_data = TensorizedDatasetInfo(output_name)
    finished = 0

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    map_fn = partial(
        batch_get_tokens,
        tokenizer=tokenizer,
        processor=processor
    )

    while more_examples:
        while len(batches) < max_batches_in_memory:
            try:
                line = next(raw_lines_iter)
            except StopIteration:
                more_examples = False
                break
            lines += 1
            buffer.append(line)
            if len(buffer) == batch_size:
                batches.append(list(map(ujson.loads, buffer)))
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
            if debug_max_samples != -1 and lines >= debug_max_samples:
                logger.warning(f"Stopping at {lines} for debugging")
                more_examples = False
                break
        if buffer:
            batches.append(list(map(ujson.loads, buffer)))
            batches_found += 1
        logger.info(f"Read {lines} lines")
        logger.info(f"Yielded {batches_found} batches")

        with mp.Pool(num_workers) as pool:
            for result in tqdm(pool.imap(map_fn, batches), total=len(batches), desc='Tokenizing'):
                for token_counts in result:
                    tensorized_data.add_instance(token_counts)
                    finished += 1
                    if finished % 50000 == 0:
                        ram_pct = f"{psutil.virtual_memory()[2]:0.2f}%"
                        logger.debug(f"Found {finished:>8}"
                                     f"| RAM Used={ram_pct:<6}")
        del batches
        batches = []
    logger.info(f"{tensorized_data.total_tokens:.3e} total tokens found")
    for k, v in tensorized_data.token_counts.items():
        logger.info(f"{v:.3e} tokens for {k.replace('_', ' ')} found")

    return tensorized_data
