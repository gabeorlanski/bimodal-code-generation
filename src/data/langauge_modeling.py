"""
This comes from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot
"""
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import IterableDataset
import torch
import logging

logger = logging.getLogger(__name__)


def raw_group_texts(texts: List, concat_token, seq_length, padding=False):
    buffer = []
    for text in texts:
        buffer.append(text)
    all_token_ids = []

    for tokenized_input in buffer:
        all_token_ids.extend(tokenized_input + [concat_token])

    all_input_ids = []
    all_attention_mask = []
    for i in range(0, len(all_token_ids), seq_length):
        input_ids = all_token_ids[i: i + seq_length]
        input_len = len(input_ids)
        attention_mask = [1] * input_len

        # We do not want to skip any examples, so we must pad some of
        # them with the concat id. But, we also want those to be ignored
        # so we need to create the attention mask.
        if not padding and len(input_ids) != seq_length:
            continue
        elif padding:
            pad_amount = seq_length - input_len
            input_ids.extend([concat_token] * pad_amount)
            attention_mask.extend([0] * pad_amount)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    out = {
        "input_ids"     : all_input_ids,
        "attention_mask": all_attention_mask,
        "labels"        : all_input_ids.copy()
    }

    return out


class ConstantLengthDataset(IterableDataset):
    def __init__(
            self,
            tokenizer,
            data_path: Path,
            max_steps=-1,
            seq_length=1024,
            effective_batch_size=256,
            seed=1,
            local_rank=-1
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.data_path = data_path
        self.seq_length = seq_length
        self.max_buffer_size = seq_length * effective_batch_size
        self.effective_batch_size = effective_batch_size
        self.epoch = 0
        self.infinite = max_steps != -1
        if max_steps != -1:
            self.length = max_steps * effective_batch_size * seq_length
        else:
            self.length = float('inf')

        self.rng = np.random.default_rng(seed)
        self.local_rank = local_rank
        self.cache = []
        self.concat_delim = self.tokenizer.encode('\n')

    def get_next_sequence(self):
        for line in map(json.loads, self.data_path.open()):
            yield line['input_ids'] + self.concat_delim + line['label']

    def __iter__(self):
        iterator = iter(self.get_next_sequence())
        more_examples = True
        total_yielded = 0
        while more_examples and total_yielded < self.length:
            buffer = []
            buffer_size = 0
            while buffer_size < self.max_buffer_size:

                try:
                    sequence = next(iterator)
                    buffer.append(sequence)
                    buffer_size += len(sequence)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.get_next_sequence())
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            self.rng.shuffle(buffer)
            all_token_ids = []
            for tokenized_input in buffer:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    total_yielded += 1
                    if not self.infinite:
                        self.cache.append(input_ids)
                    yield torch.tensor(input_ids)

    def __len__(self):
        return self.length


class IterableDatasetShard(IterableDataset):
    """
    Wraps a PyTorch :obj:`IterableDataset` to generate samples for one of the processes only. Instances of this class
    will always yield a number of samples that is a round multiple of the actual batch size (depending of the value of
    :obj:`split_batches`, this is either :obj:`batch_size` or :obj:`batch_size x num_processes`). Depending on the
    value of the :obj:`drop_last` attribute of the batch sampler passed, it will either stop the iteration at the first
    batch that would be too small or loop with indices from the beginning.
    Args:
        dataset (:obj:`torch.utils.data.dataset.IterableDataset`):
            The batch sampler to split in several shards.
        batch_size (:obj:`int`, `optional`, defaults to 1):
            The size of the batches per shard (if :obj:`split_batches=False`) or the size of the batches (if
            :obj:`split_batches=True`).
        drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to drop the last incomplete batch or complete the last batches by using the samples from the
            beginning.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.
        split_batches (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the shards should be created by splitting a batch to give a piece of it on each process, or by
            yielding different full batches on each process.
            On two processes with an iterable dataset yielding of :obj:`[0, 1, 2, 3, 4, 5, 6, 7]`, this will result in:
            - the shard on process 0 to yield :obj:`[0, 1, 2, 3]` and the shard on process 1 to yield :obj:`[4, 5, 6,
              7]` if this argument is set to :obj:`False`.
            - the shard on process 0 to yield :obj:`[0, 1, 4, 5]` and the sampler on process 1 to yield :obj:`[2, 3, 6,
              7]` if this argument is set to :obj:`True`.
    """

    def __init__(
            self,
            dataset: IterableDataset,
            batch_size: int = 1,
            drop_last: bool = False,
            num_processes: int = 1,
            process_index: int = 0,
            split_batches: bool = False,
    ):
        if split_batches and batch_size > 1 and batch_size % num_processes != 0:
            raise ValueError(
                f"To use `IterableDatasetShard` in `split_batches` mode, the batch size ({batch_size}) "
                f"needs to be a round multiple of the number of processes ({num_processes})."
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.split_batches = split_batches
        logger.info(f"Iterable Shard initialized with {num_processes=} and {process_index=} ")

    def __iter__(self):
        real_batch_size = self.batch_size if self.split_batches else (
                    self.batch_size * self.num_processes)
        process_batch_size = (
                    self.batch_size // self.num_processes) if self.split_batches else self.batch_size
        process_slice = range(self.process_index * process_batch_size,
                              (self.process_index + 1) * process_batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            current_batch.append(element)
            # Wait to have a full batch before yielding elements.
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        # Finished if drop_last is True, otherwise complete the last batch with elements from the beginning.
        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]
