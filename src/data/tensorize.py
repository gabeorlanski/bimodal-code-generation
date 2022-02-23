import json
import logging
import pickle
import random
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Callable
import multiprocessing as mp
from transformers import AutoTokenizer
from src.data.parse_so.util import log_process
from tqdm import tqdm
from src.common.file_util import human_readable_size

logger = logging.getLogger(__name__)
__all__ = [
    "TensorizedDataset",
    "tensorize"
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
