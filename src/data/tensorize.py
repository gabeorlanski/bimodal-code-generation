import json
import logging
import pickle
import random
import threading
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
    "TensorizeProcessor",
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


class TensorizeProcessor(mp.Process):
    def __init__(
            self,
            worker_id,
            task_queue,
            result_queue,
            log_queue,
            model_name,
            data_processor
    ):
        super().__init__()
        self.worker_id = worker_id
        self.tasks = task_queue
        self.results = result_queue
        self.logs = log_queue
        self.model_name = model_name
        self.processor = data_processor

    def _log(self, level, message):
        self.logs.put((level, f"WORKER {self.worker_id}: {message}"))

    def run(self):
        completed = 0
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._log(logging.INFO, "Started")
        while True:
            next_task = self.tasks.get()

            # Poison pill means shutdown.
            if next_task is None:
                self.tasks.task_done()
                self._log(logging.INFO, "Finished")
                return

            batch_num, instance_batch = next_task
            try:
                parsed_instances = self.processor(instance_batch, tokenizer)
            except Exception as e:
                parsed_instances = None
                self._log(logging.ERROR, f"{batch_num=} failed with exception {e}")

            self.results.put((batch_num, parsed_instances))

            self.tasks.task_done()
            completed += 1
            if completed % 1000 == 0:
                self._log(logging.INFO, f"Finished {completed}")


def tensorize_main_process(
        save_name,
        raw_data_path,
        workers,
        task_queue,
        result_queue,
        batch_size
) -> TensorizedDataset:
    logger.info(f"Reading {raw_data_path}")
    lines = 0
    buffer = []
    batches_found = 0
    last_logged_batch = 0
    for line in map(json.loads, raw_data_path.open('r')):
        lines += 1
        buffer.append(line)
        if len(buffer) == batch_size:
            task_queue.put((batches_found, buffer))
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
    for _ in workers:
        task_queue.put(None)

    task_queue.join()

    failure_count = 0
    found = 0
    tensorized_data = TensorizedDataset(save_name)

    pbar = tqdm(total=batches_found, desc='Processing')
    while not result_queue.empty():
        batch_num, result = result_queue.get(timeout=5.0)
        found += 1
        pbar.update()
        if result is None:
            failure_count += 1
            logger.warning(f"{batch_num} failed to tensorize")
            continue
        tensorized_data.add_instances(result)

    pbar.close()

    logger.info(f"{failure_count}/{batches_found} failed")
    logger.info(f"{tensorized_data.total_tokens:e} total tokens found")
    logger.info(f"{tensorized_data.input_token_count:e} input tokens found")
    logger.info(f"{tensorized_data.target_token_count:e} target tokens found")
    return tensorized_data


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
    log_queue = mp.Queue()
    task_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue()

    # Setup processors
    processor_init_fn = partial(
        TensorizeProcessor,
        task_queue=task_queue,
        result_queue=result_queue,
        log_queue=log_queue,
        model_name=model_name,
        data_processor=data_processor
    )
    logger.info(f"Creating {num_workers} workers")
    workers = [processor_init_fn(i) for i in range(num_workers-1)]
    log_thread = threading.Thread(
        target=log_process,
        args=(log_queue, num_workers)
    )
    log_thread.start()
    for w in workers:
        w.start()
    try:
        to_save = tensorize_main_process(
            out_path.stem,
            raw_data_path,
            workers,
            task_queue,
            result_queue,
            batch_size
        )
    except Exception as e:
        # Making sure we cleanup
        for w in workers:
            w.terminate()
        log_queue.put('KILL')
        log_thread.join()
        raise e

    log_queue.put('KILL')
    log_thread.join()
    for w in workers:
        w.terminate()

    logger.info(f"Saving {to_save.name} to {out_path}")
    with out_path.open('wb') as f:
        pickle.dump(to_save, f)

    logger.info(f"Size of {to_save.name} is {human_readable_size(out_path.stat().st_size)}")
