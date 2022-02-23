import json
import logging
import random
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Callable
import multiprocessing as mp
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
__all__ = [
    "TensorizeProcessor"
]


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

            instance_batch = next_task

            self.tasks.task_done()
            completed += 1
            if completed % 5000 == 0:
                self._log(logging.INFO, f"Finished {completed}")
