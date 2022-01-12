import pytest
from datasets import Dataset
from pathlib import Path
from typing import Dict

from tio import Task

__all__ = [
    "DummyTask"
]


@Task.register('dummy')
class DummyTask(Task):

    @staticmethod
    def map_to_standard_entries(sample: Dict) -> Dict:
        sample['input_sequence'] = sample['input']
        sample['target'] = sample['output']
        return sample

    def dataset_load_fn(self, split: str) -> Dataset:
        return Dataset.from_dict({
            "idx"   : [0, 1, 2, 3],
            "input" : ["The comment section is ", "The butcher of ", "Get ", "I hate"],
            "output": ["out of control.", "Blevkin.", "Some.", "tf.data"]
        })
