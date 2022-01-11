import pytest
from datasets import Dataset
from pathlib import Path
from typing import Dict

from yamrf import Task

__all__ = [
    "DummyTask"
]


@Task.register('dummy')
class DummyTask(Task):

    @staticmethod
    def _map_to_standard_entries(sample: Dict) -> Dict:
        sample['input_sequence'] = sample['input']
        sample['target'] = sample['output']
        return sample

    def _load_dataset(self, data_path: Path) -> Dataset:
        return Dataset.from_dict({
            "idx"   : [0, 1, 2],
            "input" : ["The comment section is ", "The butcher of ", "Get "],
            "output": ["out of control.", "Blevkin.", "Some."]
        })
