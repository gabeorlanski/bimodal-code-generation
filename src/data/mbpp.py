"""
Code for handling the mostly basic programming problems dataset from
https://arxiv.org/pdf/2108.07732.pdf
"""
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Optional, Union, Callable, List, Dict
from datasets import Dataset
from dataclasses import dataclass, field
from omegaconf import MISSING
from src.data.dataset_reader import DatasetReader, DatasetReaderConfig

logger = logging.getLogger(__name__)


@dataclass
class MBPPConfig(DatasetReaderConfig):
    name: str = 'mbpp'
    train_path: str = "MBPP/train.jsonl"
    validation_path: str = "MBPP/validation.jsonl"


@DatasetReader.register('mbpp')
class MBPP(DatasetReader):
    """
    DatasetReader for the Mostly Basic Programming Problems Dataset.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable] = None,
            postprocessors: List[Callable] = None,
    ):
        super(MBPP, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors
        )

        self._tokenizer = tokenizer
        self.dataset = None
        self.raw = None

    def _load_dataset(self, data_path: Path) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return Dataset.from_json(str(data_path))

    @staticmethod
    def _map_to_standard_entries(sample: Dict) -> Dict:
        sample['input_sequence'] = sample['text'] + '\n' + '\n'.join(sample['test_list'])
        sample['target'] = sample['code']
        return sample
