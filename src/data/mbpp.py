"""
Code for handling the mostly basic programming problems dataset from
https://arxiv.org/pdf/2108.07732.pdf
"""
from pathlib import Path
import logging
from transformers import PreTrainedTokenizer
from typing import Callable, List, Dict
from datasets import Dataset
from src.data.task import Task

logger = logging.getLogger(__name__)


@Task.register("mbpp")
class MBPP(Task):
    """
    Task for the Mostly Basic Programming Problems Dataset.
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
            postprocessors=postprocessors,
        )

        self._tokenizer = tokenizer
        self.dataset = None
        self.raw = None

    def _load_dataset(self, data_path: Path) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return Dataset.from_json(str(data_path))

    @staticmethod
    def _map_to_standard_entries(sample: Dict) -> Dict:
        sample["input_sequence"] = (
            sample["text"] + "\n" + "\n".join(sample["test_list"])
        )
        sample["target"] = sample["code"]
        return sample
