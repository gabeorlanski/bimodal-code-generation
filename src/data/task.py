from pathlib import Path
from typing import Dict, List, Callable

from src.common import Registrable
from datasets import Dataset
from transformers import PreTrainedTokenizer


class Task(Registrable):
    def __init__(
            self,
            data_path: Path,
            preprocessors: List[Callable],
            tokenizer: PreTrainedTokenizer,
            postprocessors: List[Callable]
    ):
        self.input_sequence_key = "input_sequence",
        self.target_key = "target"
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
        self._dataset: Dataset = None  # type:ignore

    def _load_dataset(self) -> Dataset:
        raise NotImplementedError()

    def read_data(self, num_procs: int = 1):
        self._dataset = self._load_dataset()

        def preprocess(example, idx):
            for fn in self.preprocessors:
                example = fn(example)

            out = self.tokenizer(example.pop('input_sequence'))
            target_tokenized = self.tokenizer(example.pop('target'))
            out.update({
                'labels'              : target_tokenized['input_ids'],
                'label_attention_mask': target_tokenized['attention_mask']
            })
            out['idx'] = idx
            out.update(example)
            return out

        self._dataset = self._dataset.map(
            preprocess,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=self._dataset.column_names
        )
