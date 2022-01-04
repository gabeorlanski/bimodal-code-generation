from pathlib import Path
from typing import Dict, List, Callable, Tuple

from src.common import Registrable
from datasets import Dataset
from transformers import PreTrainedTokenizer


class DatasetReader(Registrable):
    """
    Base class for reading in datasets

    Args:
        tokenizer (PreTrainedTokenizer): HuggingFace Tokenizer.

        preprocessors (List[Callable]): List of callables for preprocessing.
            Each must take in a single argument of type ``Dict`` and return
            a ``Dict``.

            Each ``example`` passed to the preprocessors will have an input
            sequence and a target entry.

        postprocessors: (List[Callable]): List of callables for postprocessing.
            Each must take in a single argument of type ``Dict`` and return
            a ``Dict``.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable]
    ):
        self.input_sequence_key = "input_sequence",
        self.target_key = "target"
        self.tokenizer = tokenizer
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []

    def _load_dataset(
            self,
            data_path: Path
    ) -> Dataset:
        """
        Method to read in the raw data that is to be implemented by subclasses.
        Args:
            data_path (Path): Path to the data.

        Returns:
            Dataset: The processed HuggingFace Dataset.

        """
        raise NotImplementedError()

    def read_data(
            self,
            data_path: Path,
            num_procs: int = 1
    ) -> Tuple[Dataset, Dataset]:
        """
        Method to read and preprocess dataset.

        Args:
            data_path (Path): Path to the data.
            num_procs (int): Number of processes to use in preprocessing.

        Returns:
            Tuple[Dataset]: The preprocessed and tokenized Datasets.
        """
        dataset = self._load_dataset(data_path)

        def preprocess(example, idx):
            for fn in self.preprocessors:
                example = fn(example)
            return {"idx": idx, **example}

        def tokenize(example, idx):
            out = {"idx": idx, **self.tokenizer(example['input_sequence'])}
            target_tokenized = self.tokenizer(example.pop('target'))
            out.update({
                'labels'              : target_tokenized['input_ids'],
                'label_attention_mask': target_tokenized['attention_mask']
            })
            return out

        preprocessed = dataset.map(
            preprocess,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names
        )
        tokenized = preprocessed.map(
            tokenize,
            with_indices=True,
            num_proc=num_procs,
            remove_columns=dataset.column_names
        )

        return preprocessed, tokenized
