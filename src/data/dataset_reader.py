from pathlib import Path
from typing import Dict, List, Callable, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
from dataclasses import dataclass, field
from omegaconf import MISSING
from src.common import Registrable


@dataclass
class DatasetReaderConfig:
    name: str
    train_path: str = MISSING
    validation_path: str = MISSING


class DatasetReader(Registrable):
    """
    Base class for reading in dataset

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
        self.preprocessors = [self._map_to_standard_entries, *(preprocessors or [])]
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

        It returns the preprocessed dataset as well so that the tokenized
        dataset can be re-aligned with the original data for evaluation later
        on.

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

    @staticmethod
    def _map_to_standard_entries(sample: Dict) -> Dict:
        """
        Function that must be implemented by sub-classes for mapping dataset
        specific columns to standardized ones.

        The output dict must have the keys ``"input_sequence"`` and
        ``"target"``.
        Args:
            sample (Dict): The dict for a given sample in the dataset.

        Returns:
            Dict: The sample with the added standard entries.

        """
        raise NotImplementedError()
