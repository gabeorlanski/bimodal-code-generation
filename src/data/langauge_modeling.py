"""
This comes from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot
"""
import math

from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
import torch
import logging

logger = logging.getLogger(__name__)


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for processing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences: Number of token sequences to keep in buffer.
            chars_per_token: Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
            self,
            tokenizer,
            dataset,
            infinite=False,
            seq_length=1024,
            num_of_sequences=1024,
            chars_per_token=3.6
    ):
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.pad_token = tokenizer.eos_token_id

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["input_ids"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                    else:
                        more_examples = False
                        break
            all_token_ids = []
            for tokenized_input in buffer:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield {"input_ids": torch.tensor(input_ids)}
                elif not more_examples:
                    yield {
                        "input_ids": torch.tensor(
                            input_ids + [self.pad_token] * (self.seq_length - len(input_ids))
                        )
                    }


def create_dataloaders(args, train_data: Dataset, val_data: Dataset, cfg: DictConfig, tokenizer):
    train_data = train_data.shuffle(seed=cfg.seed)
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        infinite=True,
        seq_length=cfg.data_args.seq_length,
        num_of_sequences=cfg.data_args.num_sequences
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        val_data,
        infinite=False,
        seq_length=cfg.data_args.seq_length,
        num_of_sequences=cfg.data_args.num_sequences
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader
