"""
This comes from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot
"""
import math

from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch
from torch.nn import functional as F
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
            tokenizer: PreTrainedTokenizer,
            dataset,
            infinite=False,
            seq_length=1024,
            num_of_sequences=3,
            chars_per_token=3.6
    ):
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_chars_in_buffer = seq_length * chars_per_token * num_of_sequences
        self.infinite = infinite
        self.tokenizer = tokenizer

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_chars_in_buffer:
                    break
                try:
                    buffer.append(next(iterator)["input_sequence"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:

                        # Reset the iterator if it is infinite.
                        iterator = iter(self.dataset)

                    else:
                        more_examples = False
                        break
            all_token_ids = []

            for tokenized_input in self.tokenizer(buffer, add_special_tokens=False)['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i: i + self.seq_length]
                input_len = len(input_ids)
                attention_mask = [1] * input_len

                # We do not want to skip any examples, so we must pad some of
                # them with the concat id. But, we also want those to be ignored
                # so we need to create the attention mask.
                pad_amount = self.seq_length - input_len
                input_ids.extend([self.concat_token_id]*pad_amount)
                attention_mask.extend([0]*pad_amount)

                yield {
                    "input_ids"     : torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask)
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
