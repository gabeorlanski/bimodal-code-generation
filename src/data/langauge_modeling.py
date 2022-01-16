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


class LanguageModelingDataset(IterableDataset):
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
            chars_per_token=3.6,
            streaming=True,
            batches_per_epoch=100,
    ):
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.max_chars_in_buffer = seq_length * chars_per_token * num_of_sequences
        self.streaming = streaming
        self.infinite = infinite and streaming
        self.tokenizer = tokenizer
        self.samples_per_epoch = batches_per_epoch
        self._ds_iterator = None
        self.samples = []
        self.samples_yielded = 0
        if not self.streaming:
            self.initialize()

    def initialize(self):
        logger.info("Streaming is disabled, initializing the samples.")
        for b in self._get_batch():
            self.samples.append(b)
        self.samples_per_epoch = len(self.samples)
        logger.info(f"{len(self.samples)} total samples.")

    def _get_batch(self):
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
                input_ids.extend([self.concat_token_id] * pad_amount)
                attention_mask.extend([0] * pad_amount)

                self.samples_yielded += 1

                yield {
                    "input_ids"     : torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask)
                }

                if self.samples_yielded == self.samples_per_epoch:
                    return

    def __iter__(self):
        if self.streaming:
            iterator_for_yielding = self._get_batch()
        else:
            iterator_for_yielding = self.samples
        for b in iterator_for_yielding:
            yield b

    def __len__(self):
        return self.samples_per_epoch


def create_dataloaders(args, train_dataset: Dataset, eval_dataset: Dataset, cfg: DictConfig,
                       tokenizer):
    train_data = train_dataset.shuffle(seed=cfg.seed)
    train_dataset = LanguageModelingDataset(
        tokenizer,
        train_data,
        infinite=cfg.data_args.get('infinite', False),
        seq_length=cfg.data_args.seq_length,
        num_of_sequences=cfg.data_args.num_sequences,
        streaming=cfg.data_args.streaming,
        # batches_per_epoch=args.steps_per_epoch * args.train_batch_size,
    )
    valid_dataset = LanguageModelingDataset(
        tokenizer,
        eval_dataset,
        infinite=False,
        seq_length=cfg.data_args.seq_length,
        num_of_sequences=cfg.data_args.num_sequences,
        streaming=cfg.data_args.streaming,
        # batches_per_epoch=args.steps_per_epoch * args.eval_batch_size,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size)
    return train_dataloader, eval_dataloader
