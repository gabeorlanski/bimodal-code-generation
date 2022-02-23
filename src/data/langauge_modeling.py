"""
This comes from
https://github.com/huggingface/transformers/blob/master/examples/research_projects/codeparrot
"""
import math
from typing import Dict, List

from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


def group_texts(texts: List, concat_token, seq_length, padding=False):
    buffer = []
    for text in texts:
        buffer.append(text)
    all_token_ids = []

    for tokenized_input in buffer:
        all_token_ids.extend(tokenized_input + [concat_token])

    all_input_ids = []
    all_attention_mask = []
    for i in range(0, len(all_token_ids), seq_length):
        input_ids = all_token_ids[i: i + seq_length]
        input_len = len(input_ids)
        attention_mask = [1] * input_len

        # We do not want to skip any examples, so we must pad some of
        # them with the concat id. But, we also want those to be ignored
        # so we need to create the attention mask.
        if not padding and len(input_ids) != seq_length:
            continue
        elif padding:
            pad_amount = seq_length - input_len
            input_ids.extend([concat_token] * pad_amount)
            attention_mask.extend([0] * pad_amount)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)

    out = {
        "input_ids"     : all_input_ids,
        "attention_mask": all_attention_mask,
        "labels"        : all_input_ids.copy()
    }

    return out


class TensorizedDataSet(IterableDataset):
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
            data_path,
            objective,
            tokenizer: PreTrainedTokenizer,
            infinite=False,
            seq_length=1024,
            streaming=True,
            max_steps=-1,
    ):
        self.concat_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.seq_length = seq_length
        self.infinite = infinite and streaming
        self.tokenizer = tokenizer
        self._ds_iterator = None
        self.samples, = self._load_samples(data_path)
        self.samples_yielded = 0

