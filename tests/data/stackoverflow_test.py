"""
Tests for the Language Modeling datasets
"""
import pytest
from transformers import AutoTokenizer
from datasets import Dataset

from src.data import langauge_modeling

from src.common import FIXTURES_ROOT, PROJECT_ROOT


@pytest.fixture()
def example_dataset(tokenizer):
    # Reference for the data:
    # https://www.youtube.com/watch?v=6GN20jud6MI
    example_data = [
        "Are you thinking what I am thinking?",
        "I don't know...",
        "Were you thinking holy **** holy **** a swordfish almost went through my head?",
        "If so, yes."
    ]
    example_tokenized = tokenizer(example_data, add_special_tokens=False)['input_ids']

    seq_length = 20
    expected_examples, rem = divmod(
        sum(map(lambda t: len(t) + 1, example_tokenized)),
        seq_length
    )
    expected_examples += rem > 0
    yield example_data, example_tokenized, seq_length, expected_examples


class TestConstantLengthDataset:
    @pytest.mark.parametrize('infinite', [True, False], ids=['Infinite', 'Finite'])
    def test_iter(self, tokenizer, example_dataset, infinite):
        ds, ds_tokenized, seq_length, expected_examples = example_dataset
        dataset = langauge_modeling.LanguageModelingDataset(
            tokenizer,
            [{'input_sequence': t} for t in ds],
            infinite=infinite,
            seq_length=seq_length,
            num_of_sequences=4,
            chars_per_token=3.5
        )

        # Go through the steps we know will not trigger the stop iteration
        result_tokens = []
        result_attention_mask = []
        ds_iter = iter(dataset)
        for _ in range(expected_examples):
            result = next(ds_iter)
            result_tokens.extend(result['input_ids'].tolist())
            result_attention_mask.extend(result['attention_mask'].tolist())

        expected_tokens = []
        expected_attention_mask = []
        for t in ds_tokenized:
            expected_tokens.extend(t + [dataset.concat_token_id])
            expected_attention_mask.extend([1] * (len(t) + 1))

        pad_amount = expected_examples * seq_length - len(expected_tokens)
        if infinite:
            spill = expected_tokens[:pad_amount]
            expected_tokens.extend(spill)
            expected_attention_mask.extend([1] * pad_amount)

        else:
            expected_tokens.extend([dataset.concat_token_id] * pad_amount)
            expected_attention_mask.extend([0] * pad_amount)
        assert len(expected_attention_mask) == 60
        assert len(expected_tokens) == 60

        assert result_tokens == expected_tokens
        assert result_attention_mask == expected_attention_mask

    @pytest.mark.parametrize('streaming', [True, False], ids=["Streaming", "No Streaming"])
    def test_initialize(self, example_dataset, tokenizer, streaming):
        ds, ds_tokenized, seq_length, expected_examples = example_dataset
        dataset = langauge_modeling.LanguageModelingDataset(
            tokenizer,
            [{'input_sequence': t} for t in ds],
            infinite=True,
            seq_length=seq_length,
            num_of_sequences=4,
            chars_per_token=3.5,
            streaming=streaming,
            batches_per_epoch=5
        )
        number_yielded = 0
        for _ in dataset:
            number_yielded += 1

        assert number_yielded == len(dataset)
