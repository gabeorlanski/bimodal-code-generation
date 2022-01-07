"""
Tests for the MBPP dataset features
"""
from transformers import AutoTokenizer
from datasets import Dataset

from src.common import FIXTURES_ROOT
from src.data import mbpp


class TestMBPPReader:
    def test_read(self):
        mbpp_task = mbpp.MBPP(None)  # Type:ignore

        expected_mbpp = Dataset.from_json(
            str(FIXTURES_ROOT.joinpath("MBPP", "mbpp.jsonl"))
        )

        actual = mbpp_task._load_dataset(FIXTURES_ROOT.joinpath("MBPP", "mbpp.jsonl"))
        assert actual.to_dict() == expected_mbpp.to_dict()

    def test_preprocess(self):
        tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
        mbpp_task = mbpp.MBPP(tokenizer)

        preprocessed, tokenized = mbpp_task.read_data(
            FIXTURES_ROOT.joinpath("MBPP", "mbpp.jsonl")
        )
        for example, example_tok in zip(preprocessed, tokenized):
            expected_input_sequence = (
                example["text"] + "\n" + "\n".join(example["test_list"])
            )
            expected_target = example["code"]
            assert example["input_sequence"] == expected_input_sequence
            assert example["target"] == expected_target
            assert (
                example_tok["input_ids"]
                == tokenizer(expected_input_sequence)["input_ids"]
            )
            assert example_tok["labels"] == tokenizer(expected_target)["input_ids"]
