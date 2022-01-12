"""
Tests for the MBPP dataset features
"""
import pytest
from transformers import AutoTokenizer
from datasets import Dataset

from tio.task import Task

from src.common import FIXTURES_ROOT, PROJECT_ROOT


@pytest.fixture(scope='module')
def mbpp_task() -> Task:
    yield Task.by_name('mbpp')(
        tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random"),
        preprocessors=[],
        postprocessors=[],
        metric_fns=[]
    )


@pytest.mark.parametrize('split', ['train', 'test', 'validation'])
def test_mbpp_task(mbpp_task, split):
    expected = Dataset.from_json(str(
        PROJECT_ROOT.joinpath('data', 'MBPP', f'{split}.jsonl')
    ))
    expected = expected.map(lambda ex: ex, remove_columns=[
        "challenge_test_list",
        "test_setup_code"
    ])

    result = mbpp_task.dataset_load_fn(split)
    assert result[:10] == expected[:10]


def test_mbpp_map_to_standard_entries(mbpp_task):
    sample = {
        "text"     : "A",
        "code"     : "B",
        "test_list": ["C", "D"]
    }
    result = mbpp_task.map_to_standard_entries(sample)
    assert result == {
        "input_sequence": "A\nC\nD",
        "target"        : "B",
        **sample
    }
