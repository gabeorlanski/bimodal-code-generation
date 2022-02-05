"""
Tests for the MBPP dataset features
"""
import pytest
from transformers import AutoTokenizer
from datasets import Dataset

from tio import Task

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

    result = mbpp_task._load_samples(split)
    assert result[:10] == expected[:10]


@pytest.mark.parametrize('add_def', [True, False], ids=['add_def', 'no_def'])
def test_mbpp_map_to_standard_entries(mbpp_task, add_def):
    mbpp_task.add_def_to_prompt = add_def
    def_str = 'import math\ndef '
    sample = {
        "text"     : "A",
        "code"     : f"{def_str}B",
        "test_list": ["C", "D"]
    }
    result = mbpp_task.map_to_standard_entries(sample)
    expected_def_str = ('\r\n' if add_def else '') + def_str
    assert result == {
        "input_sequence": f"A\r\nC\r\nD{expected_def_str if add_def else ''}",
        "target"        : f"{def_str if not add_def else ''}B",
        **sample
    }
