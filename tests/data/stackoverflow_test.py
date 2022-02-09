"""
Tests for the StackOverflow dataset
"""
import json

import pytest
from transformers import AutoTokenizer
from datasets import Dataset
from unittest.mock import MagicMock

from src.data import stackoverflow

from src.common import FIXTURES_ROOT, PROJECT_ROOT


def test_load_samples(sample_parsed_so):
    task = stackoverflow.StackOverflowTask(
        'test',
        AutoTokenizer.from_pretrained('gpt2'),
        preprocessors=[],
        postprocessors=[],
        metric_fns=[],
        max_samples=2,
        max_val_samples=1,
        split_mapping={'train': sample_parsed_so}
    )
    task.get_samples_mask = MagicMock(return_value=[True, True, False])

    result = task._load_samples('train')
    assert len(result) == 2
    assert result['id'] == ["13454", "13941"]


@pytest.mark.parametrize("answer_sorting", ['accepted', 'ascending', 'descending'])
def test_map_to_standard_entries(sample_parsed_so, answer_sorting):
    sample = list(map(json.loads, sample_parsed_so.open('r')))[-1]
    task = stackoverflow.StackOverflowTask(
        'test',
        AutoTokenizer.from_pretrained('gpt2'),
        preprocessors=[],
        postprocessors=[],
        metric_fns=[],
        max_samples=2,
        max_val_samples=1,
        answer_sorting=answer_sorting,
        answers_per_sample=1,
        split_mapping={'train': sample_parsed_so}
    )

    sample['answers'] = list(sample['answers'].values())
    result = task.map_to_standard_entries(sample)

    expected = "Title 3\nQuestion Body 3\n"
    if answer_sorting == "ascending":
        expected += "Answer 16"
    elif answer_sorting == "descending":
        expected += "Answer 12"
    else:
        expected += "Answer 9"

    assert result['input_sequence'] == expected
