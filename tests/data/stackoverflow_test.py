"""
Tests for the StackOverflow dataset
"""
import json

import pytest
from transformers import AutoTokenizer
from src.data import stackoverflow



@pytest.mark.parametrize('max_steps', [-1, 36])
def test_init(sample_parsed_so, max_steps):
    task = stackoverflow.StackOverflowTask(
        'test',
        str(sample_parsed_so),
        AutoTokenizer.from_pretrained('gpt2'),
        max_samples=1,
        seed=1,
        sequence_length=2,
        max_steps=max_steps
    )
    assert len(task.data) == 1
    result_seq = task.tokenizer.decode(task.data[0])
    assert result_seq == 'Title 2\nQuestion Body 2\nAnswer 6\nAnswer 8\nAnswer 9\nAnswer 7\nAnswer 10\nAnswer 11'
    expected_size = task.tokenizer(result_seq, add_special_tokens=False)['input_ids']
    expected_size = (len(expected_size) + 1) // 2
    assert len(task) == (expected_size if max_steps == -1 else max_steps)


@pytest.mark.parametrize("answer_sorting", ['accepted', 'ascending', 'descending'])
def test_map_to_standard_entries(sample_parsed_so, answer_sorting):
    sample = list(map(json.loads, sample_parsed_so.open('r')))[-1]
    task = stackoverflow.StackOverflowTask(
        'test',
        sample_parsed_so,
        AutoTokenizer.from_pretrained('gpt2'),
        max_samples=2,
        answer_sorting=answer_sorting,
        answers_per_sample=1,
    )

    result = task.get_text_from_sample(sample)

    expected = "Title 3\nQuestion Body 3\n"
    if answer_sorting == "ascending":
        expected += "Answer 16"
    elif answer_sorting == "descending":
        expected += "Answer 12"
    else:
        expected += "Answer 9"

    assert result == expected
