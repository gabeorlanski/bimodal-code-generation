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
def test_get_text_from_sample(sample_parsed_so, answer_sorting):
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


@pytest.mark.parametrize('repeat_mode', [None, 'title', 'full'], ids=['NoRepeat', 'Title', 'Full'])
@pytest.mark.parametrize('answer_prompt', [True, False], ids=['APrompt', 'NoAPrompt'])
@pytest.mark.parametrize('question_prompt', [True, False], ids=['QPrompt', 'NoQPrompt'])
@pytest.mark.parametrize('use_eos_token', [True, False], ids=['EOS', 'NewLine'])
def test_get_text_from_sample_repeat(sample_parsed_so, repeat_mode, answer_prompt, question_prompt,
                                     use_eos_token):
    sample = {
        "line" : 5991, "body": "Body", "type": 1, "id": "13454",
        "date" : "2008-08-17T01:23:50.067", "score": 13, "comment_count": 0,
        "tags" : ["python", "string", "escaping"], "title": "Title", "answer_count": 5,
        "views": 8027, "accepted_answer": None, "answers": {
            "13608"   : {
                "line"     : 6083, "body": "Answer 1", "type": 2, "id": "13608",
                "date"     : "2008-08-17T12:55:25.100", "score": -1, "comment_count": 0,
                "parent_id": "13454"
            }, "13456": {
                "line"     : 5993, "body": "Answer 2", "type": 2, "id": "13456",
                "date"     : "2008-08-17T01:26:52.043", "score": 0, "comment_count": 0,
                "parent_id": "13454"
            }, "13598": {
                "line"     : 6077, "body": "Answer 3", "type": 2, "id": "13598",
                "date"     : "2008-08-17T12:15:13.170", "score": 13, "comment_count": 0,
                "parent_id": "13454"
            }
        }
    }

    if answer_prompt:
        answer_prompt_template = "__QUALITY__"
        expected_answer_strs = [
            "good\nAnswer 3",
            "ok\nAnswer 2",
            "bad\nAnswer 1"
        ]
    else:
        answer_prompt_template = None
        expected_answer_strs = [
            "Answer 3",
            "Answer 2",
            "Answer 1"
        ]
    if question_prompt:
        question_prompt_template = "Title:__TITLE__\nBody:__BODY__"
        expected_question_str = "Title:Title\nBody:Body"
    else:
        question_prompt_template = None

        expected_question_str = "Title\nBody"

    task = stackoverflow.StackOverflowTask(
        'test',
        sample_parsed_so,
        AutoTokenizer.from_pretrained('gpt2'),
        max_samples=2,
        use_eos_token_when_repeat=use_eos_token,
        answer_prompt=answer_prompt_template,
        question_prompt=question_prompt_template,
        repeat_question_for_each_answer=repeat_mode
    )

    result = task.get_text_from_sample(sample)

    if repeat_mode == "title":
        expected = [
            expected_question_str + '\n' + expected_answer_strs[0],
            f"Title\n{expected_answer_strs[1]}",
            f"Title\n{expected_answer_strs[2]}",
        ]
    elif repeat_mode == "full":
        expected = [f"{expected_question_str}\n{a}" for a in expected_answer_strs]
    else:
        expected = [
            expected_question_str + '\n' + expected_answer_strs[0],
            expected_answer_strs[1],
            expected_answer_strs[2],
        ]

    if repeat_mode is not None and use_eos_token:
        expected = f"{task.tokenizer.eos_token}".join(expected)
    else:
        expected = '\n'.join(expected)

    assert result == expected
