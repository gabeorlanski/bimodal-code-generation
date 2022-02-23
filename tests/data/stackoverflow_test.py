"""
Tests for the StackOverflow dataset
"""
import json

import pytest
from transformers import AutoTokenizer
from src.data import stackoverflow


#
# @pytest.mark.parametrize('max_steps', [-1, 36])
# def test_init(sample_parsed_so, max_steps):
#     task = stackoverflow.StackOverflowTask(
#         'test',
#         str(sample_parsed_so),
#         AutoTokenizer.from_pretrained('gpt2'),
#         max_samples=1,
#         seed=1,
#         sequence_length=2,
#         max_steps=max_steps
#     )
#     assert len(task.data) == 1
#     result_seq = task.tokenizer.decode(task.data[0])
#     assert result_seq == 'Title 2\nQuestion Body 2\nAnswer 6\nAnswer 8\nAnswer 9\nAnswer 7\nAnswer 10\nAnswer 11'
#     expected_size = task.tokenizer(result_seq, add_special_tokens=False)['input_ids']
#     expected_size = (len(expected_size) + 1) // 2
#     assert len(task) == (expected_size if max_steps == -1 else max_steps)


class TestStackOverflowProcessor:

    @pytest.mark.parametrize('repeat_mode', [None, 'title', 'full'],
                             ids=['NoRepeat', 'Title', 'Full'])
    @pytest.mark.parametrize('answer_prompt', [True, False], ids=['APrompt', 'NoAPrompt'])
    @pytest.mark.parametrize('question_prompt', [True, False], ids=['QPrompt', 'NoQPrompt'])
    def test_call(
            self,
            repeat_mode,
            answer_prompt,
            question_prompt
    ):
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
            expected_question_str = "Title:Title\nBody:Body\n"
        else:
            question_prompt_template = None

            expected_question_str = "Title\nBody\n"

        processor = stackoverflow.StackOverflowTextProcessor(
            answer_prompt=answer_prompt_template,
            question_prompt=question_prompt_template,
            repeat_question_for_each_answer=repeat_mode
        )

        result = processor(sample)

        if repeat_mode == "title":
            expected = [
                {'input': expected_question_str, 'target': expected_answer_strs[0]},
                {'input': "Title", 'target': expected_answer_strs[1]},
                {'input': "Title", 'target': expected_answer_strs[2]}
            ]
        elif repeat_mode == "full":
            expected = [
                {'input': expected_question_str, 'target': expected_answer_strs[0]},
                {'input': expected_question_str, 'target': expected_answer_strs[1]},
                {'input': expected_question_str, 'target': expected_answer_strs[2]}]
        else:
            answer = '\n'.join(expected_answer_strs)
            expected = [{'input': expected_question_str, 'target': answer}]

        assert result == expected

    @pytest.mark.parametrize("answer_sorting", ['accepted', 'ascending', 'descending'])
    def test_answer_sorting(self, sample_parsed_so, answer_sorting):
        sample = list(map(json.loads, sample_parsed_so.open('r')))[-1]
        processor = stackoverflow.StackOverflowTextProcessor(
            answer_sorting=answer_sorting,
            answers_per_sample=1,
        )

        result = processor(sample)

        expected_input = "Title 3\nQuestion Body 3\n"
        if answer_sorting == "ascending":
            expected_answer = "Answer 16"
        elif answer_sorting == "descending":
            expected_answer = "Answer 12"
        else:
            expected_answer = "Answer 9"

        assert len(result) == 1
        assert result[0]['input'] == expected_input
        assert result[0]['target'] == expected_answer
