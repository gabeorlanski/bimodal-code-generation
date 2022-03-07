"""
Tests for the StackOverflow dataset
"""
import json

import pytest
from transformers import AutoTokenizer
from src.data import stackoverflow


class TestStackOverflowProcessor:

    @pytest.mark.parametrize('repeat_mode', ['title', 'full', 'none'],
                             ids=['Title', 'Full', 'None'])
    @pytest.mark.parametrize('answer_prompt', [True, False], ids=['APrompt', 'NoAPrompt'])
    @pytest.mark.parametrize('question_prompt', [True, False], ids=['QPrompt', 'NoQPrompt'])
    def test_make_instances_from_question(
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
            answer_prompt_template = "__QUALITY__\n__ANSWER__"
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
            title_prompt = "Title:__TITLE__"
            question_prompt_template = "Body:__BODY__"
            expected_title_str = "Title:Title"
            expected_question_str = f"{expected_title_str}\nBody:Body"
        else:
            question_prompt_template = None
            title_prompt = None

            expected_title_str = "Title"
            expected_question_str = f"{expected_title_str}\nBody"

        processor = stackoverflow.StackOverflowProcessor(
            answer_prompt=answer_prompt_template,
            question_prompt=question_prompt_template,
            repeat_question_for_each_answer=repeat_mode,
            title_prompt=title_prompt
        )

        result = processor.make_instances_from_question(sample)
        if repeat_mode == "full":
            expected = [
                {'input': expected_question_str, 'labels': expected_answer_strs[0]},
                {'input': expected_question_str, 'labels': expected_answer_strs[1]},
                {'input': expected_question_str, 'labels': expected_answer_strs[2]}
            ]
        elif repeat_mode == 'title':
            expected = [
                {'input': expected_question_str, 'labels': expected_answer_strs[0]},
                {'input': expected_title_str, 'labels': expected_answer_strs[1]},
                {'input': expected_title_str, 'labels': expected_answer_strs[2]}
            ]
        else:
            expected = [{
                "input": expected_question_str, 'labels': '\n'.join(expected_answer_strs)
            }]

        assert result == expected

    @pytest.mark.parametrize("answer_sorting", ['accepted', 'ascending', 'descending'])
    def test_answer_sorting(self, sample_parsed_so, answer_sorting):
        sample = list(map(json.loads, sample_parsed_so.open('r')))[-1]
        processor = stackoverflow.StackOverflowProcessor(
            answer_sorting=answer_sorting,
            answers_per_sample=1,
        )

        result = processor.make_instances_from_question(sample)

        expected_input = "Title 3\nQuestion Body 3"
        if answer_sorting == "ascending":
            expected_answer = "Answer 16"
        elif answer_sorting == "descending":
            expected_answer = "Answer 12"
        else:
            expected_answer = "Answer 9"

        assert len(result) == 1
        assert result[0]['input'] == expected_input
        assert result[0]['labels'] == expected_answer

    def test_call(self):
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
        processor = stackoverflow.StackOverflowProcessor(
            repeat_question_for_each_answer='full'
        )

        result = processor([sample])
        expected_answer_strs = [
            "Answer 3",
            "Answer 2",
            "Answer 1"
        ]
        assert len(result['inputs']) == len(result['labels']) == 3
        for i, v in enumerate(expected_answer_strs):
            assert result['inputs'][i] == 'Title\nBody'
            assert result['labels'][i] == v

    @pytest.mark.parametrize('remove_modality', [None, "CODE", "NL"])
    @pytest.mark.parametrize('clean', [True, False], ids=['Clean', 'Dirty'])
    def test_clean(self, remove_modality, clean):
        processor = stackoverflow.StackOverflowProcessor(
            clean=clean,
            remove_modality=remove_modality
        )

        input_text = "<p>NL <code> Inline Code</code></p>\n<pre><code>CodeBlock</code></pre>"
        result = processor.clean_html_body(input_text)
        if not clean:
            if remove_modality is None:
                assert result == input_text
            elif remove_modality == 'CODE':
                assert result == "<p>NL <code> Inline Code</code></p>"
            else:
                assert result == "<pre><code>CodeBlock</code></pre>"
        else:
            if remove_modality is None:
                assert result == "NL  Inline Code\nCodeBlock"
            elif remove_modality == "CODE":
                assert result == "NL  Inline Code"
            else:
                assert result == "CodeBlock"
