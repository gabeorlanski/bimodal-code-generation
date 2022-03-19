"""
Tests for the StackOverflow dataset
"""
import json
import re
from copy import deepcopy
from functools import partial

from jinja2 import BaseLoader, Environment, StrictUndefined
import pytest
from transformers import AutoTokenizer

from src.common import PROJECT_ROOT
from src.data import stackoverflow
from bs4 import BeautifulSoup

space_cleaner = re.compile(r'\s{2,}')


class TestStackOverflowProcessor:
    @pytest.mark.parametrize('repeat_prompt', [True, False], ids=['Repeat', 'Single'])
    @pytest.mark.parametrize('comment_type', ['NONE', "BLOCK", "LINE"])
    @pytest.mark.parametrize('wrap_answer', ['NONE', "BLOCK", "LINE"])
    @pytest.mark.parametrize('repeat_question', [True, False], ids=['RepeatQ', 'SingleQ'])
    @pytest.mark.parametrize('include_scores', [True, False], ids=['Score', 'NoScore'])
    @pytest.mark.parametrize('include_date', [True, False], ids=['Date', 'NoDate'])
    @pytest.mark.parametrize('include_tags', [True, False], ids=['Tags', 'NoTags'])
    @pytest.mark.parametrize('answer_sorting', ['accepted', 'ascending'])
    def test_call(
            self,
            repeat_prompt,
            comment_type,
            wrap_answer,
            repeat_question,
            include_date,
            include_scores,
            include_tags,
            answer_sorting
    ):

        sample = {
            "line" : 5991, "body": "<p>Body</p>", "type": 1, "id": "13454",
            "date" : "2008-08-17T01:23:50.067", "score": 13, "comment_count": 0,
            "tags" : ["python", "string", "escaping"], "title": "Title", "answer_count": 5,
            "views": 8027, "accepted_answer": None, "answers": {
                "13608"   : {
                    "line"     : 6083, "body": "<pre><code>Answer 1</code></pre>", "type": 2,
                    "id"       : "13608",
                    "date"     : "2008-08-17T12:55:25.100", "score": -1, "comment_count": 0,
                    "parent_id": "13454"
                }, "13456": {
                    "line"     : 5993, "body": "<p>Answer 2</p>", "type": 2, "id": "13456",
                    "date"     : "2008-08-17T01:26:52.043", "score": 0, "comment_count": 0,
                    "parent_id": "13454"
                }, "13598": {
                    "line"     : 6077, "body": "<p>Answer 3</p>", "type": 2, "id": "13598",
                    "date"     : "2008-08-17T12:15:13.170", "score": 13, "comment_count": 0,
                    "parent_id": "13454"
                }
            }
        }

        prompt = Environment().from_string(
            PROJECT_ROOT.joinpath("templates/so/base_question.txt").read_text())

        processor = stackoverflow.StackOverflowProcessor(
            prompt_file='templates/so/base_question.txt',
            repeat_prompt_each_answer=repeat_prompt,
            comment_type_for_question=comment_type,
            wrap_answer_character=wrap_answer,
            repeat_body_for_each_answer=repeat_question,
            include_date=include_date,
            include_tags=include_tags,
            include_question_score=include_scores,
            answer_sorting=answer_sorting
        )

        result = processor.__call__(sample)

        prompt_kwargs = {
            "title"         : "Title",
            "question_score": "13" if include_scores else None,
            "question"      : "Body",
            "tags"          : "python,string,escaping" if include_tags else None,
            "comment_type"  : comment_type,
            "question_date" : "2008-08-17" if include_date else None

        }

        quality_strs = [
            ("GREAT", 13),
            ("OK", 0),
            ("BAD", -1)
        ]
        if answer_sorting == 'ascending':
            quality_strs = reversed(quality_strs)

        expected_inputs = []
        if repeat_prompt:
            for i, v in enumerate(quality_strs):
                if i > 0 and not repeat_question:
                    prompt_kwargs['question'] = None
                prompt_kwargs['quality'] = v[0]
                prompt_kwargs['answer_score'] = v[1]
                expected_inputs.append(prompt.render(**prompt_kwargs).strip())
        else:
            prompt_kwargs['quality'] = 'ACCEPTED'
            expected_inputs.append(prompt.render(**prompt_kwargs).strip())

        expected_answers = [
            "Answer 3",
            "Answer 2",
            "Answer 1"
        ]
        if wrap_answer == "LINE":
            expected_answers = [
                "# Answer 3",
                "# Answer 2",
                "Answer 1"
            ]
        elif wrap_answer == 'BLOCK':
            expected_answers = [
                '"""\nAnswer 3\n"""',
                '"""\nAnswer 2\n"""',
                'Answer 1',
            ]

        if answer_sorting == 'ascending':
            expected_answers = list(reversed(expected_answers))
        if not repeat_prompt:
            expected_answers = ['\n'.join(expected_answers)]
        assert len(result) == len(expected_inputs)
        for i, (actual, expected_input, expected_answer) in enumerate(
                zip(result, expected_inputs, expected_answers)):
            assert actual['input'] == expected_input, i
            assert actual['labels'] == expected_answer, i

    @pytest.mark.parametrize('remove_modality', ["NONE", "CODE", "NL"])
    def test_clean(self, remove_modality):
        processor = stackoverflow.StackOverflowProcessor(
            'templates/so/base_question.txt',
            remove_modality=remove_modality
        )

        input_text = "<p>NL <code>Inline Code</code></p><p>NL 2</p><pre><code>CodeBlock</code></pre>"
        result = processor.clean_html_body(input_text)
        result = ''.join(map(repr, result))
        if remove_modality == "NONE":
            assert result == "<p>NL Inline Code\nNL 2</p><code>CodeBlock</code>"
        elif remove_modality == "CODE":
            assert result == "<p>NL Inline Code\nNL 2</p>"
        else:
            assert result == "<code>CodeBlock</code>"
