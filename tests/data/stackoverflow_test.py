"""
Tests for the StackOverflow dataset
"""
import re
from unittest.mock import MagicMock

import pytest
from src.common import PROJECT_ROOT
from src.data import stackoverflow

space_cleaner = re.compile(r'\s{2,}')


class TestStackOverflowProcessor:
    @pytest.mark.parametrize('repeat_prompt', [True, False], ids=['Repeat', 'Single'])
    @pytest.mark.parametrize('wrap_answer', ['NONE', "BLOCK", "LINE"])
    @pytest.mark.parametrize('repeat_question', [True, False], ids=['RepeatQ', 'SingleQ'])
    @pytest.mark.parametrize('answer_sorting', ['accepted', 'ascending'])
    @pytest.mark.parametrize('date_fmt', ['%Y', '%m'], ids=['Year', 'Month'])
    @pytest.mark.parametrize('best_question', ['Highest', 'Lowest', 'Accepted'])
    def test_call(
            self,
            repeat_prompt,
            wrap_answer,
            repeat_question,
            answer_sorting,
            date_fmt,
            best_question
    ):

        sample = {
            "line" : 5991, "body": "<p>Body</p>", "type": 1, "id": "13454",
            "date" : "2008-08-17T01:23:50.067", "score": 13, "comment_count": 0,
            "tags" : ["python", "string", "escaping"], "title": "Title", "answer_count": 5,
            "views": 8027, "accepted_answer": "13456", "answers": {
                "13608"   : {
                    "line"     : 6083, "body": "<pre><code>Answer 1</code></pre>", "type": 2,
                    "id"       : "13608",
                    "date"     : "2009-11-17T12:55:25.100", "score": -1, "comment_count": 0,
                    "parent_id": "13454"
                }, "13456": {
                    "line"     : 5993, "body": "<p>Answer 2</p>", "type": 2, "id": "13456",
                    "date"     : "2009-11-17T01:26:52.043", "score": 0, "comment_count": 0,
                    "parent_id": "13454"
                }, "13598": {
                    "line"     : 6077, "body": "<p>Answer 3</p>", "type": 2, "id": "13598",
                    "date"     : "2009-11-17T12:15:13.170", "score": 10, "comment_count": 0,
                    "parent_id": "13454"
                }
            }
        }

        highest_is_best = False
        worst_is_best = False
        if best_question == 'Highest':
            highest_is_best = True
            worst_is_best = False
        elif best_question == 'Lowest':
            highest_is_best = False
            worst_is_best = True

        prompt_fn = MagicMock()
        if repeat_prompt:
            prompt_fn.side_effect = lambda f: (
                f"{f['question_date']} {f['tags']} "
                f"{f['answer_date']} {f['quality']} "
                f"{f['input_sequence']} {f['context']}")
        else:
            prompt_fn.side_effect = lambda f: (
                f"{f['question_date']} {f['tags']} "
                f"{f['input_sequence']} {f['context']}")
        processor = stackoverflow.StackOverflowProcessor(
            prompt_fn=prompt_fn,
            repeat_prompt_each_answer=repeat_prompt,
            wrap_answer_character=wrap_answer,
            repeat_body_for_each_answer=repeat_question,
            answer_sorting=answer_sorting,
            date_format_str=date_fmt,
            highest_is_best=highest_is_best,
            worst_is_best=worst_is_best
        )

        result = processor(sample)
        prompt_kwargs = {
            "input_sequence": "Title",
            "question_score": "13",
            "context"      : "Body",
            "tags"          : "python,string,escaping",
            "question_date" : "2008" if date_fmt == '%Y' else '08',
            "answer_date"   : "2009" if date_fmt == '%Y' else '11'

        }

        if highest_is_best:
            quality_strs = [
                ("OK", 0),
                ("BEST", 10),
                ("BAD", -1)
            ]
        elif worst_is_best:
            quality_strs = [
                ("OK", 0),
                ("GOOD", 10),
                ("BEST", -1)
            ]
        else:
            quality_strs = [
                ("BEST", 0),
                ("GOOD", 10),
                ("BAD", -1)
            ]
        if answer_sorting == 'ascending':
            quality_strs = list(sorted(quality_strs, key=lambda l: l[1]))

        expected_inputs = []
        if repeat_prompt:
            for i, v in enumerate(quality_strs):
                if i>0 and not repeat_question:
                    prompt_kwargs['context'] = None
                prompt_kwargs['quality'] = v[0]
                prompt_kwargs['answer_score'] = v[1]
                expected_inputs.append(
                    prompt_fn.side_effect(prompt_kwargs).strip()
                )
        else:
            prompt_kwargs['quality'] = 'BEST'
            expected_inputs.append(
                prompt_fn.side_effect(prompt_kwargs).strip()
            )

        expected_answers = [
            "Answer 2",
            "Answer 3",
            "Answer 1"
        ]
        if wrap_answer == "LINE":
            expected_answers = [
                "# Answer 2",
                "# Answer 3",
                "Answer 1"
            ]
        elif wrap_answer == 'BLOCK':
            expected_answers = [
                '"""\nAnswer 2\n"""',
                '"""\nAnswer 3\n"""',
                'Answer 1',
            ]

        # if date_fmt == '%Y' and

        if answer_sorting == 'ascending':
            expected_answers = [
                expected_answers[2],
                expected_answers[0],
                expected_answers[1]
            ]
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
            lambda f: f,
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

    def test_relative_quality(self):
        sample = {
            "line" : 5991, "body": "<p>Body</p>", "type": 1, "id": "13454",
            "date" : "2008-08-17T01:23:50.067", "score": 13, "comment_count": 0,
            "tags" : ["python", "string", "escaping"], "title": "Title", "answer_count": 5,
            "views": 8027, "accepted_answer": "13456", "answers": {
                "13608"   : {
                    "line"     : 6083, "body": "<pre><code>Answer 1</code></pre>", "type": 2,
                    "id"       : "13608",
                    "date"     : "2009-11-17T12:55:25.100", "score": -1, "comment_count": 0,
                    "parent_id": "13454"
                }, "13456": {
                    "line"     : 5993, "body": "<p>Answer 2</p>", "type": 2, "id": "13456",
                    "date"     : "2009-11-17T01:26:52.043", "score": 0, "comment_count": 0,
                    "parent_id": "13454"
                }, "13598": {
                    "line"     : 6077, "body": "<p>Answer 3</p>", "type": 2, "id": "13598",
                    "date"     : "2009-11-17T12:15:13.170", "score": 10, "comment_count": 0,
                    "parent_id": "13454"
                }
            }
        }

        prompt_fn = MagicMock()
        prompt_fn.side_effect = lambda f: (
            f"{f['quality']} {f['input_sequence']} {f['context']}"
        )
        processor = stackoverflow.StackOverflowProcessor(
            prompt_fn=prompt_fn,
            relative_quality=True,
            repeat_prompt_each_answer=True,
            repeat_body_for_each_answer=True
        )
        expected_qualities = [
            'BEST',
            '2ND',
            '3RD'
        ]

        expected_inputs = []
        prompt_kwargs = {
            "input_sequence": "Title",
            "question_score": "13",
            "context"       : "Body",
            "tags"          : "python,string,escaping",
        }
        for q in expected_qualities:
            prompt_kwargs['quality'] = q
            expected_inputs.append(prompt_fn.side_effect(prompt_kwargs))

        expected_answers = [
            "Answer 2",
            "Answer 3",
            "Answer 1"
        ]
        result = processor(sample)
        assert len(result) == len(expected_inputs)
        for i, (actual, expected_input, expected_answer) in enumerate(
                zip(result, expected_inputs, expected_answers)):
            assert actual['input'] == expected_input, i
            assert actual['labels'] == expected_answer, i
