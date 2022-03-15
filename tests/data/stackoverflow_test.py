"""
Tests for the StackOverflow dataset
"""
import json
import re
from copy import deepcopy
from functools import partial

import pytest
from transformers import AutoTokenizer
from src.data import stackoverflow
from bs4 import BeautifulSoup

space_cleaner = re.compile(r'\s{2,}')


class TestStackOverflowProcessor:

    @pytest.mark.parametrize('repeat_mode', ['title', 'full', 'none'],
                             ids=['Title', 'Full', 'None'])
    @pytest.mark.parametrize('quality_prompt', [True, False], ids=['QPrompt', 'NoQPrompt'])
    @pytest.mark.parametrize('title_prompt', [True, False], ids=['TPrompt', 'NoTPrompt'])
    @pytest.mark.parametrize('wrap', [True, False], ids=['Wrap', 'NoWrap'])
    def test_make_instances_from_question(
            self,
            repeat_mode,
            quality_prompt,
            title_prompt,
            wrap
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

        if quality_prompt:
            quality_prompt = "__QUALITY__"
            expected_quality_strs = [
                '\nGreat',
                '\nOk',
                '\nBad'
            ]

        else:
            quality_prompt = None
            expected_quality_strs = [''] * 3
        if title_prompt:
            title_prompt = "Title:__TITLE__"
            expected_title_str = "Title:Title"
            expected_question_str = f'{expected_title_str}\nBody'

        else:
            title_prompt = None

            expected_title_str = "Title"
            expected_question_str = f"{expected_title_str}\nBody"

        if wrap:
            q_wrap = 'BLOCK'
            a_wrap = 'LINE'
            expected_title_str = f'"""\n{expected_title_str}'
            expected_question_str = f'"""\n{expected_question_str}'
            if repeat_mode == 'none':
                if quality_prompt:
                    expected_answer_strs = [
                        '# Great\n# Answer 3',
                        '# Ok\n# Answer 2',
                        '# Bad\nAnswer 1'
                    ]
                else:

                    expected_answer_strs = [
                        '# Answer 3',
                        '# Answer 2',
                        'Answer 1'
                    ]
                expected_inputs = expected_question_str + '\n"""'
            else:
                expected_answer_strs = [
                    '# Answer 3',
                    '# Answer 2',
                    'Answer 1'
                ]
                expected_inputs = [
                    f'{expected_title_str if i > 0 and repeat_mode == "title" else expected_question_str}' \
                    f'{expected_quality_strs[i]}\n"""'
                    for i, v in enumerate(expected_quality_strs)
                ]
        else:

            q_wrap = None
            a_wrap = None
            if repeat_mode == 'none':
                if quality_prompt:
                    expected_answer_strs = [
                        'Great\nAnswer 3',
                        'Ok\nAnswer 2',
                        'Bad\nAnswer 1'
                    ]
                else:
                    expected_answer_strs = [
                        'Answer 3',
                        'Answer 2',
                        'Answer 1'
                    ]
                expected_inputs = expected_question_str
            else:
                expected_answer_strs = [
                    'Answer 3',
                    'Answer 2',
                    'Answer 1'
                ]
                expected_inputs = [
                    f"{expected_title_str if i > 0 and repeat_mode == 'title' else expected_question_str}" \
                    f"{expected_quality_strs[i]}"
                    for i, v in enumerate(expected_quality_strs)
                ]

        processor = stackoverflow.StackOverflowProcessor(
            quality_prompt=quality_prompt,
            repeat_question_for_each_answer=repeat_mode,
            title_prompt=title_prompt,
            wrap_question_character=q_wrap,
            wrap_answer_character=a_wrap
        )

        result = processor.make_instances_from_question(sample)
        if repeat_mode != "none":
            expected_samples = [
                {'input': a, 'labels': b} for a, b in
                zip(expected_inputs, expected_answer_strs)
            ]
        else:
            expected_samples = [{
                'input' : expected_inputs,
                'labels': '\n'.join(
                    expected_answer_strs
                )
            }]

        assert len(result) == len(expected_samples)
        for i, (actual, expected) in enumerate(zip(result, expected_samples)):
            assert actual['input'] == expected['input']
            assert actual['labels'] == expected['labels']

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

    @pytest.mark.parametrize('remove_modality', ["NONE", "CODE", "NL"])
    def test_clean(self, remove_modality):
        processor = stackoverflow.StackOverflowProcessor(
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

    @pytest.mark.parametrize('remove_modality', ['NONE', "CODE", "NL"],
                             ids=["Both", "OnlyNL", "OnlyCode"])
    @pytest.mark.parametrize('repeat_mode', ['title', 'full', 'none'],
                             ids=['Title', 'Full', 'None'])
    @pytest.mark.parametrize('force_include_question', [True, False], ids=['ForceQ', 'NoForceQ'])
    @pytest.mark.parametrize('force_include_title', [True, False], ids=['ForceT', 'NoForceT'])
    def test_remove_modality_repeats(
            self,
            remove_modality,
            repeat_mode,
            force_include_question,
            force_include_title
    ):
        processor = stackoverflow.StackOverflowProcessor(
            remove_modality=remove_modality,
            force_include_question=force_include_question,
            force_include_title=force_include_title,
            repeat_question_for_each_answer=repeat_mode
        )

        sample = {
            "answers"     : {
                "56537578"   : {
                    "line"     : 44432429,
                    "body"     : "<p>Answer 1.p1</p>\n\n<pre><code>Answer 1.c1</code></pre>\n\n<p> Answer 1.p2 <code>Answer 1.i1</code></p>",
                    "result"   : "PASS", "type": 2, "id": "56537578",
                    "date"     : "2019-06-11T06:16:49.870", "score": 0, "comment_count": 0,
                    "parent_id": "56537551"
                }, "56537596": {
                    "line"     : 44432442,
                    "body"     : "<p> Answer 2.p1 <code>Answer 2.i1</code></p>\n\n<pre><code>Answer 2.c1</code></pre>",
                    "result"   : "PASS", "type": 2, "id": "56537596",
                    "date"     : "2019-06-11T06:18:25.427", "score": -1, "comment_count": 0,
                    "parent_id": "56537551"
                }
            }, "line"     : 44432410,
            "body"        : "<p>Question Paragraph</p>\n\n<pre><code>Question Code</code></pre>\n",
            "result"      : "PASS", "type": 1, "id": "56537551", "date": "2019-06-11T06:14:53.697",
            "score"       : -1, "comment_count": 1, "tags": ["python", "python-2.7"],
            "title"       : "Title",
            "answer_count": 2, "views": 32, "accepted_answer": None
        }

        bodies_with_removed = {
            1: {
                "CODE": "Answer 1.p1 Answer 1.p2 Answer 1.i1",
                "NL"  : "Answer 1.c1",
                "NONE": "Answer 1.p1 Answer 1.c1 Answer 1.p2 Answer 1.i1",
            },
            2: {
                "CODE": "Answer 2.p1 Answer 2.i1",
                "NL"  : "Answer 2.c1",
                "NONE": "Answer 2.p1 Answer 2.i1 Answer 2.c1",
            },
            0: {
                "CODE": "Question Paragraph",
                "NL"  : "Question Code",
                "NONE": "Question Paragraph Question Code"
            }
        }

        result = processor.make_instances_from_question(sample)
        for i in range(len(result)):
            result[i]['input'] = space_cleaner.sub(' ', result[i]['input'].replace('\n', ' '))
            result[i]['labels'] = space_cleaner.sub(' ', result[i]['labels'].replace('\n', ' '))

        expected_input = ""
        if force_include_title or remove_modality != 'NL':
            expected_input = sample['title'] + " "

        if force_include_question:
            expected_input += bodies_with_removed[0]['NONE']
        else:
            expected_input += bodies_with_removed[0][remove_modality]

        if repeat_mode == 'none':
            expected_target = bodies_with_removed[1][remove_modality]
            expected_target += " "
            expected_target += bodies_with_removed[2][remove_modality]
            assert len(result) == 1
            actual_input = result[0]['input']
            assert actual_input == expected_input

            actual_target = result[0]['labels']
            assert actual_target == expected_target
        else:
            assert len(result) == 2

            expected_repeat = ''
            if remove_modality != 'NL' or force_include_title:
                expected_repeat += sample['title']

            if repeat_mode == 'full':
                expected_repeat += " "
                if force_include_question or remove_modality == 'NONE':
                    expected_repeat += bodies_with_removed[0]["NONE"]
                else:
                    expected_repeat += bodies_with_removed[0][remove_modality]

            assert result[0]['input'] == expected_input
            assert result[0]['labels'] == bodies_with_removed[1][remove_modality]

            assert result[1]['input'] == expected_repeat.strip()
            assert result[1]['labels'] == bodies_with_removed[2][remove_modality]

    @pytest.mark.parametrize('remove_modality', ['NONE', "NL"])
    @pytest.mark.parametrize('force_include_tags', [True, False], ids=['Force', 'NoForce'])
    @pytest.mark.parametrize('repeat_mode', ['title', 'full'])
    def test_add_tags(self, remove_modality, force_include_tags, repeat_mode):
        processor = stackoverflow.StackOverflowProcessor(
            remove_modality=remove_modality,
            force_include_tags=force_include_tags,
            repeat_question_for_each_answer=repeat_mode,
            tags_prompt='__TAGS__'
        )
        soup = BeautifulSoup('', 'lxml')
        body = soup.new_tag('p')
        body.string = 'Body'
        processor_fn = partial(
            processor.apply_question_prompt,
            title="Title",
            score=1,
            views=1,
            tags=["TAG1", "TAG2"],
        )

        first_question = processor_fn(is_first_answer=True,body=deepcopy([body]))
        repeat_question = processor_fn(is_first_answer=False,body=deepcopy([body]))

        if remove_modality == 'NL':
            if force_include_tags:
                expected_first_question = 'TAG1 TAG2\nBody'
            else:
                expected_first_question = "Body"
            if repeat_mode == 'full':
                expected_second_question = expected_first_question
            else:
                expected_second_question = "TAG1 TAG2" if force_include_tags else ''
        else:
            expected_first_question = 'TAG1 TAG2\nTitle\nBody'
            if repeat_mode == 'full':
                expected_second_question = expected_first_question
            else:
                expected_second_question = "TAG1 TAG2\nTitle"

        assert first_question == expected_first_question
        assert repeat_question == expected_second_question

    @pytest.mark.parametrize('wrap', ['LINE', 'BLOCK', None])
    def test_no_body_title_repeat(self, wrap):
        processor = stackoverflow.StackOverflowProcessor(
            repeat_question_for_each_answer='title',
            remove_body_title_repeat=True,
            wrap_question_character=wrap
        )

        soup = BeautifulSoup('', 'lxml')
        body = soup.new_tag('p')
        body.string = 'Body'

        result = processor.apply_question_prompt(
            title="Title",
            body=[body],
            score=1,
            views=1,
            tags=["TAG1", "TAG2"],
            is_first_answer=True
        )
        if not wrap:
            assert result == 'Title'
        elif wrap=='BLOCK':
            assert result == f'"""\nTitle\n"""'
        else:
            assert result == f"# Title"

