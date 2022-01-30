import logging
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch

from src.common import FIXTURES_ROOT
from scripts.parse_so_data import process_file


def test_parse_so(tmpdir):
    tag_filters = ['terminology']
    out_path = Path(tmpdir).joinpath('so_dumps')
    out_path.mkdir(parents=True)
    logger = logging.getLogger()
    actual_dump_stats = process_file(
        logger,
        FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        2,
        out_path,
        tag_filters,
        True
    )

    expected_files = {
        'questions.jsonl'    : {
            '1' : {
                'line'           : 2,
                'body'           : 'Body1',
                'id'             : '1',
                'date'           : '2016-08-02T15:39:14.947',
                'score'          : 10,
                'comment_count'  : 0,
                'tags'           : ['neural-networks', 'backpropagation', 'terminology',
                                    'definitions'],
                'title'          : 'What is "backprop"?',
                'answer_count'   : 5,
                'views'          : 625,
                'accepted_answer': '3',
                'type'           : 1,
            },
            '10': {
                'line'           : 8,
                'body'           : 'Body7',
                'id'             : '10',
                'date'           : '2016-08-02T15:47:56.593',
                'score'          : 48,
                'comment_count'  : 0,
                'tags'           : ['deep-neural-networks', 'terminology', 'fuzzy-logic'],
                'title'          : 'What is fuzzy logic?',
                'answer_count'   : 6,
                'views'          : 2302,
                'accepted_answer': '32',
                'type'           : 1,
            },
        },
        'answers.jsonl'      : {
            '3': {
                'line'         : 4,
                'body'         : 'Body3',
                'id'           : '3',
                'date'         : '2016-08-02T15:40:24.820',
                'score'        : 15,
                'comment_count': 0,
                'parent_id'    : '1',
                'type'         : 2,
            }
        },
        'wiki.jsonl'         : {},
        'wiki_excerpts.jsonl': {},
    }
    for file_name, expected in expected_files.items():
        assert out_path.joinpath(file_name).exists(), file_name

        actual = {
            d['id']: d for d in
            map(json.loads, out_path.joinpath(file_name).read_text().splitlines(False))
        }
        assert actual == expected

    valid_questions = out_path.joinpath('valid_questions.txt').read_text().splitlines(False)
    assert sorted(valid_questions) == ["1", '10']
    file_dump_stats = json.load(out_path.joinpath('stats.json').open())
    assert file_dump_stats == actual_dump_stats
    assert file_dump_stats == {
        "post_types"      : {
            "questions"    : 2,
            "answers"      : 1,
            "wiki_excerpts": 0,
            "wiki"         : 0,
        },
        "failures"        : {
            "PARSE_FAIL"  : 3,
            "FILTERED_OUT": 3
        },
        "orphaned_answers": 1,
        "tags"            : {
            "neural-networks"     : 1,
            "deep-neural-networks": 1,
            "backpropagation"     : 1,
            "terminology"         : 2,
            "definitions"         : 1,
            "fuzzy-logic"         : 1,
        }
    }
