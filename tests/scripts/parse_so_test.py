import logging
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch

from src.common import FIXTURES_ROOT
from scripts.parse_so_data import process_file


def test_parse_so():
    tag_filters = ['terminology']
    logger = logging.getLogger()
    questions, failures = process_file(
        logger,
        FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        2,
        tag_filters,
        True
    )
    expected_questions = {
        '1' : {
            'line'           : 2,
            'body'           : 'Body1',
            'id'             : '1',
            'date'           : '2016-08-02T15:39:14.947',
            'score'          : 10,
            'comment_count'  : 0,
            'tags'           : ['neural-networks', 'backpropagation', 'terminology', 'definitions'],
            'title'          : 'What is "backprop"?',
            'answer_count'   : 5,
            'views'          : 625,
            'accepted_answer': '3',
            'answers'        : {
                '3': {
                    'line' : 4, 'body': 'Body3', 'id': '3', 'date': '2016-08-02T15:40:24.820',
                    'score': 15, 'comment_count': 0, 'parent_id': '1'
                }
            }
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
            'answers'        : {}
        }
    }
    expected_failures = {
        "PARSE_FAIL"  : [0, 1, 9],
        "NO_VALID_TAG": [3, 5, 6]
    }
    failures = {k: list(sorted(v)) for k, v in failures.items()}
    assert questions == expected_questions
    assert failures == expected_failures
