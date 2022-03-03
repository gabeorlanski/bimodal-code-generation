"""
Tests for the parse so data scripts
"""
import json
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.data.parse_so.filtering import *


@pytest.mark.parametrize('buffer_size', [1, 10000])
def test_consolidate_so_data(
        tmpdir,
        parsed_arduino_dump,
        arduino_questions,
        buffer_size
):
    questions_ignore = {
        'json'       : ['65744'],
        'web-service': ['57', '36'],
        'array'      : []
    }

    tmpdir_path = Path(tmpdir)
    filter_path = tmpdir_path.joinpath('test.json')
    filter_dict = {
        k: [qid for qid in v if qid not in questions_ignore[k]]
        for k, v in arduino_questions.items()
    }
    with filter_path.open('w') as f:
        json.dump(
            filter_dict,
            f
        )

    questions = [qid for t in filter_dict.values() for qid in t]
    val_questions = [questions[i] for i in [0, 2, 4]]
    expected_train = {}
    expected_val = {}
    for k, v in filter_dict.items():
        for qid, question in arduino_questions[k].items():
            if qid not in v:
                continue
            question_keep = {
                k: v for k, v in question.items() if k not in ['title', 'answers', 'body']
            }
            if qid in val_questions:
                expected_val[qid] = question_keep
            else:
                expected_train[qid] = question_keep

    with patch('src.data.parse_so.filtering.np.random.default_rng') as rng_mock:
        rng_mock.return_value = MagicMock()
        rng_mock.return_value.choice = lambda *args, **kwargs: np.array([0, 2, 4])
        rng_mock.return_value.shuffle = lambda elements: elements
        consolidate_so_data(
            name='TEST',
            filter_file=str(filter_path.absolute()),
            dump_path=str(parsed_arduino_dump.absolute()),
            max_buffer_size=buffer_size,
            seed=1,
            debug=True,
            output_path=str(tmpdir_path.absolute()),
            max_val_size=1000
        )
    train_path = tmpdir_path.joinpath('TEST.jsonl')
    val_path = tmpdir_path.joinpath('TEST_val.jsonl')

    train_data = {d['id']: {
        k: v for k, v in d.items() if k not in ['title', 'answers', 'body']
    } for d in map(json.loads, train_path.open())}
    val_data = {d['id']: {
        k: v for k, v in d.items() if k not in ['title', 'answers', 'body']
    } for d in map(json.loads, val_path.open())}

    assert train_data == expected_train
    assert val_data == expected_val
