import logging
from collections import Counter
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch

from src.common import FIXTURES_ROOT
from src.data.parse_so.dump_parsing import initial_parse_dump, parse_so_dump


def test_initial_pass(tmpdir):
    question_overview_data, tag_counts, dump_stats = initial_parse_dump(
        FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        Path(tmpdir),
        False
    )

    assert question_overview_data == {
        "1"    : {
            "tags" : ["neural-networks", "backpropagation", "terminology", "definitions"],
            "score": 10, "views": 625, "answer_count": 5, "accepted_answer": "3"
        },
        "2"    : {
            "tags" : ["neural-networks", "machine-learning", "statistical-ai", "generalization"],
            "score": 14, "views": 801, "answer_count": 3, "accepted_answer": "9"
        },
        "7"    : {
            "tags" : ["agi", "superintelligence", "singularity", "ai-safety", "ai-takeover"],
            "score": 10, "views": 544, "answer_count": 6, "accepted_answer": None,
        }, "10": {
            "tags" : ["deep-neural-networks", "terminology", "fuzzy-logic"], "score": 48,
            "views": 2302, "answer_count": 6, "accepted_answer": "32",
        },
    }
    assert tag_counts == Counter([
        "agi", "superintelligence", "singularity", "ai-safety", "ai-takeover",
        "deep-neural-networks", "terminology", "fuzzy-logic",
        "neural-networks", "backpropagation", "terminology", "definitions",
        "neural-networks", "machine-learning", "statistical-ai", "generalization"
    ])

    assert dump_stats == {
        'failures'  : Counter({'PARSE_FAIL': 3, 'NO_TITLE': 1}),
        'post_types': Counter({
            'questions'    : 4,
            'answers'      : 2,
            'wiki_excerpts': 1,
            'wiki'         : 1
        }),
        'tag_counts': Counter({
            'neural-networks'     : 2,
            'terminology'         : 2,
            'backpropagation'     : 1,
            'definitions'         : 1,
            'machine-learning'    : 1,
            'statistical-ai'      : 1,
            'generalization'      : 1,
            'agi'                 : 1,
            'superintelligence'   : 1,
            'singularity'         : 1,
            'ai-safety'           : 1,
            'ai-takeover'         : 1,
            'deep-neural-networks': 1,
            'fuzzy-logic'         : 1
        })
    }

    found_questions = [
        json.loads(d)['id'] for d in
        Path(tmpdir, 'questions.jsonl').read_text().splitlines()
    ]
    found_answers = [
        json.loads(d)['id'] for d in
        Path(tmpdir, 'answers.jsonl').read_text().splitlines()
    ]
    found_wiki = [
        json.loads(d)['id'] for d in
        Path(tmpdir, 'wiki.jsonl').read_text().splitlines()
    ]
    found_excerpts = [
        json.loads(d)['id'] for d in
        Path(tmpdir, 'wiki_excerpts.jsonl').read_text().splitlines()
    ]

    assert set(found_questions) == {"1", "2", "7", "10"}
    assert set(found_answers) == {"3", "9"}
    assert set(found_wiki) == {"12"}
    assert set(found_excerpts) == {"11"}


def test_parse_so_dump(tmpdir):
    parse_so_dump(
        posts_path=FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        num_workers=1,
        out_dir=Path(tmpdir),
        debug=False
    )

    questions_dir = Path(tmpdir, 'questions')
    expected_tags = {
        "backpropagation"     : {
            "1": {"3"}
        },
        "machine-learning"    : {
            "2": {"9"},
        },
        "agi"                 : {"7": set()},
        "deep-neural-networks": {"10": set()}
    }

    for tag_name, expected_ids in expected_tags.items():
        assert questions_dir.joinpath(f"{tag_name}.jsonl").exists()

        result = {
            d['id']: set(d['answers']) for d in
            map(json.loads, questions_dir.joinpath(f"{tag_name}.jsonl").open('r'))
        }
        assert result == expected_ids
