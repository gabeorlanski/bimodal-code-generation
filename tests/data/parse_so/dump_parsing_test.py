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
    question_overview_data,_, tag_counts, dump_stats = initial_parse_dump(
        FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        Path(tmpdir),
        debug=False,
        max_buffer_size=1,
        tmp_dir=Path(tmpdir)
    )

    assert question_overview_data == {
        '1' : {
            'accepted_answer': '3',
            'answer_count'   : 5,
            'score'          : 10,
            'tag_to_use'     : 'neural-networks',
            'tags'           : ['neural-networks',
                                'backpropagation',
                                'terminology',
                                'definitions'],
            'views'          : 625
        },
        '10': {
            'accepted_answer': '32',
            'answer_count'   : 6,
            'score'          : 48,
            'tag_to_use'     : 'deep-neural-networks',
            'tags'           : ['deep-neural-networks', 'terminology', 'fuzzy-logic'],
            'views'          : 2302
        },
        '2' : {
            'accepted_answer': '9',
            'answer_count'   : 3,
            'score'          : 14,
            'tag_to_use'     : 'neural-networks',
            'tags'           : ['neural-networks',
                                'machine-learning',
                                'statistical-ai',
                                'generalization'],
            'views'          : 801
        },
        '7' : {
            'accepted_answer': None,
            'answer_count'   : 6,
            'score'          : 10,
            'tag_to_use'     : 'agi',
            'tags'           : ['agi',
                                'superintelligence',
                                'singularity',
                                'ai-safety',
                                'ai-takeover'],
            'views'          : 544
        }
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

    assert set(found_answers) == {"3", "9"}
    assert set(found_wiki) == {"12"}
    assert set(found_excerpts) == {"11"}


@pytest.mark.parametrize('buffer_size', [1000, 1])
def test_parse_so_dump(tmpdir, buffer_size):
    parse_so_dump(
        posts_path=FIXTURES_ROOT.joinpath('so_dumps', 'Posts.xml'),
        num_workers=1,
        out_dir=Path(tmpdir),
        debug=False,
        buffer_size=buffer_size
    )

    questions_dir = Path(tmpdir, 'questions')
    expected_tags = {
        "neural-networks" : {
            "1": {"3"},
            "2": {"9"},
        },
        "agi"           : {"7": set()},
        "deep-neural-networks": {"10": set()}
    }

    for tag_name, expected_ids in expected_tags.items():
        assert questions_dir.joinpath(f"{tag_name}.jsonl").exists()

        result = {
            d['id']: set(d['answers']) for d in
            map(json.loads, questions_dir.joinpath(f"{tag_name}.jsonl").open('r'))
        }
        assert result == expected_ids

    overview_data = json.load(Path(tmpdir, 'question_overview.json').open())

    assert overview_data == {
        '1' : {
            'accepted_answer': '3',
            'answer_count'   : 5,
            'score'          : 10,
            'tag_to_use'     : 'neural-networks',
            'tags'           : ['neural-networks',
                                'backpropagation',
                                'terminology',
                                'definitions'],
            'views'          : 625
        },
        '10': {
            'accepted_answer': '32',
            'answer_count'   : 6,
            'score'          : 48,
            'tag_to_use'     : 'deep-neural-networks',
            'tags'           : ['deep-neural-networks', 'terminology', 'fuzzy-logic'],
            'views'          : 2302
        },
        '2' : {
            'accepted_answer': '9',
            'answer_count'   : 3,
            'score'          : 14,
            'tag_to_use'     : 'neural-networks',
            'tags'           : ['neural-networks',
                                'machine-learning',
                                'statistical-ai',
                                'generalization'],
            'views'          : 801
        },
        '7' : {
            'accepted_answer': None,
            'answer_count'   : 6,
            'score'          : 10,
            'tag_to_use'     : 'agi',
            'tags'           : ['agi',
                                'superintelligence',
                                'singularity',
                                'ai-safety',
                                'ai-takeover'],
            'views'          : 544
        }
    }
