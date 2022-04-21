import json

import pytest
from src.common import FIXTURES_ROOT, PROJECT_ROOT
from src.data import npv


def test_parse_mbpp():
    file_path = FIXTURES_ROOT.joinpath('npv', 'MBPP.jsonl')

    lines = list(map(json.loads, file_path.open()))

    result = npv.parse_mbpp(file_path)
    assert len(result) == 3

    expected_io_pairs = [
        [
            {
                'input' : ['12'],
                'output': '7',
                'ops': '=='
            },
            {
                'input' : ['105'],
                'output': '15',
                'ops': '=='
            },
            {
                'input' : ['2'],
                'output': '2',
                'ops': '=='
            }
        ],
        [
            {
                'input' : ['(5, 6, (5, 6), 7, (8, 9), 9)'],
                'output': '{5: 2, 6: 2, 7: 1, 8: 1, 9: 2}',
                'ops': '=='
            },
            {
                'input' : ['(6, 7, (6, 7), 8, (9, 10), 10)'],
                'output': '{6: 2, 7: 2, 8: 1, 9: 1, 10: 2}',
                'ops': '=='
            },
            {
                'input' : ['(7, 8, (7, 8), 9, (10, 11), 11)'],
                'output': '{7: 2, 8: 2, 9: 1, 10: 1, 11: 2}',
                'ops': '=='
            }
        ],
        [
            {
                'input' : ["(5, 6, 7, 4, 9)", "'FDF'"],
                'output': "[5, 'FDF', 6, 'FDF', 7, 'FDF', 4, 'FDF', 9, 'FDF']",
                'ops': '=='
            },
            {
                'input' : ["(7, 8, 9, 10)", "'PF'"],
                'output': "[7, 'PF', 8, 'PF', 9, 'PF', 10, 'PF']",
                'ops': '=='
            },
            {
                'input' : ["(11, 14, 12, 1, 4)", "'JH'"],
                'output': "[11, 'JH', 14, 'JH', 12, 'JH', 1, 'JH', 4, 'JH']",
                'ops': '=='
            }
        ]
    ]

    line_to_func = [
        'find_Min_Sum',
        'count_element_freq',
        'add_str'
    ]

    for i, (actual, expected, line) in enumerate(zip(result, expected_io_pairs, lines)):
        assert actual == npv.serialize_instance_to_dict(
            task='MBPP',
            task_id=line['task_id'],
            description=line['text'].replace('\r', ''),
            programs={line_to_func[i]: line['code'].replace('\r', '')},
            input_output_pairs=expected,
            context=line['test_setup_code'].replace('\r', '')
        ), f"{i} failed"


def test_parse_human_eval():
    file_path = FIXTURES_ROOT.joinpath('npv', 'HUMAN_EVAL.jsonl')

    result = npv.parse_human_eval(file_path)
    assert len(result) == 1

    expected_io_pairs = [{
        'input' : ['[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]', '0.3'],
        'ops': '==',
        'output': 'True'
    },
        {
            'input' : ['[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]', '0.05'],
            'ops': '==',
            'output': 'False'
        },
        {
            'input' : ['[1.0, 2.0, 5.9, 4.0, 5.0]', '0.95'],
            'ops': '==',
            'output': 'True'
        },
        {
            'input' : ['[1.0, 2.0, 5.9, 4.0, 5.0]', '0.8'],
            'ops': '==',
            'output': 'False'
        },
        {
            'input' : ['[1.0, 2.0, 3.0, 4.0, 5.0, 2.0]', '0.1'],
            'ops': '==',
            'output': 'True'
        },
        {
            'input' : ['[1.1, 2.2, 3.1, 4.1, 5.1]', '1.0'],
            'ops': '==',
            'output': 'True'
        },
        {
            'input' : ['[1.1, 2.2, 3.1, 4.1, 5.1]', '0.5'],
            'ops': '==',
            'output': 'False'
        }
    ]

    assert result[0] == {
        'task'              : 'HUMAN_EVAL',
        'task_id'           : 'HumanEval/0',
        'description'       : 'Check if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue',
        'programs'          : {
            'has_close_elements': "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False"
        },
        'input_output_pairs': expected_io_pairs,
        'context'           : 'from typing import List'
    }
