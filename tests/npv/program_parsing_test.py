import json
from pathlib import Path

from src.common import FIXTURES_ROOT
from src.npv import program_parsing


def test_parse_mbpp():
    file_path = FIXTURES_ROOT.joinpath('npv', 'MBPP.jsonl')

    lines = list(map(json.loads, file_path.open()))

    result, _ = program_parsing.parse_mbpp(file_path)
    assert len(result) == 3

    expected_io_pairs = [
        [
            {
                'input' : ['12'],
                'output': '7',
                'ops'   : '=='
            },
            {
                'input' : ['105'],
                'output': '15',
                'ops'   : '=='
            },
            {
                'input' : ['2'],
                'output': '2',
                'ops'   : '=='
            }
        ],
        [
            {
                'input' : ['(5, 6, (5, 6), 7, (8, 9), 9)'],
                'output': '{5: 2, 6: 2, 7: 1, 8: 1, 9: 2}',
                'ops'   : '=='
            },
            {
                'input' : ['(6, 7, (6, 7), 8, (9, 10), 10)'],
                'output': '{6: 2, 7: 2, 8: 1, 9: 1, 10: 2}',
                'ops'   : '=='
            },
            {
                'input' : ['(7, 8, (7, 8), 9, (10, 11), 11)'],
                'output': '{7: 2, 8: 2, 9: 1, 10: 1, 11: 2}',
                'ops'   : '=='
            }
        ],
        [
            {
                'input' : ["(5, 6, 7, 4, 9)", "'FDF'"],
                'output': "[5, 'FDF', 6, 'FDF', 7, 'FDF', 4, 'FDF', 9, 'FDF']",
                'ops'   : '=='
            },
            {
                'input' : ["(7, 8, 9, 10)", "'PF'"],
                'output': "[7, 'PF', 8, 'PF', 9, 'PF', 10, 'PF']",
                'ops'   : '=='
            },
            {
                'input' : ["(11, 14, 12, 1, 4)", "'JH'"],
                'output': "[11, 'JH', 14, 'JH', 12, 'JH', 1, 'JH', 4, 'JH']",
                'ops'   : '=='
            }
        ]
    ]

    line_to_func = [
        'find_Min_Sum',
        'count_element_freq',
        'add_str'
    ]

    for i, (actual, expected, line) in enumerate(zip(result, expected_io_pairs, lines)):
        expected = [
            {k: v if k != 'input' else f"{line_to_func[i]}({', '.join(v)})" for k, v in x.items()}
            for x in expected
        ]

        assert actual == program_parsing.serialize_instance_to_dict(
            func_name=line_to_func[i],
            source_file='MBPP.jsonl',
            task='MBPP',
            task_id=line['task_id'],
            description=line['text'].replace('\r', ''),
            program=line['code'].replace('\r', '').strip(),
            input_output_pairs=expected,
            context=line['test_setup_code'].replace('\r', '')
        ), f"{i} failed"


def test_parse_human_eval():
    file_path = FIXTURES_ROOT.joinpath('npv', 'HUMAN_EVAL.jsonl')

    result, _ = program_parsing.parse_human_eval(file_path)
    assert len(result) == 1

    io_pairs = [
        {
            'input' : ['[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]', '0.3'],
            'ops'   : '==',
            'output': 'True'
        },
        {
            'input' : ['[1.0, 2.0, 3.9, 4.0, 5.0, 2.2]', '0.05'],
            'ops'   : '==',
            'output': 'False'
        },
        {
            'input' : ['[1.0, 2.0, 5.9, 4.0, 5.0]', '0.95'],
            'ops'   : '==',
            'output': 'True'
        },
        {
            'input' : ['[1.0, 2.0, 5.9, 4.0, 5.0]', '0.8'],
            'ops'   : '==',
            'output': 'False'
        },
        {
            'input' : ['[1.0, 2.0, 3.0, 4.0, 5.0, 2.0]', '0.1'],
            'ops'   : '==',
            'output': 'True'
        },
        {
            'input' : ['[1.1, 2.2, 3.1, 4.1, 5.1]', '1.0'],
            'ops'   : '==',
            'output': 'True'
        },
        {
            'input' : ['[1.1, 2.2, 3.1, 4.1, 5.1]', '0.5'],
            'ops'   : '==',
            'output': 'False'
        }
    ]

    expected_io_pairs = []
    for p in io_pairs:
        p_dict = {}
        for k, v in p.items():
            if k == 'input':
                p_dict[k] = f"has_close_elements({', '.join(v)})"
            else:
                p_dict[k] = v
        expected_io_pairs.append(p_dict)

    assert result[0] == {
        'function'          : 'has_close_elements',
        'source_file'       : 'HUMAN_EVAL.jsonl',
        'task'              : 'HUMAN_EVAL',
        'task_id'           : 'HumanEval/0',
        'description'       : 'Check if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue',
        'code'              : "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False",
        'input_output_pairs': expected_io_pairs,
        'context'           : 'from typing import List'
    }


def test_special_mbpp(tmpdir):
    tmpdir_path = Path(tmpdir)
    with tmpdir_path.joinpath('test.jsonl').open('w') as f:
        f.write(json.dumps({
            "text"               : "Write a function to remove characters from the first string which are present in the second string.",
            "code"               : "NO_OF_CHARS = 256\r\ndef str_to_list(string): \r\n\ttemp = []",
            "task_id"            : 18,
            "test_setup_code"    : "",
            "test_list"          : [
                "assert remove_dirty_chars(\"probasscurve\", \"pros\") == 'bacuve'",
            ],
            "challenge_test_list": []
        }))

    result, _ = program_parsing.parse_mbpp(tmpdir_path.joinpath('test.jsonl'))
    assert len(result) == 1
    assert result[0] == {
        'function'          : 'remove_dirty_chars',
        'source_file'       : 'test.jsonl',
        'task'              : 'MBPP',
        'task_id'           : 18,
        'description'       : 'Write a function to remove characters from the first string which are present in the second string.',
        'code'              : "def str_to_list(string): \n\ttemp = []",
        'input_output_pairs': [
            {
                'input': "remove_dirty_chars('probasscurve', 'pros')", 'output': "'bacuve'",
                'ops'  : '=='
            }
        ],
        'context'           : 'NO_OF_CHARS = 256'
    }
