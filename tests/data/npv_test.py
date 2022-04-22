import contextlib
import io
import json
from pathlib import Path

import pytest

from transformers import AutoTokenizer
from datasets import Dataset, set_caching_enabled

from tio import Task
from src.common import FIXTURES_ROOT, PROJECT_ROOT
from src.evaluation.execute import create_tempdir
from src.data import npv


def test_parse_mbpp():
    file_path = FIXTURES_ROOT.joinpath('npv', 'MBPP.jsonl')

    lines = list(map(json.loads, file_path.open()))

    result, _ = npv.parse_mbpp(file_path)
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

        assert actual == npv.serialize_instance_to_dict(
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

    result, _ = npv.parse_human_eval(file_path)
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

    result, _ = npv.parse_mbpp(tmpdir_path.joinpath('test.jsonl'))
    assert len(result) == 1
    assert result[0] == {
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


@pytest.fixture(scope='module')
def npv_task() -> npv.NPV:
    out = Task.by_name('npv')(
        tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random"),
        preprocessors=[],
        postprocessors=[],
        metric_fns=[]
    )
    out.SPLIT_MAPPING = {
        'test': FIXTURES_ROOT.joinpath('npv', 'npv.jsonl')
    }
    out.excluded_columns_data = {}
    out.dataset = None
    out.excluded_columns_data = {}
    out._dataset_mapping = out.initialize_data()
    yield out


class TestNPVTask:
    def test_initialize(self, npv_task: npv.NPV):
        result = npv_task._dataset_mapping
        assert len(npv_task.excluded_columns_data) == 3
        assert len(result) == 1
        all_results = result['test']
        assert len(all_results) == 42
        result = all_results[:9]

        for k in ['description', 'code', 'context', 'idx']:
            assert all(v == result[k][0] for v in result[k]), k

        assert result['input'] == ['count_binary_seq(1)', 'count_binary_seq(1)',
                                   'count_binary_seq(1)', 'count_binary_seq(2)',
                                   'count_binary_seq(2)', 'count_binary_seq(2)',
                                   'count_binary_seq(3)', 'count_binary_seq(3)',
                                   'count_binary_seq(3)']
        assert result['op'] == ["=="] * 9
        assert result['output'] == ["2.0", "6.0", "20.0"] * 3
        assert result['result'] == ['True', 'False', 'False', 'False', 'True', 'False', 'False',
                                    'False', 'True']

        result = all_results[34:]

        assert result['input'] == ['is_equal_to_sum_even(4)', 'is_equal_to_sum_even(4)',
                                   'is_equal_to_sum_even(6)', 'is_equal_to_sum_even(6)',
                                   'is_equal_to_sum_even(8)', 'is_equal_to_sum_even(8)',
                                   'is_equal_to_sum_even(10)', 'is_equal_to_sum_even(10)']
        assert result['output'] == ['False', 'True', 'False', 'True', 'False', 'True', 'False',
                                    'True']
        assert result['result'] == ['True', 'False', 'True', 'False', 'False', 'True', 'False',
                                    'True']

    @staticmethod
    def assert_code_executes(code):
        with create_tempdir():
            try:
                stdout_f = io.StringIO()
                stderr_f = io.StringIO()
                with contextlib.redirect_stdout(stdout_f):
                    with contextlib.redirect_stderr(stderr_f):
                        # sys.stdout.write = lambda *args, **kwargs: None
                        exec(code, globals(), locals())
            except Exception as e:
                raise AssertionError(str(e))

    def test_all_execute(self):
        set_caching_enabled(False)
        npv_task = Task.by_name('npv')(
            tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random"),
            preprocessors=[],
            postprocessors=[],
            metric_fns=[]
        )
        npv_task.prompt = npv_task.JINJA_ENV.from_string(
            "def test_fn():{%- for line in context.split('\n') %}\n    {{line}}\n{%-endfor%}"
            "\n{% for line in code.split('\n') %}\n    {{line}}\n{%- endfor %}"
            "\n    assert ({{ test_stmt }}) == {{ target }}"
            "\ntest_fn()"
        )
        npv_task.include_target_in_prompt_kwargs = True

        ds = npv_task.preprocess('test', overwrite_cache=True)

        for i, c in enumerate(ds['input_sequence']):
            self.assert_code_executes(c)
