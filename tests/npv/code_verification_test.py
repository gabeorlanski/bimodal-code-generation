from collections import Counter

from src.npv import code_verifcation


def test_verify_code():
    samples = [
        {
            'source_file'  : 'mbpp.jsonl', 'function': 'min_cost', 'task': 'MBPP',
            'description'  : 'Test',
            'code'         : 'import re\ndef testing(inputs):'
                             '\n    re.compile(r\'\')\n    return inputs % 2 == 0',
            'context'      : 'R = 3\nC = 3', 'instance_idx': 0, 'original_task_id': 1,
            'task_id'      : 'MBPP_0_0',
            'input'        : 'testing(1)',
            'output'       : 'False', 'op': '==', 'is_original': True, 'result': 'True',
            'is_manual_fix': False, 'is_negation_of': None
        },
        {
            'source_file'  : 'mbpp.jsonl', 'function': 'min_cost', 'task': 'MBPP',
            'description'  : 'Test',
            'code'         : 'def testing(inputs):\n    return inputs % 2 == 0',
            'context'      : 'R = 3\nC = 3', 'instance_idx': 0, 'original_task_id': 1,
            'task_id'      : 'MBPP_0_1',
            'input'        : 'testing(1)',
            'output'       : 'True', 'op': '==', 'is_original': True, 'result': 'False',
            'is_manual_fix': False, 'is_negation_of': 'MBPP_0_0'
        },
        {
            'source_file'  : 'mbpp.jsonl', 'function': 'min_cost', 'task': 'MBPP',
            'description'  : 'Test',
            'code'         : 'def bad_test(inputs):\n    return inputs % 2 == 0',
            'context'      : 'R = 3\nC = 3', 'instance_idx': 0, 'original_task_id': 1,
            'task_id'      : 'MBPP_0_2',
            'input'        : 'testing(6)',
            'output'       : 'False', 'op': '==', 'is_original': True, 'result': 'False',
            'is_manual_fix': False, 'is_negation_of': None
        },
        {
            'source_file'  : 'mbpp.jsonl', 'function': 'min_cost', 'task': 'MBPP',
            'description'  : 'Test',
            'code'         : 'def testing(inputs):\n    return inputs % 2 == 0',
            'context'      : 'R = 3\nC = 3', 'instance_idx': 0, 'original_task_id': 1,
            'task_id'      : 'MBPP_0_3',
            'input'        : 'testing(4)',
            'output'       : 'True', 'op': '==', 'is_original': True, 'result': 'False',
            'is_manual_fix': False, 'is_negation_of': 'MBPP_0_2'
        }
    ]

    rtr = code_verifcation.check_io_sample_executes_correctly(
        'test', samples, 1
    )
    rtr_values, results = rtr
    assert rtr_values == {
        0: {
            'MBPP_0_0': {'type': 'bool', 'value': 'False'},
            'MBPP_0_1': {'type': 'bool', 'value': 'False'},
            'MBPP_0_3': {'type': 'bool', 'value': 'True'}
        }
    }
    assert results == {
        'failed_counts': Counter({
            0: 2
        }),
        'passed_counts': Counter({
            0: 2
        }),
        "failed_tests" : {
            0: {
                "MBPP_0_3": "testing(4) True"
            }
        },
        "had_errors"   : {
            0: {
                "MBPP_0_2": "testing(6) False"
            }
        }
    }
