import json
from pathlib import Path

from src.common import FIXTURES_ROOT
from src.npv import instance_creation


def test_make_instances_with_negation():
    instance = {
        'source_file'       : 'mbpp_train.jsonl', 'task': 'MBPP', 'task_id': 667,
        'description'       : 'Test',
        'code'              : 'Test',
        'input_output_pairs': [
            {'input': "Check_Vow('corner', 'AaEeIiOoUu')", 'output': '2', 'ops': '=='},
            {'input': "Check_Vow('aeo', 'AaEeIiOoUu')", 'output': '3', 'ops': '=='},
        ],
        'context'           : '',
        'instance_idx'      : 656,
        'test_negations'    : [],
        'exclude_tests'     : []
    }
    result = instance_creation.make_samples_from_dict(instance, True)
    assert result == [{
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_0',
        'input'        : "Check_Vow('corner', 'AaEeIiOoUu')", 'output': '2',
        'op'           : '==', 'is_original': True, 'result': 'True',
        'is_manual_fix': False, 'is_negation_of': None
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_1',
        'input'        : "Check_Vow('corner', 'AaEeIiOoUu')", 'output': '2',
        'op'           : '!=', 'is_original': True, 'result': 'False',
        'is_manual_fix': False, 'is_negation_of': 'MBPP_656_0'
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_2',
        'input'        : "Check_Vow('aeo', 'AaEeIiOoUu')", 'output': '3',
        'op'           : '==', 'is_original': True, 'result': 'True',
        'is_manual_fix': False, 'is_negation_of': None
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_3',
        'input'        : "Check_Vow('aeo', 'AaEeIiOoUu')", 'output': '3',
        'op'           : '!=', 'is_original': True, 'result': 'False',
        'is_manual_fix': False, 'is_negation_of': 'MBPP_656_2'
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_4',
        'input'        : "Check_Vow('corner', 'AaEeIiOoUu')", 'output': '3',
        'op'           : '==', 'is_original': False, 'result': 'False',
        'is_manual_fix': False, 'is_negation_of': None
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_5',
        'input'        : "Check_Vow('corner', 'AaEeIiOoUu')", 'output': '3',
        'op'           : '!=', 'is_original': False, 'result': 'True',
        'is_manual_fix': False, 'is_negation_of': 'MBPP_656_4'
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_6',
        'input'        : "Check_Vow('aeo', 'AaEeIiOoUu')", 'output': '2',
        'op'           : '==', 'is_original': False, 'result': 'False',
        'is_manual_fix': False, 'is_negation_of': None
    }, {
        'source_file'  : 'mbpp_train.jsonl', 'task': 'MBPP',
        'description'  : 'Test', 'code': 'Test', 'context': '',
        'instance_idx' : 656, 'original_task_id': 667, 'task_id': 'MBPP_656_7',
        'input'        : "Check_Vow('aeo', 'AaEeIiOoUu')", 'output': '2',
        'op'           : '!=', 'is_original': False, 'result': 'True',
        'is_manual_fix': False, 'is_negation_of': 'MBPP_656_6'
    }]
