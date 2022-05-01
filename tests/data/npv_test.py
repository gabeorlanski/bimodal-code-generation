import random

import pytest
from transformers import AutoTokenizer
from datasets import Dataset
import json

from tio import Task
from src.data.npv import NPV
from src.common import FIXTURES_ROOT, PROJECT_ROOT


@pytest.fixture(scope='module')
def npv_task() -> NPV:
    NPV.SPLIT_MAPPING = {
        "test": str(FIXTURES_ROOT.joinpath('npv', 'sample_test.jsonl'))
    }
    out: NPV = Task.by_name('npv')(
        tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random"),
        preprocessors=[],
        postprocessors=[],
        metric_fns=[],
        false_ctx_examples=2,
        true_ctx_examples=2,
        ctx_pool_sorting_method='output_length'
    )

    yield out


@pytest.fixture()
def npv_sample_data():
    yield list(map(json.loads, FIXTURES_ROOT.joinpath('npv', 'sample_test.jsonl').open()))


@pytest.mark.parametrize('sorting_method',
                         ['random', 'output_length', 'input_length', 'total_length'])
def test_npv_mk_ctx_examples_list_sorting(npv_task, npv_sample_data, sorting_method):
    random.seed(1)
    npv_task.ctx_pool_sorting_method = sorting_method

    sort_fn = lambda idx, _tid: _tid
    if sorting_method == 'output_length':
        sort_fn = lambda idx, _tid: len(
            npv_sample_data[idx]['all_tasks'][_tid]['output'])  # type: ignore
    elif sorting_method == 'input_length':
        sort_fn = lambda idx, _tid: len(
            npv_sample_data[idx]['all_tasks'][_tid]['input'])  # type: ignore
    elif sorting_method == 'total_length':
        sort_fn = lambda idx, _tid: (
                len(npv_sample_data[idx]['all_tasks'][_tid]['input'])  # type: ignore
                + len(npv_sample_data[idx]['all_tasks'][_tid]['output'])  # type: ignore
        )
    for i in range(len(npv_sample_data)):
        true_examples, false_examples = npv_task.get_true_false_examples(npv_sample_data[i])
        if sorting_method == 'random':
            if i == 0:
                assert true_examples == [
                    'MBPP_14_4', 'MBPP_14_8', 'MBPP_14_49', 'MBPP_14_43', 'MBPP_14_47', 'MBPP_14_2',
                    'MBPP_14_45', 'MBPP_14_31', 'MBPP_14_35', 'MBPP_14_25', 'MBPP_14_27',
                    'MBPP_14_37', 'MBPP_14_0', 'MBPP_14_21', 'MBPP_14_11', 'MBPP_14_19',
                    'MBPP_14_29', 'MBPP_14_41', 'MBPP_14_33', 'MBPP_14_6', 'MBPP_14_15',
                    'MBPP_14_23', 'MBPP_14_13', 'MBPP_14_39', 'MBPP_14_17'
                ]
                assert false_examples == [
                    'MBPP_14_16', 'MBPP_14_24', 'MBPP_14_10', 'MBPP_14_34',
                    'MBPP_14_12', 'MBPP_14_32', 'MBPP_14_38', 'MBPP_14_26',
                    'MBPP_14_5', 'MBPP_14_3', 'MBPP_14_14', 'MBPP_14_22',
                    'MBPP_14_42', 'MBPP_14_7', 'MBPP_14_40', 'MBPP_14_20',
                    'MBPP_14_44', 'MBPP_14_30', 'MBPP_14_18', 'MBPP_14_28',
                    'MBPP_14_46', 'MBPP_14_36', 'MBPP_14_9', 'MBPP_14_48',
                    'MBPP_14_1'
                ]
            else:
                assert true_examples == [
                    'MBPP_0_7', 'MBPP_0_4', 'MBPP_0_15', 'MBPP_0_2',
                    'MBPP_0_17', 'MBPP_0_13', 'MBPP_0_0', 'MBPP_0_11',
                    'MBPP_0_9'
                ]
                assert false_examples == [
                    'MBPP_0_6', 'MBPP_0_1', 'MBPP_0_8', 'MBPP_0_12', 'MBPP_0_5', 'MBPP_0_3',
                    'MBPP_0_14', 'MBPP_0_16', 'MBPP_0_10'
                ]
        else:
            assert true_examples == list(sorted(
                true_examples, key=lambda tid: sort_fn(i, tid)
            ))
            assert false_examples == list(sorted(
                false_examples, key=lambda tid: sort_fn(i, tid)
            ))


@pytest.fixture()
def true_example_pool():
    yield [{
        'input'              : "split_lowerstring('AbCd')", 'op': '!=', 'output': '[]',
        'is_manual_fix'      : False, 'is_negation_of': 'MBPP_14_14', 'is_original': False,
        'task_id'            : 'MBPP_14_15', 'result': 'True', 'is_input_generated': False,
        'is_output_generated': True
    }, {
        'input'              : "split_lowerstring('0')", 'op': '==', 'output': '[]',
        'is_manual_fix'      : False, 'is_negation_of': None, 'is_original': True,
        'task_id'            : 'MBPP_14_6', 'result': 'True', 'is_input_generated': True,
        'is_output_generated': True
    }, {
        'input'             : "split_lowerstring('AbCd')", 'op': '==',
        'output'            : "['bC', 'd']", 'is_manual_fix': False, 'is_negation_of': None,
        'is_original'       : True, 'task_id': 'MBPP_14_0', 'result': 'True',
        'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'              : "split_lowerstring('Python')", 'op': '!=',
        'output'             : "['bC', 'd']", 'is_manual_fix': False,
        'is_negation_of'     : 'MBPP_14_18', 'is_original': False, 'task_id': 'MBPP_14_19',
        'result'             : 'True', 'is_input_generated': False,
        'is_output_generated': False
    }, {
        'input'         : "split_lowerstring('Programming')", 'op': '!=',
        'output'        : "['bC', 'd']", 'is_manual_fix': False,
        'is_negation_of': 'MBPP_14_26', 'is_original': False, 'task_id': 'MBPP_14_27',
        'result'        : 'True', 'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'         : "split_lowerstring('Python')", 'op': '==',
        'output'        : "['y', 't', 'h', 'o', 'n']", 'is_manual_fix': False,
        'is_negation_of': None, 'is_original': True, 'task_id': 'MBPP_14_2',
        'result'        : 'True', 'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'         : "split_lowerstring(u'überbür')", 'op': '==',
        'output'        : "['b', 'e', 'r', 'bü', 'r']", 'is_manual_fix': False,
        'is_negation_of': None, 'is_original': True, 'task_id': 'MBPP_14_8',
        'result'        : 'True', 'is_input_generated': True, 'is_output_generated': True
    }]


@pytest.fixture()
def false_example_pool():
    yield [{
        'input'              : "split_lowerstring('AbCd')", 'op': '==', 'output': '[]',
        'is_manual_fix'      : False, 'is_negation_of': None, 'is_original': False,
        'task_id'            : 'MBPP_14_14', 'result': 'False', 'is_input_generated': False,
        'is_output_generated': True
    }, {
        'input'              : "split_lowerstring('0')", 'op': '!=', 'output': '[]',
        'is_manual_fix'      : False, 'is_negation_of': 'MBPP_14_6', 'is_original': True,
        'task_id'            : 'MBPP_14_7', 'result': 'False', 'is_input_generated': True,
        'is_output_generated': True
    }, {
        'input'             : "split_lowerstring('Python')", 'op': '==',
        'output'            : "['bC', 'd']", 'is_manual_fix': False, 'is_negation_of': None,
        'is_original'       : False, 'task_id': 'MBPP_14_18', 'result': 'False',
        'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'         : "split_lowerstring('Python')", 'op': '!=',
        'output'        : "['y', 't', 'h', 'o', 'n']", 'is_manual_fix': False,
        'is_negation_of': 'MBPP_14_2', 'is_original': True, 'task_id': 'MBPP_14_3',
        'result'        : 'False', 'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'             : "split_lowerstring('Programming')", 'op': '==',
        'output'            : "['bC', 'd']", 'is_manual_fix': False, 'is_negation_of': None,
        'is_original'       : False, 'task_id': 'MBPP_14_26', 'result': 'False',
        'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'         : "split_lowerstring('Programming')", 'op': '==',
        'output'        : "['y', 't', 'h', 'o', 'n']", 'is_manual_fix': False,
        'is_negation_of': None, 'is_original': False, 'task_id': 'MBPP_14_28',
        'result'        : 'False', 'is_input_generated': False, 'is_output_generated': False
    }, {
        'input'         : "split_lowerstring('0')", 'op': '==',
        'output'        : "['y', 't', 'h', 'o', 'n']", 'is_manual_fix': False,
        'is_negation_of': None, 'is_original': False, 'task_id': 'MBPP_14_36',
        'result'        : 'False', 'is_input_generated': True, 'is_output_generated': False
    }]


@pytest.mark.parametrize('neg_priority', [True, False], ids=['Neg', 'AvoidNeg'])
@pytest.mark.parametrize('gen_output_priority', [True, False], ids=['GenO', 'AvoidGenO'])
def test_npv_get_ctx_examples(
        npv_task,
        npv_sample_data,
        false_example_pool,
        true_example_pool,
        neg_priority,
        gen_output_priority
):
    npv_task.allow_negated_ctx = neg_priority
    npv_task.allow_generated_ctx = gen_output_priority
    input_example = {
        'input'             : "split_lowerstring('AbCd')", 'op': '==',
        'output'            : "['bC', 'd']", 'is_manual_fix': False, 'is_negation_of': None,
        'is_original'       : True, 'task_id': 'MBPP_14_0', 'result': 'True',
        'is_input_generated': False, 'is_output_generated': False
    }

    npv_task.num_true_ctx_pairs = npv_task.num_false_ctx_pairs = 2
    results = list(npv_task.get_ctx_examples_from_pool(
        input_example,
        true_example_pool,
        false_example_pool
    ))
    result = [d['task_id'] for d in results]

    if not neg_priority and not gen_output_priority:
        assert result == ['MBPP_14_6', 'MBPP_14_18', 'MBPP_14_19', 'MBPP_14_28']
    elif not neg_priority:
        assert result == ['MBPP_14_6', 'MBPP_14_18', 'MBPP_14_8', 'MBPP_14_28']
    elif not gen_output_priority:
        assert result == ['MBPP_14_6', 'MBPP_14_18', 'MBPP_14_19', 'MBPP_14_28']
    else:
        assert result == ['MBPP_14_6', 'MBPP_14_18', 'MBPP_14_8', 'MBPP_14_28']
