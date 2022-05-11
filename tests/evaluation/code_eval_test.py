from collections import Counter

import numpy as np

from src.evaluation.code_eval import (
    evaluate_code_from_file, estimate_pass_at_k, BASE_ERROR_TYPES, get_metrics_from_list
)


def test_evaluate_code(code_preds_dir):
    results = evaluate_code_from_file(
        'testing',
        code_preds_dir.joinpath('split_predictions.jsonl'),
        4,
        2,
        3.0,
    )
    expected_overview = {
        "all_invalid": 1,
        "all_valid"  : 1,
        "preds_total": 12,
        f"pass@1"    : estimate_pass_at_k([4, 4, 4], [1, 1, 0], 1).mean() * 100
    }
    runtime_error_pcts = [25, 50, 0.0]
    valid_syntax = [75, 100, 0]
    expected_outcomes = {
        "SyntaxError" : 5,
        "Failed_Tests": 2,
        "Correct"     : 2,
        "TypeError"   : 2,
        "KeyError"    : 1
    }
    expected_overview.update({
        'runtime_error_pct_mean': np.mean(runtime_error_pcts),
        "valid_syntax_pct_mean" : np.mean(valid_syntax),
        'correct_pct_ovr'       : expected_outcomes['Correct'] / 12 * 100,
        'syntax_error_pct_ovr'  : expected_outcomes['SyntaxError'] / 12 * 100,
        'failed_tests_pct_ovr'  : expected_outcomes['Failed_Tests'] / 12 * 100,
        'runtime_error_pct_ovr' : 3 / 12 * 100,
        'problems_correct_pct'  : 2 / 3 * 100,
    })

    for k in [5, 10, 25, 50, 80, 100]:
        expected_overview[f"pass@{k}"] = 0.0

    for e in BASE_ERROR_TYPES:
        if e not in expected_outcomes:
            expected_outcomes[e] = 0
    expected_outcomes_pct = {}
    for k, v in expected_outcomes.items():
        key = k.replace(' ', '_')
        expected_outcomes_pct[key] = v / 12 * 100

    assert results['overview'] == expected_overview

    # Again, small floats that we only care are present.
    for k in results['results_by_task_id']:
        v = results['results_by_task_id'][k]
        for p in v['pred_results']:
            assert isinstance(v['pred_results'][p].pop('time'), float)

    assert results['results_by_task_id'] == {
        '939'   : {
            'correct'    : 1,
            'total'      : 4,
            'error_types': Counter({'SyntaxError': 1, 'Failed Tests': 1, 'TypeError': 1}),
            'correct_pct': 25.0, 'runtime_error_pct': 25.0, 'pred_results': {
                0   : {'result': 'SyntaxError', 'is_failure': True},
                1   : {'is_failure': False, 'result': 'Failed Tests'}, 2: {
                    'is_failure': True,
                    'result'    : "TypeError: unsupported operand type(s) for -: 'int' and 'str'"
                }, 3: {'is_failure': False, 'result': 'Passed'}
            }
        }, '940': {
            'correct'    : 1, 'total': 4, 'error_types': Counter(
                {'KeyError': 1, 'TypeError': 1, 'Failed Tests': 1, 'SyntaxError': 0}),
            'correct_pct': 25.0, 'runtime_error_pct': 50.0, 'pred_results': {
                0: {'is_failure': True, 'result': 'KeyError: 4'},
                1: {'is_failure': True, 'result': "TypeError: 'int' object is not subscriptable"},
                2: {'is_failure': False, 'result': 'Failed Tests'},
                3: {'is_failure': False, 'result': 'Passed'}
            }
        }, '941': {
            'correct'       : 0, 'total': 4, 'error_types': Counter({'SyntaxError': 4}),
            'pred_results'  : {
                0: {'result': 'SyntaxError', 'is_failure': True},
                1: {'result': 'SyntaxError', 'is_failure': True},
                2: {'result': 'SyntaxError', 'is_failure': True},
                3: {'result': 'SyntaxError', 'is_failure': True}
            }, 'correct_pct': 0.0, 'runtime_error_pct': 0.0
        }
    }
    assert results['outcome_pcts'] == expected_outcomes_pct
