import logging
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
__all__ = [
    "parse_eval_results_dir"
]


def parse_eval_results_dir(task, dir_path: Path):
    logger.debug(f"Parsing {dir_path} from {task}")
    logger.debug(f"Loading execution metrics from {dir_path}")

    execution_metrics = json.loads(dir_path.joinpath('execution_metrics.json').read_text())
    logger.debug(f"Loading the predictions from {dir_path.joinpath('test.jsonl')}")
    predictions = {str(d['task_id']): d for d in
                   map(json.loads, dir_path.joinpath('test.jsonl').open())}

    if 'test' not in execution_metrics:
        raise KeyError(f"'test' not in {dir_path.joinpath('execution_metrics.json')}")

    all_outcomes = set(execution_metrics['test']['outcome_pcts'])
    results_by_task_id = execution_metrics['test']['results_by_task_id']
    mean_tracker = defaultdict(list)

    preds_to_time_check = []

    # Make sure that every one of the keys are present
    task_result_counter = {k: 0 for k in [
        'no_correct',
        'all_correct',
        'all_runtime_error',
        'all_syntax_error',
        'all_failed_tests',
        'has_runtime_errors'
    ]}

    for tid, task_results in results_by_task_id.items():
        total_preds = task_results['total']
        if task_results['correct'] == 0:
            task_result_counter['no_correct'] += 1

        error_types = task_results['error_types']
        syntax_errors = error_types.get('SyntaxError', 0)
        failed_tests = error_types.get('Failed Tests', 0)
        if syntax_errors == total_preds:
            task_result_counter['all_syntax_error'] += 1
        elif failed_tests == total_preds:
            task_result_counter['all_failed_tests'] += 1
        elif task_results['correct'] == total_preds:
            task_result_counter['all_correct'] += 1

        mean_tracker['correct'].append(task_results['correct'])
        total_runtime_errors = 0
        for outcome in all_outcomes:
            if outcome == 'Correct':
                continue

            outcome_key = outcome.replace("_", '')
            outcome_count = error_types.get(outcome.replace('_', ' '), 0)
            if outcome_key not in ['TimedOut', 'SyntaxError', 'FailedTests']:
                total_runtime_errors += outcome_count
            mean_tracker[outcome_key].append(outcome_count)

        if total_runtime_errors == total_preds:
            task_result_counter['all_runtime_error'] += 1
        mean_tracker['TotalRuntimeErrors'].append(total_runtime_errors)
        if 'error_messages' in task_results and total_runtime_errors > 0:
            task_result_counter['has_runtime_errors'] += 1
            num_unique_errors = len(set(task_results['error_messages'].values()))
            mean_tracker['UniqueErrors'].append(num_unique_errors)
            mean_tracker['PCT_UniqueErrors'].append(
                num_unique_errors / len(task_results['error_messages']) * 100
            )

        preds_to_get = (
                task_results.get('failed_tests', [])
                + task_results.get('passed', [])
                + task_results.get('timed_out', [])
        )
        if preds_to_get:

            preds_for_task = predictions[tid]
            task_info = {
                k: preds_for_task[k] for k in
                ['task_id', 'idx', 'tests']
            }
            task_info['test_setup_code'] = preds_for_task.get('test_setup_code', '')
            for pred_idx in preds_to_get:
                preds_to_time_check.append(
                    {'prediction': preds_for_task['prediction'][pred_idx], **task_info}
                )

    out = {}
    for k, v in task_result_counter.items():
        out[f"{k}_pct"] = v / len(results_by_task_id) * 100
        out[f"{k}_total"] = v

    for k, v in mean_tracker.items():
        out[f"{k}_mean"] = np.mean(v)
        out[f"{k}_std"] = np.std(v)

    return out, preds_to_time_check
