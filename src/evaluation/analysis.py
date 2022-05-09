import contextlib
import io
import multiprocessing
import signal

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import ast

from src.evaluation.execute import TimeoutException, create_tempdir, time_limit

logger = logging.getLogger(__name__)
__all__ = [
    "parse_eval_results_dir",
    "execute_time_check"
]


def parse_eval_results_dir(task, dir_path: Path):
    import warnings
    warnings.filterwarnings("ignore")
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

        preds_for_task = predictions[tid]
        task_info = {
            k: preds_for_task[k] for k in
            ['task_id', 'idx', 'tests']
        }
        task_info['test_setup_code'] = preds_for_task.get('test_setup_code', '')

        # I miscalculated the idx for the predictions that passed. So I need to
        # remove those with bad syntax prior.
        predictions_w_valid_syntax = []
        for p in preds_for_task['prediction']:
            try:
                ast.parse(p)
                predictions_w_valid_syntax.append(p)
            except (SyntaxError, MemoryError):
                continue

        for k in ['passed']:

            for pred_idx in task_results.get(k, []):
                preds_to_time_check.append(
                    {'prediction': predictions_w_valid_syntax[pred_idx], **task_info}
                )
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

    out = {}
    for k, v in task_result_counter.items():
        out[f"{k}_pct"] = v / len(results_by_task_id) * 100
        out[f"{k}_total"] = v

    for k, v in mean_tracker.items():
        out[f"{k}_mean"] = np.mean(v)
        out[f"{k}_std"] = np.std(v)

    return out, preds_to_time_check


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def get_runtime(args_list):
    run_info, check_program, timeout, task_id = args_list
    RESULT_DICT = {}
    had_error = False
    had_timeout = False
    with create_tempdir():

        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # reliability_guard()
        try:
            stdout_f = io.StringIO()
            stderr_f = io.StringIO()
            with time_limit(timeout):
                with contextlib.redirect_stdout(stdout_f):
                    with contextlib.redirect_stderr(stderr_f):
                        _locals = locals()
                        exec(check_program, globals(), _locals)
                        RESULT_DICT = _locals['RESULT_DICT']
        except TimeoutException:
            RESULT_DICT['TIME'] = timeout
            had_timeout = True
        except Exception as e:
            RESULT_DICT['TIME'] = str(e)
            had_error = True
    stdout_f.close()
    stderr_f.close()
    return dict(
        run_info=run_info,
        task_id=task_id,
        had_error=had_error,
        had_timeout=had_timeout,
        runtime=RESULT_DICT['TIME'],
    )


def execute_time_check(to_time_check, num_workers, timeit_number=100, timeout=3):
    logger.info(f"{len(to_time_check)} predictions to time check")
    logger.info(f"Running each program {timeit_number} time(s)")
    mp_args = []
    for sample in to_time_check:
        test_str = '\n'.join([sample['test_setup_code']] + sample['tests'])
        test_str = test_str.replace('assert', 'ASSERT_PLACEHOLDER=')
        task_id = sample['task_id']
        test_program = sample['prediction'] + "\n" + test_str

        # Wrap the test function with another function so that the
        # entire thing can be called from timeit.
        wrapped_func = []
        for line in test_program.split('\n'):
            wrapped_func.append(f'\t{line}')

        wrapped_func = '\n'.join(wrapped_func)
        wrapped_func = f"def TEST_CANDIDATE():\n{wrapped_func}"

        test_program = [
            "import timeit",
            wrapped_func,
            f"RESULT_DICT['TIME']=timeit.timeit(TEST_CANDIDATE,number={timeit_number})"
        ]

        mp_args.append((sample['run_info'], '\n'.join(test_program), timeout, task_id))

    results = defaultdict(lambda: defaultdict(list))
    with_errors = []
    with_timeout = 0
    with multiprocessing.Pool(num_workers) as pool:
        raw_results = list(tqdm(
            pool.imap_unordered(get_runtime, mp_args),
            total=len(to_time_check),
            desc='Getting Runtime')
        )

        for r in raw_results:
            if r['had_error']:
                with_errors.append((r['run_info'], r['task_id'], r['runtime']))
                continue
            if r['had_timeout']:
                with_timeout += 1
            results[r['run_info']][r['task_id']].append(r)
    logger.info(f"{len(with_errors)}/{len(to_time_check)} had errors")
    logger.info(f"{with_timeout}/{len(to_time_check)} timed out")
    return results, with_errors
