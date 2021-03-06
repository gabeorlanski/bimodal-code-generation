import itertools
import json
import multiprocessing
from copy import deepcopy
from typing import Union, List, Tuple, Dict
import logging
from pathlib import Path
import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluation.execute import check_correctness, unsafe_execute, time_limit, TimeoutException

logger = logging.getLogger(__name__)

Sample = namedtuple('Sample', ['idx', 'predictions', 'tests'])

BASE_ERROR_TYPES = [
    "TypeError", "StopIteration", "ImportError", "RuntimeError", "NameError",
    "AttributeError", "LookupError", "ValueError", "ArithmeticError",
    "ReferenceError", "BufferError"
]


def get_metrics_from_list(name, list_of_values):
    metrics = pd.Series(list_of_values).describe()
    metrics_dict = {
        "mean"      : metrics['mean'],
        "median"    : metrics['50%'],
        "quartile_1": metrics['25%'],
        "quartile_3": metrics['75%'],
        "std"       : metrics['std']
    }
    # Cast so it is not a numpy type
    return {f"{name + '/' if name else ''}{k}": float(v) for k, v in metrics_dict.items()}


def get_samples(code_items, samples_per_problem):
    failed = 0
    sample_num = 0
    total_valid_preds = []
    metrics = {
        "preds_total": 0,
    }
    all_invalid = []
    invalid_syntax = {}
    all_samples = {}
    pred_count = {}
    task_id_to_gold_idx = {}
    pred_results_by_task_idx = defaultdict(dict)
    for sample_dict in tqdm(code_items, desc='Reading Preds'):
        sample_num += 1

        if any(k not in sample_dict for k in ['prediction', 'tests']):
            logger.error(f"Sample {sample_num} is missing either 'prediction' or 'tests' keys")
            failed += 1
            continue
        idx = sample_dict['task_id']
        task_id_to_gold_idx[idx] = sample_dict['idx']
        if idx is None:
            logger.error(f"Sample {sample_num} is missing an idx key")
            failed += 1
            continue

        valid_predictions = []
        pred_count[idx] = len(sample_dict['prediction'])
        assert len(sample_dict['prediction']) == samples_per_problem
        for i, pred in enumerate(sample_dict['prediction']):
            metrics['preds_total'] += 1
            try:
                ast.parse(pred)
            except SyntaxError:
                pred_results_by_task_idx[idx][i] = {
                    'result'    : 'SyntaxError', 'time': -1.0,
                    'is_failure': True
                }
                continue
            except Exception as e:
                logger.error(f"Could not parse prediction {i} for {idx=} "
                             f"due to {type(e).__name__}:{str(e)}")
                pred_results_by_task_idx[idx][i] = {
                    'result'    : 'SyntaxError', 'time': -1.0,
                    'is_failure': True
                }
                continue
            valid_predictions.append((i, sample_dict['idx'], pred))

        invalid_syntax[idx] = len(sample_dict['prediction']) - len(valid_predictions)
        total_valid_preds.append(len(valid_predictions))
        if not valid_predictions:
            logger.info(f"Task {idx} had no valid syntax predictions")
            all_invalid.append(idx)
            continue
        all_samples[idx] = Sample(idx, valid_predictions, sample_dict['tests'])

    if failed > 0:
        logger.error(f"{failed}/{sample_num} had failures.")

    metrics["valid_syntax_pct_mean"] = np.mean(
        np.array(total_valid_preds) / samples_per_problem * 100
    )
    logger.info(f"{len(all_samples)} unique tasks")

    # Adding the / to some metrics to better separate them
    metrics['preds_total'] = metrics.pop('preds_total')
    metrics['all_invalid'] = len(all_invalid)
    metrics['all_valid'] = sum(map(lambda l: l == samples_per_problem, total_valid_preds))

    # Yes this is messy.
    return (
        list(all_samples.values()),
        (all_invalid, pred_count, invalid_syntax),
        metrics,
        (task_id_to_gold_idx, pred_results_by_task_idx)
    )


def evaluate_code(task, code_dicts, samples_per_problem, num_workers, timeout):
    samples, stats, overview_metrics, pred_info = get_samples(
        code_dicts,
        samples_per_problem
    )
    task_id_to_gold_idx, pred_results_by_task_idx = pred_info
    all_invalid, pred_count, invalid_syntax_by_idx = stats
    results = execute_code(
        task, samples, num_workers, timeout)

    results_by_task_id, global_error_tracker, metrics, counts = parse_results(
        results,
        pred_count,
        invalid_syntax_by_idx,
        all_invalid,
        samples_per_problem,
        task_id_to_gold_idx,
        pred_results_by_task_idx
    )
    correct, runtime_errors = counts
    overview_metrics.update(metrics)

    # Calculate the pass @ k metric across multiple k values.
    all_correct = np.array(correct)
    all_total = np.array([samples_per_problem] * len(results_by_task_id))
    total = int(sum(all_total))
    for k in [1, 5, 10, 25, 50, 80, 100]:
        if (all_total < k).all():
            overview_metrics[f"pass@{k}"] = 0.0
            continue
        overview_metrics[f"pass@{k}"] = estimate_pass_at_k(all_total, all_correct, k).mean() * 100

    # Calculate the % outcomes for different events.
    outcome_counts = {}
    outcome_pcts = {}
    global_error_tracker['Correct'] = int(sum(correct))
    for error_type, count in global_error_tracker.items():
        key = error_type.replace(" ", "_")
        outcome_counts[key] = count
        outcome_pcts[key] = count / total * 100

    logger.info("Metrics:")
    overview_metrics['problems_correct_pct'] = (all_correct > 0).sum() / all_correct.shape[0] * 100
    overview_metrics['runtime_error_pct_ovr'] = sum(
        v for k, v in outcome_counts.items()
        if k not in ["Correct", "Failed_Tests", "SyntaxError"]
    ) / total * 100

    overview_metrics['correct_pct_ovr'] = outcome_pcts.get('Correct', 0)
    overview_metrics['failed_tests_pct_ovr'] = outcome_pcts.get('Failed_Tests', 0)
    overview_metrics['syntax_error_pct_ovr'] = outcome_pcts.get('SyntaxError', 0)
    for k in sorted(overview_metrics.keys()):
        logger.info(f"\t{k:>32} = {overview_metrics[k]:0.3f}")

    result_metrics = {
        "overview"          : overview_metrics,

        # No int keys in json files, so make them match.
        "results_by_task_id": {str(k): v for k, v in results_by_task_id.items()},
        "outcome_pcts"      : outcome_pcts
    }

    return result_metrics


def evaluate_code_from_file(
        task,
        predictions_file: Union[str, Path, os.PathLike],
        samples_per_problem: int,
        num_workers: int,
        timeout: float = 3.0,
):
    predictions_file = Path(predictions_file)
    logger.info(
        f"Starting Code Evaluation with predictions in {predictions_file.resolve().absolute()}")

    if not predictions_file.exists():
        logger.error(f"{predictions_file.resolve().absolute()} is missing 'predictions.jsonl")
        raise FileExistsError(f"The predictions directory must have a predictions.jsonl")

    logger.info(f"Reading {predictions_file}")
    code_dicts = list(map(json.loads, predictions_file.read_text().splitlines(False)))

    return evaluate_code(task, code_dicts, samples_per_problem, num_workers, timeout)


def parse_results(
        execution_results,
        pred_count,
        invalid_syntax_by_idx,
        all_invalid,
        samples_per_problem,
        task_id_to_gold_idx,
        pred_results_by_task_idx
):
    global_error_tracker = Counter({k: 0 for k in BASE_ERROR_TYPES})
    results_by_task_id = {}

    correct, runtime_errors, runtimes = [], [], []

    for task_id, task_results in execution_results.items():

        task_results.sort()
        if pred_count[task_id] == 0:
            logger.error(f"Task {task_id} has no prediction count but has results")
            continue

        # Setup the dict for tracking metrics for a given task.
        task_runtimes_dict = defaultdict(list)
        task_metrics = {
            'correct'    : 0,
            'total'      : pred_count[task_id],
            'error_types': Counter({'SyntaxError': invalid_syntax_by_idx[task_id]}),

        }
        task_correct = task_runtime_errors = 0

        task_pred_results = pred_results_by_task_idx[task_id]
        assert len(task_pred_results) == invalid_syntax_by_idx[task_id]
        # Go through and calculate metrics on both a task and global level.
        for pred_idx, result_dict in task_results:

            assert task_id_to_gold_idx[result_dict['task_id']] == result_dict['gold_idx']
            assert task_id_to_gold_idx[task_id] == result_dict['gold_idx']
            assert result_dict['pred_idx'] == pred_idx
            assert result_dict['task_id'] == task_id
            assert pred_idx not in task_pred_results

            result_str = result_dict['result']

            task_runtimes_dict[result_str].append(result_dict['time'])

            task_metrics['correct'] += result_dict['passed']

            if not result_dict['passed']:
                assert not result_dict['passed']
                task_metrics['error_types'][result_str] += 1
                assert result_dict['result'] != 'Passed'

                if result_str != 'Timed Out' and result_str != 'Failed Tests':
                    task_runtime_errors += 1
                    task_pred_results[
                        result_dict['pred_idx']] = {
                        'is_failure': True,
                        'result'    : f"{result_str}: {result_dict['error']}",
                        'time'      : result_dict['time']
                    }
                elif result_str == 'Failed Tests':
                    task_pred_results[result_dict['pred_idx']] = {
                        'is_failure': False,
                        'result'    : result_str,
                        'time'      : result_dict['time']
                    }
                else:
                    task_pred_results[result_dict['pred_idx']] = {
                        'is_failure': False,
                        'result'    : result_str,
                        'time'      : result_dict['time']
                    }

            else:
                assert result_dict['passed']
                task_pred_results[result_dict['pred_idx']] = {
                    'result'    : result_str,
                    'time'      : result_dict['time'],
                    'is_failure': False
                }
                task_correct += 1

        # Calculate Percents for the task.
        task_metrics['correct_pct'] = task_metrics['correct'] / task_metrics['total'] * 100
        task_metrics['runtime_error_pct'] = task_runtime_errors / task_metrics['total'] * 100
        task_metrics['pred_results'] = deepcopy(task_pred_results)
        correct.append(task_correct)
        runtime_errors.append(task_runtime_errors)
        # runtimes_for_executions.extend(execution_runtimes)

        # Add the error types to the global tracker.
        for error_type, count in task_metrics['error_types'].items():
            global_error_tracker[error_type] += count

        results_by_task_id[task_id] = task_metrics

    # Need to add tasks that are all invalid to the results by task id
    for task_id in all_invalid:
        results_by_task_id[task_id] = {
            'correct'          : 0,
            'total'            : samples_per_problem,
            'error_types'      : Counter({'SyntaxError': samples_per_problem}),
            'pred_results'     : pred_results_by_task_idx[task_id],
            'correct_pct'      : 0.0,
            'runtime_error_pct': 0.0
        }
        global_error_tracker['SyntaxError'] += samples_per_problem
        correct.append(0)
        runtime_errors.append(0)

    correct = np.array(correct)
    runtime_errors = np.array(runtime_errors)
    metrics = {
        'runtime_error_pct_mean': np.mean(runtime_errors / samples_per_problem * 100),
    }

    return results_by_task_id, global_error_tracker, metrics, (correct, runtime_errors)


def mp_check_correctness(args):
    check_program, timeout, completion_id = args
    result = None
    try:
        with time_limit(5):
            result = unsafe_execute(check_program, timeout)
    except TimeoutException:
        pass
    if result is None:
        result = ("Timed Out", timeout, None)

    return dict(
        passed=result[0] == "Passed",
        result=result[0],
        time=result[1],
        error=result[2],
        completion_id=completion_id,
    )


def execute_code(task, samples, num_workers, timeout):
    to_run = sum(map(lambda s: len(s.predictions), samples))
    logger.info(f"{to_run} predictions to check")
    mp_args = []

    completion_id_to_pred_info = {}
    for sample in samples:
        task_id = sample.idx
        for pred_idx, gold_idx, candidate in sample.predictions:
            if task.lower() == 'mbpp':
                candidate = candidate.split('# Solution')[0]
            test_program = candidate + "\n" + '\n'.join(sample.tests)
            completion_id = len(completion_id_to_pred_info)
            completion_id_to_pred_info[completion_id] = dict(
                task_id=task_id,
                pred_idx=pred_idx,
                gold_idx=gold_idx,
            )

            mp_args.append(deepcopy((
                test_program, timeout, completion_id
            )))

            completion_id += 1

    found_results = {}

    timeout_result = dict(
        passed=False,
        result="Timed Out",
        time=timeout,
        error=None
    )
    all_results =[]
    with tqdm(total=to_run, desc='Running Code') as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            n_samples = 0
            for args in mp_args:
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                n_samples += 1

            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                pbar.update(1)

    for result in all_results:
        found_results[result['completion_id']] = result

    results = defaultdict(list)
    num_killed = 0
    for completion_id, pred_dict in completion_id_to_pred_info.items():
        if completion_id not in found_results:
            num_killed += 1
        result = found_results.get(completion_id, timeout_result)
        result.pop('completion_id', None)
        result.update(pred_dict)

        results[result["task_id"]].append((result["pred_idx"], result))
    logger.info(f"{num_killed} were killed")
    return results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])
