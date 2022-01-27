import itertools
import json
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

from src.evaluation.execute import check_correctness

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
    return {f"{name}/{k}": float(v) for k, v in metrics_dict.items()}


def get_samples(file_path, samples_per_problem) -> Tuple[List[Sample], List, Dict, Dict, Dict]:
    logger.info(f"Reading {file_path}")
    total_lines = len(file_path.read_text().splitlines(False))
    lines = map(json.loads, file_path.read_text().splitlines(False))
    failed = 0
    line_num = 0
    total_valid_preds = []
    metrics = {
        "preds_total": 0,
    }
    all_invalid = []
    invalid_syntax = {}
    all_samples = {}
    pred_count = {}
    for sample_dict in tqdm(lines, total=total_lines, desc='Reading Preds'):
        line_num += 1

        if any(k not in sample_dict for k in ['prediction', 'tests']):
            logger.error(f"Line {line_num} is missing either 'prediction' or 'tests' keys")
            failed += 1
            continue

        idx = sample_dict.get('task_id', sample_dict.get('idx'))
        if idx is None:
            logger.error(f"Line {line_num} is missing an idx key")
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
                continue
            except Exception as e:
                logger.error(f"Could not parse prediction {i} on line "
                             f"{line_num} due to {type(e).__name__}:{str(e)}")
                continue
            valid_predictions.append(pred)

        invalid_syntax[idx] = len(sample_dict['prediction']) - len(valid_predictions)
        total_valid_preds.append(len(valid_predictions))
        if not valid_predictions:
            logger.info(f"Task {idx} had no valid syntax predictions")
            all_invalid.append(idx)
            continue
        all_samples[idx] = Sample(idx, valid_predictions, sample_dict['tests'])

    if failed > 0:
        logger.error(f"{failed}/{line_num} had failures.")

    metrics.update(
        get_metrics_from_list(
            'valid_syntax_pct',
            np.array(total_valid_preds) / samples_per_problem * 100
        )
    )
    logger.info(f"{len(all_samples)} unique tasks")

    # Adding the / to some metrics to better separate them
    metrics['info/preds_total'] = metrics.pop('preds_total')
    metrics['info/all_invalid'] = len(all_invalid)
    metrics['info/all_valid'] = sum(map(lambda l: l == samples_per_problem, total_valid_preds))

    return list(all_samples.values()), all_invalid, pred_count, invalid_syntax, metrics


def evaluate_code(
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

    samples, all_invalid, pred_count, invalid_syntax_by_idx, overview_metrics = get_samples(
        predictions_file,
        samples_per_problem
    )

    results = execute_code(samples, num_workers, timeout)

    results_by_task_id, global_error_tracker, metrics, counts = parse_results(
        results,
        pred_count,
        invalid_syntax_by_idx,
        all_invalid,
        samples_per_problem
    )
    correct, runtime_errors = counts
    overview_metrics.update(metrics)

    # Calculate the pass @ k metric across multiple k values.
    all_correct = np.array(correct)
    all_total = np.array([samples_per_problem] * len(correct))
    total = int(sum(all_total))
    for k in [1, 5, 10, 25, 50]:
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
    for k in sorted(overview_metrics.keys()):
        logger.info(f"\t{k:>32} = {overview_metrics[k]:0.3f}")

    result_metrics = {
        "overview"          : overview_metrics,

        # No int keys in json files, so make them match.
        "results_by_task_id": {str(k): v for k, v in results_by_task_id.items()},
        "outcome_counts"    : outcome_counts,
        "outcome_pcts"      : outcome_pcts
    }

    return result_metrics


def parse_results(results, pred_count, invalid_syntax_by_idx, all_invalid, samples_per_problem):
    global_error_tracker = Counter({k: 0 for k in BASE_ERROR_TYPES})
    results_by_task_id = {}

    correct, runtime_errors = [], []
    for task_id, task_results in results.items():
        task_results.sort()
        if pred_count[task_id] == 0:
            logger.error(f"Task {task_id} has no prediction count but has results")
            continue

        # Setup the dict for tracking metrics for a given task.
        task_metrics = {
            'correct'    : 0,
            'total'      : pred_count[task_id],
            'error_types': Counter({'SyntaxError': invalid_syntax_by_idx[task_id]})
        }
        task_correct = task_runtime_errors = 0

        # Go through and calculate metrics on both a task and global level.
        for completion_id, result in task_results:
            assert result['completion_id'] == completion_id
            assert result['task_id'] == task_id

            task_metrics['correct'] += result['passed']

            if not result['passed']:
                task_metrics['error_types'][result['result']] += 1
                assert result['result'] != 'Passed'
                if result['result'] != 'Failed Tests' and result['result'] != 'Timed Out':
                    task_runtime_errors += 1
            else:
                task_correct += 1

        # Calculate Percents for the task.
        task_metrics['correct_pct'] = task_metrics['correct'] / task_metrics['total'] * 100
        task_metrics['runtime_error_pct'] = task_runtime_errors / task_metrics['total'] * 100

        correct.append(task_correct)
        runtime_errors.append(task_runtime_errors)

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
            'correct_pct'      : 0.0,
            'runtime_error_pct': 0.0
        }
        global_error_tracker['SyntaxError'] += samples_per_problem
        correct.append(0)
        runtime_errors.append(0)

    correct = np.array(correct)
    runtime_errors = np.array(runtime_errors)

    metrics = get_metrics_from_list(
        'runtime_error_pct',
        runtime_errors / samples_per_problem * 100
    )

    return results_by_task_id, global_error_tracker, metrics, (correct, runtime_errors)


def execute_code(samples, num_workers, timeout):
    to_run = sum(map(lambda s: len(s.predictions), samples))
    logger.info(f"{to_run} predictions to check")
    with tqdm(total=to_run, desc='Running Code') as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)

            for sample in samples:
                task_id = sample.idx
                for candidate in sample.predictions:
                    test_program = candidate + "\n" + '\n'.join(sample.tests)
                    args = (test_program, timeout, task_id, completion_id[task_id])
                    future = executor.submit(check_correctness, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

            for future in as_completed(futures):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))
                pbar.update(1)
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
