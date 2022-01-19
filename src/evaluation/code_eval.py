import itertools
import json
from typing import Union, List, Tuple, Dict
import logging
from pathlib import Path
import os
import ast
import numpy as np
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


def get_samples(file_path) -> Tuple[List[Sample], Dict, Dict, Dict]:
    logger.info(f"Reading {file_path}")
    total_lines = len(file_path.read_text().splitlines(False))
    lines = map(json.loads, file_path.read_text().splitlines(False))
    failed = 0
    line_num = 0
    total_valid_preds = []
    metrics = {
        "preds_total"   : 0,
        "all_invalid"   : 0,
        "valid_pct_list": [],
        "total_tests"   : []
    }
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
        for i, pred in enumerate(sample_dict['prediction']):
            metrics['preds_total'] += 1
            try:
                ast.parse(pred)
            except SyntaxError:
                continue
            except MemoryError:
                logger.error(f"Could not parse prediction {i} on line "
                             f"{line_num} due to a memory error.")
                continue
            valid_predictions.append(pred)
        metrics['valid_pct_list'].append(
            len(valid_predictions) / len(sample_dict['prediction']) * 100
        )

        invalid_syntax[idx] = len(sample_dict['prediction']) - len(valid_predictions)
        total_valid_preds.append(len(valid_predictions))
        metrics['total_tests'].append(len(sample_dict['tests']))
        if not valid_predictions:
            metrics['all_invalid'] += 1
            logger.info(f"Task {idx} had no valid syntax predictions")
            continue
        all_samples[idx] = Sample(idx, valid_predictions, sample_dict['tests'])

    if failed > 0:
        logger.error(f"{failed}/{line_num} had failures.")

    metrics['valid_syntax_total'] = sum(total_valid_preds)
    metrics['valid_syntax_mean'] = np.mean(total_valid_preds)
    metrics['valid_syntax_pct'] = metrics['valid_syntax_total'] / metrics['preds_total'] * 100
    metrics['tests_mean'] = np.mean(metrics.pop('total_tests'))
    metrics['valid_syntax_pct_mean'] = np.mean(metrics.pop('valid_pct_list'))

    return list(all_samples.values()), pred_count, invalid_syntax, metrics


def evaluate_code(
        split_name: str,
        predictions_dir: Union[str, Path, os.PathLike],
        num_workers: int,
        timeout: float = 3.0,
        out_dir: Union[str, Path, os.PathLike] = None
):
    predictions_dir = Path(predictions_dir)
    out_dir = Path(out_dir) if out_dir else predictions_dir
    logger.info(
        f"Starting Code Evaluation with predictions in {predictions_dir.resolve().absolute()}")
    path_to_predictions = predictions_dir.joinpath(f'{split_name}_predictions.jsonl')

    if not path_to_predictions.exists():
        logger.error(f"{predictions_dir.resolve().absolute()} is missing 'predictions.jsonl")
        raise FileExistsError(f"The predictions directory must have a predictions.jsonl")

    samples, pred_count, invalid_syntax_by_idx, overview_metrics = get_samples(path_to_predictions)

    global_error_tracker = Counter({k: 0 for k in BASE_ERROR_TYPES})
    results_by_task_id = {}

    # Need to add tasks that are all invalid to the results by task id
    for task_id in filter(lambda idx: invalid_syntax_by_idx[idx] == pred_count[idx],
                          invalid_syntax_by_idx):
        results_by_task_id[task_id] = {
            'correct'          : 0,
            'total'            : pred_count[task_id],
            'error_types'      : Counter({'SyntaxError': invalid_syntax_by_idx[task_id]}),
            'correct_pct'      : 0.0,
            'runtime_error_pct': 0.0
        }
        global_error_tracker['SyntaxError'] += invalid_syntax_by_idx[task_id]

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

    correct = runtime_errors = 0
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
        task_runtime_errors = 0

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
                correct += 1

        # Calculate Percents for the task.
        task_metrics['correct_pct'] = task_metrics['correct'] / task_metrics['total'] * 100
        task_metrics['runtime_error_pct'] = task_runtime_errors / task_metrics['total'] * 100

        # Add the error types to the global tracker.
        for error_type, count in task_metrics['error_types'].items():
            global_error_tracker[error_type] += count
        runtime_errors += task_runtime_errors

        results_by_task_id[task_id] = task_metrics

    total = overview_metrics['preds_total']
    correct = int(np.sum(correct))
    overview_metrics['runtime_error_pct'] = runtime_errors / total * 100
    global_error_tracker['Correct'] = correct

    # Calculate the pass @ k metrics
    all_correct, all_total = [], []
    for d in results_by_task_id.values():
        all_correct.append(d['correct'])
        all_total.append(d['total'])

    all_total = np.array(all_total)
    all_correct = np.array(all_correct)
    for k in [1, 5, 10, 25, 50, 100]:
        if (all_total < k).all():
            overview_metrics[f"pass@{k}"] = 0.0
            continue
        overview_metrics[f"pass@{k}"] = estimate_pass_at_k(all_total, all_correct, k).mean() * 100

    for error_type, count in global_error_tracker.items():
        key = error_type.replace(" ", "_")
        overview_metrics[key] = count
        overview_metrics[f'{key}_pct'] = count / total * 100

    logger.info("Metrics:")
    for k in sorted(overview_metrics.keys()):
        logger.info(f"\t{k:>24} = {overview_metrics[k]:0.3f}")

    result_metrics = {
        "overview"          : overview_metrics,
        # No int keys in json files, so make them match.
        "results_by_task_id": {str(k): v for k, v in results_by_task_id.items()}
    }
    with out_dir.joinpath('execution_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(result_metrics, f, indent=True)

    return result_metrics, list(global_error_tracker), out_dir.joinpath('execution_metrics.json')


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
