import ast
import contextlib
import inspect
import io
import logging
import math
import multiprocessing as mp
import pickle
import random
import signal
from collections import defaultdict, Counter
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.evaluation.execute import create_tempdir

logger = logging.getLogger(__name__)

__all__ = [
    "check_io_sample_executes_correctly",
]


def execute_code(code):
    result = None
    with create_tempdir():
        try:
            stdout_f = io.StringIO()
            stderr_f = io.StringIO()
            with contextlib.redirect_stdout(stdout_f):
                with contextlib.redirect_stderr(stderr_f):
                    # sys.stdout.write = lambda *args, **kwargs: None
                    exec(code, globals(), locals())
        except Exception as e:
            result = e
    return result


def make_executable_fn(
        code_str,
        context,
        input_expr,
        output_expr,
        expected_result,
        operator,
        with_assert=False
):
    code = [
        'def test_fn():',
        '\tresult_dict = {}'
    ]
    raw_code = [context, code_str]
    for block in map(lambda b: b.split('\n'), raw_code):
        for line in filter(lambda b: b.strip(), block):
            code.append(f"\t{line}")

    eval_stmt = f"{operator} __OUTPUT__"
    test_stmt = [
        f"result_dict['value'] = __INPUT__",
        f"result_dict['result']= (result_dict['value'] {eval_stmt})==__RESULT__",

    ]
    if with_assert:
        test_stmt.append("assert result_dict['result']")
    test_stmt.append('return result_dict')

    for line in test_stmt:
        code.append(f"\t{line}")
    code.append("RESULT_DICT=test_fn()")
    code = '\n'.join(code)
    code = code.replace('__INPUT__', input_expr).replace('__OUTPUT__', output_expr)
    return code.replace('__RESULT__', expected_result)


def check_io_sample_executes_correctly(split, unverified_samples, num_workers, with_assert=False):
    multiprocessing_args = []

    logger.info(f"Creating MP args for {len(unverified_samples)} "
                f"unverified samples in '{split}'")
    for i, program_dict in enumerate(unverified_samples):
        cleaned_output = program_dict['output']
        cleaned_input = program_dict['input']
        multiprocessing_args.append((
            program_dict['instance_idx'],
            i,
            make_executable_fn(
                program_dict['code'],
                program_dict['context'],
                cleaned_input,
                cleaned_output,
                program_dict['result'],
                program_dict['op'],
                with_assert
            )
        ))

    had_errors = defaultdict(dict)
    failed_tests = defaultdict(dict)
    num_passed_by_idx = Counter()
    num_failed_by_idx = Counter()

    rtr_values = defaultdict(dict)

    with mp.Pool(num_workers) as pool:
        exec_results = list(tqdm(
            pool.imap_unordered(is_sample_valid, multiprocessing_args),
            total=len(multiprocessing_args),
            desc='Executing'
        ))

        for result in exec_results:
            instance = unverified_samples[result['code_idx']]
            task_id = instance['task_id']
            instance_str = f"{instance['input']} {instance['output']}"
            if result['had_error']:
                num_failed_by_idx[result['idx']] += 1
                had_errors[result['idx']][task_id] = instance_str

            else:
                rtr_values[result['idx']][instance['task_id']] = {
                    'type' : result['result_type'],
                    'value': result['result']
                }
                if not result['passed']:
                    failed_tests[result['idx']][task_id] = instance_str
                    num_failed_by_idx[result['idx']] += 1
                else:
                    num_passed_by_idx[result['idx']] += 1

    return dict(rtr_values), {
        'failed_counts': num_failed_by_idx, 'passed_counts': num_passed_by_idx,
        'failed_tests' : failed_tests, 'had_errors': had_errors
    }


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def is_sample_valid(args):
    instance_idx, code_idx, code_sample = args
    # RESULT_VALUE = {}

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        with create_tempdir():
            try:
                stdout_f = io.StringIO()
                stderr_f = io.StringIO()
                with contextlib.redirect_stdout(stdout_f):
                    with contextlib.redirect_stderr(stderr_f):
                        _locals = locals()
                        exec(code_sample, globals(), _locals)
                        results = _locals['RESULT_DICT']
            except Exception as e:
                return {
                    'idx'      : instance_idx,
                    'code_idx' : code_idx,
                    'had_error': True,
                    'passed'   : False,
                    'result'   : str(e)
                }
        output = {
            'idx'        : instance_idx,
            'code_idx'   : code_idx,
            'had_error'  : False,
            'passed'     : results['result'],
            'result'     : repr(results['value']),
            'result_type': type(results['value']).__name__
        }

        if inspect.isfunction(results['value']) or inspect.isclass(results['value']):
            return {
                'idx'      : instance_idx,
                'code_idx' : code_idx,
                'passed'   : False,
                'had_error': True,
                'result'   : 'Not Literal'
            }
        try:
            pickle.dumps(output)
        except Exception as e:
            return {
                'idx'   : instance_idx, 'code_idx': code_idx,
                'passed': False, 'result': str(e), 'had_error': True,
            }
        return output
    except Exception as e:
        return {
            'idx'      : instance_idx, 'code_idx': code_idx,
            'had_error': True, 'passed': False,
            'result'   : str(e)
        }