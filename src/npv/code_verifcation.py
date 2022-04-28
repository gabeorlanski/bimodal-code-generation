import ast
import astor
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
from .program_parsing import CustomSourceGenerator

logger = logging.getLogger(__name__)

__all__ = [
    "check_io_sample_executes_correctly",
    "add_random_inputs"
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


def input_types(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        return args, kwargs, result

    return inner


def make_executable_fn(
        func_name,
        code_str,
        context,
        input_expr,
        output_expr,
        expected_result,
        operator,
        with_assert=False,
        wrap_with_decorator=False
):
    code = [
        'def test_fn():',
        '\tresult_dict = {}'
    ]
    for line in context.split('\n'):
        code.append(f"\t{line}")
    for line in code_str.split('\n'):
        if f'def {func_name}' in line or line.startswith('def'):
            if wrap_with_decorator:
                code.append('\t@input_types')
        code.append(f"\t{line}")

    eval_stmt = f"{operator} __OUTPUT__"
    if wrap_with_decorator:
        test_stmt = [
            f"result_dict['args'],result_dict['kwargs'],result_dict['value'] = __INPUT__",
        ]
    else:
        test_stmt = [
            f"result_dict['value'] = __INPUT__"
        ]
    test_stmt.append(
        f"result_dict['result']= (result_dict['value'] {eval_stmt})==__RESULT__",
    )
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
    for i, program_dict in tqdm(enumerate(unverified_samples)):
        cleaned_output = program_dict['output']
        cleaned_input = program_dict['input']
        multiprocessing_args.append((
            program_dict['instance_idx'],
            i,
            make_executable_fn(
                program_dict['function'],
                program_dict['code'],
                program_dict['context'],
                cleaned_input,
                cleaned_output,
                program_dict['result'],
                program_dict['op'],
                with_assert,
                wrap_with_decorator=True
            )
        ))

    had_errors = defaultdict(dict)
    failed_tests = defaultdict(dict)
    num_passed_by_idx = Counter()
    num_failed_by_idx = Counter()

    rtr_values = defaultdict(dict)

    signature_counter = defaultdict(Counter)
    argument_pool = defaultdict(set)

    with mp.Pool(num_workers) as pool:
        exec_results = list(tqdm(
            pool.imap_unordered(abortable_worker, multiprocessing_args),
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

                # assert not result['kwargs']
                result_sig = []
                for arg in result['args']:
                    arg_type = type(arg).__name__
                    argument_pool[arg_type].add(repr(arg))
                    result_sig.append(arg_type)

                signature_counter[result['idx']][tuple(result_sig)] += 1

                if not result['passed']:
                    failed_tests[result['idx']][task_id] = instance_str
                    num_failed_by_idx[result['idx']] += 1
                else:
                    num_passed_by_idx[result['idx']] += 1

    logger.info(
        f"{sum(map(len, argument_pool.values()))} total unique inputs "
        f"found for {len(argument_pool)} types found"
    )
    # Get the most common signatures for each example.
    signatures = {}
    for k in signature_counter:
        signatures[k] = signature_counter[k].most_common(1)[0][0]

    return (argument_pool, signatures), dict(rtr_values), {
        'failed_counts': num_failed_by_idx, 'passed_counts': num_passed_by_idx,
        'failed_tests' : failed_tests, 'had_errors': had_errors
    }


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def is_sample_valid(args):
    instance_idx, code_idx, code_sample = args
    # RESULT_VALUE = {}
    to_return = None
    with create_tempdir():
        try:
            raised_exception = False
            try:
                stdout_f = io.StringIO()
                stderr_f = io.StringIO()
                with contextlib.redirect_stdout(stdout_f):
                    with contextlib.redirect_stderr(stderr_f):
                        _locals = locals()
                        exec(code_sample, globals(), _locals)
                        results = _locals['RESULT_DICT']
            except Exception as e:

                raised_exception = True
                to_return = {
                    'idx'      : instance_idx,
                    'code_idx' : code_idx,
                    'had_error': True,
                    'passed'   : False,
                    'result'   : str(e)
                }
            if not raised_exception:
                output = {
                    'idx'        : instance_idx,
                    'code_idx'   : code_idx,
                    'had_error'  : False,
                    'passed'     : results['result'],
                    'result'     : repr(results['value']),
                    'result_type': type(results['value']).__name__,
                    'args'       : results['args'],
                    'kwargs'     : results['kwargs']
                }

                if inspect.isfunction(results['value']) or inspect.isclass(results['value']):
                    to_return = {
                        'idx'      : instance_idx,
                        'code_idx' : code_idx,
                        'passed'   : False,
                        'had_error': True,
                        'result'   : 'Not Literal'
                    }
                else:
                    try:
                        pickle.dumps(output)

                        to_return = output
                    except Exception as e:
                        to_return = {
                            'idx'   : instance_idx, 'code_idx': code_idx,
                            'passed': False, 'result': str(e), 'had_error': True,
                        }
        except Exception as e:
            to_return = {
                'idx'      : instance_idx, 'code_idx': code_idx,
                'had_error': True, 'passed': False,
                'result'   : str(e)
            }
    return to_return


def abortable_worker(args,timeout=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        out = is_sample_valid(args)
        signal.alarm(0)
        return out
    except TimeoutError:
        return {
            'idx'      : -1, 'code_idx': -1,
            'had_error': True, 'passed': False,
            'result'   : "Timeout"
        }


def add_random_inputs(programs, signatures, pool, random_inputs_to_add, workers):
    if random_inputs_to_add == 0:
        return programs

    funcs_to_test = []
    sampled_inputs = []
    for program_idx, program in tqdm(enumerate(programs), total=len(programs)):
        all_inputs = [v['input'] for v in program['input_output_pairs']]

        func_node = None
        for n in ast.walk(ast.parse(all_inputs[0])):
            if isinstance(n, ast.Call) and n.func.id == program['function']:
                func_node = n
                break

        num_sigs_to_generate = 2*random_inputs_to_add
        tries = 0
        sampled_functions = set()
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            while len(sampled_functions) < num_sigs_to_generate and tries < 100:
                sampled_sig = [
                    ast.parse(random.choice(pool[sig_type])).body[0].value
                    for sig_type in signatures[program['instance_idx']]
                ]

                func_node.args = sampled_sig
                func_string = astor.to_source(
                    func_node,
                    source_generator_class=CustomSourceGenerator
                ).strip()
                if func_string not in sampled_functions and func_string not in all_inputs:
                    sampled_functions.add(func_string)
                tries += 1
            signal.alarm(0)
        except TimeoutError:
            logger.error(f"{program_idx} timed out")
            continue

        for i, func in enumerate(sampled_functions):
            funcs_to_test.append(
                (
                    program_idx,
                    len(sampled_inputs),
                    make_executable_fn(
                        program['function'],
                        program['code'],
                        program['context'],
                        func,
                        'None',
                        "True",
                        'is not',
                    ))
            )
            sampled_inputs.append(func)
    logger.info(f"{len(funcs_to_test)} random functions to test")

    passed_by_idx = defaultdict(lambda: defaultdict(list))

    for to_test in tqdm(funcs_to_test,desc='Executing'):
        result = abortable_worker(to_test,timeout=2)
        if not result['passed'] or result['had_error']:
            continue

        passed_by_idx[result['idx']][result['result']].append(
            sampled_inputs[result['code_idx']]
        )

    logger.info(f'RoundRobin Selecting new inputs for {len(passed_by_idx)}/'
                f'{len(programs)} programs')
    for prog_idx, raw_output_dict in tqdm(passed_by_idx.items()):
        program = programs[prog_idx]
        output_dict = {}
        for k, v in raw_output_dict.items():
            output_dict[k] = list(v)
            random.shuffle(output_dict[k])

        all_outputs = set(v['output'] for v in program['input_output_pairs'])
        if len(all_outputs) < len(output_dict):

            # Want a diverse set of inputs AND outputs.
            num_from_new_outputs = sum(
                len(v) for k, v in output_dict.items() if k not in all_outputs
            )
            if num_from_new_outputs > random_inputs_to_add:
                for output in all_outputs:
                    output_dict.pop(output, None)

        # Want a diverse set of new outputs added, so do round robin on them.
        to_add = []
        func_iter = iter(sorted(output_dict, key=lambda key: len(output_dict[key])))
        while output_dict and len(to_add) < random_inputs_to_add:
            try:
                to_take_key = next(func_iter)
            except StopIteration:
                if not output_dict:
                    break
                func_iter = iter(sorted(output_dict, key=lambda key: len(output_dict[key])))
                continue

            to_add.append({
                'input'       : output_dict[to_take_key].pop(0),
                'output'      : to_take_key,
                'ops'         : '==',
                'is_random'   : True,
                'is_generated': False
            })
            if not output_dict[to_take_key]:
                output_dict.pop(to_take_key)
        programs[prog_idx]['input_output_pairs'].extend(to_add)

    return programs
