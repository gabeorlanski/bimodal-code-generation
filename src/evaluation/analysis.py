import contextlib
import io
import multiprocessing
import signal
import astor
from copy import deepcopy

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


class CustomSourceGenerator(astor.SourceGenerator):
    def visit_Dict(self, node):
        astor.code_gen.set_precedence(astor.op_util.Precedence.Comma, *node.keys)  # type: ignore
        astor.code_gen.set_precedence(astor.op_util.Precedence.Comma, *node.values)  # type: ignore
        with self.delimit('{}'):
            for idx, (key, value) in enumerate(zip(node.keys, node.values)):
                self.write(', ' if idx else '',
                           key if key else '',
                           ': ' if key else '**', value)

    def visit_Tuple(self, node):
        # with self.delimit(node) as delimiters:
        # Two things are special about tuples:
        #   1) We cannot discard the enclosing parentheses if empty
        # #   2) We need the trailing comma if only one item
        # elts = node.elts
        # delimiters.discard = delimiters.discard and elts

        astor.code_gen.set_precedence(
            astor.op_util.Precedence.Comma,  # type: ignore
            *node.elts
        )
        self.write('(')
        for idx, item in enumerate(node.elts):
            self.write(', ' if idx else '', item)
        if len(node.elts) == 1:
            self.write(',')
        self.write(')')


class AssertTransformer(ast.NodeTransformer):
    def visit_Assert(self, node):
        if isinstance(node.test, ast.Compare):
            out = node.test.left
        else:
            out = node.test

        return ast.Assign(targets=[ast.Name(id='TEST_RESULT')], value=out)


class MarkForSkip(Exception):
    pass


class VariableTracer(ast.NodeVisitor):
    def __init__(self):
        self.func_traces = []
        self.imported_libraries = []
        self.trace = defaultdict(
            lambda: {
                'func_name': None, 'func_args': [], 'defined': [], 'used': []
            }
        )
        self.in_aug_assign = False
        self.in_func = False

        self.tracker_maps = {
            'control_flow' : {
                ast.If,
                ast.For,
                ast.With,
                ast.While,
                ast.Try
            },
            'comprehension': {
                ast.ListComp,
                ast.DictComp,
                ast.GeneratorExp,
                ast.SetComp
            },
            'subscripts'   : {
                ast.Slice,
                ast.Index,
                ast.ExtSlice
            }
        }
        self.tracker_counts = {
            cat: {c: 0 for c in cat_classes}
            for cat, cat_classes in self.tracker_maps.items()
        }

    def _clear(self):
        self.in_aug_assign = False
        self.in_func = False

    def finish_trace(self):
        self.func_traces.append(deepcopy(self.trace))
        self.trace = defaultdict(
            lambda: {
                'func_name': None, 'func_args': [], 'defined': [], 'used': []
            }
        )

    @staticmethod
    def parse_trace(trace_dict):
        unused_vars = set()
        redefined_vars_no_usage = []
        used_vars = set()
        defined_vars = set()
        func_name = None
        func_signature = []
        nested_func_names = []
        for line_no, line_trace in trace_dict.items():
            if line_trace['func_name'] is not None:
                if func_name is None:
                    func_name = line_trace['func_name']
                    func_signature = line_trace['func_args']
                else:
                    nested_func_names.append(line_trace['func_name'])
                unused_vars.update(line_trace['func_args'])
            else:
                for var_name in line_trace['used']:
                    if var_name in unused_vars:
                        unused_vars.remove(var_name)
                    used_vars.add(var_name)

                for var_name in line_trace['defined']:
                    if var_name in unused_vars:
                        redefined_vars_no_usage.append(var_name)
                    unused_vars.add(var_name)
                    defined_vars.add(var_name)

        out = {
            'func_name'              : func_name,
            'unused_vars'            : list(unused_vars),
            'redefined_vars_no_usage': redefined_vars_no_usage,
            'used_vars'              : list(used_vars),
            'defined_vars'           : list(defined_vars),
            'line_count'             : max(trace_dict) if trace_dict else 0,
            'func_signature'         : func_signature,
            'nested_func_names'      : nested_func_names,
        }
        return out

    def __call__(self, code_str):

        tree = ast.parse(code_str)

        for body in tree.body:
            self._clear()
            self.visit(body)  # type:ignore
        self.func_traces.append(self.trace)

        parsed_traces = [
            self.parse_trace(td) for td in self.func_traces
        ]

        trace_stats = {}
        for group, group_dict in self.tracker_counts.items():
            trace_stats[group] = {c.__name__: v for c, v in group_dict.items()}

        return parsed_traces, trace_stats, self.imported_libraries

    def _handle_import(self, node):
        for n in node.names:
            if n.asname is not None:
                self.imported_libraries.append(n.asname)
            else:
                self.imported_libraries.append(n.name)

    def visit_Import(self, node):
        return self._handle_import(node)

    def visit_ImportFrom(self, node):
        return self._handle_import(node)

    def visit_FunctionDef(self, node):
        # We only want to trace functions within their scope
        if self.trace and not self.in_func:
            self.finish_trace()
        self.trace[node.lineno]['func_name'] = node.name  # type: ignore
        self.in_func = True

        # Some functions are only the comment. We want to skip those, so we
        # mark them to be skipped.
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                raise MarkForSkip('Invalid function')

        args_found = self.handle_arguments(node.args)
        self.trace[node.lineno]['func_args'] = args_found
        return self.generic_visit(node)

    def handle_arguments(self, node: ast.arguments):
        args_found = []
        for arg in node.args:
            args_found.append(arg.arg)

        for arg in node.posonlyargs:
            args_found.append(arg.arg)
        for arg in node.kwonlyargs:
            args_found.append(arg.arg)

        if node.vararg is not None:
            args_found.append(node.vararg.arg)
        if node.kwarg is not None:
            args_found.append(node.kwarg.arg)
        return args_found

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store) and not self.in_aug_assign:
            self.trace[node.lineno]['defined'].append(node.id)
        else:
            self.trace[node.lineno]['used'].append(node.id)

    def visit_AugAssign(self, node: ast.AugAssign):
        self.in_aug_assign = True
        self.generic_visit(node)

    def generic_visit(self, node):
        for group, group_map in self.tracker_maps.items():
            if type(node) in group_map:
                self.tracker_counts[group][type(node)] += 1
                break
        return super(VariableTracer, self).generic_visit(node)


def get_stats_for_programs(code_str):
    try:

        # Single parse out here to make sure there are no errors
        # with the line of code.
        ast.parse(code_str)

        visitor = VariableTracer()
        try:
            parsed_traces, trace_stats, imported = visitor(code_str)
        except MarkForSkip:
            return {}, True

        out = {}

        for group, group_dict in trace_stats.items():
            for c, v in group_dict.items():
                out[f"{group}_{c}"] = v

        # Multiply by the number of idx with this prediction.
        out['imports'] = imported
        out['parsed_bodies'] = parsed_traces

    except (SyntaxError, MemoryError):
        return None, False
    return out, False


def calc_stats_for_task(task, task_prediction_dict, task_results):
    task_info = {
        k: task_prediction_dict[k] for k in
        ['task_id', 'idx', 'tests']
    }
    task_info['test_setup_code'] = task_prediction_dict.get('test_setup_code', '')

    # Create a dict to map unique predictions to the list of indices they correspond too.
    unique_pred_to_idx = defaultdict(list)
    with_runtime_errors = 0
    with_syntax_errors = 0
    with_signature_errors = 0
    unique_errors = set()

    for pred_idx, pred_result in task_results['pred_results'].items():
        if pred_result['result'] == 'SyntaxError':
            with_syntax_errors += 1
            continue
        elif pred_result['is_failure']:

            if all(w in pred_result['result'] for w in ['positional', 'argument', 'takes']):
                with_signature_errors += 1
            with_runtime_errors += 1
            unique_errors.add(pred_result['result'])

        pred = task_prediction_dict['prediction'][int(pred_idx)]
        if task == 'MBPP':
            pred = pred.split('# Solution')[0]

        unique_pred_to_idx[pred.strip()].append(int(pred_idx))

    out = {
        'with_runtime_errors'     : with_runtime_errors,
        'with_syntax_errors'      : with_syntax_errors,
        "with_signature_errors"   : with_signature_errors,
        "runtime_unique_per_total": len(unique_errors) / task_results['total'],
        "unique_errors"           : len(unique_errors),
        "unique_programs"         : len(unique_pred_to_idx),
        "prog_unique_per_total"   : len(unique_pred_to_idx) / task_results['total']
    }

    return out


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
    task_result_counter = Counter({k: 0 for k in [
        'no_correct',
        'all_correct',
        'all_runtime_error',
        'all_syntax_error',
        'all_failed_tests',
        'has_runtime_errors',
        'unique_programs'
    ]})

    solved_tasks = []
    program_stats_by_tid = defaultdict(dict)

    for tid, task_results in tqdm(results_by_task_id.items(), desc='Parsing'):
        total_preds = task_results['total']

        preds_for_task = predictions[tid]

        task_stats = calc_stats_for_task(task, preds_for_task, task_results)
        for k, v in task_stats.items():
            task_result_counter[k] += v
            mean_tracker[k].append(v)

        if task_results['correct'] == 0:
            task_result_counter['no_correct'] += 1
        else:
            solved_tasks.append(tid)

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
        # for outcome in all_outcomes:
        #     if outcome == 'Correct':
        #         continue
        #
        #     outcome_key = outcome.replace("_", '')
        #     outcome_count = error_types.get(outcome.replace('_', ' '), 0)
        #     if outcome_key not in ['TimedOut', 'SyntaxError', 'FailedTests']:
        #         total_runtime_errors += outcome_count
        #     mean_tracker[outcome_key].append(outcome_count)

        if total_runtime_errors == total_preds:
            task_result_counter['all_runtime_error'] += 1
        mean_tracker['TotalRuntimeErrors'].append(total_runtime_errors)

    out = {}
    for k, v in task_result_counter.items():
        out[f"{k}_pct"] = v / len(results_by_task_id) * 100
        out[f"{k}_total"] = v

    for k, v in mean_tracker.items():
        out[f"{k}_mean"] = np.mean(v)
        out[f"{k}_std"] = np.std(v)
    out['solved_tasks'] = solved_tasks
    out['solved_pct'] = len(solved_tasks)/len(results_by_task_id)*100
    out.update(execution_metrics['test']['overview'])

    return out, program_stats_by_tid, preds_to_time_check


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def get_runtime(args_list):
    run_info, check_program, timeout, sample_idx, task_id = args_list
    # Allows us to save values from the exec call
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
            RESULT_DICT['TIME'] = f"{type(e).__name__}: {str(e)}"
            had_error = True
    stdout_f.close()
    stderr_f.close()
    return dict(
        run_info=run_info,
        task_id=task_id,
        sample_idx=sample_idx,
        had_error=had_error,
        had_timeout=had_timeout,
        runtime=RESULT_DICT['TIME'],
    )


def execute_time_check(to_time_check, num_workers, timeit_number=100, timeout=3):
    logger.info(f"{len(to_time_check)} predictions to time check")
    logger.info(f"Running each program {timeit_number} time(s)")
    mp_args = []
    with_syntax_errors = 0
    for sample_idx, sample in tqdm(enumerate(to_time_check), total=len(to_time_check),
                                   desc='Creating Arguments'):
        test_str = '\n'.join([sample['test_setup_code']] + sample['tests'])

        # test_str = test_str.replace('assert', 'ASSERT_PLACEHOLDER=')
        task_id = sample['task_id']
        test_program = sample['prediction'] + "\n" + test_str

        # Wrap the test function with another function so that the
        # entire thing can be called from timeit.
        wrapped_func = []
        for line in test_program.split('\n'):
            wrapped_func.append(f'\t{line}')

        wrapped_func = '\n'.join(wrapped_func)
        wrapped_func = f"def TEST_CANDIDATE():\n{wrapped_func}"

        wrapped_program = [
            "import timeit",
            wrapped_func,
            f"RESULT_DICT['TIME']=timeit.timeit(TEST_CANDIDATE,number={timeit_number})"
        ]

        wrapped_program = '\n'.join(wrapped_program)
        try:
            ast.parse(wrapped_program)
        except (SyntaxError, MemoryError) as e:
            with_syntax_errors += 1
            continue
        mp_args.append(
            (sample['run_info'], wrapped_program, timeout, sample_idx, task_id)
        )

    logger.info(f"{with_syntax_errors}/{len(to_time_check)} had syntax errors")

    task_info = defaultdict(dict)
    with_errors = []
    with_timeout = 0
    with multiprocessing.Pool(num_workers) as pool:
        raw_results = list(tqdm(
            pool.imap_unordered(get_runtime, mp_args),
            total=len(to_time_check),
            desc='Getting Runtime')
        )

        for r in raw_results:
            sample_idx = r.pop('sample_idx')
            if r['had_error']:
                # passed_with_errors += r['passed']
                with_errors.append(
                    ('_'.join(r['run_info']), r['task_id'], sample_idx, r['runtime'])
                )
                continue
            if r['had_timeout']:
                with_timeout += 1

            runtime = r.pop('runtime')
            if r['task_id'] not in task_info[r['run_info']]:
                task_info[r['run_info']][r['task_id']] = {
                    'runtimes': [], 'passed': [], **r
                }

            task_info[r['run_info']][r['task_id']]['runtimes'].append(runtime)
            task_info[r['run_info']][r['task_id']]['passed'].append(sample_idx)

    logger.info(f"{len(with_errors)}/{len(to_time_check)} had errors")
    logger.info(f"{with_timeout}/{len(to_time_check)} timed out")
    return task_info, with_errors
