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
        'has_runtime_errors',
        'unique_programs'
    ]}

    solved_tasks = []
    program_stats_by_tid = defaultdict(dict)

    for tid, task_results in tqdm(results_by_task_id.items(), desc='Parsing'):
        total_preds = task_results['total']

        preds_for_task = predictions[tid]
        task_info = {
            k: preds_for_task[k] for k in
            ['task_id', 'idx', 'tests']
        }
        task_info['test_setup_code'] = preds_for_task.get('test_setup_code', '')

        # I miscalculated the idx for the predictions that passed. So I need to
        # remove those with bad syntax prior.
        # First, create a list of bool mappings for if we should keep a
        # prediction
        to_keep = [None] * len(preds_for_task['prediction'])
        force_skip = [False] * len(preds_for_task['prediction'])

        # Create a dict to map unique predictions to the list of indices they correspond too.
        unique_pred_to_idx = defaultdict(list)
        predicted_by_idx = []
        for i, p in enumerate(preds_for_task['prediction']):
            if task == 'MBPP':
                p = p.split('# Solution')[0]

            unique_pred_to_idx[p.strip()].append(i)
            predicted_by_idx.append(p.strip())

        task_result_counter['unique_programs'] += len(unique_pred_to_idx)
        mean_tracker['unique_programs'].append(len(unique_pred_to_idx))

        for p, idx_list in unique_pred_to_idx.items():
            p_stats, to_skip = get_stats_for_programs(p)
            if p_stats is None:
                continue
            for idx in idx_list:
                force_skip[idx] = to_skip
                to_keep[idx] = p_stats

        predictions_w_valid_syntax = []
        valid_force_skip = []
        idx_to_prog_stats = {}

        for i, p_stats in enumerate(to_keep):
            if p_stats is None:
                continue
            if tid == '42' and len(valid_force_skip) >= 195:
                print("???")
            valid_force_skip.append(force_skip[i])
            idx_to_prog_stats[len(predictions_w_valid_syntax)] = p_stats
            predictions_w_valid_syntax.append(predicted_by_idx[i])

        for k in ['passed', 'failed_tests']:

            for pred_idx in task_results.get(k, []):
                if valid_force_skip[pred_idx]:
                    continue
                program_stats_by_tid[tid][pred_idx] = idx_to_prog_stats[pred_idx]
                preds_to_time_check.append(
                    {
                        'prediction': predictions_w_valid_syntax[pred_idx],
                        'passed'    : k == 'passed',
                        **task_info
                    }
                )
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
    out['solved_tasks'] = solved_tasks

    return out, program_stats_by_tid, preds_to_time_check


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def get_runtime(args_list):
    run_info, passed, check_program, timeout, sample_idx, task_id = args_list
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
        passed=passed,
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
    passed_with_syntax_errors = 0
    for sample_idx, sample in tqdm(enumerate(to_time_check), total=len(to_time_check),
                                   desc='Creating Arguments'):
        test_str = '\n'.join([sample['test_setup_code']] + sample['tests'])
        try:
            visitor = AssertTransformer()
            test_str = astor.to_source(
                visitor.visit(ast.parse(test_str)),
                source_generator_class=CustomSourceGenerator
            )
        except (SyntaxError, MemoryError):
            if sample['passed']:
                passed_with_syntax_errors += 1
            continue

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

        test_program = [
            "import timeit",
            wrapped_func,
            f"RESULT_DICT['TIME']=timeit.timeit(TEST_CANDIDATE,number={timeit_number})"
        ]

        test_program = '\n'.join(test_program)
        try:
            ast.parse(test_program)
        except (SyntaxError, MemoryError):
            with_syntax_errors += 1
            if sample['passed']:
                passed_with_syntax_errors += 1
            continue
        mp_args.append(
            (sample['run_info'], sample['passed'], test_program, timeout, sample_idx, task_id)
        )

    logger.info(f"{with_syntax_errors}/{len(to_time_check)} had syntax errors")
    logger.info(
        f"{passed_with_syntax_errors} passed examples had syntax errors")

    task_info = defaultdict(dict)
    with_errors = []
    passed_with_errors = 0
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
                passed_with_errors += r['passed']
                with_errors.append(
                    ('_'.join(r['run_info']), r['passed'], r['task_id'], sample_idx, r['runtime'])
                )
                continue
            if r['had_timeout']:
                with_timeout += 1

            runtime = r.pop('runtime')
            passed = r.pop('passed')
            if r['task_id'] not in task_info[r['run_info']]:
                task_info[r['run_info']][r['task_id']] = {
                    'passed_runtimes': [], 'failed_runtimes': [], **r
                }

            if passed:
                task_info[r['run_info']][r['task_id']]['passed_runtimes'].append(runtime)
            else:
                task_info[r['run_info']][r['task_id']]['failed_runtimes'].append(runtime)

    logger.info(f"{len(with_errors)}/{len(to_time_check)} had errors")
    logger.info(f"{passed_with_errors} passed examples had errors")
    logger.info(f"{with_timeout}/{len(to_time_check)} timed out")
    return task_info, with_errors
