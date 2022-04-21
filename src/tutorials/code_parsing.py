import json
import logging
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import ast
from multiprocessing import Manager, Process
from typing import List, Dict, Tuple, Union
import contextlib
import io
import astor
import numpy as np
from dataclasses import dataclass, field, asdict
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm
import signal
from datetime import datetime
from arrow import Arrow

from src.evaluation.execute import swallow_io, create_tempdir

from .node_visitors import (mk_valid_syntax, PrintTransformer, VariableTracer,
                            NotSupportedException)
from .saving import get_context_tags_for_code
from .code_sample import CodeSample, FailedCodeSample

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

GET_ROWS_NUMPY = re.compile(r'\[[^\[\]]+\]')
FIX_RAW_NUMPY = re.compile(r'( *[0-9\.-]+ *)')

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

logger = logging.getLogger(__name__)


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def get_context(snippet, prior_context, excluded_used=None):
    """
    Get the context needed from the snippet and the global context.
    """

    excluded_used = excluded_used or []
    visitor = VariableTracer()
    _, snippet_trace, _, _ = visitor(snippet)

    remaining_vars_needed = set()
    snippet_iter = iter(reversed(snippet_trace))
    while True:
        try:
            current_trace = next(snippet_iter)
        except StopIteration:
            break

        for d in current_trace['defined']:
            if d in remaining_vars_needed:
                remaining_vars_needed.remove(d)

        remaining_vars_needed.update(current_trace['used'])

    out = []
    context_body, context_traced, imports, import_names = visitor(prior_context)
    excluded_used += import_names
    for n in import_names:
        if n in remaining_vars_needed:
            remaining_vars_needed.remove(n)

    i = len(context_body) - 1
    while remaining_vars_needed and i >= 0:
        body = context_body[i]
        c_trace = context_traced[i]
        i -= 1
        found = list(filter(lambda var_def: var_def in remaining_vars_needed, c_trace['defined']))

        if found:
            out.append(body)
            for d in found:
                if d in remaining_vars_needed:
                    remaining_vars_needed.remove(d)
            remaining_vars_needed.update([u for u in c_trace['used'] if u not in import_names])
        else:
            found_used = []
            for u in c_trace['used']:
                if u not in excluded_used and u in remaining_vars_needed:
                    found_used.append(u)
            if found_used:
                out.append(body)
                remaining_vars_needed.update([u for u in c_trace['used'] if u not in excluded_used])

    out = list(map(lambda o: astor.to_source(o).strip(), reversed(out)))
    imports = list(set(map(lambda o: astor.to_source(o).strip(), imports)))
    return out, imports


def get_snippets(name, code_str) -> List[Dict]:
    """
    Get the list of snippet dictionaries from a code string
    """
    block = []
    out = []
    output = ''
    first_start = 0
    code_str = code_str.replace(';', '\n>>>')

    for match in GET_CODE_BLOCK.finditer(code_str):
        leading_space, snippet, output = match.groups()

        # Set output to empty string if None to reduce number of annoying checks
        if output is None:
            output = ''

        num_leading_space = len(leading_space)
        code = []
        for i, line in enumerate(snippet.split('\n')):
            if i == 0:
                code.append(line[max(num_leading_space - 1, 0):])
            else:
                assert line.startswith('...')
                code.append(line[num_leading_space + 3:])
        code = '\n'.join(code)
        try:
            ast.parse(code)
        except SyntaxError:
            tmp_code = code
            result_split = output.split('\n')
            last_idx = 0
            found = False
            while last_idx < len(result_split):
                tmp_code += f'\n{result_split[last_idx]}'
                try:
                    ast.parse(tmp_code)
                    found = True
                    break
                except SyntaxError:
                    last_idx += 1
            if found:
                output = '\n'.join(result_split[last_idx + 1:])
                code = tmp_code

        if output is not None and output.strip():
            visitor = PrintTransformer(name)
            cleaned_code = []
            try:
                transformed_code = visitor(code)
                for new_context, cleaned_snip in transformed_code:
                    block.extend(new_context)
                    cleaned_code.append(cleaned_snip)

            except NotSupportedException:
                cleaned_code = []
            if cleaned_code:
                if len(cleaned_code) != 1:
                    assert len(cleaned_code) == 1
                if not block and out:
                    out[-1]['code'].extend(cleaned_code)
                    out[-1]['result'].append(output.strip())
                else:
                    out.append(
                        {
                            'context'       : block,
                            'code'          : cleaned_code,
                            'result'        : [output.rstrip()],
                            'start_char_idx': first_start
                        })
                    block = []
                    first_start = match.span()[1] + 1
        else:
            block.append(code)
    if block:
        out.append({
            'context'       : block,
            'code'          : [],
            'result'        : [output.rstrip()] if output is not None and output.strip() else [],
            'start_char_idx': first_start
        })
    return out


def does_code_raise_errors(code):
    """
    Executed the code and see if it raises errors. Wrapper for the main
    function so that we can call it from another process.
    """
    result = None
    with create_tempdir():
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            stdout_f = io.StringIO()
            stderr_f = io.StringIO()
            with contextlib.redirect_stdout(stdout_f):
                with contextlib.redirect_stderr(stderr_f):
                    plt.show = lambda: None
                    # sys.stdout.write = lambda *args, **kwargs: None
                    exec(code)
        except AssertionError:
            result = 'AssertionError'
        except Exception as e:
            try:
                result = f"{type(e).__name__}: {str(e)}"
            except Exception as e:
                try:
                    result = f"{type(e).__name__}"
                except:
                    result = "BAD ERROR OCCUR"
    signal.alarm(0)
    return result


def get_code_from_content(
        name,
        content,
        context,
        global_context,
        path=None,
        exclude_idx=None
) -> Tuple[List[Union[CodeSample, FailedCodeSample]], List[str]]:
    path = path or []
    out = []
    name_str = '-'.join(name)
    exclude_idx = exclude_idx or []

    for i, c in enumerate(content):
        if c['tag'] == 'section':
            child_out, context = get_code_from_content(
                name + [c['title']],
                c['content'],
                context,
                global_context,
                path + [i]
            )
            out.extend(child_out)
        if c['tag'] != 'code' or '>>>' not in c['text']:
            continue
        if c['idx'] in exclude_idx:
            continue

        for snip_num, block in enumerate(get_snippets(name_str, c['text'])):
            block_context = mk_valid_syntax('\n'.join(block['context']))

            if block['result']:
                valid_code = mk_valid_syntax('\n'.join(block['code']))

                if valid_code is not None and block_context is not None:
                    snippet = '\n'.join(block_context + valid_code)
                    snippet_context, imports = get_context(snippet, '\n'.join(context))

                    for line in imports:
                        if line not in global_context and line not in snippet_context:
                            snippet_context = [line] + snippet_context

                    run_context = list(deepcopy(global_context)) + snippet_context
                    to_run = run_context + block_context + valid_code
                    code_error = does_code_raise_errors('\n'.join(to_run))

                    if code_error is None:
                        out.append(CodeSample(
                            section_path=name,
                            idx=c['idx'],
                            snippet_idx=snip_num,
                            body_code=block_context,
                            start_char_idx=block['start_char_idx'],
                            return_code=block['code'],
                            expected_result=block['result'],
                            context=run_context
                        ))
                    else:
                        out.append(FailedCodeSample(
                            section_name=name,
                            idx=c['idx'],
                            snippet_idx=snip_num,
                            error=str(code_error),
                            code=to_run
                        ))
            if block_context is not None:
                for line in block_context:
                    if 'plt.' in line:
                        continue
                    code_error = does_code_raise_errors('\n'.join(context + [line]))
                    if not code_error:
                        context.append(line)
    return out, context


def convert_raw_result_to_supported_type(raw_result_str, target_type):
    if target_type == bytes:
        return False, f"b'{raw_result_str}'"
    if target_type == bool:
        if raw_result_str in ['True', 'False']:
            return False, raw_result_str
        return False, f'bool({raw_result_str})'
    if target_type == np.ndarray:
        if '[' in raw_result_str and ']' in raw_result_str:
            array_rows = GET_ROWS_NUMPY.findall(raw_result_str)
            row_vals = [','.join([v.strip() for v in FIX_RAW_NUMPY.findall(row)])
                        for row in array_rows]
            raw_result_str = '[' + ','.join(map(lambda v: f'[{v}]', row_vals)) + ']'

        if 'array' in raw_result_str:
            raw_result_str = raw_result_str.replace('array', 'numpy.array')
            return False, raw_result_str
        else:
            return False, f"numpy.array({raw_result_str})"
    # Check for primitive types
    if target_type in [int, float, complex]:
        return False, raw_result_str

    end_single_quote = raw_result_str.startswith("'") and raw_result_str.endswith("'")
    end_double_quote = raw_result_str.startswith('"') and raw_result_str.endswith('"')
    if not end_single_quote and not end_double_quote:
        raw_result_str = f"'{raw_result_str}'"

    return True, raw_result_str


def get_returned_values(code, context):
    RESULT_VALUES = {}
    to_run = [*context]
    for i, c in enumerate(code):
        to_run.append(f"RESULT_VALUES['OUT_{i}'] = {c}")
    with create_tempdir():

        stdout_f = io.StringIO()
        stderr_f = io.StringIO()
        plt.show = lambda: None
        with contextlib.redirect_stdout(stdout_f):
            with contextlib.redirect_stderr(stderr_f):
                exec('\n'.join(to_run))
    out = {}
    for k, v in RESULT_VALUES.items():
        out[k] = {'val': repr(v), 'type': type(v) if v is not None else None}
    out['STDOUT'] = stdout_f.getvalue()
    out['STDERR'] = stderr_f.getvalue()
    stdout_f.close()
    stderr_f.close()
    return out


def proc_get_returned_values(code, context):
    manager = mp.Manager()

    d = manager.dict()
    p = Process(target=get_returned_values, args=(code, context, d))
    p.start()
    p.join(timeout=10)
    if p.is_alive():
        p.kill()
        raise TimeoutError('Timeout')
    return d


def get_code_passes_test(domain: str, file: str, sample: Dict):
    sample = CodeSample(**sample)
    try:
        actual_returned_values = get_returned_values(
            sample.return_code,
            sample.context + sample.body_code
        )
        sample.actual_returned = deepcopy(actual_returned_values)
        found_none = False
        for k, v in actual_returned_values.items():
            if k in ['STDOUT', 'STDERR']:
                continue
            if v['type'] is None:
                found_none = True

                sample.actual_returned[k]['type'] = None
            else:
                sample.actual_returned[k]['type'] = v['type'].__name__
        if found_none:
            sample.errors.append("NoneType")
            return {'file': file, 'sample': sample, 'passed': False}

    except Exception as e:
        sample.errors.append(str(e))
        return {'file': file, 'sample': sample, 'passed': False}

    if actual_returned_values.get('STDOUT', '').strip():
        sample.errors.append("STDOUT not empty")
        return {'file': file, 'sample': sample, 'passed': False}
    try:
        deterministic_check_returned = get_returned_values(
            sample.return_code,
            sample.context + sample.body_code
        )
    except Exception as e:
        sample.errors.append(f"DETERMINISTIC_CHECK: {str(e)}")
        return {'file': file, 'sample': sample, 'passed': False}

    if set(deterministic_check_returned) != set(actual_returned_values):
        sample.errors.append("STDOUT not empty")
        return {'file': file, 'sample': sample, 'passed': False}

    failed_deterministic_check = False
    for k, v in deterministic_check_returned.items():
        if k == 'STDOUT' or k == 'STDERR':
            continue

        if domain == 'arrow' and isinstance(v, Arrow):
            v = v.datetime

        if isinstance(v, np.ndarray):
            has_correct_value = np.isclose(v, actual_returned_values[k]).all()
        elif isinstance(v, datetime):
            actual = actual_returned_values[k]
            if isinstance(actual, Arrow):
                actual = actual.datetime
            if not isinstance(actual, datetime):
                has_correct_value = False
            else:
                has_correct_value = True
                if v.year != actual.year:
                    has_correct_value = False
                if v.month != actual.month:
                    has_correct_value = False
                if v.day != actual.day:
                    has_correct_value = False
        else:
            has_correct_value = repr(v) == repr(actual_returned_values[k])

        if not has_correct_value:
            failed_deterministic_check = True
            sample.errors.append(
                f"Failed deterministic check {k}:  "
                f"{v}!= {actual_returned_values[k]}"
            )
            break

    if failed_deterministic_check:
        return {'file': file, 'sample': sample, 'passed': False}
    return {'file': file, 'sample': sample, 'passed': True}


def mp_get_code_passes_tests_wrapper(args):
    return get_code_passes_test(**args)


def get_code_samples_from_tutorial(name, parsed_tutorial, global_context, fixes_by_section):
    logger.debug(f"{len(parsed_tutorial)} top level section(s) for {name}")
    logger.debug(f"The global context is {global_context}")

    out = []
    failed = defaultdict(list)
    for i, section in enumerate(parsed_tutorial):
        logger.debug(f"Getting code from top level section '{section['title']}'")
        found_code = []
        results, _ = get_code_from_content(
            [section['title']],
            section['content'],
            deepcopy(global_context),
            tuple(global_context),
            [i],
            exclude_idx=fixes_by_section.get('exclude_idx', [])

        )
        for sample in results:
            if isinstance(sample, FailedCodeSample):
                failed['/'.join(sample.section_name)].append(sample.to_dict())
            else:
                found_code.append(sample)
        logger.debug(f"Found {len(found_code)} code sample(s)")
        out.extend(
            found_code
        )
    return out, failed


def unravel_code_list_into_tree(code):
    def make_tree(data, current_level, path):
        current, *rem = path
        if not rem:
            if isinstance(current_level, list):
                return {'/': current_level, current: [data]}
            if current not in current_level:
                current_level[current] = [data]
            else:
                if isinstance(current_level[current], dict):
                    current_level[current]['/'].append(data)
                else:
                    current_level[current].append(data)
            return current_level

        if isinstance(current_level, list):
            current_level = {'/': current_level}

        current_level[current] = make_tree(
            data,
            current_level.get(current, {}),
            rem
        )
        return current_level

    out = {}
    for code_dict in code:
        p = code_dict.pop('name')
        out = make_tree(code_dict, out, p)
    return out


def parse_domain_path(
        domain_path,
        global_context_dict,
        fixes_by_section,
        out_dir,
        parsed_idx_offset,
        num_prior_for_ctx,
        url_dict,
        annotations
):
    domain_context_cfg = global_context_dict.get(domain_path.stem, {})
    domain_context = domain_context_cfg.get('global', [])
    domain_fixes = fixes_by_section.get(domain_path.stem, {})

    parse_fails_by_file = {}
    parsed_by_file = {}
    num_failed = 0
    num_runnable = 0
    files = list(domain_path.glob('*'))
    stem_to_path = {}

    for file in tqdm(files, desc=f'Parsing {domain_path.stem}'):
        file_context = []
        if domain_context_cfg:
            file_context = domain_context_cfg['files'].get(file.stem, [])
        tutorial_fixes = {}
        if domain_fixes:
            tutorial_fixes = domain_fixes.get(file.stem, {})
        stem_to_path[file.stem] = file
        parsed_file = json.loads(file.read_text())
        parsed, failed = get_code_samples_from_tutorial(
            file.stem,
            parsed_file,
            domain_context + file_context,
            tutorial_fixes
        )
        num_failed += sum(map(len, failed.values()))
        num_runnable += len(parsed)
        parse_fails_by_file[file.stem] = failed
        parsed_by_file[file.stem] = parsed
        logger.debug(f"{file} had {num_failed} failures")
        logger.debug(f"{file} had {num_runnable} code snippets")

    passed_by_file = defaultdict(list)
    failed_by_file = defaultdict(lambda: defaultdict(list))

    mp_test_args = []

    bad_result = re.compile(r'<[^\s]+ [^>]+(?:\.\.\.)>')

    total_passed_tests = 0
    for file, samples in parsed_by_file.items():
        tutorial_fixes = {}
        if domain_fixes:
            tutorial_fixes = domain_fixes.get(file, {})
        override_all = tutorial_fixes.get('override_all', False)
        for sample in samples:
            if (
                    any(bad_result.search(v) is not None for v in sample.expected_result)
                    and domain_path.stem not in ['delorean', 'arrow']
            ):
                sample.errors.append('BAD EXPECTED')
                failed_name = '/'.join(sample.section_path)
                failed_by_file[failed_name][sample.idx].append(asdict(sample))
                continue

            sample_fixes = tutorial_fixes.get('/'.join(sample.section_path), {})
            sample_overrides = sample_fixes.get('overrides', [])

            # Some tests will fail because of bad issues, instead override them
            # because we know they are correct.
            if (
                    sample.idx in sample_overrides
                    or override_all
            ):
                passed_by_file[file].append(sample)
                total_passed_tests += 1
                continue

            mp_test_args.append(
                {'domain': domain_path.stem, 'file': file, 'sample': asdict(sample)}
            )

    logger.debug(f"{domain_path.stem} has {len(mp_test_args)} programs to check")
    total_failed_tests = 0

    with mp.Pool(4) as pool:
        results = list(tqdm(
            pool.imap_unordered(mp_get_code_passes_tests_wrapper, mp_test_args),
            total=len(mp_test_args),
            desc=f"Testing {domain_path.stem} code"
        ))
        for result in results:
            if result['passed']:
                passed_by_file[result['file']].append(result['sample'])
                total_passed_tests += 1
            else:
                failed_sample = result['sample']
                failed_name = '/'.join(failed_sample.section_path)
                failed_by_file[failed_name][failed_sample.idx].append(
                    asdict(failed_sample)
                )
                total_failed_tests += 1

    parsed_files = {}

    for file, passed in tqdm(passed_by_file.items(), desc=f"Saving {domain_path.stem}"):
        parsed = json.loads(stem_to_path[file].read_text())
        combined = get_context_tags_for_code(
            domain_path.stem,
            file,
            parsed,
            passed,
            num_context_to_keep=num_prior_for_ctx,
            url=url_dict[file],
            annotation_mode=annotations
        )
        combined['idx'] = parsed_idx_offset + len(parsed_files)
        parsed_files[combined['name']] = combined

    fail_dir = out_dir.joinpath(f'{domain_path.stem}')
    fail_dir.mkdir()
    with fail_dir.joinpath('parse.json').open('w') as f:
        json.dump(parse_fails_by_file, f, indent=True)

    with fail_dir.joinpath('test.json').open('w') as f:
        json.dump(failed_by_file, f, indent=True)

    return parsed_files, num_runnable, total_passed_tests, num_failed
