import json
import logging
import re
import sys
from collections import defaultdict
from copy import deepcopy
import ast
from typing import List, Dict, Tuple, Union

import astor
import numpy as np
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.evaluation.execute import swallow_io, create_tempdir

from .node_visitors import (mk_valid_syntax, PrintTransformer, VariableTracer)
from .saving import combine_code_samples_with_parsed
from .code_sample import CodeSample, FailedCodeSample

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

GET_ROWS_NUMPY = re.compile(r'\[[^\[\]]+\]')
FIX_RAW_NUMPY = re.compile(r'( *[0-9\.-]+ *)')

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

logger = logging.getLogger(__name__)


def get_context(snippet, global_context, excluded_used=None):
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
    context_body, context_traced, imports, import_names = visitor(global_context)
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

    out = imports + list(reversed(out))
    return list(map(lambda o: astor.to_source(o).strip(), out))


def get_snippets(name, code_str) -> List[Dict]:
    """
    Get the list of snippet dictionaries from a code string
    """
    block = []
    out = []
    output = ''
    first_start = 0

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
            for new_context, cleaned_snip in visitor(code):
                block.extend(new_context)
                cleaned_code.append(cleaned_snip)
            if cleaned_code:

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
    Executed the code and see if it raises errors
    """
    result = None
    orig_write = sys.stdout.write
    with create_tempdir():
        orig_plt_fn = plt.show
        try:
            with swallow_io():
                plt.show = lambda: None
                print = lambda f: None
                sys.stdout.write = lambda *args, **kwargs: None
                exec(code)
        except AssertionError:
            result = 'AssertionError'
        except Exception as e:
            result = f"{type(e).__name__}: {str(e)}"
    sys.stdout.write = orig_write
    return result


def get_code_from_content(
        name,
        content,
        context,
        global_context,
        path=None
) -> Tuple[List[Union[CodeSample, FailedCodeSample]], List[str]]:
    path = path or []
    out = []
    name_str = '-'.join(name)

    for i, c in enumerate(content):
        if c['tag'] == 'section':
            child_out, context = get_code_from_content(
                name + [c['title']],
                c['content'],
                context,
                deepcopy(global_context),
                path + [i]
            )
            out.extend(child_out)
        if c['tag'] != 'code' or '>>>' not in c['text']:
            continue

        for snip_num, block in enumerate(get_snippets(name_str, c['text'])):
            block_context = mk_valid_syntax('\n'.join(block['context']))

            if block['result']:
                valid_code = mk_valid_syntax('\n'.join(block['code']))

                if valid_code is not None and block_context is not None:
                    snippet = '\n'.join(block_context + valid_code)
                    snippet_context = get_context(snippet, '\n'.join(context))
                    to_run = snippet_context + block_context + valid_code
                    if to_run[:len(global_context)] != list(global_context):
                        to_run = [*global_context, *to_run]
                        snippet_context = [*global_context, *snippet_context]

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
                            context=snippet_context
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
    to_run = context
    RESULT_VALUES = {}
    # to_run.append('RESULT_VALUES={}')
    for i, c in enumerate(code):
        to_run.append(f"RESULT_VALUES['OUT_{i}'] = {c}")
    exec('\n'.join(to_run))

    out = {}
    for k, v in RESULT_VALUES.items():
        out[k] = {'val': str(v), 'type': type(v)}

    return out


def get_code_passes_test(code: List[CodeSample], fixes_by_section, override_all=False):
    passes_test = 0
    passes_type_conversion = 0
    samples_passed = []
    failed = []

    # This mess is to try and check if the code samples passes the mined tests

    for sample in code:

        sample_fixes = fixes_by_section.get('/'.join(sample.section_path), {})
        sample_overrides = sample_fixes.get('overrides', [])

        # Some tests will fail because of bad issues, instead override them
        # because we know they are correct.
        if (
                sample.idx in sample_overrides
                or override_all
        ):
            samples_passed.append(sample)
            passes_test += 1
            continue

        original_to_run = sample.context + sample.body_code
        for c, r in sample.aligned_returns_and_results():
            original_to_run.append(f"assert {c} == {r}")

        exec_errors = does_code_raise_errors('\n'.join(original_to_run))
        if exec_errors is None:
            samples_passed.append(sample)
            passes_test += 1
            continue
        sample.errors.append(exec_errors)
        try:
            actual_returned_values = get_returned_values(
                sample.return_code,
                sample.context + sample.body_code
            )
            sample.actual_returned = deepcopy(actual_returned_values)
            for k in actual_returned_values.keys():
                sample.actual_returned[k]['type'] = actual_returned_values[k]['type'].__name__
        except Exception as e:
            sample.errors.append(str(e))
            failed.append(sample)
            continue

        # Try with converting the output and expected result to a string.
        type_conversion_to_run = sample.context + sample.body_code
        testing_code = []
        for i, (c, r) in enumerate(sample.aligned_returns_and_results()):
            aligned_out = actual_returned_values[f"OUT_{i}"]

            should_convert_to_str, result_str = convert_raw_result_to_supported_type(
                r,
                aligned_out['type']
            )

            # Some types need special code to convert them. This does that.
            if aligned_out['type'] == np.ndarray:
                if not any('import numpy' == line for line in type_conversion_to_run):
                    type_conversion_to_run = ['import numpy'] + type_conversion_to_run
                testing_code.append(f"assert numpy.isclose({c},{result_str}).all()")
            elif 'class' in aligned_out['val']:
                type_str_name = aligned_out['val'].split('\'')[1]

                testing_code.append(
                    f"assert repr({c}).split(\"\'\")[1] == \"{type_str_name}\""
                )
            else:

                remove_addr = r'(?<=at) [^>]+(?=>)'
                if re.search(remove_addr, result_str):
                    result_str = re.sub(remove_addr, '', result_str)
                    testing_code.append(
                        f"v = re.sub(r'{remove_addr}','',repr({c}))"
                    )
                    testing_code.append(
                        f"assert v == {result_str}"
                    )
                else:
                    if should_convert_to_str:
                        testing_code.append(f"assert repr({c}) == {result_str}")
                    else:
                        testing_code.append(f"assert {c} == {result_str}")
        type_conversion_to_run.extend(testing_code)

        exec_errors = does_code_raise_errors('\n'.join(type_conversion_to_run))
        if exec_errors is None:
            sample.errors = []
            sample.testing_code = testing_code
            samples_passed.append(sample)
            passes_type_conversion += 1
            continue
        sample.errors.append(exec_errors)
        sample.testing_code = testing_code
        sample.errors.append(exec_errors)
        failed.append(sample)

    return samples_passed, failed, passes_test, passes_type_conversion


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
            [i]
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
    passed = []
    failed_tests = []
    if out:
        if fixes_by_section.get('override_all'):
            logger.warning(f"{name} Has override all enabled")
        passed, failed_tests, pass_test, pass_str = get_code_passes_test(
            out,
            fixes_by_section,
            override_all=fixes_by_section.get('override_all', False)
        )
        logger.debug(
            f"{name} has {pass_str + pass_test}/{len(out)} pass, "
            f"{pass_str}/{pass_str + pass_test} required type conversion."
        )
    return failed, passed, failed_tests


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
        parsed_idx_offset
):
    logger.info(f"Parsing {domain_path}")
    domain_context_cfg = global_context_dict.get(domain_path.stem, {})
    domain_context = domain_context_cfg.get('global', [])
    domain_fixes = fixes_by_section.get(domain_path.stem, {})

    total_num_runnable = 0
    total_num_fails = 0
    failures = {}
    domain_passed = {}
    domain_fail_tests = {}
    parsed_files = []
    files = list(domain_path.glob('*'))

    for file in tqdm(files, desc=f'Parsing {domain_path.stem}'):
        file_context = []
        if domain_context_cfg:
            file_context = domain_context_cfg['files'].get(file.stem, [])
        tutorial_fixes = {}
        if domain_fixes:
            tutorial_fixes = domain_fixes.get(file.stem, {})
        parsed = json.loads(file.read_text())
        fail, passed, fail_tests = get_code_samples_from_tutorial(
            file.stem,
            parsed,
            domain_context + file_context,
            tutorial_fixes
        )
        domain_passed[file.stem] = list(map(asdict, passed))

        if passed:
            combined = combine_code_samples_with_parsed(
                domain_path.stem,
                file.stem,
                parsed,
                passed
            )
            combined['idx'] = parsed_idx_offset + len(parsed_files)
            parsed_files.append(combined)

        fail_tests_by_idx = {}
        for failed_sample in fail_tests:
            failed_name = '/'.join(failed_sample.section_path)
            if failed_name not in fail_tests_by_idx:
                fail_tests_by_idx[failed_name] = {}
            if failed_sample.idx not in fail_tests_by_idx[failed_name]:
                fail_tests_by_idx[failed_name][failed_sample.idx] = []
            fail_tests_by_idx[failed_name][failed_sample.idx].append(asdict(failed_sample))

        for name in fail_tests_by_idx:
            for idx in fail_tests_by_idx[name]:
                fail_tests_by_idx[name][idx] = list(sorted(
                    fail_tests_by_idx[name][idx],
                    key=lambda snip: snip['snippet_idx']
                ))

        num_runnable = len(passed) + len(fail_tests)
        domain_fail_tests[file.stem] = fail_tests_by_idx
        failures[file.stem] = fail
        total_num_runnable += num_runnable
        total_num_fails += sum(map(len, fail.values()))
        logger.debug(f"{file} had {sum(map(len, fail.values()))} failures")
        logger.debug(f"{file} had {num_runnable} code snippets")

    fail_dir = out_dir.joinpath(f'{domain_path.stem}_fails')
    fail_dir.mkdir()
    with fail_dir.joinpath('parse.json').open('w') as f:
        json.dump(failures, f, indent=True)

    with fail_dir.joinpath('test.json').open('w') as f:
        json.dump(domain_fail_tests, f, indent=True)

    return parsed_files, domain_passed, total_num_runnable, total_num_fails
