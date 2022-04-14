import logging
import re
from collections import defaultdict
from copy import deepcopy
import ast
import astor
import numpy as np

from src.evaluation.execute import swallow_io, create_tempdir

from .node_visitors import (mk_valid_syntax, PrintTransformer, VariableTracer)

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

GET_ROWS_NUMPY = re.compile(r'\[[^\[\]]+\]')
FIX_RAW_NUMPY = re.compile(r'( *[0-9\.-]+ *)')

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

logger = logging.getLogger(__name__)


def get_context(snippet, global_context, excluded_used=None):
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


def get_snippets(name, code_str):
    block = []
    out = []
    output = ''

    for leading_space, snippet, output in GET_CODE_BLOCK.findall(code_str):
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

        if output.strip():
            visitor = PrintTransformer(name)
            cleaned_code = []
            for new_context, cleaned_snip in visitor(code):
                block.extend(new_context)
                cleaned_code.append(cleaned_snip)
            if cleaned_code:

                if len(cleaned_code) != 1:
                    assert len(cleaned_code) == 1
                if not block and out:
                    out[-1]['code'].extend(cleaned_code)
                    out[-1]['result'].append(output.strip())
                else:
                    out.append(
                        {'context': block, 'code': cleaned_code, 'result': [output.rstrip()]})
                    block = []
        else:
            block.append(code)
    if block:
        out.append({
            'context': block,
            'code'   : [],
            'result' : [output.rstrip()] if output.strip() else []
        })
    return out


def does_code_raise_errors(code):
    result = None
    with create_tempdir():
        try:
            with swallow_io():
                exec(code)
        except Exception as e:
            result = str(e)
    return result


def get_code_from_content(name, content, context, global_context, path=None):
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
                # if any('traceback' in r.lower() for r in block['result']):

                if valid_code is not None and block_context is not None:
                    snippet = '\n'.join(block_context + valid_code)
                    snippet_context = get_context(snippet, '\n'.join(context))
                    to_run = snippet_context + block_context + valid_code
                    if to_run[:len(global_context)] != list(global_context):
                        to_run = [*global_context, *to_run]
                        snippet_context = [*global_context, *snippet_context]

                    code_error = does_code_raise_errors('\n'.join(to_run))

                    if code_error is None:
                        out.append({
                            'prior_context': snippet_context,
                            'context'      : block_context,
                            'code'         : block['code'],
                            'result'       : block['result'],
                            "name"         : name_str,
                            'idx'          : c['idx'],
                            'snip_idx'     : snip_num
                        })
                    else:
                        out.append({
                            "result"  : "FAILED", "name": name, 'idx': c['idx'],
                            "error"   : str(code_error),
                            'code'    : to_run,
                            'snip_idx': snip_num
                        })
            if block_context is not None:
                for line in block_context:
                    code_error = does_code_raise_errors('\n'.join(context + [line]))
                    if not code_error:
                        context.append(line)
    return out, context


def convert_raw_result_to_supported_type(raw_result_str, target_type):
    if target_type == bytes:
        return f"b'{raw_result_str}'"
    if target_type == bool:
        if raw_result_str in ['True', 'False']:
            return raw_result_str
        return f'bool({raw_result_str})'
    if target_type == np.ndarray:
        if '[' in raw_result_str and ']' in raw_result_str:
            array_rows = GET_ROWS_NUMPY.findall(raw_result_str)
            row_vals = [','.join([v.strip() for v in FIX_RAW_NUMPY.findall(row)])
                        for row in array_rows]
            raw_result_str = '[' + ','.join(map(lambda v: f'[{v}]', row_vals)) + ']'

        if 'array' in raw_result_str:
            raw_result_str = raw_result_str.replace('array', 'numpy.array')
            return raw_result_str
        else:
            return f"numpy.array({raw_result_str})"

    end_single_quote = raw_result_str.startswith("'") and raw_result_str.endswith("'")
    end_double_quote = raw_result_str.startswith('"') and raw_result_str.endswith('"')
    if not end_single_quote and not end_double_quote:
        raw_result_str = f"'{raw_result_str}'"
    return raw_result_str


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


def get_code_passes_test(code, fixes_by_section, override_all=False):
    passes_test = 0
    passes_type_conversion = 0
    samples_passed = []
    failed = []

    # This mess is to try and check if the code samples passes the mined tests

    for parsed_code_dict in code:

        # Some tests will fail because of bad issues, instead override them
        # because we know they are correct.
        if (
                (parsed_code_dict['idx'] in fixes_by_section.get(parsed_code_dict['name'], {}).get('overrides', []))
                or override_all
        ):
            samples_passed.append(parsed_code_dict)
            passes_test += 1
            continue

        original_to_run = parsed_code_dict['prior_context'] + parsed_code_dict['context']
        for c, r in zip(parsed_code_dict['code'], parsed_code_dict['result']):
            original_to_run.append(f"assert {c} == {r}")

        parsed_code_dict['errors'] = []
        exec_errors = does_code_raise_errors('\n'.join(original_to_run))
        if exec_errors is None:
            parsed_code_dict.pop('errors')
            samples_passed.append(parsed_code_dict)
            passes_test += 1
            continue
        parsed_code_dict['errors'].append(exec_errors)
        try:
            actual_returned_values = get_returned_values(
                parsed_code_dict['code'],
                parsed_code_dict['prior_context'] + parsed_code_dict['context']
            )
        except Exception as e:
            parsed_code_dict['errors'].append(str(e))
            failed.append(parsed_code_dict)
            continue

        # Try with converting the output and expected result to a string.
        type_conversion_to_run = parsed_code_dict['prior_context'] + parsed_code_dict['context']
        for (i, c), r in zip(enumerate(parsed_code_dict['code']), parsed_code_dict['result']):
            aligned_out = actual_returned_values[f"OUT_{i}"]

            result_str = convert_raw_result_to_supported_type(
                r,
                aligned_out['type']
            )
            if aligned_out['type'] == np.ndarray:
                if not any('import numpy' == line for line in type_conversion_to_run):
                    type_conversion_to_run = ['import numpy'] + type_conversion_to_run
                type_conversion_to_run.append(f"assert numpy.isclose({c},{result_str}).all()")
            else:
                type_conversion_to_run.append(f"assert {c} == {result_str}")

        exec_errors = does_code_raise_errors('\n'.join(type_conversion_to_run))
        if exec_errors is None:
            samples_passed.append(parsed_code_dict)
            passes_type_conversion += 1
            continue
        parsed_code_dict['errors'].append(exec_errors)
        parsed_code_dict['testing_code'] = type_conversion_to_run[-len(actual_returned_values):]
        for k in actual_returned_values.keys():
            actual_returned_values[k]['type'] = actual_returned_values[k]['type'].__name__
        parsed_code_dict['actual_returned'] = actual_returned_values
        parsed_code_dict['error'] = exec_errors
        failed.append(parsed_code_dict)

    return samples_passed, failed, passes_test, passes_type_conversion


def get_code_samples_from_tutorial(name, parsed_tutorial, global_context, fixes_by_section):
    logger.info(f"{len(parsed_tutorial)} top level section(s) for {name}")
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
        for result in results:
            if result['result'] == 'FAILED':
                result.pop('result')
                result_title = result.pop('name', 'NA')
                failed['/'.join(result_title)].append(result)
            else:
                found_code.append(result)
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
        logger.info(
            f"{name} has {pass_str + pass_test}/{len(out)} pass, "
            f"{pass_str}/{pass_str + pass_test} required type conversion."
        )
    return failed, out, passed, failed_tests


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
