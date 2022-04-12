import logging
import re
from collections import defaultdict
from copy import deepcopy
import ast
from io import StringIO
from typing import Dict, Any
from contextlib import redirect_stdout
import astor

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

logger = logging.getLogger(__name__)


def mk_valid_syntax(code_str):
    code_lines = code_str.split('\n')
    failed = False
    while True:
        try:
            ast.parse('\n'.join(code_lines))
        except SyntaxError as e:
            if code_lines[e.lineno - 1].startswith('#'):
                failed = True
                break
            code_lines[e.lineno - 1] = f"# {code_lines[e.lineno - 1]}"

            continue
        break
    if all(l.startswith('#') for l in code_lines) or failed:
        return None
    return [line for line in code_lines if line.strip()]


class NotSupportedException(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)


class PrintTransformer(ast.NodeTransformer):
    def __init__(self, name):
        self.in_for_loop = False
        self.in_if_statement = False
        self.found_print = False
        self.var_num = 0
        self.name = name

    def __call__(self, code_snippet):
        try:
            tree = ast.parse(code_snippet)
        except SyntaxError:
            return [], code_snippet
        except NotSupportedException as e:
            logger.warning(f"Found a '{e.msg}', not supported")
            return [], code_snippet

        for b in tree.body:
            self.in_for_loop = False
            self.in_if_statement = False
            self.found_print = False
            added_out_var = False
            result = self.visit(b)  # type: ignore

            block = [result]

            if self.found_print:
                logger.debug(f"Found Print in {self.name}")
                if self.in_for_loop:
                    added_out_var = True
                    block = [ast.Assign(
                        targets=[ast.Name(id=f'out_{self.var_num}', ctx=ast.Store)],
                        value=ast.List(elts=[], ctx=ast.Load),
                        type_comment=None,
                    )] + block

                elif self.in_if_statement:
                    added_out_var = True
                    block = [ast.Assign(
                        targets=[ast.Name(id=f'out_{self.var_num}', ctx=ast.Store)],
                        value=ast.Constant(value=None, kind=None),
                        type_comment=None,
                    )] + block
            block = [astor.to_source(c).strip() for c in block]
            if added_out_var:
                yield block, f"out_{self.var_num}"
                self.var_num += 1
            else:
                yield [], '\n'.join(block)

    def visit_If(self, node):
        self.in_if_statement = True
        return self.generic_visit(node)

    def visit_For(self, node):
        self.in_for_loop = True
        return self.generic_visit(node)

    def visit_While(self, node):
        self.in_for_loop = True
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        raise NotSupportedException('Function')

    def visit_ClassDef(self, node):
        raise NotSupportedException('Class')

    def visit_Call(self, node):
        if not hasattr(node, 'func'):
            return self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)
        if node.func.id != 'print':
            return self.generic_visit(node)

        line_number = node.lineno
        end_line_number = node.end_lineno
        self.found_print = True
        if self.in_for_loop:
            node.func = ast.Attribute(
                lineno=line_number,
                end_lineno=end_line_number,
                value=ast.Name(
                    lineno=line_number,
                    end_lineno=end_line_number,
                    id=f'out_{self.var_num}',
                    ctx=ast.Load
                ),
                attr='append',
                ctx=ast.Load,
            )
        elif self.in_if_statement:
            return ast.Assign(
                lineno=line_number,
                end_lineno=end_line_number,
                targets=[ast.Name(
                    lineno=line_number,
                    end_lineno=end_line_number,
                    id=f'out_{self.var_num}',
                    ctx=ast.Store
                )],
                value=node.args[0],
                type_comment=None
            )
        else:
            node = ast.Expr(value=node.args[0])

        return node


class VariableTracer(ast.NodeVisitor):
    def __init__(self):
        self.defined = []
        self.used = []
        self.imported = []
        self.in_aug_assign = False

    def _clear(self):
        self.defined = []
        self.used = []

        self.imported = []
        self.in_aug_assign = False

    def __call__(self, code_str):

        tree = ast.parse(code_str)

        traced_bodies = []
        bodies = []
        imports = []
        import_names = []
        for body in tree.body:
            bodies.append(body)
            self._clear()
            self.visit(body)  # type:ignore
            if self.imported:
                imports.append(body)
                import_names.extend(self.imported)
            traced_bodies.append({
                'defined': deepcopy(self.defined),
                'used'   : deepcopy(self.used)
            })
        return bodies, traced_bodies, imports, import_names

    def _handle_import(self, node):
        for n in node.names:
            if n.asname is not None:
                self.imported.append(n.asname)
            else:
                self.imported.append(n.name)

    def visit_Import(self, node):
        return self._handle_import(node)

    def visit_ImportFrom(self, node):
        return self._handle_import(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Store) and not self.in_aug_assign:
            self.defined.append(node.id)
        else:
            self.used.append(node.id)

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self.in_aug_assign = True
        self.generic_visit(node)


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
            out.append({'context': block, 'code': cleaned_code, 'result': [output.rstrip()]})
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


def is_runnable(code):
    try:
        exec(code)
    except Exception as e:
        return e
    return None


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

                    code_error = is_runnable('\n'.join(to_run))

                    if code_error is None:
                        out.append({
                            'prior_context': snippet_context,
                            'context'      : block_context,
                            'code'         : block['code'],
                            'result'       : block['result'],
                            "name"         : name,
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
                    code_error = is_runnable('\n'.join(context + [line]))
                    if not code_error:
                        context.append(line)
    return out, context


def get_returned_values(code, context):
    to_run = context
    RESULT_VALUES = {}
    # to_run.append('RESULT_VALUES={}')
    for i, c in enumerate(code):
        to_run.append(f"RESULT_VALUES['OUT_{i}'] = {c}")
    exec('\n'.join(to_run))
    return {k: str(v) for k, v in RESULT_VALUES.items()}


def get_code_passes_test(code):
    passes_test = 0
    passes_str_test = 0
    samples_passed = []
    failed = []

    # This mess is to try and check if the code samples passes the mined tests

    for result in code:
        to_run = result['prior_context'] + result['context']
        for c, r in zip(result['code'], result['result']):
            to_run.append(f"assert {r} == {c}")

        try:
            exec('\n'.join(to_run))
            passes_test += 1
            samples_passed.append(result)
        except Exception:
            to_run = result['prior_context'] + result['context']
            for c, r in zip(result['code'], result['result']):

                result_str = r
                end_single_quote = r.startswith("'") and r.endswith("'")
                end_double_quote = r.startswith('"') and r.endswith('"')
                if not end_single_quote and not end_double_quote:
                    result_str = f"'{result_str}'"
                to_run.append(f"assert {result_str} ==str({c})")
            try:
                exec('\n'.join(to_run))
                passes_str_test += 1
                samples_passed.append(result)
            except Exception:

                result['testing_code'] = to_run
                result['actual_return'] = get_returned_values(
                    result['code'],
                    result['prior_context'] + result['context']
                )
                failed.append(result)
                continue

    return samples_passed, failed, passes_test, passes_str_test


def get_code_samples_from_tutorial(name, parsed_tutorial, global_context):
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
        passed, failed_tests, pass_test, pass_str = get_code_passes_test(out)
        logger.info(
            f"{name} has {pass_str + pass_test}/{len(out)} pass, "
            f"{pass_str}/{pass_str + pass_test} required string."
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
