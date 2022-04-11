import logging
import re
from copy import deepcopy
import ast
import astor

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

logger = logging.getLogger(__name__)


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


def get_snippets(name, code_str):
    context = []
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


def get_code_from_parsed_tutorial(domain, name, parsed_tutorial):
    total_updated = 0
    for section_num, section in enumerate(parsed_tutorial):
        if section['tag'] != 'code' or '>>>' not in section['text']:
            continue
        section['snippets'] = get_snippets(f"{domain}-{name}-{section['id']}", section['text'])
        total_updated += 1
        parsed_tutorial[section_num] = section

    logger.debug(f"{name} had {total_updated} total code snippets")
    return total_updated, parsed_tutorial
