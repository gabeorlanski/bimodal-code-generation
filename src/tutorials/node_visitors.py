import logging
import re
from copy import deepcopy
import ast
from typing import Dict, Any
import astor

OLD_PRINT_FIX = re.compile(r'print (.+)$', flags=re.DOTALL)

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
            test_snippet = OLD_PRINT_FIX.sub(r'print(\1)', code_snippet)
            try:
                tree = ast.parse(test_snippet)
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
