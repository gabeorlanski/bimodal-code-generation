import ast
import logging

import astor

from .node_visitors import VariableTracer

logger = logging.getLogger(__name__)


def prepare_sample_with_func(sample):
    visitor = VariableTracer()
    lines, snippet_trace, target_imports, _ = visitor(
        sample['body_code'] + '\n' + '\n'.join(sample['return_code'])
    )

    remaining_vars_needed = []
    snippet_iter = iter(reversed(snippet_trace))
    while True:
        try:
            current_trace = next(snippet_iter)
        except StopIteration:
            break

        for d in current_trace['defined']:
            if d in remaining_vars_needed:
                remaining_vars_needed.remove(d)

        for v in current_trace['used']:
            if v not in remaining_vars_needed:
                remaining_vars_needed.append(v)

    context_lines, _, imports, _ = visitor(sample['context'])
    context_lines = [
        line for line in context_lines if
        not isinstance(line, (ast.Import, ast.ImportFrom))
    ]

    context_return_stmt = []
    target_inputs = []
    for var in sorted(remaining_vars_needed):
        context_return_stmt.append(ast.keyword(arg=var, value=ast.Name(id=var, ctx=ast.Load)))
        target_inputs.append(
            ast.arg(arg=var, annotation=None, type_comment=None)
        )

    context = ast.FunctionDef(
        name='get_inputs_from_context',
        args=ast.arguments(
            # These default arguments are needed otherwise the to_source will not work
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
            defaults=[]
        ),
        body=context_lines + [ast.Return(value=ast.Call(
            func=ast.Name(id='dict', ctx=ast.Load),
            args=[],
            keywords=context_return_stmt
        ))],
        decorator_list=[]
    )
    context = astor.to_source(context).strip()

    target_body = ast.parse(sample['body_code']).body
    target_return_values = []
    for v in sample['return_code']:
        tree = ast.parse(v)
        assert len(tree.body) == 1
        tree = tree.body[0]
        if isinstance(tree, ast.Expr):
            target_return_values.append(tree.value)
        else:
            target_return_values.append(tree)
    target_return_values = ast.Tuple(
        elts=target_return_values,
        ctx=ast.Load
    )
    target = ast.FunctionDef(
        name='solution',
        args=ast.arguments(
            # These default arguments are needed otherwise the to_source will not work
            posonlyargs=[],
            args=target_inputs,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        ),
        body=target_body + [ast.Return(value=target_return_values)],
        decorator_list=[]
    )
    target = astor.to_source(target).strip()
    out = {
        k: v for k, v in sample.items() if
        k in ['idx', 'snippet_idx', 'start_char_idx', 'returned', 'section_path']
    }
    out['target'] = target
    out['imports'] = list(map(lambda l: astor.to_source(l).strip(), imports + target_imports))
    out['inputs'] = set(remaining_vars_needed)
    out['context'] = context
    return out
