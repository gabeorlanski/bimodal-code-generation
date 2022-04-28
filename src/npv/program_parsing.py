import ast
import json
import logging
import re
from typing import List, Dict, Tuple

import astor
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "SUPPORTED_TASKS",
    "CustomSourceGenerator"
]
OP_TO_STR = {
    ast.Eq   : '==',
    ast.NotEq: '!=',
    ast.GtE  : '>=',
    ast.LtE  : '<=',
    ast.Lt   : '<',
    ast.Gt   : '>',
    ast.Is   : "is",
    ast.IsNot: "is not",
    ast.In   : "in",
    ast.NotIn: "not in"
}

PROG_SPLIT = re.compile(r'(class |def )')


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


class IOPairsFromAssertVisitor(ast.NodeVisitor):
    def __init__(self):
        self.io_pairs = []
        self.func_name = None

    def get_func_io(self, node):
        if not isinstance(node, ast.Call):
            return node

        assert not node.keywords
        self.func_name = node.func.id  # type: ignore
        return node.args

    def visit_Call(self, node):
        try:
            self.func_name = node.func.id
        except:
            pass
        self.generic_visit(node)

    def visit_Compare(self, node):
        # if not node.ops or not isinstance(node.left, list):  # type: ignore
        #     return

        assert len(node.comparators) == 1
        outputs = astor.to_source(
            ast.Expr(value=node.comparators[0]),
            source_generator_class=CustomSourceGenerator
        ).strip()  # type:ignore

        outputs = outputs.replace('"""', "'")

        assert len(node.ops) == 1

        self.io_pairs.append({
            'input' : astor.to_source(node.left).strip(),
            'output': outputs,
            'ops'   : OP_TO_STR[type(node.ops[0])],  # type: ignore
        })

        self.generic_visit(node)


def serialize_instance_to_dict(
        source_file,
        task: str,
        task_id: str,
        description: str,
        program: str,
        func_name: str,
        input_output_pairs: List[Dict],
        context: str = ''
):
    return {
        'source_file'       : source_file,
        'function'          : func_name,
        "task"              : task,
        "task_id"           : task_id,
        "description"       : description,
        "code"              : program,
        "input_output_pairs": input_output_pairs,
        "context"           : context
    }


#####################################################################
# Parsing instances from list of dicts for different task           #
#####################################################################

def parse_human_eval(file_path) -> Tuple[List[Dict], List[Dict]]:
    logger.info("Getting data for HUMAN_EVAL")
    out = []
    fails = []
    for line_number, line in tqdm(enumerate(map(json.loads, file_path.open())),
                                  desc='Parsing'):

        if line['task_id'] in ['HumanEval/38']:
            fails.append({
                'source_file': file_path.stem + file_path.suffix,
                'task'       : 'HUMAN_EVAL',
                'line_number': line_number,
                "exception"  : 'Skipped program',
                **line
            })
            continue
        try:
            visitor = IOPairsFromAssertVisitor()
            visitor.visit(ast.parse(line['test']))
        except Exception as e:
            fails.append({
                'source_file': file_path.stem + file_path.suffix,
                'task'       : 'HUMAN_EVAL',
                'line_number': line_number,
                "exception"  : str(e),
                **line
            })
            continue

        if not visitor.io_pairs:
            fails.append({
                'source_file': file_path.stem + file_path.suffix,
                'task'       : 'HUMAN_EVAL',
                'line_number': line_number,
                "exception"  : "No IO Pairs",
                **line
            })
            continue

        if 'FIX' in line['prompt']:
            _, _, line['prompt'] = line['prompt'].split('"""', 2)
            line['prompt'].lstrip()
        try:
            program, description, *_ = line['prompt'].split('"""')
        except:
            program, description, *_ = line['prompt'].replace("'''", '"""').split('"""')

        description = '\n'.join([d_l.strip() for d_l in description.split('\n')]).strip()

        context, delim, program = PROG_SPLIT.split(program, 1)
        program = f"{delim}{program}".strip()
        if not program.endswith('\n') and not line['canonical_solution'].startswith('\n'):
            program += '\n'
        program = f"{program}{line['canonical_solution']}"

        ast.parse(program)

        io_pairs = []
        for p in visitor.io_pairs:
            pair_dict = {}
            for k, v in p.items():
                if k == 'input':
                    pair_dict[k] = v.replace('candidate', line['entry_point'])
                else:
                    pair_dict[k] = v
            io_pairs.append(pair_dict)

        out.append(serialize_instance_to_dict(
            source_file=file_path.stem + file_path.suffix,
            task='HUMAN_EVAL',
            func_name=line['entry_point'],
            task_id=line['task_id'],
            description=description,
            program=program.strip(),
            input_output_pairs=io_pairs,
            context=context.strip()
        ))
    return out, fails


def parse_mbpp(file_path) -> Tuple[List[Dict], List[Dict]]:
    logger.info("Getting data for HumanEval")
    out = []
    fails = []
    for line_number, line in tqdm(enumerate(map(json.loads, file_path.open())),
                                  desc='Parsing'):
        try:
            visitor = IOPairsFromAssertVisitor()
            visitor.visit(ast.parse('\n'.join(line['test_list'] + line['challenge_test_list'])))
        except Exception as e:
            fails.append({
                'source_file': file_path.stem + file_path.suffix,
                'task'       : 'HUMAN_EVAL',
                'line_number': line_number,
                "exception"  : str(e),
                **line
            })
            continue
        if not visitor.io_pairs:
            fails.append({
                'source_file': file_path.stem + file_path.suffix,
                'task'       : 'HUMAN_EVAL',
                'line_number': line_number,
                "exception"  : "No IO Pairs",
                **line
            })
            continue
        program = line['code'].replace('\r', '')
        context = line.get('test_setup_code', '').replace('\r', '')
        prog_context, delim, program = PROG_SPLIT.split(program, 1)
        program = f"{delim}{program}".strip()
        context = f"{prog_context.strip()}\n{context}"
        ast.parse(program)
        out.append(serialize_instance_to_dict(
            source_file=file_path.stem + file_path.suffix,
            task='MBPP',
            func_name=visitor.func_name,
            task_id=line['task_id'],
            description=line['text'].replace('\r', ''),
            program=program,
            input_output_pairs=visitor.io_pairs,
            context=context.strip()
        ))
    return out, fails


SUPPORTED_TASKS = {
    'mbpp'      : parse_mbpp,
    'human_eval': parse_human_eval
}
