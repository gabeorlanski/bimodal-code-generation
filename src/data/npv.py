import json
import logging
import re
from typing import List, Dict
from tqdm import tqdm
import ast
import astor

logger = logging.getLogger(__name__)

__all__ = [
    "make_npv_data_from_dicts",
    "SUPPORTED_TASKS"
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

    def visit_Compare(self, node):
        inputs = []
        for c in self.get_func_io(node.left):  # type: ignore
            input_str = astor.to_source(
                ast.Expr(value=c),
                source_generator_class=CustomSourceGenerator).strip()

            inputs.append(input_str.replace('"""', "'"))

        assert len(node.comparators) == 1
        outputs = astor.to_source(
            ast.Expr(value=node.comparators[0]),
            source_generator_class=CustomSourceGenerator
        ).strip()  # type:ignore

        assert len(node.ops) == 1

        self.io_pairs.append({
            'input' : inputs,
            'output': outputs,
            'ops'   : OP_TO_STR[type(node.ops[0])]  # type:ignore
        })


def serialize_instance_to_dict(
        task: str,
        task_id: str,
        description: str,
        programs: Dict[str, str],
        input_output_pairs: List[Dict],
        context: str = ''
):
    return {
        "task"              : task,
        "task_id"           : task_id,
        "description"       : description,
        "programs"          : programs,
        "input_output_pairs": input_output_pairs,
        "context"           : context
    }


#####################################################################
# Parsing instances from list of dicts for different task           #
#####################################################################

def parse_human_eval(file_path) -> List[Dict]:
    logger.info("Getting data for HUMAN_EVAL")
    out = []

    double_newlines = re.compile(r'\n{2,}')

    for line_number, line in tqdm(enumerate(map(json.loads, file_path.open())),
                                  desc='Parsing'):
        visitor = IOPairsFromAssertVisitor()
        visitor.visit(ast.parse(line['test']))

        program, description, *_ = line['prompt'].split('"""')
        description = '\n'.join([d_l.strip() for d_l in description.split('\n')]).strip()

        if not program.startswith('def'):
            context, program = program.split('def')
            program = f'def{program}'.strip()
            context = context
        else:
            context = ''
        if not program.endswith('\n') and not line['canonical_solution'].startswith('\n'):
            program += '\n'
        program = f"{program}{line['canonical_solution']}"

        out.append(serialize_instance_to_dict(
            task='HUMAN_EVAL',
            task_id=line['task_id'],
            description=description,
            programs={line['entry_point']: program.strip()},
            input_output_pairs=visitor.io_pairs,
            context=context.strip()
        ))
    return out


def parse_mbpp(file_path) -> List[Dict]:
    logger.info("Getting data for HumanEval")
    out = []
    for line_number, line in tqdm(enumerate(map(json.loads, file_path.open())),
                                  desc='Parsing'):
        visitor = IOPairsFromAssertVisitor()
        visitor.visit(ast.parse('\n'.join(line['test_list'] + line['challenge_test_list'])))

        out.append(serialize_instance_to_dict(
            task='MBPP',
            task_id=line['task_id'],
            description=line['text'].replace('\r', ''),
            programs={visitor.func_name: line['code'].replace('\r', '')},
            input_output_pairs=visitor.io_pairs,
            context=line.get('test_setup_code', '').replace('\r', '')
        ))
    return out


SUPPORTED_TASKS = {
    'mbpp'      : parse_mbpp,
    'human_eval': parse_human_eval
}


def make_npv_data_from_dicts(file_path, task):
    return SUPPORTED_TASKS[task](file_path)
