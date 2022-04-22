import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Callable

import yaml
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import ast
import astor
from tio import Task
from transformers import PreTrainedTokenizer
from src.common import PathType, PROJECT_ROOT

from jinja2 import BaseLoader, Environment, StrictUndefined

logger = logging.getLogger(__name__)

__all__ = [
    "make_npv_data_from_dicts",
    "SUPPORTED_TASKS",
    "NPV"
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


@Task.register("npv")
class NPV(Task):
    SPLIT_MAPPING = {
        "test": str(PROJECT_ROOT.joinpath('data', 'NPV', 'test.jsonl')),
    }

    EXCLUDE_KEYS = [
        'source_file', 'task', 'task_id'
    ]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None,
            prompt: str = "base",
            choices: List[str] = None
    ):
        super(NPV, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors,
            metric_fns=metric_fns,
            split_mapping=split_mapping
        )
        self.dataset = None
        self.excluded_columns_data = {}
        self._dataset_mapping = self.initialize_data()
        self.choices = choices or ["False", 'True']
        prompt_dict = yaml.load(PROJECT_ROOT.joinpath('templates/npv_prompts.yaml').open(),
                                yaml.Loader)

        self.JINJA_ENV = Environment(loader=BaseLoader)  # type:ignore

        # Allow the python function zip()
        self.JINJA_ENV.globals.update(zip=zip)
        self.JINJA_ENV.undefined = StrictUndefined
        self.prompt = self.JINJA_ENV.from_string(prompt_dict[prompt])
        self.include_target_in_prompt_kwargs = False

    def initialize_data(self):
        out = {}
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in map(json.loads, Path(path).read_text('utf-8').splitlines(False)):

                excluded = {}
                for k in self.EXCLUDE_KEYS:
                    excluded[k] = d.pop(k)

                task_name = excluded['task']

                io_pairs = d.pop('input_output_pairs')
                test_fixes = d.pop('test_negations', [])
                exclude_tests = d.pop('exclude_tests', [])
                io_combos = set()
                for i, left in enumerate(io_pairs):
                    to_keep = []
                    for j, right in enumerate(io_pairs):
                        op = left['ops']
                        result = right['output'] == left['output']
                        is_manual_fix = False
                        io_pair = f"{left['input']} {right['output']}"
                        if io_pair in exclude_tests:
                            continue
                        if io_pair in test_fixes:
                            result = True
                            is_manual_fix = True

                        combo = f"{left['input']} {op} {right['output']}"
                        if combo not in io_combos:
                            io_combos.add(combo)
                            to_keep.append(
                                [left['input'], right['output'], op, result, is_manual_fix]
                            )
                    for pred_idx, (input_val, output_val, op, res, is_manual_fix) in enumerate(
                            to_keep):
                        for k, v in d.items():
                            split_dict[k].append(v)

                        split_dict['task_id'].append(f"{task_name}_{d['instance_idx']}_{pred_idx}")
                        self.excluded_columns_data[f"{task_name}_{d['instance_idx']}_{pred_idx}"] = excluded
                        split_dict['input'].append(input_val)
                        split_dict['op'].append(op)
                        split_dict['output'].append(output_val)
                        split_dict['result'].append(str(res))
                        split_dict['is_manual_fix'].append(is_manual_fix)

            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def _load_samples(self, split: str) -> Dataset:
        return self._dataset_mapping[split]

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = sample['result']
        test_stmt = f"{sample['input']} " \
                    f"{sample['op']} {sample['output']}"

        prompt_kwargs = {
            "context"  : sample['context'] + '\n',
            'code'     : sample['code'].lstrip(),
            'test_stmt': test_stmt
        }
        if self.include_target_in_prompt_kwargs:
            prompt_kwargs['target'] = sample['result']

        sample['input_sequence'] = self.prompt.render(prompt_kwargs)
        return sample

    def serialize_task_features(self, idx: int, predictions: List, processed_sample: Dict) -> Dict:
        return {
            'task'       : self.excluded_columns_data[idx]['task'],
            'source_file': self.excluded_columns_data[idx]['source_file'],
            **processed_sample
        }


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
            'ops'   : OP_TO_STR[type(node.ops[0])]  # type:ignore
        })


def serialize_instance_to_dict(
        source_file,
        task: str,
        task_id: str,
        description: str,
        program: str,
        input_output_pairs: List[Dict],
        context: str = ''
):
    return {
        'source_file'       : source_file,
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


def make_npv_data_from_dicts(file_path, task):
    return SUPPORTED_TASKS[task](file_path)
