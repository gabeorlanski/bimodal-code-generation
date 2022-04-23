import json
import logging
import math
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Tuple, Callable
from itertools import chain, zip_longest
import contextlib
import io
import yaml
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import ast
import astor
from tio import Task
from transformers import PreTrainedTokenizer
from src.common import PathType, PROJECT_ROOT

from src.evaluation.execute import create_tempdir
from jinja2 import BaseLoader, Environment, StrictUndefined

logger = logging.getLogger(__name__)

__all__ = [
    "make_samples_from_dict",
    "SUPPORTED_TASKS",
    "NPV",
    "check_code_executes_properly"
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


JINJA_ENV = Environment(loader=BaseLoader)  # type: ignore
JINJA_ENV.globals.update(zip=zip)
JINJA_ENV.undefined = StrictUndefined

PROMPT_TO_USE = None


@Task.register("npv")
class NPV(Task):
    SPLIT_MAPPING = {
        "test": str(PROJECT_ROOT.joinpath('data', 'NPV', 'test.jsonl')),
    }

    EXCLUDE_KEYS = [
        'source_file', 'task', 'original_task_id'
    ]

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable],
            postprocessors: List[Callable],
            metric_fns: List[Callable],
            split_mapping: Dict[str, PathType] = None,
            prompt: str = "base",
            choices: List[str] = None,
            n_ctx_pairs: int = 0,
            ctx_true_pct: float = 0.5,
            shuffle_ctx_pairs: bool = False,
            stmt_prompt: str = "{stmt}",
            trailing_newline: bool = False
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

        global PROMPT_TO_USE
        PROMPT_TO_USE = JINJA_ENV.from_string(prompt_dict[prompt])
        self.include_target_in_prompt_kwargs = False
        self.num_context_pairs = n_ctx_pairs
        self.num_true_ctx_pairs = math.ceil(ctx_true_pct * self.num_context_pairs)
        self.num_false_ctx_pairs = max(0, self.num_context_pairs - self.num_true_ctx_pairs)
        self.shuffle_ctx_pairs = shuffle_ctx_pairs
        self.stmt_prompt = stmt_prompt
        self.trailing_newline = trailing_newline

    def initialize_data(self):
        out = {}
        for split, path in self.SPLIT_MAPPING.items():
            split_dict = defaultdict(list)
            for d in tqdm(
                    map(json.loads, Path(path).read_text('utf-8').splitlines(False)),
                    desc=f"Reading '{split}'"
            ):

                excluded = {}
                for k, v in d.items():
                    if k in self.EXCLUDE_KEYS:
                        excluded[k] = v
                    else:
                        split_dict[k].append(v)
                self.excluded_columns_data[d['task_id']] = excluded

            out[split] = Dataset.from_dict(split_dict, split=split)
        return DatasetDict(out)

    def _load_samples(self, split: str) -> Dataset:
        return self._dataset_mapping[split]

    def make_stmt_from_io(self, input_stmt, op, output_stmt, target=None):
        out = self.stmt_prompt.format(stmt=f"{input_stmt} {op} {output_stmt}")
        if self.trailing_newline:
            return f"{out}\n"
        return out

    def map_to_standard_entries(self, sample: Dict) -> Dict:
        sample['target'] = sample['result']
        test_stmt = self.make_stmt_from_io(sample['input'], sample['op'], sample['output'])

        true_examples = random.sample(
            sample['context_io_pairs']['True'],
            k=min(self.num_true_ctx_pairs, len(sample['context_io_pairs']['True']))
        )
        false_examples = random.sample(
            sample['context_io_pairs']['False'],
            k=min(self.num_false_ctx_pairs, len(sample['context_io_pairs']['False']))
        )

        context_examples = []
        for true_example, false_example in zip_longest(true_examples, false_examples):
            if true_example is not None:
                context_examples.append([self.make_stmt_from_io(
                    true_example['input'], true_example['op'], true_example['output']
                ), 'True'])
            if false_example is not None:
                context_examples.append([self.make_stmt_from_io(
                    false_example['input'], false_example['op'], false_example['output']
                ), 'False'])

        if self.shuffle_ctx_pairs:
            random.shuffle(context_examples)

        # Some are VERY long, so we need to adapt for that by removing context
        # examples until we either have no context or have under the threshold.
        # num_required_chars = len(sample['context'] + sample['code'].lstrip())
        # context_example_chars = sum(map(len, context_examples))
        # if num_required_chars + context_example_chars >= 1000:
        #     logger.warning(f"{sample['task_id']} has too many characters, "
        #                    f"removing some context examples")
        #     while num_required_chars + context_example_chars >= 1000 and context_examples:
        #         context_example_chars -= len(context_examples.pop(-1))

        prompt_kwargs = {
            "context_code"    : sample['context'],
            'context_examples': context_examples,
            'description'     : sample['description'],
            'code'            : sample['code'].lstrip(),
            'test_stmt'       : test_stmt
        }
        if self.include_target_in_prompt_kwargs:
            prompt_kwargs['target'] = sample['result']
        assert PROMPT_TO_USE is not None
        sample['input_sequence'] = PROMPT_TO_USE.render(prompt_kwargs)
        return sample

    def serialize_task_features(self, idx: int, predictions: List, processed_sample: Dict) -> Dict:
        return {
            **self.excluded_columns_data[idx]
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


def execute_code(code):
    result = None
    with create_tempdir():
        try:
            stdout_f = io.StringIO()
            stderr_f = io.StringIO()
            with contextlib.redirect_stdout(stdout_f):
                with contextlib.redirect_stderr(stderr_f):
                    # sys.stdout.write = lambda *args, **kwargs: None
                    exec(code, globals(), locals())
        except Exception as e:
            result = e
    return result


def check_code_executes_properly(split, unverified_samples):
    results = defaultdict(list)
    test_negations = defaultdict(list)
    exclude_programs = defaultdict(list)
    exec_fails = []
    num_failed_tests = 0
    instances_passed = 0
    for i, program_dict in tqdm(enumerate(unverified_samples), desc='Executing',
                                total=len(unverified_samples)):

        code = ['def test_fn():']
        raw_code = [program_dict['context'], program_dict['code']]
        for block in map(lambda b: b.split('\n'), raw_code):
            for line in filter(lambda b: b.strip(), block):
                code.append(f"\t{line}")

        test_code = f"{program_dict['input']} {program_dict['op']} {program_dict['output']}"
        code.append(f"\tassert ({test_code})=={program_dict['result']}")
        code.append("test_fn()")
        result = execute_code('\n'.join(code))
        if result is None:

            instances_passed += 1
        else:
            results[program_dict['instance_idx']].append(program_dict['task_id'])
            num_failed_tests += 1

            exec_fails.append(program_dict)
            if isinstance(result, AssertionError):
                test_negations[program_dict['instance_idx']].append(
                    f"{program_dict['input']} {program_dict['output']}"
                )

            else:
                exclude_programs[program_dict['instance_idx']].append(
                    f"{program_dict['input']} {program_dict['output']}"
                )
    logger.info(
        f"{num_failed_tests}/{num_failed_tests + instances_passed} total "
        f"failed verification for '{split}'"
    )

    return results, test_negations, exclude_programs, exec_fails


def make_samples_from_dict(single_instance):
    io_pairs = single_instance.pop('input_output_pairs')
    specific_fixes = single_instance.pop('test_negations', [])
    excluded = single_instance.pop('exclude_tests', [])

    single_instance['original_task_id'] = single_instance.pop("task_id")
    out = []

    io_combos = set()
    pred_idx = 0
    for i, left in enumerate(io_pairs):
        to_keep = []
        for j, right in enumerate(io_pairs):
            op = left['ops']
            result = right['output'] == left['output']
            is_manual_fix = False
            io_pair = f"{left['input']} {right['output']}"
            if io_pair in excluded:
                continue
            if io_pair in specific_fixes:
                result = not result
                is_manual_fix = True

            combo = f"{left['input']} {op} {right['output']}"
            if combo not in io_combos:
                io_combos.add(combo)
                exec_info = {
                    'input': left['input'], 'output': right['output'], 'op': op
                }
                to_keep.append(
                    [exec_info, result, is_manual_fix]
                )
        for execute_info, res, is_manual_fix in to_keep:
            pred_dict = deepcopy(single_instance)
            pred_dict['task_id'] = f"{pred_dict['task']}_{pred_dict['instance_idx']}_{pred_idx}"
            pred_dict.update(execute_info)
            pred_dict['result'] = str(res)
            pred_dict['is_manual_fix'] = is_manual_fix
            out.append(pred_dict)
            pred_idx += 1

    return out
