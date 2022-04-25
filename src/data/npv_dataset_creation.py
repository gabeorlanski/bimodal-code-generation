import ast
import contextlib
import io
import json
import logging
import math
import random
import re
import signal
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import astor
from tqdm import tqdm
import multiprocessing as mp
import pickle
import inspect
from src.evaluation.execute import create_tempdir

logger = logging.getLogger(__name__)

__all__ = [
    "make_samples_from_dict",
    "SUPPORTED_TASKS",
    "check_io_sample_executes_correctly",
    "generate_more_io_pairs",
    "get_true_and_false_instances_from_verified"
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

OP_NEGATION_MAP = {
    k: v for a, b in [
        ("!=", '=='),
        ("<", '>='),
        (">", '<='),
        ("is not", "is"),
        ("not in", "in"),
    ] for k, v in [(a, b), (b, a)]
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
            'ops'   : OP_TO_STR[type(node.ops[0])]  # type:ignore
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


def make_code_sample(code_str, context, test_stmt):
    code = ['def test_fn():']
    raw_code = [context, code_str]
    for block in map(lambda b: b.split('\n'), raw_code):
        for line in filter(lambda b: b.strip(), block):
            code.append(f"\t{line}")

    for line in test_stmt.split('\n'):
        code.append(f"\t{line}")
    code.append("RETURN_VALUE = test_fn()")
    return '\n'.join(code)


def check_io_sample_executes_correctly(split, unverified_samples):
    results = defaultdict(list)
    test_negations = defaultdict(list)
    exclude_programs = defaultdict(list)
    exec_fails = []
    num_failed_tests = 0
    instances_passed = 0
    for i, program_dict in tqdm(enumerate(unverified_samples), desc='Executing',
                                total=len(unverified_samples)):
        test_stmt = f"{program_dict['input']} {program_dict['op']} {program_dict['output']}"
        test_stmt = f"assert ({test_stmt})=={program_dict['result']}"

        result = execute_code(
            make_code_sample(program_dict['code'], program_dict['context'], test_stmt)
        )
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


def generate_more_io_pairs(
        instances,
        model_name,
        temperature,
        p_val,
        batch_size,
        workers
):
    logger.info(f"Generating more io pairs for {len(instances)} instances")
    device = torch.device('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    eos_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.eos_token = eos_token
    tokenizer.bos_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    num_rtr_sequences = 5
    num_iters = 2
    generation_kwargs = {
        'do_sample'           : True,
        'temperature'         : temperature,
        'top_p'               : p_val,
        'num_return_sequences': num_rtr_sequences
    }
    generated_io_by_instance_idx = defaultdict(list)
    idx_to_instance_idx = {}

    def get_func_from_line(line, prompt_str, func_name):
        if not line.startswith(">>>"):
            return None

        if func_name not in line:
            return None

        func_str = line.split('>>>')[-1].strip().split('#')[0].strip()
        if func_str in prompt_str:
            return None

        try:
            ast.parse(func_str)
        except SyntaxError as e:
            return None
        return func_str

    torch.backends.cudnn.benchmark = True
    model.eval()
    with torch.inference_mode():
        pbar = tqdm(total=num_rtr_sequences * num_iters * math.ceil(len(instances) / batch_size),
                    desc='Generating')

        sorted_instances = list(sorted(
            instances,
            key=lambda inst: sum(map(lambda io_p: len(io_p['input']), inst['input_output_pairs'])),
            reverse=True)
        )
        for i in range(0, len(instances), batch_size):
            prompts = []
            batch = sorted_instances[i:i + batch_size]
            for idx, instance in enumerate(batch):
                idx_to_instance_idx[instance['instance_idx']] = i + idx
                prompt = ""
                for input_str in map(lambda d: d['input'], instance['input_output_pairs']):
                    prompt += f">>> {input_str}\n"
                prompt += f">>> {instance['function']}("
                prompts.append(prompt)

            prompts_tok = tokenizer(prompts, return_tensors='pt', padding='longest')
            max_length = prompts_tok['input_ids'].size(1) + 128

            for _ in range(num_iters):
                results = model.generate(
                    max_length=max_length,
                    **{k: v.to(device) for k, v in prompts_tok.items()},
                    **generation_kwargs
                )

                decoded = tokenizer.batch_decode(results.tolist(), skip_special_tokens=True)

                for j, seq in enumerate(decoded):
                    batch_idx = j // num_rtr_sequences
                    generated_io_by_instance_idx[batch[batch_idx]['instance_idx']].extend(
                        list(filter(
                            lambda x: x,
                            map(
                                lambda l: get_func_from_line(
                                    l,
                                    prompts[batch_idx],
                                    batch[batch_idx]['function']
                                ),
                                seq.split('\n')
                            )
                        ))
                    )
                pbar.update(num_rtr_sequences)
        pbar.close()

    mp_args = []
    for instance_idx, generated in generated_io_by_instance_idx.items():
        program_dict = instances[instance_idx]
        assert program_dict['instance_idx'] == instance_idx
        for i, test_stmt in enumerate(generated):
            mp_args.append(
                (
                    instance_idx,
                    i,
                    (
                            program_dict['code'] + '\n'
                            + program_dict['context'] + '\n'
                            + f"RESULT_VALUE['result'] = {test_stmt}"
                    )
                )
            )

    logger.info(f"{len(mp_args)} potential code samples generated")

    passed_by_idx = defaultdict(dict)
    fails_by_idx = defaultdict(dict)
    passed = 0
    with mp.Pool(workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(is_sample_valid, mp_args),
            total=len(mp_args),
            desc='Validating Samples'
        ))

        for result in results:
            if not result['passed']:
                fails_by_idx[result['idx']][result['code_idx']] = result['result']
                continue
            if result['result'] is None:
                fails_by_idx[result['idx']][result['code_idx']] = "None"
                continue
            elif len(str(result['result'])) >= 256:
                fails_by_idx[result['idx']][result['code_idx']] = "Length"
                continue

            passed_by_idx[result['idx']][result['code_idx']] = result['result']
            passed += 1
    logger.info(f"{passed}/{len(mp_args)} passed")

    all_generated = defaultdict(list)
    for instance_idx, passed_code in passed_by_idx.items():
        for code_idx, output in passed_code.items():
            sample = {
                'input' : generated_io_by_instance_idx[instance_idx][code_idx],
                'output': output,
                'ops'   : "=="
            }
            all_generated[instance_idx].append(sample)
            instances[instance_idx]['input_output_pairs'].append(
                sample
            )

    return instances, all_generated


def timeout_handler(signum, frame):
    raise TimeoutError("Failed to process")


def is_sample_valid(args):
    instance_idx, code_idx, code_sample = args
    RESULT_VALUE = {}

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)
    try:
        with create_tempdir():
            try:
                stdout_f = io.StringIO()
                stderr_f = io.StringIO()
                with contextlib.redirect_stdout(stdout_f):
                    with contextlib.redirect_stderr(stderr_f):
                        _locals = locals()
                        exec(code_sample, globals(), _locals)
                        RESULT_VALUE = _locals['RESULT_VALUE']
            except Exception as e:
                return {
                    'idx'   : instance_idx, 'code_idx': code_idx, 'passed': False,
                    'result': str(e)
                }
        output = {
            'idx'     : instance_idx,
            'code_idx': code_idx,
            'passed'  : True,
            'result'  : repr(RESULT_VALUE['result'])
        }

        if inspect.isfunction(RESULT_VALUE['result']) or inspect.isclass(RESULT_VALUE['result']):
            return {
                'idx'   : instance_idx, 'code_idx': code_idx, 'passed': False,
                'result': 'Not Literal'
            }
        try:
            pickle.dumps(output)
        except Exception as e:
            return {'idx': instance_idx, 'code_idx': code_idx, 'passed': False, 'result': str(e)}
        return output
    except Exception as e:
        return {'idx': instance_idx, 'code_idx': code_idx, 'passed': False, 'result': str(e)}


def make_samples_from_dict(single_instance, with_negation=False):
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
            original_pred_id = f"{single_instance['task']}_{single_instance['instance_idx']}_{pred_idx}"

            # Add in the correct pair first, then add in the negated pair.
            pred_dict = deepcopy(single_instance)
            pred_dict['task_id'] = original_pred_id
            pred_dict.update(execute_info)
            pred_dict['result'] = str(res)
            pred_dict['is_manual_fix'] = is_manual_fix
            pred_dict['is_negation_of'] = None
            out.append(pred_dict)
            pred_idx += 1

            if res in [True, False] and with_negation:
                negation_pred_id = f"{single_instance['task']}_{single_instance['instance_idx']}_{pred_idx}"
                negation_pred_dict = deepcopy(single_instance)
                negation_pred_dict['task_id'] = negation_pred_id
                execute_info['op'] = OP_NEGATION_MAP[execute_info['op']]
                negation_pred_dict.update(execute_info)
                negation_pred_dict['result'] = str(not res)
                negation_pred_dict['is_manual_fix'] = is_manual_fix
                negation_pred_dict['is_negation_of'] = original_pred_id
                out.append(negation_pred_dict)
                pred_idx += 1

    return out


def get_true_and_false_instances_from_verified(verified_samples_by_idx):
    count_tracker = Counter()
    count_tracker['no_true_pairs'] = 0
    count_tracker['not_eq_pair_keys'] = 0
    mean_tracker = defaultdict(list)

    all_false_instances = []
    all_true_instances = []
    to_save_false = []

    false_count = Counter()
    true_count = Counter()
    for program_idx, sample_dict in tqdm(verified_samples_by_idx.items()):
        false_count[program_idx] = 0
        correct_io_pairs = defaultdict(list)
        incorrect_io_pairs = defaultdict(list)
        num_true_pairs = 0
        num_false_pairs = 0
        for sample in sample_dict.values():
            io_pair_dict = {
                'input'         : sample['input'],
                'op'            : sample['op'],
                'output'        : sample['output'],
                'is_manual_fix' : sample['is_manual_fix'],
                'is_negation_of': sample['is_negation_of'],
                'task_id'       : sample['task_id']
            }
            if str(sample['result']) == 'True':
                num_true_pairs += 1
                correct_io_pairs[sample['input']].append(io_pair_dict)
            else:
                num_false_pairs += 1
                incorrect_io_pairs[sample['input']].append(io_pair_dict)

        if num_true_pairs == 0:
            count_tracker['no_true_pairs'] += 1

        if set(incorrect_io_pairs) != set(correct_io_pairs):
            count_tracker['not_eq_pair_keys'] += 1

        true_count[program_idx] = num_true_pairs
        mean_tracker['created_false_pairs'].append(num_false_pairs)

        # Want to keep a dict of IO examples that are NOT the same as
        # the one that is tested. So make a map storing it.
        context_io_pair_map = {k: {'True': [], 'False': []} for k in
                               set(correct_io_pairs).union(set(incorrect_io_pairs))}
        for input_str, outputs in correct_io_pairs.items():
            for k in context_io_pair_map:
                if k == input_str:
                    continue
                context_io_pair_map[k]['True'].extend(outputs)
                context_io_pair_map[k]['False'].extend(incorrect_io_pairs[input_str])

        program_false_samples = []
        for sample in sample_dict.values():
            sample['context_io_pairs'] = context_io_pair_map[sample['input']]
            if str(sample['result']) == 'True':
                all_true_instances.append(sample)
            else:
                program_false_samples.append(sample)
        if not program_false_samples:
            raise ValueError()

        false_to_save = program_false_samples.pop(
            random.choice(range(len(program_false_samples)))
        )
        false_count[program_idx] += 1
        to_save_false.append(false_to_save)
        all_false_instances.extend(program_false_samples)

    stats = (true_count, false_count, mean_tracker, count_tracker)
    return all_true_instances, to_save_false, all_false_instances, stats
