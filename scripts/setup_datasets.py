import contextlib
import io
from collections import defaultdict
from copy import deepcopy
from itertools import chain
from pathlib import Path
import logging
import json
import random
from typing import Union

import yaml
from tqdm import tqdm
import click
import numpy as np

import sys

# If this file is called by itself (for creating the splits) then it will
# have import issues.
from transformers import AutoTokenizer

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT
from src.common import setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.npv import make_samples_from_dict, SUPPORTED_TASKS, NPV
from src.evaluation.execute import create_tempdir


def setup_mbpp(
        data_path,
        test_size: int = 500,
        few_shot_size: int = 10,
        fine_tuning_size: int = 374
):
    """
    Setup the splits for the mostly basic programming problems dataset.

    The default values for the split sizes come from the paper. The size of the
     validation split is equal to total - (test+few shot+fine tuning).

    Args:
        data_path (str): Path to the data dir that has mbpp.jsonl and
            sanitized-mbpp.json in them.
        test_size (int): Size of the test set.
        few_shot_size (int): Size of the set used for few-shot prompting.
        fine_tuning_size (int): Size of the set used for fine tuning.
    """

    out_path = PROJECT_ROOT.joinpath(data_path)
    data_path = PROJECT_ROOT.joinpath(data_path)

    logger = logging.getLogger("setup_datasets")
    logger.info(
        f"Setting up splits for MBPP files located in " f"'{data_path.resolve()}'"
    )

    logger.info("Validating directory")
    try:
        mbpp_path, sanitized_path = validate_files_exist(
            data_path, ["mbpp.jsonl", "sanitized-mbpp.json"]
        )
    except FileExistsError as e:
        logger.error(f"Missing '{e.file}' in '{data_path.resolve()}' ")
        raise e

    logger.info("Loading data from files")
    logger.debug(f"Loading json lines from '{mbpp_path.resolve()}'")
    mbpp_data = []
    for line in tqdm(
            mbpp_path.read_text("utf-8").splitlines(False),
            desc="Reading mbpp.jsonl",
            file=sys.stdout,
    ):
        mbpp_data.append(json.loads(line))
    logger.debug(f"Loading json from '{sanitized_path.resolve()}'")
    sanitized_data = json.loads(sanitized_path.read_text("utf-8"))
    logger.info(
        f"{len(mbpp_data)} items in MBPP and {len(sanitized_data)} in Sanitized"
    )

    logger.info(f"Saving sanitized to '{out_path.joinpath('edited.jsonl')}'")
    with out_path.joinpath("edited.jsonl").open("w", encoding="utf-8") as f:
        for i in sanitized_data:
            f.write(json.dumps(i) + "\n")

    validation_size = len(mbpp_data) - (test_size + few_shot_size + fine_tuning_size)
    splits = [
        ("Few-Shot", "few_shot.jsonl", few_shot_size),
        ("Test", "test.jsonl", test_size),
        ("Fine-Tuning", "train.jsonl", fine_tuning_size),
        ("Validation", "validation.jsonl", validation_size),
    ]
    progress_bar = tqdm(total=len(mbpp_data), desc="Saving Splits", file=sys.stdout)
    current = 0
    for name, out_file_name, size in splits:

        logger.info(
            f"Saving split {name} with {size} items to {out_file_name}"
        )
        with out_path.joinpath(out_file_name).open("w", encoding="utf-8") as split_file:
            for i in mbpp_data[current: current + size]:
                split_file.write(json.dumps(i) + "\n")
                progress_bar.update()
            current += size

    progress_bar.close()


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


def setup_npv():
    logger = logging.getLogger('setup_datasets')
    data_path = Path(PROJECT_ROOT.joinpath('data/raw_npv'))
    logger.info(f"Making NPV data from files {data_path}")
    assert data_path.joinpath('cfg.yaml').exists()
    cfg = yaml.load(data_path.joinpath('cfg.yaml').open(), yaml.Loader)

    out_path = PROJECT_ROOT.joinpath('data/NPV')
    raw_path = out_path.joinpath('raw')
    if not out_path.exists():
        out_path.mkdir(parents=True)
    if not raw_path.exists():
        raw_path.mkdir(parents=True)
    fails = []
    exec_fails = []
    total_fail_exec = 0
    for split, file_cfg in cfg.items():
        raw_instances = []
        for task, files in file_cfg.items():
            logger.info(f"{len(files)} files to use for task {task} in split {split}")
            for file_name in files:
                logger.info(f"Parsing {file_name}")
                file_path = data_path.joinpath(split, file_name)

                parsed_dataset, parse_fails = SUPPORTED_TASKS[task](file_path)
                for instance in parsed_dataset:
                    instance['instance_idx'] = len(raw_instances)
                    raw_instances.append(instance)
                raw_instances.extend(parsed_dataset)
                if parse_fails:
                    logger.info(f"{file_name} had {len(parse_fails)} fail(s)")
                    fails.extend(parse_fails)

        logger.info(f"Found {len(raw_instances)} samples for {split}")
        random.shuffle(raw_instances)
        #
        # logger.info(f"Saving to {out_path.joinpath(f'{split}.jsonl')}")
        # with raw_path.joinpath(f'{split}.jsonl').open('w') as f:
        #     for i, v in enumerate(raw_instances):
        #         f.write(f"{json.dumps(v)}\n")

        logger.info(f"Making samples from {len(raw_instances)} raw instances")
        unverified_samples = []
        for i, instance in tqdm(enumerate(raw_instances), total=len(raw_instances)):
            unverified_samples.extend(make_samples_from_dict(deepcopy(instance)))

        logger.info(f"{len(unverified_samples)} total samples to verify")

        results = {}
        test_negations = defaultdict(list)
        exclude_tests = defaultdict(list)
        num_failed_tests = 0
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
                results[program_dict['instance_idx']] = True
            else:
                results[program_dict['instance_idx']] = False
                num_failed_tests += 1

                exec_fails.append(program_dict)
                if isinstance(result, AssertionError):
                    test_negations[program_dict['instance_idx']].append(
                        f"{program_dict['input']} {program_dict['output']}"
                    )
                else:
                    exclude_tests[program_dict['instance_idx']].append(
                        f"{program_dict['input']} {program_dict['output']}"
                    )
        logger.info(f"{num_failed_tests} total failed the test for {split}")
        passed_programs = []
        with raw_path.joinpath(f'{split}.jsonl').open('w') as f:
            for i, v in enumerate(raw_instances):
                if not results[v['instance_idx']]:
                    total_fail_exec += 1
                    continue

                v['test_negations'] = test_negations[v['instance_idx']]
                v['exclude_tests'] = exclude_tests[v['instance_idx']]
                passed_programs.append(v)
                out_str = f"{json.dumps(v)}\n"
                f.write(out_str)

        logger.info(f"Saving {len(passed_programs)} passed programs")
        with out_path.joinpath(f'{split}.jsonl').open('w') as f:
            for program_dict in tqdm(passed_programs):
                all_samples = make_samples_from_dict(deepcopy(program_dict))
                all_io_pairs = defaultdict(list)
                for sample in all_samples:
                    all_io_pairs[sample['input']].append({
                        'input': sample['input'], 'op': sample['op'], 'ouptut': sample['output']
                    })

                # Want to keep a dict of IO examples that are NOT the same as
                # the one that is tested. So make a map storing it.
                context_io_pair_map = {k: [] for k in all_io_pairs}
                for input_str, outputs in all_io_pairs.items():
                    for k in context_io_pair_map:
                        if k == input_str:
                            continue
                        context_io_pair_map[k].extend(outputs)

                for sample in all_samples:
                    sample['context_io_pairs'] = context_io_pair_map[sample['input']]
                    f.write(json.dumps(sample) + '\n')

    with out_path.joinpath('parse_fails.jsonl').open('w') as f:
        for fail in fails:
            f.write(json.dumps(fail) + '\n')

    logger.info(f"{total_fail_exec} failed execution")
    with out_path.joinpath('exec_fails.jsonl').open('w') as f:
        for fail in exec_fails:
            f.write(json.dumps(fail) + '\n')


@click.group()
@click.option('--seed', default=1, type=int, help="Seed")
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.pass_context
def setup_datasets(ctx, seed, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['SEED'] = seed
    random.seed(seed)
    setup_global_logging("setup_datasets", PROJECT_ROOT, disable_issues_file=True, debug=debug)


@setup_datasets.command('mbpp')
@click.argument('data_path')
@click.pass_context
def setup_mbpp_cli(
        ctx,
        data_path: Union[str, Path],
) -> None:
    setup_mbpp(data_path)


@setup_datasets.command('npv')
@click.pass_context
def setup_npv_cli(ctx):
    setup_npv()


if __name__ == "__main__":
    setup_datasets()
