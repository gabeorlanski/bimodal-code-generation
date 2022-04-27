from collections import defaultdict, Counter
from copy import deepcopy
from itertools import chain
from pathlib import Path
import logging
import json
import random
from typing import Union

import torch
import yaml
from tqdm import tqdm
import click
import numpy as np

import sys

# If this file is called by itself (for creating the splits) then it will
# have import issues.
# print(Path().resolve().absolute())
CWD = Path(__file__).resolve().absolute()
if str(CWD.parents[1]) not in sys.path:
    sys.path.insert(0, str(CWD.parents[1]))
from src.common import PROJECT_ROOT
from src.common import setup_global_logging
from src.common.file_util import validate_files_exist
from src.npv import (
    make_samples_from_dict, SUPPORTED_TASKS,
    check_io_sample_executes_correctly,
    get_instances_to_save, parse_raw_examples_for_split, verify_raw_programs
)


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


def setup_raw_npv(
        debug,
        use_negation,
        workers
):
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
    total_fail_exec = 0
    for split, file_cfg in cfg.items():
        split_fails, split_failed_exec = parse_raw_examples_for_split(
            split,
            file_cfg,
            raw_path,
            data_path,
            debug,
            use_negation,
            workers
        )
        fails.extend(split_fails)
        total_fail_exec += split_failed_exec

    with out_path.joinpath('parse_fails.jsonl').open('w') as f:
        for fail in fails:
            f.write(json.dumps(fail) + '\n')

    logger.info(f"{total_fail_exec} failed execution")


@click.group()
@click.option('--seed', default=1, type=int, help="Seed")
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.pass_context
def setup_datasets(ctx, seed, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['SEED'] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    setup_global_logging("setup_datasets", PROJECT_ROOT, disable_issues_file=True, debug=debug)


@setup_datasets.command('mbpp')
@click.argument('data_path')
@click.pass_context
def setup_mbpp_cli(
        ctx,
        data_path: Union[str, Path],
) -> None:
    setup_mbpp(data_path)


@setup_datasets.command('raw_npv')
@click.option('--negation', is_flag=True, default=False,
              help='Use negation for creating more samples')
@click.option(
    '--workers', '-n', type=int, default=1,
    help=f"# Workers to use"
)
@click.pass_context
def setup_raw_npv_cli(
        ctx,
        negation,
        workers,
        # batch_size
):
    setup_raw_npv(
        ctx.obj['DEBUG'],
        negation,
        workers=workers
    )


@setup_datasets.command('verify_npv')
@click.option(
    '--num-false-pairs-mod', '-fratio', type=float, default=-1,
    help=f"Float ratio for number of false samples to number of true samples"
)
@click.option('--negation', is_flag=True, default=False,
              help='Use negation for creating more samples')
@click.option(
    '--workers', '-n', type=int, default=1,
    help=f"# Workers to use"
)
@click.pass_context
def verify_npv(
        ctx,
        num_false_pairs_mod,
        negation,
        workers,
        # batch_size
):
    logger = logging.getLogger('setup_datasets')
    data_path = Path(PROJECT_ROOT.joinpath('data/raw_npv'))
    logger.info("Verifying RAW NPV data")
    assert data_path.joinpath('cfg.yaml').exists()
    cfg = yaml.load(data_path.joinpath('cfg.yaml').open(), yaml.Loader)

    out_path = PROJECT_ROOT.joinpath('data/NPV')
    raw_path = out_path.joinpath('raw')
    if not out_path.exists():
        out_path.mkdir(parents=True)
    assert raw_path.exists()
    for split, file_cfg in cfg.items():
        raw_file = raw_path.joinpath(f'{split}.jsonl')
        verify_raw_programs(
            raw_file,
            out_path,
            num_false_pairs_mod,
            negation,
            workers
        )


if __name__ == "__main__":
    setup_datasets()
