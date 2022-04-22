import contextlib
import io
from collections import defaultdict
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
from src.data.npv import make_npv_data_from_dicts, SUPPORTED_TASKS, NPV
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
    if not out_path.exists():
        out_path.mkdir(parents=True)
    fails = []
    exec_fails = []
    for split, file_cfg in cfg.items():
        out = []
        for task, files in file_cfg.items():
            logger.info(f"{len(files)} files to use for task {task} in split {split}")
            for file_name in files:
                logger.info(f"Parsing {file_name}")
                file_path = data_path.joinpath(split, file_name)

                parsed_dataset, parse_fails = make_npv_data_from_dicts(file_path, task)
                out.extend(parsed_dataset)
                if parse_fails:
                    logger.info(f"{file_name} had {len(parse_fails)} fail(s)")
                    fails.extend(parse_fails)

        logger.info(f"Found {len(out)} samples for {split}")
        random.shuffle(out)

        logger.info(f"Saving to {out_path.joinpath(f'{split}.jsonl')}")
        with out_path.joinpath(f'tmp_{split}.jsonl').open('w') as f:
            for i, v in enumerate(out):
                v['instance_idx'] = i
                out_str = f"{json.dumps(v)}\n"
                f.write(out_str)
        NPV.SPLIT_MAPPING = {
            split: out_path.joinpath(f'tmp_{split}.jsonl')
        }

        task = NPV(
            tokenizer=AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random"),
            preprocessors=[],
            postprocessors=[],
            metric_fns=[]
        )
        task.prompt = task.JINJA_ENV.from_string(
            "def test_fn():{%- for line in context.split('\n') %}\n    {{line}}\n{%-endfor%}"
            "\n{% for line in code.split('\n') %}\n    {{line}}\n{%- endfor %}"
            "\n    assert ({{ test_stmt }}) == {{ target }}"
            "\ntest_fn()"
        )
        task.include_target_in_prompt_kwargs = True

        ds = task.preprocess('test', overwrite_cache=True)
        results = {}
        test_negations = defaultdict(list)
        exclude_tests = defaultdict(list)
        num_failed_tests = 0
        for i, (idx, c) in tqdm(enumerate(zip(ds['instance_idx'], ds['input_sequence'])), desc='Executing',
                                total=len(ds)):

            task_id = task.excluded_columns_data[idx]['task_id']

            result = execute_code(c)
            if result is None:
                results[task_id] = True
            else:
                results[task_id] = False
                num_failed_tests += 1
                if isinstance(result, AssertionError):
                    test_negations[idx].append(f"{ds[i]['input']} {ds[i]['output']}")
                else:
                    exclude_tests[idx].append(f"{ds[i]['input']} {ds[i]['output']}")
        logger.info(f"{num_failed_tests} total failed the test")
        with out_path.joinpath(f'{split}.jsonl').open('w') as f:
            for i, v in enumerate(out):
                if not results[v['task_id']]:
                    exec_fails.append(v)
                    continue

                v['test_negations'] = test_negations[v['instance_idx']]
                v['exclude_tests'] = exclude_tests[v['instance_idx']]
                out_str = f"{json.dumps(v)}\n"
                f.write(out_str)
        out_path.joinpath(f'tmp_{split}.jsonl').unlink()

    with out_path.joinpath('parse_fails.jsonl').open('w') as f:
        for fail in fails:
            f.write(json.dumps(fail) + '\n')

    logger.info(f"{len(exec_fails)} failed execution")
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
