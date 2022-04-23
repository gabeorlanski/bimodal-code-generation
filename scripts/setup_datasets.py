from collections import defaultdict, Counter
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
from src.data.npv import make_samples_from_dict, SUPPORTED_TASKS, check_code_executes_properly


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


def setup_npv(debug, num_false_pair_mod):
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
                if parse_fails:
                    logger.info(f"{file_name} had {len(parse_fails)} fail(s)")
                    fails.extend(parse_fails)

        logger.info(f"Found {len(raw_instances)} samples for {split}")

        if debug:
            raw_instances = raw_instances[:25]
        logger.info(f"Making samples from {len(raw_instances)} programs")
        unverified_samples = []
        num_samples_per = {}
        for i, instance in tqdm(enumerate(raw_instances), total=len(raw_instances)):
            instance_samples = make_samples_from_dict(deepcopy(instance))
            num_samples_per[instance['instance_idx']] = len(instance_samples)
            unverified_samples.extend(instance_samples)

        logger.info(f"{len(unverified_samples)} total samples to verify")
        results, test_negations, exclude_pairs, split_exec_fails = check_code_executes_properly(
            split,
            unverified_samples
        )
        passed_programs = []
        split_failed_execution = 0
        with raw_path.joinpath(f'{split}.jsonl').open('w') as f:
            for i, v in enumerate(raw_instances):
                if len(results[v['instance_idx']]) >= num_samples_per[v['instance_idx']]:
                    split_failed_execution += 1
                    continue

                v['test_negations'] = test_negations[v['instance_idx']]
                v['exclude_tests'] = exclude_pairs[v['instance_idx']]
                passed_programs.append(v)
                out_str = f"{json.dumps(v)}\n"
                f.write(out_str)

        logger.info(f"{split_failed_execution} programs failed all sample execution for '{split}'")
        total_fail_exec += split_failed_execution

        # Verify the samples again, and this time if they fail, discard them.
        unverified_samples = []
        total_true_examples = 0
        total_false_examples = 0
        verified_samples_by_idx = defaultdict(dict)
        for i, instance in tqdm(enumerate(passed_programs), total=len(raw_instances)):
            samples = make_samples_from_dict(deepcopy(instance))
            has_true = False
            has_false = False
            for sample in samples:
                verified_samples_by_idx[instance['instance_idx']][sample['task_id']] = deepcopy(
                    sample
                )
                if sample['result'] == 'True':
                    total_true_examples += 1
                    has_true = True
                else:
                    has_false = True
                    total_false_examples += 1
            if not has_true or not has_false:
                raise ValueError(
                    f"{instance['source_file']}[{instance['task']}] {instance['task_id']}")
            unverified_samples.extend(samples)
        logger.info(
            f"After first verification pass for {split}, {total_true_examples} "
            f"true examples and {total_false_examples} false examples"
        )
        logger.info(f"Doing second verification pass for '{split}'")
        results, test_negations, _, _ = check_code_executes_properly(
            split,
            unverified_samples
        )

        removed_failed = 0
        for instance_idx, failed_samples in results.items():
            for task_id in failed_samples:
                verified_samples_by_idx[instance_idx].pop(task_id)
                removed_failed += 1
        logger.info(f"Removed {removed_failed} program(s) because they failed twice")
        total_fail_exec += removed_failed

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
                if str(sample['result']) == 'True':
                    num_true_pairs += 1
                    correct_io_pairs[sample['input']].append({
                        'input': sample['input'], 'op': sample['op'], 'output': sample['output']
                    })
                else:
                    num_false_pairs += 1
                    incorrect_io_pairs[sample['input']].append({
                        'input': sample['input'], 'op': sample['op'], 'output': sample['output']
                    })

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
        to_save_samples = all_true_instances

        num_false_to_save = max(0, num_false_pair_mod * len(to_save_samples) - len(to_save_false))
        num_false_to_save = int(min(len(all_false_instances), num_false_to_save))

        # Add them back to the list of ones to save after calculating the number
        # of false to save so that we do not factor them back into the pool.
        to_save_samples.extend(to_save_false)
        logger.info(f"{len(to_save_samples)}/{len(all_false_instances)} are guaranteed to be saved")
        logger.info(f"Randomly selecting {num_false_to_save}/{len(all_false_instances)} "
                    f"from the false samples")

        for idx in random.sample(range(len(all_false_instances)), k=num_false_to_save):
            sample = all_false_instances[idx]
            false_count[sample['instance_idx']] += 1  # type:ignore
            to_save_samples.append(sample)

        for k, v in true_count.items():
            mean_tracker['selected_false'].append(false_count[k])
            mean_tracker['true_pairs'].append(v)
            mean_tracker['total_pairs'].append(v + false_count[k])
        random.shuffle(to_save_samples)
        with out_path.joinpath(f'{split}.jsonl').open('w') as f:
            for sample in to_save_samples:
                f.write(json.dumps(sample) + '\n')
        logger.info(f"Stats for '{split}':")
        for k, v in count_tracker.items():
            logger.info(f"\t{k:>20} = {v}")
        for k, v in mean_tracker.items():
            logger.info(f"\t{k:>20} = {np.mean(v):.2f}")

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
@click.option(
    '--num-false-pairs-mod', '-fratio', type=float, default=1.0,
    help=f"Float ratio for number of false samples to number of true samples"
)
@click.pass_context
def setup_npv_cli(ctx, num_false_pairs_mod):
    setup_npv(ctx.obj['DEBUG'], num_false_pairs_mod)


if __name__ == "__main__":
    setup_datasets()
