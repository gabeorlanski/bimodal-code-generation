import json
import logging
import random
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np
from tqdm import tqdm

from src.npv import (
    make_samples_from_dict, check_io_sample_executes_correctly,
    get_instances_to_save
)
from .program_parsing import SUPPORTED_TASKS

logger = logging.getLogger(__name__)

__all__ = [
    "parse_raw_examples_for_split",
    "verify_raw_programs"
]


def parse_raw_examples_for_split(
        split,
        file_cfg,
        raw_path,
        data_path,
        debug,
        use_negation,
        workers,
        generated_tests,
        seed,
        generated_mod
):
    rng = np.random.default_rng(seed)
    logger.info(f"{generated_mod=}")

    fails = []
    raw_instances = []
    generated_found = 0
    for task, files in file_cfg.items():
        logger.info(f"{len(files)} files to use for task {task} in split {split}")
        for file_name in files:
            logger.info(f"Parsing {file_name}")
            file_path = data_path.joinpath(split, file_name)

            parsed_dataset, parse_fails = SUPPORTED_TASKS[task](file_path)
            for instance in parsed_dataset:
                for io_idx in range(len(instance['input_output_pairs'])):
                    instance['input_output_pairs'][io_idx]['is_generated'] = False

                generated_for_instance = generated_tests.get(instance['function'], [])
                if generated_for_instance:

                    # Never take more than the number of existing samples
                    # from generated
                    if generated_mod != -1:
                        generated_for_instance = rng.choice(
                            generated_for_instance,
                            size=int(min(
                                len(generated_for_instance),
                                math.floor(len(instance['input_output_pairs']) * generated_mod)
                            ))
                        )
                    generated_found += len(generated_for_instance)
                    for generated in generated_for_instance:
                        instance['input_output_pairs'].append({'is_generated': True, **generated})
                instance['instance_idx'] = len(raw_instances)
                raw_instances.append(instance)
            if parse_fails:
                logger.info(f"{file_name} had {len(parse_fails)} fail(s)")
                fails.extend(parse_fails)

    logger.info(f"Found {len(raw_instances)} samples for {split}")
    logger.info(f"{generated_found} generated used")

    if debug:
        raw_instances = raw_instances[:50]

    logger.info(f"Making samples from {len(raw_instances)} programs")
    unverified_samples = []
    num_samples_per = {}
    for i, instance in tqdm(enumerate(raw_instances), total=len(raw_instances)):
        instance_samples = make_samples_from_dict(deepcopy(instance),
                                                  with_negation=use_negation)
        num_samples_per[instance['instance_idx']] = len(instance_samples)
        unverified_samples.extend(instance_samples)

    logger.info(f"{len(unverified_samples)} total samples to verify")
    returned_values, results = check_io_sample_executes_correctly(
        split,
        unverified_samples,
        workers
    )

    logger.info(f"{sum(map(len, results['failed_tests'].values()))} functions failed tests.")
    logger.info(f"{sum(map(len, results['had_errors'].values()))} had errors.")

    failed_counts = results['failed_counts']

    split_failed_execution = 0
    passed_programs = []
    for i, v in enumerate(raw_instances):
        if failed_counts[v['instance_idx']] >= num_samples_per[v['instance_idx']]:
            split_failed_execution += 1
            continue
        passed_programs.append(v)
    total_overridden = 0
    with raw_path.joinpath(f'{split}.jsonl').open('w') as f:
        for i, v in enumerate(passed_programs):
            returned = returned_values[v['instance_idx']]
            valid_inputs_to_override = {}
            skip_this = False
            for j, io_pair in enumerate(v['input_output_pairs']):
                if 'Index Minimal' in io_pair['input']:
                    skip_this = True
                    break
                if io_pair.get('is_generated'):
                    valid_inputs_to_override[(io_pair['input'], io_pair['ops'])] = j
            if skip_this:
                logger.info(f"Skipping {v['instance_idx']}")
                continue
            negated = []
            for tid, failed in results['failed_tests'][v['instance_idx']].items():
                found_key = valid_inputs_to_override.get((failed[0], failed[1]), )
                if found_key:
                    if v['input_output_pairs'][found_key]['output'] != returned[tid]['value']:
                        logger.info(
                            f"Override ({failed[0]},{failed[1]}) from"
                            f" {v['input_output_pairs'][found_key]['output']}"
                            f" to {returned[tid]['value']}"
                        )
                        v['input_output_pairs'][found_key]['output'] = returned[tid]['value']
                        total_overridden += 1
                else:
                    negated.append(' '.join(failed))

            v['test_negations'] = list()
            v['exclude_tests'] = list(results['had_errors'][v['instance_idx']].values())
            out_str = f"{json.dumps(v)}\n"
            f.write(out_str)
    logger.info(f"{total_overridden} total overridden")
    logger.info(f"{split_failed_execution} programs failed all sample execution for '{split}'")
    return fails, split_failed_execution


def verify_raw_programs(
        debug,
        file_path,
        out_path,
        num_false_pair_mod,
        use_negation,
        workers,
        seed,
        gold_to_generated_ratio
):
    rng = np.random.default_rng(seed)
    passed_programs = list(map(json.loads, file_path.open()))
    split = file_path.stem

    # Verify the samples again, and this time if they fail, discard them.
    unverified_samples = []
    total_true_examples = 0
    total_false_examples = 0
    total_fail_exec = 0
    verified_samples_by_idx = defaultdict(dict)
    for i, instance in tqdm(enumerate(passed_programs), total=len(passed_programs)):
        samples = make_samples_from_dict(
            deepcopy(instance),
            with_negation=use_negation
        )
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
    # if not debug:
    _, exec_results = check_io_sample_executes_correctly(
        split,
        unverified_samples,
        num_workers=workers,
        with_assert=True
    )

    logger.debug("Removing failed tests")
    for instance_idx, failed_samples in exec_results['failed_tests'].items():
        for task_id in failed_samples:
            verified_samples_by_idx[instance_idx].pop(task_id)
            total_fail_exec += 1

    logger.debug("Removing had errors")
    for instance_idx, failed_samples in exec_results['had_errors'].items():
        for task_id in failed_samples:
            verified_samples_by_idx[instance_idx].pop(task_id)
            total_fail_exec += 1

    logger.info(f"Removed {total_fail_exec} program(s) because they failed twice")
    # else:
    #     logger.warning(f"DEBUG IS ENABLED, SKIPPING EXECUTION")
    to_save_samples, stats = get_instances_to_save(
        verified_samples_by_idx,
        false_to_true_num_mod=num_false_pair_mod,
        rng=rng,
        gold_to_generated_ratio=gold_to_generated_ratio
    )
    true_count, false_count, mean_tracker, count_tracker = stats

    logger.info(f"{len(to_save_samples)} unique programs to save")

    for k, v in true_count.items():
        mean_tracker['non_neg_selected_false'].append(false_count[k])
        mean_tracker['non_neg_true_pairs'].append(v)
        mean_tracker['non_neg_total_pairs'].append(v + false_count[k])
    rng.shuffle(to_save_samples)
    with out_path.joinpath(f'{split}.jsonl').open('w') as f:
        for sample in to_save_samples:
            f.write(json.dumps(sample) + '\n')
    logger.info(f"Stats for '{split}':")

    for k, v in count_tracker.items():
        logger.info(f"\t{k:>24} = {v}")

    for k, v in mean_tracker.items():
        logger.info(f"\t{k:>24} = {np.mean(v):.2f}")

    logger.info(f"{total_fail_exec} failed execution for {split}")
