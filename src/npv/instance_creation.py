import logging
import math
from collections import defaultdict, Counter
from copy import deepcopy

from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = [
    "make_samples_from_dict",
    "get_instances_to_save"
]

OP_NEGATION_MAP = {
    k: v for a, b in [
        ("!=", '=='),
        ("<", '>='),
        (">", '<='),
        ("is not", "is"),
        ("not in", "in"),
    ] for k, v in [(a, b), (b, a)]
}


def make_samples_from_dict(single_instance, with_negation=False):
    io_pairs = single_instance.pop('input_output_pairs')
    specific_fixes = single_instance.pop('test_negations', [])
    excluded = single_instance.pop('exclude_tests', [])

    single_instance['original_task_id'] = single_instance.pop("task_id")
    out = []

    io_combos = set()

    pred_idx = 0
    to_keep_by_result = defaultdict(list)
    for i, left in enumerate(io_pairs):

        for j, right in enumerate(io_pairs):
            op = left['ops']
            result = right['output'] == left['output']
            is_manual_fix = False
            combo = f"{left['input']} {left['ops']} {right['output']}"
            if combo in specific_fixes and combo in io_combos:
                raise ValueError(combo)

            if combo in excluded:
                continue
            if combo in specific_fixes:
                result = not result
                is_manual_fix = True

            if combo not in io_combos:
                io_combos.add(combo)
                exec_info = {
                    'input'              : left['input'],
                    'output'             : right['output'],
                    'op'                 : op,
                    'is_original'        : i == j,
                    'is_input_generated' : left.get('is_generated', False),
                    'is_output_generated': right.get('is_generated', False),
                }
                to_keep_by_result[str(result)].append(
                    [exec_info, result, is_manual_fix]
                )

    to_keep = to_keep_by_result['True'] + to_keep_by_result['False']
    comparisons = set()
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
        comparisons.add(
            f"{execute_info['input']} {execute_info['op']} {execute_info['output']} {res}"
        )
        pred_idx += 1

        if res in [True, False] and with_negation:
            negated_op = OP_NEGATION_MAP[execute_info['op']]
            negate_comparison = f"{execute_info['input']} {negated_op} " \
                                f"{execute_info['output']} {not res}"
            if negate_comparison in comparisons:
                continue
            negation_pred_id = f"{single_instance['task']}_{single_instance['instance_idx']}_{pred_idx}"
            negation_pred_dict = deepcopy(single_instance)
            negation_pred_dict['task_id'] = negation_pred_id
            execute_info['op'] = negated_op
            negation_pred_dict.update(execute_info)
            negation_pred_dict['result'] = str(not res)
            negation_pred_dict['is_manual_fix'] = is_manual_fix
            negation_pred_dict['is_negation_of'] = original_pred_id
            out.append(negation_pred_dict)
            pred_idx += 1
            comparisons.add(negate_comparison)

    return out


def get_instances_to_save(
        verified_samples_by_idx,
        false_to_true_num_mod,
        rng,
        gold_to_generated_ratio,
):
    logger.info(f"{false_to_true_num_mod=}")
    pct_gold = gold_to_generated_ratio / (gold_to_generated_ratio + 1)
    logger.info(f"{pct_gold=:.2%} will be from gold")

    count_tracker = Counter()
    mean_tracker = defaultdict(list)

    to_save = []
    to_save_col = [
        'source_file',
        'task',
        'function',
        'description',
        'code',
        'context',
        'instance_idx',
        'original_task_id'
    ]
    false_count = {}
    true_count = {}
    for program_idx, sample_dict in tqdm(verified_samples_by_idx.items()):

        instance_dict = {
            k: v for k, v in next(iter(sample_dict.values())).items()
            if k in to_save_col
        }
        false_count[program_idx] = 0
        tid_by_result_and_input = defaultdict(lambda: defaultdict(list))

        false_pairs = defaultdict(list)
        true_tids = defaultdict(list)
        tid_to_io_dict = {}
        negations = {}
        num_true_pairs = 0
        num_false_pairs = 0

        num_gen_true = 0
        num_gen_false = 0

        has_true = has_false = False
        for sample in sample_dict.values():
            io_pair_dict = {
                'input'              : sample['input'],
                'op'                 : sample['op'],
                'output'             : sample['output'],
                'is_manual_fix'      : sample['is_manual_fix'],
                'is_negation_of'     : sample['is_negation_of'],
                'is_original'        : sample['is_original'],
                'task_id'            : sample['task_id'],
                'result'             : sample['result'],
                'is_input_generated' : sample['is_input_generated'],
                'is_output_generated': sample['is_output_generated'],
            }

            result_str = str(sample['result'])
            tid_to_io_dict[io_pair_dict['task_id']] = io_pair_dict
            tid_by_result_and_input[result_str][io_pair_dict['input']].append(
                io_pair_dict['task_id'])
            is_gen = io_pair_dict['is_input_generated'] or io_pair_dict['is_output_generated']
            if io_pair_dict['is_negation_of'] is None:
                if result_str == 'True':
                    if is_gen:
                        num_gen_true+=1
                        continue
                    has_true = True
                    num_true_pairs += 1
                    true_tids[sample['input']].append(io_pair_dict['task_id'])
                else:
                    if is_gen:
                        num_gen_false+=1
                        continue
                    has_false = True
                    num_false_pairs += 1
                    false_pairs[sample['input']].append(io_pair_dict)
            else:
                assert io_pair_dict['is_negation_of'] not in negations
                negations[io_pair_dict['is_negation_of']] = io_pair_dict['task_id']
        if not has_true or not has_false:
            logger.error(
                f"{program_idx} had {num_true_pairs} TRUE and {num_false_pairs} "
                f"FALSE pairs. Expecting at least 1 for each. Skipping."
            )
            continue

        true_examples_to_use = [ex for v in true_tids.values() for ex in v]
        # First add 1 false example for each input
        false_examples_to_use = []

        remaining_generated_false = []
        remaining_gold_false = []
        num_remaining_false = 0

        for input_str, io_pairs in false_pairs.items():

            # We give preference to non-generated outputs
            non_generated = [
                i for i, c in enumerate(io_pairs)
                if not c['is_output_generated']
            ]
            to_keep_idx = rng.choice(non_generated or range(len(io_pairs)))
            for i, v in enumerate(io_pairs):

                # We only want to force keep certain pairs if the input also
                # has a corresponding true pair.
                if i == to_keep_idx and input_str in true_tids:
                    false_examples_to_use.append(v['task_id'])
                else:
                    if v['is_output_generated'] or v['is_input_generated']:
                        remaining_generated_false.append(v['task_id'])
                    else:
                        remaining_gold_false.append(v['task_id'])
                    num_remaining_false += 1

        logger.debug(
            f"There are {len(remaining_gold_false) + len(remaining_generated_false)} "
            f"in the false pool for {program_idx}"
        )

        if false_to_true_num_mod != -1 and num_remaining_false > 0:
            total_to_select = min(min(num_false_pairs, num_true_pairs),
                                  math.floor(num_true_pairs * false_to_true_num_mod))
            total_to_select = int(max(0, total_to_select - len(false_examples_to_use)))
            logger.debug(
                f"{program_idx} has {num_true_pairs} true pairs. "
                f"Selecting {total_to_select} False"
            )

            num_gold_to_select = int(min(
                math.ceil(total_to_select * pct_gold),
                len(remaining_gold_false)
            ))
            num_generated_to_select = min(
                total_to_select - num_gold_to_select,
                len(remaining_generated_false)
            )

            logger.debug(f"Selecting {num_gold_to_select} gold False and "
                         f"{num_generated_to_select} generated False")
            selected_some = False
            if remaining_gold_false and remaining_gold_false:
                selected_some = True
                false_examples_to_use.extend(
                    rng.choice(
                        remaining_gold_false,
                        size=num_gold_to_select
                    )
                )
            if remaining_generated_false and remaining_generated_false:
                selected_some = True
                false_examples_to_use.extend(
                    rng.choice(
                        remaining_generated_false,
                        size=num_generated_to_select)

                )
            if not selected_some:
                logger.critical(f"DID NOT SELECT ANY for {program_idx}")
        else:
            false_examples_to_use.extend(remaining_gold_false + remaining_generated_false)

        true_count[program_idx] = num_true_pairs
        false_count[program_idx] = len(false_examples_to_use)
        if num_true_pairs != len(false_examples_to_use):
            logger.critical(f"Unequal {num_true_pairs} != "
                            f"{len(false_examples_to_use)} for {program_idx}")
        to_save_task_ids = []

        num_generated = 0

        for tid in true_examples_to_use:
            if tid_to_io_dict[tid]['is_input_generated']:
                num_generated += 1

            to_save_task_ids.append(tid)
            count_tracker['total_true'] += 1
            if tid in negations:
                to_save_task_ids.append(negations[tid])
                count_tracker['total_false'] += 1
        for tid in false_examples_to_use:
            if tid_to_io_dict[tid]['is_input_generated']:
                num_generated += 1

            to_save_task_ids.append(tid)
            count_tracker['total_false'] += 1
            if tid in negations:
                to_save_task_ids.append(negations[tid])
                count_tracker['total_true'] += 1

        count_tracker['all_pairs'] += len(to_save_task_ids)

        mean_tracker['is_generated'].append(num_generated)
        mean_tracker['gen_true'].append(num_gen_true)
        mean_tracker['gen_false'].append(num_gen_false)

        instance_dict['all_tasks'] = tid_to_io_dict
        instance_dict['instances'] = to_save_task_ids
        instance_dict['tid_by_result'] = dict(tid_by_result_and_input)


        to_save.append(instance_dict)

    stats = (true_count, false_count, mean_tracker, count_tracker)
    return to_save, stats
