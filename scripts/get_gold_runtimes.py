import json
import sys
from pathlib import Path

import click
import numpy as np

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT
from src.evaluation.analysis import *


@click.command()
@click.option('--num-workers', '-n', default=4, type=int)
@click.option('--timeit-number', '-tn', default=100, type=int)
@click.option('--timeout', '-to', default=3, type=int)
def get_gold_runtime(num_workers, timeit_number, timeout):
    mbpp_data = list(map(json.loads, PROJECT_ROOT.joinpath('data/MBPP/test.jsonl').open()))
    human_eval_data = list(map(json.loads, PROJECT_ROOT.joinpath('data/human_eval.jsonl').open()))

    to_time_check = []
    num_to_check = 10
    for d in mbpp_data:
        for i in range(num_to_check):
            to_time_check.append({
                'run_info'       : 'MBPP',
                'task_id'        : f"{d['task_id']}",
                'tests'          : d['test_list'],
                'idx'            : i,
                'prediction'     : d['code'],
                'test_setup_code': d['test_setup_code']
            })
    for d in human_eval_data:

        for i in range(num_to_check):
            to_time_check.append({
                'run_info'       : 'HUMAN_EVAL',
                'task_id'        : f"{d['task_id']}",
                'idx'            : i,
                'tests'          : [f"{d['test']}\ncheck({d['entry_point']})"],
                'prediction'     : d['prompt'] + d['canonical_solution'],
                'test_setup_code': ''
            })

    results, with_errors = execute_time_check(to_time_check, num_workers,
                                              timeit_number=timeit_number,
                                              timeout=timeout)
    print(f"{len(with_errors)=}")
    with PROJECT_ROOT.joinpath('data/runtime_errors.txt').open('w') as f:
        for w in with_errors:
            f.write(f"{','.join(w)}\n")
    for task_name, task_runtimes in results.items():
        to_write = {}
        for task_id, runtimes in task_runtimes.items():
            to_write[task_id] = np.mean([d['runtime'] for d in runtimes])

        with PROJECT_ROOT.joinpath(f'data/gold_{task_name}_runtimes.json').open('w') as f:
            json.dump(to_write, f, indent=True)
    print("Done.")


if __name__ == '__main__':
    get_gold_runtime()
