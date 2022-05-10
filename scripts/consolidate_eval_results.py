import json
import argparse
import logging
import random
import shutil
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path
import sys
from dataclasses import asdict
from urllib.parse import urlparse, urljoin

import psutil
import ujson
import yaml
from lxml import etree
import multiprocessing as mp

from bs4 import BeautifulSoup
import click
import numpy as np
from tqdm import tqdm

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging, flatten
from src.evaluation.analysis import *


@click.command()
@click.argument('eval_dir')
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--num-workers', '-n', default=4, type=int)
@click.option('--timeit-number', '-tn', default=100, type=int)
@click.option('--timeout', '-to', default=3, type=int)
@click.option('--file-path', '-f', default=None)
def consolidate_results(eval_dir, debug, num_workers, timeit_number, timeout, file_path):
    setup_global_logging(f"consolidate_eval", PROJECT_ROOT.joinpath('logs'),
                         debug=debug, disable_issues_file=True)
    logger = logging.getLogger("consolidate_eval")

    if file_path is not None:
        file_path = PROJECT_ROOT.joinpath(file_path).absolute()

    eval_dir = PROJECT_ROOT.joinpath(eval_dir)
    logger.info(f"Getting '{eval_dir}'")

    mbpp_dir = eval_dir.joinpath('MBPP')
    assert mbpp_dir.exists()
    mbpp_eval_runs = {
        p.name.split('_Eval')[0]: p
        for p in mbpp_dir.glob('*')
        if p.is_dir() and p.joinpath('execution_metrics.json').exists()
    }
    logger.info(f"Found {len(mbpp_eval_runs)} directories in  '{mbpp_dir}'")

    human_eval_dir = eval_dir.joinpath('HUMAN_EVAL')
    assert human_eval_dir.exists()
    human_eval_runs = {
        p.name.split('_HEEval')[0]: p
        for p in human_eval_dir.glob('*')
        if p.is_dir() and p.joinpath('execution_metrics.json').exists()
    }
    logger.info(f"Found {len(human_eval_runs)} directories in  '{human_eval_dir}'")

    if set(human_eval_runs) != set(mbpp_eval_runs):
        human_eval_missing = set(human_eval_runs).difference(set(mbpp_eval_runs))
        mbpp_missing = set(mbpp_eval_runs).difference(set(human_eval_runs))
        logger.warning(f'{len(human_eval_missing) + len(mbpp_missing)} Missing run(s):')
        for r in mbpp_missing:
            logger.warning(f"\tHUMAN_EVAL {r}")
        for r in human_eval_missing:
            logger.warning(f"\tMBPP {r}")

    results = defaultdict(lambda: {'MBPP': {}, 'HUMAN_EVAL': {}})
    to_time_check = []
    all_program_stats = {}

    for task_name, task_dict in {'HUMAN_EVAL': human_eval_runs, 'MBPP': mbpp_eval_runs}.items():

        logger.info(f'Parsing {task_name} runs')
        task_program_stats = {}
        for run_name, path in task_dict.items():
            if file_path is not None and file_path != path:
                continue
            if not path.joinpath('test.jsonl'):
                logger.error(f"{run_name} for task {task_name} is missing 'test.jsonl'")
                continue
            if not path.joinpath('execution_metrics.json'):
                logger.error(f"{run_name} for task {task_name} "
                             f"is missing 'execution_metrics.json'")
                continue
            run_results, program_stats, run_to_time_check = parse_eval_results_dir(task_name, path)
            results[run_name][task_name] = run_results
            task_program_stats[run_name] = program_stats
            to_time_check.extend(
                [{'run_info': (run_name, task_name), **t} for t in run_to_time_check])
        all_program_stats[task_name] = task_program_stats

    out_dir = PROJECT_ROOT.joinpath('data', f'eval_analysis')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for task_name in ['MBPP', 'HUMAN_EVAL']:

        with out_dir.joinpath(f'{task_name}.jsonl').open('w') as f:
            for run_name, run_results in results.items():
                f.write(f"{json.dumps({'run_name': run_name, **run_results[task_name]})}\n")


        with out_dir.joinpath(f'program_stats_{task_name}.jsonl').open('w') as f:
            for run_name, run_results in all_program_stats[task_name].items():
                f.write(f"{json.dumps({'run_name': run_name, 'stats': run_results})}\n")

    # result_runtimes, errors = execute_time_check(
    #     to_time_check, num_workers,
    #     timeit_number=timeit_number,
    #     timeout=timeout
    # )
    #
    # to_write_by_task = defaultdict(lambda: defaultdict(dict))
    # num_single_passed = 0
    # for (run_name, task_name), task_runtimes in result_runtimes.items():
    #     for task_id, runtimes in task_runtimes.items():
    #         runtimes_dict = {
    #             'passed': runtimes['passed_runtimes'], 'failed': runtimes['failed_runtimes']
    #         }
    #         all_runtimes = runtimes_dict['passed'] + runtimes_dict['failed']
    #         if len(all_runtimes) == 1 or len(runtimes_dict['passed']) == 1:
    #             num_single_passed += 1
    #         runtime_result_dict = {}
    #         for k, v in runtimes_dict.items():
    #             runtime_result_dict.update({
    #                 f'{k}_mean'         : np.mean(v),
    #                 f'{k}_std'          : np.std(v),
    #                 f'{k}_median'       : np.median(v),
    #                 f'{k}_runtimes'     : v,
    #                 f'{k}_runtime_count': len(v)
    #             })
    #         runtime_result_dict.update({
    #             'mean'         : np.mean(all_runtimes),
    #             'std'          : np.std(all_runtimes),
    #             'median'       : np.median(all_runtimes),
    #             '25_percentile': np.percentile(all_runtimes, 25)
    #         })
    #         to_write_by_task[task_name][run_name][task_id] = runtime_result_dict
    #
    # logger.info(f"{num_single_passed} only had a single runtime pass")

    # with out_dir.joinpath('errors.txt').open('w') as f:
    #     for error in errors:
    #         f.write('\t'.join(map(str, error)) + '\n')
    # for task_name in ['MBPP', 'HUMAN_EVAL']:
    #     with out_dir.joinpath(f'runtimes_{task_name}.json').open('w') as f:
    #         json.dump(to_write_by_task[task_name], f)


if __name__ == '__main__':
    consolidate_results()
