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
import csv
import tldextract
from urllib.parse import urljoin
import requests

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
@click.option('--debug-samples', '-n', default=-1, type=int)
def consolidate_results(eval_dir, debug, num_workers, debug_samples):
    setup_global_logging(f"consolidate_eval", PROJECT_ROOT.joinpath('logs'),
                         debug=debug)
    logger = logging.getLogger("consolidate_eval")

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
    parsed = 0

    for task_name, task_dict in {'MBPP': mbpp_eval_runs, 'HUMAN_EVAL': human_eval_runs}.items():
        logger.info(f'Parsing {task_name} runs')
        for run_name, path in tqdm(task_dict.items(), desc='Parsing'):
            if debug_samples > 0 and parsed >= debug_samples:
                break
            if not path.joinpath('test.jsonl'):
                logger.error(f"{run_name} for task {task_name} is missing 'test.jsonl'")
                continue
            if not path.joinpath('execution_metrics.json'):
                logger.error(f"{run_name} for task {task_name} "
                             f"is missing 'execution_metrics.json'")
                continue
            run_results, run_to_time_check = parse_eval_results_dir(task_name, path)
            results[run_name][task_name] = run_results
            to_time_check.extend(
                [{'run_info': (run_name, task_name), **t} for t in run_to_time_check])
            parsed += 1

    execute_time_check(to_time_check, num_workers,debug)

    for task_name in ['MBPP', 'HUMAN_EVAL']:
        with PROJECT_ROOT.joinpath('data', f'eval_{task_name}.jsonl').open('w') as f:
            for run_name, run_results in results.items():
                f.write(f"{json.dumps({'run_name': run_name, **run_results[task_name]})}\n")


if __name__ == '__main__':
    consolidate_results()
