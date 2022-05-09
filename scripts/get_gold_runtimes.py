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


def get_gold_runtime():
    mbpp_data = list(map(json.loads, PROJECT_ROOT.joinpath('data/MBPP/test.jsonl').open()))
    human_eval_data = list(map(json.loads, PROJECT_ROOT.joinpath('data/human_eval.jsonl').open()))

    to_time_check = []
    num_to_check = 10
    for d in mbpp_data:
        for i in range(num_to_check):
            to_time_check.append({
                'run_info'  : 'MBPP',
                'task_id'   : f"{d['task_id']}.{i}",
                'tests'     : [d['test_setup_code']] + d['test_list'],
                'prediction': d['code']
            })
    for d in human_eval_data:

        for i in range(num_to_check):
            to_time_check.append({
                'run_info'  : 'HUMAN_EVAL',
                'task_id'   : f"{d['task_id']}.{i}",
                'tests'     : [f"{d['test']}\ncheck({d['entry_point']})"],
                'prediction': d['prompt'] + d['canonical_solution']
            })

    results = execute_time_check(to_time_check, 8)
    print("Done.")


if __name__ == '__main__':
    get_gold_runtime()
