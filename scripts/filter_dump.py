"""
Scripts to parse StackExchange dumps
"""

import json
import argparse
import logging
import random
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path
import sys
from dataclasses import asdict
from urllib.parse import urlparse

import psutil
import ujson
from lxml import etree
import multiprocessing as mp

from bs4 import BeautifulSoup
import click
import numpy as np
from tqdm import tqdm
import csv
import tldextract

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.parse_so import *


def get_filtered_from_file(file_path, num_answers, question_score):
    passed_filter = []
    lines_read = 0
    with file_path.open('r') as f:
        for line in map(ujson.loads, f):
            if line['answer_count'] >= num_answers and line['score'] >= question_score:
                passed_filter.append(line)

            lines_read += 1
            if lines_read % 25000 == 0:
                print(f"Passed Filter {len(passed_filter)}/{lines_read}")
    return passed_filter


# Here just to allow the grouping.
@click.command()
@click.argument('dump_name')
@click.argument('out_name')
@click.option('--num-answers', '-A', default=2, type=int)
@click.option('--question-score', '-Q', default=5, type=int)
@click.option('--seed', default=1, type=int)
@click.option('--max-val-size', default=10000, type=int)
def main(dump_name, out_name, num_answers, question_score, seed, max_val_size):
    dump_path = PROJECT_ROOT.joinpath('data', 'dumps')
    train_file = dump_path.joinpath(f'{dump_name}.jsonl')
    print(f"Reading {train_file}")
    passed_filter = get_filtered_from_file(train_file, num_answers, question_score)

    print(f"{len(passed_filter)} passed the filter from the train file")
    val_file = dump_path.joinpath(f'{dump_name}_val.jsonl')
    passed_filter.extend(get_filtered_from_file(val_file, num_answers, question_score))

    print(f"{len(passed_filter)} total passed the filter")
    print("Creating val set")
    val_questions = min(max_val_size, int(.1 * len(passed_filter)))
    print(f"Using {val_questions} questions for the val set")
    rng = np.random.default_rng(seed)
    val_set_mask = {qid['id']: False for qid in passed_filter}
    val_question_indices = rng.choice(len(passed_filter), (val_questions,), replace=False)
    for q_idx in val_question_indices:
        val_set_mask[passed_filter[q_idx]['id']] = True

    print(f"Saving to {dump_path.joinpath(out_name + '.jsonl')}")
    out_train_file = dump_path.joinpath(f"{out_name}.jsonl").open('w')
    out_val_file = dump_path.joinpath(f"{out_name}_val.jsonl").open('w')
    for q in tqdm(passed_filter, desc="Saving"):
        if val_set_mask[q['id']]:
            out_val_file.write(f"{json.dumps(q)}\n")
        else:
            out_train_file.write(f"{json.dumps(q)}\n")
    out_train_file.close()
    out_val_file.close()


if __name__ == "__main__":
    main()
