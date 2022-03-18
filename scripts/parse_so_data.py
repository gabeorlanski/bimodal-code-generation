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


# Here just to allow the grouping.
@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='data/stack_exchange',
              help='The path to save the results.')
@click.pass_context
def main(ctx, debug, output_path):
    ctx.ensure_object(dict)
    if not PROJECT_ROOT.joinpath(output_path).exists():
        PROJECT_ROOT.joinpath(output_path).mkdir(parents=True)
    ctx.obj['DEBUG'] = debug
    ctx.obj['OUT_PATH'] = output_path


@main.command('consolidate')
@click.argument("name")
@click.argument("filter_file")
@click.argument("dump_path")
@click.option(
    '--buffer-size',
    '-buffer',
    'max_buffer_size',
    type=int,
    help='Buffer Size',
    default=10000000
)
@click.option(
    '--max-val-size',
    '-val',
    'max_val_size',
    type=int,
    help='Maximum Validation Set Size',
    default=2500
)
@click.option(
    '--seed', type=int, default=1, help="Seed to use"
)
@click.pass_context
def consolidate_so_data_from_cli(
        ctx,
        name,
        filter_file,
        dump_path,
        max_buffer_size,
        seed,
        max_val_size
):
    """
    Wrapper that allows me to unittest the underlying consolidate function w/o
    needing to simulate the command line.
    """
    if not PROJECT_ROOT.joinpath('logs', 'consolidate').exists():
        PROJECT_ROOT.joinpath('logs', 'consolidate').mkdir(parents=True)
    setup_global_logging(f"consolidate_{name}", str(PROJECT_ROOT.joinpath('logs', 'consolidate')),
                         debug=ctx.obj['DEBUG'])
    logger = logging.getLogger(f"consolidate_{name}")
    logger.info("Starting Consolidate")
    consolidate_so_data(
        name=name,
        filter_file=filter_file,
        dump_path=dump_path,
        max_buffer_size=max_buffer_size,
        seed=seed,
        debug=ctx.obj['DEBUG'],
        output_path=ctx.obj['OUT_PATH'],
        max_val_size=max_val_size
    )


@main.command('parse')
@click.argument('dump_path', metavar='<Data Path>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.option('--buffer-size', '-buffer', type=int, help='Buffer Size', default=1000000)
@click.option(
    '--out-name', default=None, help="name"
)
@click.pass_context
def parse_dump(
        ctx,
        dump_path,
        num_workers,
        out_name,
        buffer_size
):
    debug = ctx.obj['DEBUG']

    setup_global_logging(f"parse_dump", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('parse_dump')
    output_path = PROJECT_ROOT.joinpath(ctx.obj['OUT_PATH'])
    path_to_dump = PROJECT_ROOT.joinpath(dump_path)
    logger.info(f"Starting parse_so with inputs {path_to_dump} "
                f"and outputting to {output_path}")
    try:
        posts_path, *_ = validate_files_exist(
            path_to_dump, ["Posts.xml"]
        )
    except FileExistsError as e:
        logger.error(f"Missing '{e.file}' in '{path_to_dump.resolve()}' ")
        raise e

    dump_name = path_to_dump.stem.split(".")[0]
    if out_name is None:
        output_path = output_path.joinpath(dump_name)
    else:
        output_path = output_path.joinpath(out_name)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    parse_so_dump(
        posts_path,
        num_workers,
        output_path,
        debug,
        buffer_size=buffer_size
    )


@main.command('filter')
@click.argument('parsed_path', metavar='<Data Path>')
@click.argument('tag_filter_file',
                metavar="<Path to list of tags to filter, use RANDOM for random selection>")
@click.argument('out_name', metavar="<Path to save to>")
@click.option(
    '--blacklist', default=None, help="Blacklist of questions to not include."
)
@click.option(
    '--tag-blacklist', default=None, help='Tag blacklist'
)
@click.option(
    '--random-size', type=int, default=5e6, help='Tag blacklist'
)
@click.option(
    '--seed', type=int, default=1, help="Seed to use"
)
@click.pass_context
def filter_tags(
        ctx,
        parsed_path,
        tag_filter_file,
        out_name,
        blacklist,
        tag_blacklist,
        random_size,
        seed
):
    debug = ctx.obj['DEBUG']
    setup_global_logging(f"filter", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('filter')
    logger.info("Starting filter")
    tag_files_to_get = create_filter_for_so_data(
        parsed_path=PROJECT_ROOT.joinpath(parsed_path, 'question_overview.json'),
        tag_filter_file=tag_filter_file,
        blacklist=blacklist if blacklist else None,
        tag_blacklist_path=tag_blacklist,
        debug=debug,
        random_size=random_size,
        seed=seed
    )

    filter_path = PROJECT_ROOT.joinpath('data', 'filters')
    if not filter_path.exists():
        logger.info(f"Creating {filter_path}")
        filter_path.mkdir()

    logger.info(f"Saving Filter to {filter_path.joinpath(f'{out_name}.json')}")

    with filter_path.joinpath(f'{out_name}.json').open('w') as filter_file:
        json.dump(tag_files_to_get, filter_file, indent=True)


@main.command('make_kg')
@click.argument('parsed_path', metavar='<Data Path>')
@click.pass_context
def make_kg(ctx, parsed_path):
    debug = ctx.obj['DEBUG']
    setup_global_logging(f"make_kg", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('make_kg')
    logger.info(f"Making the KG for {parsed_path}")
    create_tag_knowledge_graph(parsed_path)


@main.command('make_tag_info')
@click.argument('parsed_path', metavar='<Parsed Data Path>')
@click.argument('dump_path', metavar='<Dump Path>')
@click.argument('tag_synonyms_path', metavar='<Tag Synonyms Path>')
@click.pass_context
def make_tag_info(ctx, parsed_path, dump_path, tag_synonyms_path):
    setup_global_logging(f"make_tag_info", str(PROJECT_ROOT.joinpath('logs')),
                         debug=ctx.obj['DEBUG'])
    logger = logging.getLogger('make_tag_info')
    logger.info("Starting make_tag_info")
    create_tag_info_file(parsed_path, dump_path, tag_synonyms_path)


def get_urls(batch):
    out = defaultdict(Counter)
    for b in batch:
        soup = BeautifulSoup(b['body'], 'lxml')
        a_tags = soup.find_all('a', href=True)
        for answer in b['answers'].values():
            soup = BeautifulSoup(answer['body'], 'lxml')
            a_tags.extend(soup.find_all('a', href=True))
        for link in a_tags:
            try:
                url_parsed = urlparse(link.get('href'))

                subdomain, domain, suffix = tldextract.extract(url_parsed.netloc)
                url = f"{domain}.{suffix}"
                if subdomain.strip() and subdomain != 'www':
                    url = f"{subdomain.strip()}.{url}"
                if not url or url == '.':
                    continue
            except:
                continue
            out[url][url_parsed.path] += 1
    return out


@main.command('get_urls')
@click.argument('dump_path')
@click.argument('num_workers', type=int)
@click.pass_context
def get_urls_from_dump(ctx, dump_path, num_workers):
    batch_size = 32
    buffer_size = 50000
    more_examples = True
    batches = []
    lines = 0
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    out_path = PROJECT_ROOT.joinpath('data', 'urls')
    if not out_path.exists():
        out_path.mkdir()

    urls_found = defaultdict(Counter)

    raw_lines_iter = iter(dump_path.open('r').readlines())
    while more_examples:
        buffer = []
        while len(batches) < buffer_size:
            try:
                line = json.loads(next(raw_lines_iter))
            except StopIteration:
                more_examples = False
                break
            lines += 1
            buffer.append(line)
            if len(buffer) == batch_size:
                batches.append(deepcopy(buffer))
                del buffer
                buffer = []

            if lines % 50000 == 0:
                print(f"Read {lines} lines. ")
        if buffer:
            batches.append(buffer)
        print(f"Read {lines} lines")
        print(f"Yielded {len(batches)} batches")
        finished = 0
        with mp.Pool(num_workers) as pool:
            for result in tqdm(pool.imap(get_urls, batches), total=len(batches), desc='Tokenizing'):
                for domain, paths in result.items():
                    for path, v in paths.items():
                        urls_found[domain][path] += v
                finished += 1
                if finished % 10000 == 0:
                    ram_pct = f"{psutil.virtual_memory()[2]:0.2f}%"
                    tqdm.write(f"Finished {finished}| RAM Used={ram_pct:<6}")
        del batches
        batches = []

    print(f"{len(urls_found)} unique urls found")
    with out_path.joinpath(f'{dump_path.stem}.json').open('w') as f:
        json.dump(urls_found, f, indent=True, sort_keys=True)


@main.command('tag_info')
@click.argument('dump_path')
@click.pass_context
def get_tag_info(ctx, dump_path):
    lines = 0
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    out_path = PROJECT_ROOT.joinpath('data', 'tags')
    if not out_path.exists():
        out_path.mkdir()

    tag_counter = defaultdict(lambda: defaultdict(list))

    for line in map(json.loads, dump_path.open('r').readlines()):
        lines += 1
        for i in range(len(line['tags'])):
            tag_counter[str(i + 1)][','.join(line['tags'][:i + 1])].append(line['score'])
        tag_counter['all'][','.join(line['tags'])].append(line['score'])

        if lines % 50000 == 0:
            print(f"Read {lines} lines. ")

    total_tag_combos = sum(map(len, tag_counter.values()))
    print(f"{total_tag_combos} unique tag combos found found")
    pbar = tqdm(total=total_tag_combos, desc='Aggregating')
    tag_means = {}
    for k, v in tag_counter.items():
        k_means = {}
        for tag_name, scores in v.items():
            k_means[tag_name] = {
                'mean'  : np.mean(scores),
                'median': np.median(scores),
                'std'   : np.std(scores),
                'len'   : len(scores)
            }
            pbar.update(1)
        tag_means[k] = k_means
    pbar.close()

    with out_path.joinpath(f'{dump_path.stem}.json').open('w') as f:
        ujson.dump(tag_means, f)


if __name__ == "__main__":
    main()
