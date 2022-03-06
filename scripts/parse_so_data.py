"""
Scripts to parse StackExchange dumps
"""

import json
import argparse
import logging
import random
from collections import defaultdict, Counter
from pathlib import Path
import sys
from dataclasses import asdict
from lxml import etree

import click
import numpy as np
from tqdm import tqdm

import ujson

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.parse_so import parse_so_dump
from src.data.parse_so.filtering import create_filter_for_so_data, consolidate_so_data


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

    setup_global_logging(f"consolidate_{name}", str(PROJECT_ROOT.joinpath('logs')),
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
@click.argument('out_path', metavar="<Path to save to>")
@click.option(
    '--blacklist', default=None, help="Blacklist of questions to not include."
)
@click.option(
    '--tag-blacklist', default=None, help='Tag blacklist'
)
@click.option(
    '--seed', type=int, default=1, help="Seed to use"
)
@click.pass_context
def filter_tags(ctx, parsed_path, tag_filter_file, out_path, blacklist, tag_blacklist, seed):
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
        seed=seed
    )

    filter_path = PROJECT_ROOT.joinpath('data', 'filters')
    if not filter_path.exists():
        logger.info(f"Creating {filter_path}")
        filter_path.mkdir()

    logger.info(f"Saving Filter to {filter_path.joinpath(f'{out_path}.json')}")

    with filter_path.joinpath(f'{out_path}.json').open('w') as filter_file:
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
    parsed_path = PROJECT_ROOT.joinpath(parsed_path)
    question_path = parsed_path.joinpath('question_overview.json')
    logger.info("Loading the parsed question overview")
    question_overview = ujson.load(question_path.open())

    logger.info(f"{len(question_overview)} questions found")
    knowledge_graph = defaultdict(Counter)
    total_questions = 0
    first_tag_counts = Counter()
    tag_counts = Counter()
    for question_id, question_dict in tqdm(question_overview.items(), total=len(question_overview)):

        tags = question_dict.get('tags', [])
        if not tags:
            continue
        first_tag, *rem_tags = tags
        first_tag_counts[first_tag] += 1
        tag_counts[first_tag] += 1
        for t in rem_tags:
            knowledge_graph[first_tag][t] += 1
            tag_counts[t] += 1
        total_questions += 1

    logger.info(f"{len(knowledge_graph)} unique tags")
    kg_path = PROJECT_ROOT.joinpath('data', 'knowledge_graph')
    if not kg_path.exists():
        kg_path.mkdir()

    with kg_path.joinpath(f"{parsed_path.stem}_kg.json").open('w') as kg_file:
        json.dump({
            'total_questions' : total_questions,
            'total_tags'      : len(knowledge_graph),
            'first_tag_counts': first_tag_counts,
            'tag_counts'      : tag_counts,
            'knowledge_graph' : knowledge_graph
        }, kg_file, indent=True)


@main.command('make_tag_info')
@click.argument('parsed_path', metavar='<Parsed Data Path>')
@click.argument('tag_xml_path', metavar='<Tag XML File Path>')
@click.pass_context
def make_tag_info(ctx, parsed_path, tag_xml_path):
    print(f"Making tag info from {parsed_path}")
    parsed_path = PROJECT_ROOT.joinpath(parsed_path)
    tag_xml_path = PROJECT_ROOT.joinpath(tag_xml_path)
    print(f"{parsed_path=}")
    print(f"{tag_xml_path=}")

    print("Reading tag data")
    tag_information = {}
    excerpt_id_mapping = {}
    wiki_id_mapping = {}
    for line in tqdm(tag_xml_path.open('r')):
        try:
            tag_dict = etree.XML(line).attrib
        except Exception as e:
            continue

        tag_name = tag_dict['TagName']
        if tag_name in tag_information:
            raise KeyError(f"{tag_dict['TagName']} is duplicated")
        tag_information[tag_name] = {
            'count'       : tag_dict['Count'],
            'id'          : tag_dict['Id'],
            'wiki'        : None,
            'wiki_date'   : None,
            'wiki_id'     : tag_dict.get('WikiPostId'),
            'excerpt'     : None,
            'excerpt_date': None,
            'excerpt_id'  : tag_dict.get('ExcerptPostId')
        }

        if 'ExcerptPostId' in tag_dict:
            excerpt_id_mapping[tag_dict['ExcerptPostId']] = tag_name

        if 'WikiPostId' in tag_dict:
            wiki_id_mapping[tag_dict['WikiPostId']] = tag_name

    print(f"{len(tag_information)} unique tags found")
    print(f"{len(excerpt_id_mapping)} excerpts to get")
    print(f"{len(wiki_id_mapping)} wiki items to get")

    excerpt_file = parsed_path.joinpath('wiki_excerpts.jsonl')
    print(f"Reading excerpts file {excerpt_file}")

    if excerpt_file.exists():
        orphaned_excerpts = 0
        with excerpt_file.open() as f:
            for excerpt in tqdm(map(json.loads, f.readlines())):
                try:
                    tag_name = excerpt_id_mapping[excerpt['id']]
                except KeyError:
                    orphaned_excerpts += 1
                    continue

                tag_information[tag_name]['excerpt'] = excerpt['body']
                tag_information[tag_name]['excerpt_date'] = excerpt['date']
        print(f"{orphaned_excerpts} orphaned excerpts")

    else:
        print(f"Could not find {excerpt_file}")

    wiki_file = parsed_path.joinpath('wiki.jsonl')
    print(f"Reading wiki file {wiki_file}")

    if wiki_file.exists():
        orphaned_wiki = 0
        with wiki_file.open() as f:
            for wiki in tqdm(map(json.loads, f.readlines())):
                try:
                    tag_name = excerpt_id_mapping[wiki['id']]
                except KeyError:
                    orphaned_wiki += 1
                    continue

                tag_information[tag_name]['wiki'] = wiki['body']
                tag_information[tag_name]['wiki_body'] = wiki['date']
        print(f"{orphaned_wiki} orphaned wiki pages")

    else:
        print(f"Could not find {wiki_file}")

    save_path = parsed_path.joinpath('tags.json')
    print(f"Saving to {save_path}")
    with save_path.open('w') as f:
        json.dump(tag_information, f, indent=True)


if __name__ == "__main__":
    main()
