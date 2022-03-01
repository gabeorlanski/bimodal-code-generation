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
from src.data.parse_so import parse_so_dump, filter_and_parse_so_posts, QuestionFilter
from src.data.parse_so.filtering import create_filter_for_so_data


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
    '--seed', type=int, default=1, help="Seed to use"
)
@click.pass_context
def consolidate_so_data(
        ctx,
        name,
        filter_file,
        dump_path,
        seed
):
    debug = ctx.obj['DEBUG']
    setup_global_logging(f"consolidate", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('consolidate')
    logger.info("Starting Consolidate")

    output_path = PROJECT_ROOT.joinpath("data", "dumps")
    if not output_path.exists():
        output_path.mkdir()

    filter_dict = json.loads(
        PROJECT_ROOT.joinpath(filter_file).read_text()
    )

    all_questions = [qid for t in filter_dict.values() for qid in t]
    logger.info(f"Total questions={len(all_questions)}")

    val_questions = min(1000, int(.1 * len(all_questions)))
    logger.info(f"{val_questions} questions will be used for validation set")
    sample_mask = np.zeros((len(all_questions),), dtype=bool)
    rng = np.random.default_rng(seed)
    sample_mask[rng.choice(len(all_questions), (val_questions,), replace=False)] = True

    logger.info("Creating mask")
    is_in_val = {}
    for in_val, qid in zip(sample_mask, all_questions):
        is_in_val[qid] = in_val

    question_path = PROJECT_ROOT.joinpath(dump_path, 'questions')
    train_file = output_path.joinpath(f"{name}.jsonl").open('w')
    val_file = output_path.joinpath(f"{name}_val.jsonl").open('w')

    update_freq = 1000 if debug else 25000
    for tag_name, questions in filter_dict.items():
        logger.info(f"Handling tag {tag_name}")
        line_num = 0
        found = 0

        # Use a dict to check if they exist because searching dict O(1)
        questions_looking_for = {k: True for k in questions}

        for line in tqdm(question_path.joinpath(f"{tag_name}.jsonl").open()):
            parsed = json.loads(line)
            line_num += 1
            if parsed['id'] in questions_looking_for:
                questions_looking_for.pop(parsed['id'])
                if is_in_val[parsed['id']]:
                    val_file.write(line.strip() + "\n")
                else:
                    train_file.write(line.strip() + '\n')
                found += 1

            if line_num % update_freq == 0:
                logger.info(f"Finished {line_num}, found {found:>8}/{len(questions)}")

            if not questions_looking_for:
                logger.info(f"Found all looking for")
                break
    train_file.close()
    val_file.close()


@main.command('parse')
@click.argument('dump_path', metavar='<Data Path>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.option(
    '--out-name', default=None, help="name"
)
@click.pass_context
def parse_dump(
        ctx,
        dump_path,
        num_workers,
        out_name
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
        debug
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
    '--seed', type=int, default=1, help="Seed to use"
)
@click.pass_context
def filter_tags(ctx, parsed_path, tag_filter_file, out_path, blacklist, seed):
    debug = ctx.obj['DEBUG']
    setup_global_logging(f"filter", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('filter')
    logger.info("Starting filter")
    tag_files_to_get = create_filter_for_so_data(
        parsed_path=parsed_path,
        tag_filter_file=tag_filter_file,
        blacklist=blacklist,
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
    setup_global_logging(f"filter", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    logger = logging.getLogger('filter')
    logger.info(f"Making the KG for {parsed_path}")
    parsed_path = PROJECT_ROOT.joinpath(parsed_path)
    question_path = parsed_path.joinpath('question_overview.json')
    logger.info("Loading the parsed question overview")
    question_overview = ujson.load(question_path.open())

    logger.info(f"{len(question_overview)} questions found")
    knowledge_graph = defaultdict(Counter)
    for question_id, question_dict in tqdm(question_overview.items(), total=len(question_overview)):

        tags = question_dict.get('tags', [])
        if not tags:
            continue
        for i in tags:
            for j in tags:
                if i == j:
                    continue
                knowledge_graph[i][j] += 1

    logger.info(f"{len(knowledge_graph)} unique first tags")
    kg_path = PROJECT_ROOT.joinpath('data', 'knowledge_graph')
    if not kg_path.exists():
        kg_path.mkdir()
    with kg_path.joinpath(f"{parsed_path.stem}_kg.json").open('w') as kg_file:
        json.dump(knowledge_graph, kg_file, indent=True)


if __name__ == "__main__":
    main()
