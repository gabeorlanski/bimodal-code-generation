import json
import argparse
import logging
from pathlib import Path
import sys
from dataclasses import asdict

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.parse_so import filter_so_dump, filter_and_parse_so_posts, QuestionFilter
import click


# Here just to allow the grouping.
@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='data/stack_exchange',
              help='The path to save the results.')
@click.pass_context
def main(ctx, debug, output_path):
    setup_global_logging(f"{ctx.invoked_subcommand}_so", str(PROJECT_ROOT.joinpath('logs')), debug=debug)
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['OUT_PATH'] = output_path


@main.command('parse')
@click.argument('dump_path', metavar='<Data Path>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.option('--cleaner', 'clean_fn', default='BASE',
              type=click.Choice(['BASE'], case_sensitive=False),
              help='Cleaning function to use.')
@click.option('--minscore', '-min', 'min_score_allowed', default=float('-inf'),
              type=float, help='Minimum score for either a question or an answer that is allowed.')
@click.option('--maxscore', '-max', 'max_score_allowed', default=float('inf'), type=float,
              help='Maximum score for either a question or an answer that is allowed.')
@click.option('--body-contains', '-contains', 'words_body_must_have', default="",
              help='Comma separated list of words that the question body must contain.',
              callback=lambda ctx, params, l: l.split(','))
@click.pass_context
def parse_so(
        ctx,
        dump_path,
        num_workers,
        clean_fn,
        min_score_allowed,
        max_score_allowed,
        words_body_must_have
):
    logger = logging.getLogger('parse_so')
    logger.info("Starting Parse")

    output_path = PROJECT_ROOT.joinpath(ctx.obj['OUT_PATH'])
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    logger.info("Initializing the filter.")
    post_filter = QuestionFilter(
        minimum_score=min_score_allowed,
        maximum_score=max_score_allowed,
        word_whitelist=words_body_must_have
    )

    logger.info("Filtering Arguments:")
    for k, v in asdict(post_filter).items():
        logger.info(f"\t{k:<24} = {v}")

    filter_and_parse_so_posts(
        dump_path,
        output_path,
        num_workers,
        clean_fn,
        post_filter
    )


@main.command('filter')
@click.argument('dump_path', metavar='<Data Path>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.argument('tag_filter_file')
@click.pass_context
def filter_so(
        ctx,
        dump_path,
        num_workers,
        tag_filter_file,
):
    debug = ctx.obj['DEBUG']
    logger = logging.getLogger('filter_so')
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

    logger.info(f"Reading tags filters from {tag_filter_file}")
    tag_filters = PROJECT_ROOT.joinpath(tag_filter_file).read_text('utf-8').splitlines(False)
    dump_name = path_to_dump.stem.split(".")[0]
    output_path = output_path.joinpath(dump_name)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    failures = filter_so_dump(
        posts_path,
        num_workers,
        output_path,
        tag_filters,
        debug
    )

    logger.info(f"Saving stats for '{dump_name}' to {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with output_path.joinpath(f'{dump_name}_failures.json').open('w', encoding='utf-8') as f:
        json.dump(failures, f, indent=True)


if __name__ == "__main__":
    main()
