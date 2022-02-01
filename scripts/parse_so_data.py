import json
import argparse
import logging
import random
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
    setup_global_logging(f"{ctx.invoked_subcommand}_so", str(PROJECT_ROOT.joinpath('logs')),
                         debug=debug)
    ctx.ensure_object(dict)
    if not PROJECT_ROOT.joinpath(output_path).exists():
        PROJECT_ROOT.joinpath(output_path).mkdir(parents=True)
    ctx.obj['DEBUG'] = debug
    ctx.obj['OUT_PATH'] = output_path


@main.command('parse')
@click.argument('dump_path', metavar='<Data Path>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.argument('output_file_name', type=str, metavar='<Stem of the output file>')
@click.option('--cleaner', 'clean_fn_name', default='BASE',
              type=click.Choice(['BASE'], case_sensitive=False),
              help='Cleaning function to use.')
@click.option('--min-score', '-min', 'min_score_allowed', default=float('-inf'),
              type=float, help='Minimum score for either a question or an answer that is allowed.')
@click.option('--max-score', '-max', 'max_score_allowed', default=float('inf'), type=float,
              help='Maximum score for either a question or an answer that is allowed.')
@click.option('--body-contains', '-contains', 'words_body_must_have', default="",
              help='Comma separated list of words that the question body must contain.',
              callback=lambda ctx, params, l: [w for w in l.split(',') if w])
@click.option('--must-have-answers', is_flag=True, default=False,
              help='Questions must have answers.')
@click.option(
    '--question-score', is_flag=True, default=False,
    help='Only look at question score for filtering. If false, a '
         'post will pass the filter if ANY answer score (question or answer):'
         ' min score <= score <= max score.'
)
@click.option(
    '--only-question-body', is_flag=True, default=False,
    help='Only look at question body for filtering based on words'
)
@click.option(
    '--val-size', '-val', 'validation_size',
    type=int,
    default=100,
    help="Number of questions to put into the validation set."
)
@click.pass_context
def parse_so(
        ctx,
        dump_path,
        num_workers,
        output_file_name,
        clean_fn_name,
        min_score_allowed,
        max_score_allowed,
        words_body_must_have,
        must_have_answers,
        question_score,
        only_question_body,
        validation_size
):
    logger = logging.getLogger('parse_so')
    logger.info("Starting Parse")

    output_path = PROJECT_ROOT.joinpath(ctx.obj['OUT_PATH'], f"{output_file_name}.jsonl")
    val_path = PROJECT_ROOT.joinpath(ctx.obj['OUT_PATH'], f"{output_file_name}_val.jsonl")
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    logger.info("Initializing the filter.")
    post_filter = QuestionFilter(
        minimum_score=min_score_allowed,
        maximum_score=max_score_allowed,
        must_have_answer=must_have_answers,
        use_question_score=question_score,
        word_whitelist=words_body_must_have,
        only_question_body=only_question_body
    )

    logger.info("Filtering Arguments:")
    for k, v in asdict(post_filter).items():
        logger.info(f"\t{k:<24} = {v}")

    random.seed(1)

    filter_and_parse_so_posts(
        dump_path,
        output_path,
        val_path,
        num_workers,
        clean_fn_name,
        post_filter,
        validation_size
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
