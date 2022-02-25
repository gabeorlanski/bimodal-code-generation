"""
Scripts to parse StackExchange dumps
"""

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
from src.data.parse_so import parse_so_dump, filter_and_parse_so_posts, QuestionFilter
import click


# Here just to allow the grouping.
@click.command()
@click.argument('file_name')
@click.option('--dump-path', default='data/parsed_so')
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='data/equal_so',
              help='The path to save the results.')
def main(file_name, debug, dump_path):
    setup_global_logging(f"equal_groups_{file_name}_so", str(PROJECT_ROOT.joinpath('logs')),
                         debug=ctx.obj['DEBUG'])


if __name__ == "__main__":
    main()
