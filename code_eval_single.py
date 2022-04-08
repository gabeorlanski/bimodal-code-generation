import argparse
import json
import logging

import click
from omegaconf import OmegaConf, open_dict
from pathlib import Path
import os
import yaml
import wandb
from src.common import setup_global_logging, PROJECT_ROOT
from src.evaluation.code_eval import evaluate_code_from_file, BASE_ERROR_TYPES
from src.config import setup_tracking_env_from_cfg, get_config_for_tracking
from src.common.util import flatten


@click.command()
@click.argument('file', metavar="<predictions dir>")
@click.argument('num_workers', type=int, metavar="<Number of workers>")
@click.argument('seq_per_sample', type=int, metavar="<Number of workers>")
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--timeout',
    default=3.0,
    type=float,
    help="The amount to use for timeout"
)
def run(file, num_workers, seq_per_sample, debug, timeout):
    # I just needed a way to get the parent directory.
    path_to_preds = PROJECT_ROOT.joinpath(file)
    setup_global_logging(
        f'execution',
        PROJECT_ROOT.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        disable_issues_file=True,
        debug=debug
    )
    logger = logging.getLogger('execution')
    logger.info(f"CWD={Path().absolute().resolve()}")
    logger.info(f"Executing code from {path_to_preds}")
    results = evaluate_code_from_file(
        str(path_to_preds),
        samples_per_problem=seq_per_sample,
        num_workers=num_workers,
        timeout=timeout
    )

    if not PROJECT_ROOT.joinpath('data/single_exec').exists():
        PROJECT_ROOT.joinpath('data/single_exec').mkdir(parents=True)
    save_path = PROJECT_ROOT.joinpath('data/single_exec', f'{path_to_preds.stem}.json')

    with save_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=True)


if __name__ == "__main__":
    run()
