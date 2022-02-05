import json
import argparse
import logging
import random
import shutil
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import click

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging, flatten
from src.config.experiments import load_experiment_cards_from_file, save_experiment_cards


def create_experiments(
        experiment_card_path,
        config_directory,
        debug,
        output_path,
        overwrite_output_dir):
    logger = logging.getLogger('create_experiments')
    experiment_card_path = PROJECT_ROOT.joinpath(experiment_card_path)
    logger.info(f"Creating Experiment configs from "
                f"'{experiment_card_path.resolve().absolute()}'")

    config_directory = PROJECT_ROOT.joinpath(config_directory)
    logger.info(f"Looking for hydra configs in '{config_directory.resolve().absolute()}'")

    output_path = PROJECT_ROOT.joinpath(output_path)
    logger.info(f"Experiment configs will be written to '{output_path.resolve().absolute()}'")
    experiment_cards = load_experiment_cards_from_file(experiment_card_path)

    logger.debug("Looking for output dir")
    if output_path.exists():
        logger.debug("Output dir exists")
        if overwrite_output_dir:
            logger.warning(f"Deleting the directory at {output_path}")
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True)
        else:
            logger.warning(f"Directory {output_path} already exists")
    else:
        logger.info("Directory does not exist, creating it.")
        output_path.mkdir(parents=True)

    save_experiment_cards(
        experiment_cards,
        output_path,
        config_directory
    )


@click.command()
@click.argument('experiment_card_path', metavar='<Path To Experiment Card>')
@click.argument('config_directory', metavar='<Path To Config Directory>')
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='experiments',
              help='The path to save the experiments')
@click.option('--overwrite-out-dir', '-overwrite', 'overwrite_output_dir',
              is_flag=True, default=False, help='Force overwriting the output directory.')
def cli(
        experiment_card_path,
        config_directory,
        debug,
        output_path,
        overwrite_output_dir
):
    setup_global_logging(
        f"create_experiments",
        str(PROJECT_ROOT.joinpath('logs')),
        debug=debug
    )
    # Split them up like this so I can actually test the function, click makes
    # that impossible.
    create_experiments(
        experiment_card_path=experiment_card_path,
        config_directory=config_directory,
        debug=debug,
        output_path=output_path,
        overwrite_output_dir=overwrite_output_dir
    )


if __name__ == '__main__':
    cli()
