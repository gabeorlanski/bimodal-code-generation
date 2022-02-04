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
from src.config.experiments import load_experiment_cards_from_file


@click.command()
@click.argument('experiment_card_path', metavar='<Path To Experiment Card>')
@click.argument('config_directory', metavar='<Path To Config Directory>')
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='experiments',
              help='The path to save the experiments')
@click.option('--overwrite-out-dir', '-overwrite', 'overwrite_output_dir',
              is_flag=True, default=False, help='Force overwriting the output directory.')
@click.pass_context
def cli(ctx, experiment_card_path, config_directory, debug, output_path, overwrite_output_dir):
    setup_global_logging(
        f"create_experiments",
        str(PROJECT_ROOT.joinpath('logs')),
        debug=debug
    )
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

    logger.info(f"Creating experiments from the base hydra configs.")
    for experiment in experiment_cards:
        logger.debug(f"Loading hydra config {experiment.base}")

        overrides_dict = flatten(experiment.overrides, sep='.')
        overrides_list = []
        for k, v in overrides_dict.items():
            override_key = k
            if "++" in k:
                override_key = f"++{k.replace('++', '')}"
            elif "+" in k:
                override_key = f"+{k.replace('+', '')}"
            overrides_list.append(f"{override_key}={v}")
        logger.info(f"{len(overrides_list)} overrides to use for {experiment.name}")
        logger.debug(f"Overrides for {experiment.name=}: {', '.join(overrides_list)}")
        save_path = output_path.joinpath(f"{experiment.save_name}.yaml")
        with initialize_config_dir(config_dir=str(config_directory.absolute()),
                                   job_name="create_configs"):
            cfg = compose(config_name=experiment.base, overrides=overrides_list)
            logger.info(f"Loaded config, now saving to {save_path}")
            with save_path.open('w', encoding='utf-8') as f:
                with open_dict(cfg):
                    cfg['name'] = experiment.name
                    cfg['group'] = experiment.group
                f.write(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == '__main__':
    cli()
