import logging
from typing import Optional, List
from hydra import compose, initialize
import yaml
from omegaconf import OmegaConf, open_dict
import os
from pathlib import Path
import click

from src import config
from src.evaluation import evaluate
from src.common import setup_global_logging, PROJECT_ROOT
from src.data.tensorize import tensorize
from src.data.stackoverflow import StackOverflowTextProcessor


@click.command()
@click.argument('name', metavar="<Name of this dataset>")
@click.argument('objective', metavar='<Objective to use>')
@click.argument('processor_name', metavar='<Processor to use>')
@click.argument('model_name', metavar='<Model to use>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--data-path', '-I',
    help='Path to the input data',
    default='data/stackoverflow'
)
@click.option(
    '--out-path', '-O',
    help='Path for saving the data',
    default='data/tensorized'
)
@click.option(
    '--validation-file-name', '-val', default=None,
    help='Name of the validation raw data, if not provided, will use the {name}_val')
@click.option(
    '--override-str',
    help='Bash does not like lists of variable args. so pass as seperated list of overrides, seperated by spaces.',
    default=''
)
def tensorize_data(
        name: str,
        objective: str,
        processor_name: str,
        model_name: str,
        num_workers: int,
        override_str: str,
        debug: bool,
        data_path,
        validation_file_name,
        out_path
):
    override_list = [
        f"name={name}",
        f"processor={processor_name}"
    ]
    override_list.extend(override_str.split(' ') if override_str else [])

    initialize(config_path="conf", job_name="train")
    cfg = compose(config_name="tensorize", overrides=override_list)

    setup_global_logging(
        f'{name}_tensorize',
        PROJECT_ROOT.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        debug=debug
    )
    logger = logging.getLogger(f'{name}_tensorize')
    logger.info(f"Starting tensorize of {name}")
    logger.info(f"Using objective {objective}")
    logger.info(f"Using processor {processor_name}")
    logger.info(f"Using model {model_name}")
    logger.debug(f"Override string is {override_str}")
    logger.debug(f"Using {num_workers} workers")

    data_path = PROJECT_ROOT.joinpath(data_path)
    logger.info(f"Data path is {data_path}")
    out_path = PROJECT_ROOT.joinpath(out_path)
    logger.info(f"Output path is {out_path}")

    train_file_name = f"{name}.jsonl"
    validation_file = f"{validation_file_name or name + '_val'}.jsonl"

    if not out_path.exists():
        out_path.mkdir(parents=True)

    logger.debug(f"Initializing processor {cfg.processor.name}")

    logger.info("Processor arguments:")
    for k, v in cfg.processor.kwargs.items():
        logger.info(f"{k:>32} = {v}")
    if cfg.processor.name == 'stackoverflow':
        processor = StackOverflowTextProcessor(
            objective=objective,
            **OmegaConf.to_object(cfg.processor.kwargs)
        )
    else:
        raise ValueError(f'Unknown processor {cfg.processor.name}')
    tensorize(
        data_path.joinpath(train_file_name),
        out_path.joinpath(f'{name}'),
        num_workers,
        model_name,
        processor,
        cfg.batch_size
    )
    tensorize(
        data_path.joinpath(validation_file),
        out_path.joinpath(f"{validation_file_name or name + '_val'}"),
        num_workers,
        model_name,
        processor,
        cfg.batch_size
    )


if __name__ == "__main__":
    tensorize_data()
