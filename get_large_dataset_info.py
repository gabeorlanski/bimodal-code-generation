import json
import logging
from dataclasses import asdict
from typing import Optional, List

import transformers.utils.logging
from hydra import compose, initialize
import yaml
from omegaconf import OmegaConf, open_dict
import os
import click

from src.common import setup_global_logging, PROJECT_ROOT
from src.data.tensorize import get_dataset_info_with_processor
from src.data.stackoverflow import StackOverflowProcessor


def get_large_dataset_info(
        name,
        output_name,
        num_workers,
        processor_name,
        model_name,
        data_path,
        out_path,
        validation_file_name,
        cfg,
        debug,
        debug_samples,
):
    setup_global_logging(
        f'{output_name}_tensorize',
        PROJECT_ROOT.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        debug=debug
    )
    transformers.utils.logging.set_verbosity_error()
    logger = logging.getLogger(f'{name}_tensorize')
    logger.info(f"Starting tensorize of {name}")
    logger.info(f"Using processor {processor_name}")
    logger.info(f"Using model {model_name}")
    logger.debug(f"Using {num_workers} workers")
    logger.info(f"{output_name=}")

    data_path = PROJECT_ROOT.joinpath(data_path)
    logger.info(f"Data path is {data_path}")
    out_path = PROJECT_ROOT.joinpath(out_path)
    logger.info(f"Output path is {out_path}")

    train_file_name = f"{name}.jsonl"

    if not out_path.exists():
        out_path.mkdir(parents=True)

    logger.debug(f"Initializing processor {cfg.processor.name}")

    logger.info("Processor arguments:")
    for k, v in cfg.processor.params.items():
        logger.info(f"{k:>32} = {v}")
    if cfg.processor.name == 'stackoverflow':
        processor_args = OmegaConf.to_object(cfg.processor.params)

        processor = StackOverflowProcessor(
            **processor_args
        )
    else:
        raise ValueError(f'Unknown processor {cfg.processor.name}')
    logger.info(f"Saving tensorized config")
    train_cfg = get_dataset_info_with_processor(
        data_path.joinpath(train_file_name),
        output_name,
        num_workers,
        model_name,
        processor,
        cfg.tensorize_batch_size,
        debug_max_samples=debug_samples
    )
    with out_path.joinpath(f"{output_name}.cfg.json").open('w') as f:
        json.dump(train_cfg.to_dict(), f, indent=True)


@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.option('--output-path', '-out', 'output_path', default='data/ds_info',
              help='The path to save the results.')
@click.option('--debug-samples', default=-1, type=int,
              help='The path to save the results.')
@click.pass_context
def main(ctx, debug, output_path, debug_samples):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    ctx.obj['OUT_PATH'] = output_path
    ctx.obj['DEBUG_SAMPLES'] = debug_samples


@main.command('cli')
@click.argument('name', metavar="<Name of this dataset>")
@click.argument('output_name', metavar="<Name of the output file>")
@click.argument('processor_name', metavar='<Processor to use>')
@click.argument('model_name', metavar='<Model to use>')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.option(
    '--data-path', '-I',
    help='Path to the input data',
    default='data/dumps'
)
@click.option(
    '--validation-file-name', '-val', default=None,
    help='Name of the validation raw data, if not provided, will use the {name}_val')
@click.option(
    '--override-str',
    help='Bash does not like lists of variable args. so pass as seperated list of overrides, seperated by spaces.',
    default=''
)
@click.pass_context
def get_large_dataset_info_from_cli(
        ctx,
        name: str,
        output_name: str,
        processor_name: str,
        model_name: str,
        num_workers: int,
        override_str: str,
        data_path,
        validation_file_name,
):
    override_list = [
        f"name={name}",
        f"processor={processor_name}"
    ]
    override_list.extend(override_str.split(' ') if override_str else [])
    initialize(config_path="conf", job_name="train")
    cfg = compose(config_name="tensorize", overrides=override_list)
    get_large_dataset_info(
        name=name,
        output_name=output_name,
        num_workers=num_workers,
        processor_name=processor_name,
        model_name=model_name,
        data_path=data_path,
        out_path=ctx.obj['OUT_PATH'],
        validation_file_name=validation_file_name,
        cfg=cfg,
        debug=ctx.obj['DEBUG'],
        debug_samples=ctx.obj['DEBUG_SAMPLES'],
    )


@main.command('cfg')
@click.argument('config')
@click.argument('num_workers', type=int, metavar='<Number Of Workers>')
@click.pass_context
def get_large_dataset_info_from_config(ctx, config, num_workers):
    cfg = OmegaConf.create(yaml.load(
        PROJECT_ROOT.joinpath(config).open(),
        yaml.Loader
    ))

    get_large_dataset_info(
        name=cfg.task.raw_dump_name,
        output_name=cfg.tensorized_name,
        num_workers=num_workers,
        processor_name=cfg.processor.name,
        model_name=cfg.model,
        data_path=cfg.raw_dump_path,
        out_path=cfg.tensorized_path,
        validation_file_name=None,
        cfg=cfg,
        debug=ctx.obj['DEBUG'],
        debug_samples=ctx.obj['DEBUG_SAMPLES'],
    )


if __name__ == "__main__":
    main()
