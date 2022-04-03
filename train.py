import argparse
import logging

import yaml
from hydra import compose, initialize
import torch
from omegaconf import DictConfig, open_dict, OmegaConf
from pathlib import Path
import numpy as np
import os
import random
import shutil
import click
from tio import Task

from src.common import setup_global_logging, is_currently_distributed
from src.training import train_model
from src.config import setup_tracking_env_from_cfg, get_run_base_name_from_cfg, \
    get_training_args_from_cfg
from src.data import NON_REGISTERED_TASKS

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


def train_from_cfg(cfg):
    OmegaConf.resolve(cfg)
    task = cfg.task.name
    name = cfg.name
    group_name = cfg.group
    if cfg.local_rank is not None:
        os.environ['LOCAL_RANK'] = str(cfg.local_rank)
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
    if not Task.is_name_registered(task) and task not in NON_REGISTERED_TASKS:
        valid_tasks = ''
        for t in Task.list_available():
            valid_tasks += f'\t{t}\n'
        raise ValueError(f"Unknown Task '{task}'. Valid tasks are:\n{valid_tasks}")
    new_cwd = Path('outputs', group_name.lower(), name)

    with open_dict(cfg):
        cfg.training.local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
        cfg.save_path = str(new_cwd)
        if "meta" not in cfg:
            cfg['meta'] = {'base_name': get_run_base_name_from_cfg(cfg)}
        else:
            cfg['meta']['base_name'] = get_run_base_name_from_cfg(cfg)

    train_args = get_training_args_from_cfg(cfg)

    if cfg.training.local_rank <= 0:
        if not new_cwd.exists():
            new_cwd.mkdir(parents=True)
        else:
            os.environ['WANDB_RESUME'] = 'false'
            if cfg.get('resume_from_checkpoint') is None:
                shutil.rmtree(new_cwd)
                new_cwd.mkdir(parents=True)
            elif new_cwd.joinpath('wandb_id').exists():
                os.environ['WANDB_RESUME'] = 'true'
                os.environ['WANDB_RUN_ID'] = new_cwd.joinpath('wandb_id').read_text()

    if is_currently_distributed():
        torch.distributed.barrier()

    setup_global_logging(
        'train',
        new_cwd,
        debug=cfg.debug,
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        disable_issues_file=True
    )

    logger = logging.getLogger('train')
    logger.info("Starting Train")

    os.chdir(new_cwd)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    setup_tracking_env_from_cfg(cfg)

    if (
            os.environ.get("LOCAL_RANK", '-1') != '-1'
            or os.environ['WANDB_DISABLED'] != 'true'
            or cfg.training.get('dataloader_num_workers', 0) > 0
    ):
        os.environ["DISABLE_FAST_TOK"] = "true"

    if "training" not in cfg:
        raise KeyError("Missing 'training' key in config.")

    seed = cfg["seed"]
    numpy_seed = cfg["numpy_seed"]
    torch_seed = cfg["pytorch_seed"]
    logger.info(f"Seed={seed}")
    logger.info(f"NumPy Seed={numpy_seed}")
    logger.info(f"Torch Seed={torch_seed}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["numpy_seed"])
    torch.manual_seed(torch_seed)

    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model, cfg = train_model(cfg, train_args)

    if cfg.training.local_rank <= 0:
        best_models_path = Path('best_model')
        save_model = True
        if best_models_path.exists():
            logger.info(f"Overwriting {best_models_path}")
            shutil.rmtree(best_models_path)
        if save_model:
            logger.info(
                f"Saving best model to {best_models_path.absolute().resolve()}")
            model.save_pretrained(best_models_path)
            with best_models_path.joinpath('config.yaml').open('w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True))
        else:
            logger.error(f"Could not save best model, {best_models_path} exists"
                         f" and force is not enabled.")
    logger.info("Training Is Done")


@click.group()
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--notrack', 'no_track',
    is_flag=True,
    default=False,
    help="Disable Tracking"
)
@click.option('--local_rank', default=-1, type=int)
@click.pass_context
def cli(ctx, debug, no_track, local_rank):
    ctx.obj = {
        "DEBUG"   : debug, "local_rank": int(os.environ.get('LOCAL_RANK', '-1')),
        'no_track': no_track
    }


@cli.command("from_config")
@click.argument("config", metavar="<Config To Use>")
@click.pass_context
def train_from_config_file(ctx, config):
    local_rank = ctx.obj['local_rank']
    debug = ctx.obj['DEBUG']
    path_to_config = PROJECT_ROOT.joinpath(config)
    cfg = OmegaConf.create(yaml.load(
        path_to_config.open('r'),
        yaml.Loader
    ))

    with open_dict(cfg):
        cfg.local_rank = local_rank
        cfg.debug = debug
        if ctx.obj.get('no_track', False):
            cfg.tracking = False

    train_from_cfg(cfg)


@cli.command('cli')
@click.argument("name", metavar="<Name of the Run>")
@click.argument("task", metavar="<Task to use>")
@click.option("--config", "config_name", help="Name of the base config file to use.",
              default='train_config')
@click.option('--override-str',
              help='Bash does not like lists of variable args. so '
                   'pass as seperated list of overrides, seperated by ' '.',
              default=''
              )
# This lets us have virtually the same exact setup as the hydra decorator
# without their annoying working directory and logging.
@click.argument(
    'cfg_overrides', nargs=-1,
    type=click.UNPROCESSED
)
@click.pass_context
def train(ctx, name, task, config_name, override_str, cfg_overrides):
    local_rank = ctx.obj['local_rank']
    debug = ctx.obj['DEBUG']

    group_name = task.upper()
    for i in cfg_overrides:
        if 'group=' in i:
            group_name = i.split('=')[-1]
            break

    # We need to add the name and task (task uppercase is also the group) to the
    # hydra configs.
    cfg_overrides = [f"name={name}", f"task={task}", f"group={group_name}"] + list(cfg_overrides)

    if override_str:
        cfg_overrides += override_str.split(" ")
    if debug:
        cfg_overrides += ['debug=True']

    initialize(config_path="conf", job_name="train")
    cfg = compose(config_name=config_name, overrides=cfg_overrides)

    with open_dict(cfg):
        cfg.local_rank = local_rank
        cfg.debug = debug

        if ctx.obj.get('no_track', False):
            cfg.tracking = False
    train_from_cfg(cfg)


if __name__ == "__main__":
    cli()
