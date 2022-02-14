import copy
import json
import logging
import argparse
import random
import numpy as np
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import yaml
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from pathlib import Path

from src import config
from src.evaluation import evaluate
from src.common import setup_global_logging, PROJECT_ROOT


def eval_from_checkpoint(
        model_path,
        splits,
        seq_per_sample,
        task,
        override_str,
        hydra_overrides,
        dry_run: bool,
        zero_shot: bool,
        output_dir_name,
        debug
):
    # Need to load in the secret from the file to log to wandb
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()

    # Loading the model from the checkpoint and the
    model_path = Path(model_path).resolve().absolute()
    train_config_path = model_path.joinpath('config.yaml')
    train_config = yaml.load(
        train_config_path.open('r', encoding='utf-8'),
        yaml.Loader
    )

    train_cfg = OmegaConf.create(
        train_config
    )

    if task is not None:
        use_train_task = False
    else:
        use_train_task = True
        task = train_cfg.task.name

    cfg_overrides = [
        f"task={task}",
        f"is_checkpoint={not zero_shot}",
        f"model_path={str(model_path)}",
        f"seq_per_sample={seq_per_sample}",
        *hydra_overrides
    ]
    if override_str:
        cfg_overrides += override_str.split(" ")

    initialize(config_path="conf", job_name="evaluate")
    cfg = compose(config_name="eval_config", overrides=cfg_overrides)
    cfg = config.merge_configs(cfg, train_cfg, exclude_keys=['preprocessors', 'postprocessors'])
    dir_name = output_dir_name or f"{train_cfg.group}.{train_cfg.name}"
    if debug:
        dir_name = f"debug_{dir_name}"
    working_dir = PROJECT_ROOT.joinpath(
        'eval_results', task.upper(),
        dir_name
    )
    if not working_dir.exists():
        working_dir.mkdir(parents=True)

    setup_global_logging(
        'evaluate',
        working_dir.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        debug=debug
    )
    logger = logging.getLogger("evaluate")
    logger.info("Starting Evaluate")
    logger.info(f"Using model located at '{model_path.resolve().absolute()}'")
    logger.info(f"Loading config from '{model_path.joinpath('config.yaml')}'")

    if os.environ.get("WORLD_SIZE", '1') != '1' or os.environ.get('WANDB_DISABLED',
                                                                  'true') != 'true':
        os.environ['DISABLE_FAST_TOK'] = 'true'

    logger.debug(f"{seq_per_sample} sequences to be generated per sample.")
    logger.debug(f"Hydra overrides are {hydra_overrides}")
    if zero_shot:
        logger.warning("NOT using checkpoint")

    # Yes this is not a clean solution, but for distributed running this works.
    os.chdir(working_dir.resolve().absolute())
    with open_dict(cfg):
        for k in ['preprocessors', 'postprocessors']:
            train_processors = OmegaConf.to_object(train_cfg[k]) if k in train_cfg else []
            cfg_processors = OmegaConf.to_object(cfg[k]) if k in cfg else []
            cfg[k] = train_processors + cfg_processors

        if not use_train_task:
            cfg.old_name = f"{cfg.group}.{cfg.name}"
            cfg.group = task.upper()

    config.setup_tracking_env_from_cfg(cfg)

    # merge_configs gives priority to the first argument, so if we are not
    # overriding the task, we need to copy the task params from the train
    # config.
    if use_train_task:
        logger.info(
            "Task was not overridden, using the task config from training"
        )
        with open_dict(cfg):
            cfg.task = train_cfg.task

    model_cls, model = config.load_model_from_cfg(cfg, model_path)
    evaluate(
        cfg,
        model,
        splits,
        working_dir,
        dry_run
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar="<Path To Model Directory>",
                        help="Path to the model directory created by train.py")
    parser.add_argument('splits', metavar="<Comma Seperated Splits>",
                        help="Name of the splits to use.")
    parser.add_argument(
        '--workers', default=1, type=int
    )
    parser.add_argument(
        '--seq-per-sample', '-seqs', type=int, default=1,
        help="Number of sequences per sample to generate"
    )
    parser.add_argument('--task', default=None,
                        help="The task to use that is "
                             "not the one specified in the training config.")
    parser.add_argument('--output-dir-name', '-o', default=None,
                        help="The output dir name for saving.")
    parser.add_argument('--dry-run',
                        action="store_true",
                        default=False,
                        help="Dry run")
    parser.add_argument('--debug',
                        action="store_true",
                        default=False,
                        help="Debug")
    parser.add_argument('--nochk',
                        action="store_true",
                        default=False,
                        help="Do not use a checkpoint.")
    parser.add_argument('--override-str',
                        help='Bash does not like lists of variable args. so '
                             'pass as seperated list of overrides, seperated by ' '.',
                        default=None
                        )
    parser.add_argument('--hydra-overrides', '-hydra', nargs=argparse.REMAINDER)
    argv = parser.parse_args()
    os.environ['WORLD_SIZE'] = str(argv.workers)
    eval_from_checkpoint(
        argv.model_path,
        argv.splits,
        argv.seq_per_sample,
        argv.task,
        argv.override_str,
        argv.hydra_overrides or [],
        argv.dry_run,
        argv.nochk,
        argv.output_dir_name,
        argv.debug
    )
