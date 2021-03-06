import argparse
import json
import logging

import click
from omegaconf import OmegaConf, open_dict
from pathlib import Path
import os
import yaml
import wandb
from src.common import setup_global_logging
from src.evaluation.code_eval import evaluate_code_from_file, BASE_ERROR_TYPES
from src.config import setup_tracking_env_from_cfg, get_config_for_tracking
from src.common.util import flatten


@click.command()
@click.argument('preds_dir', metavar="<predictions dir>")
@click.argument('num_workers', type=int, metavar="<Number of workers>")
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--notrack', 'disable_tracking',
    is_flag=True,
    default=False,
    help="Disable Tracking"
)
@click.option(
    '--timeout',
    default=3.0,
    type=float,
    help="The amount to use for timeout"
)
def run(preds_dir, num_workers, debug, disable_tracking, timeout):
    # I just needed a way to get the parent directory.
    path_to_preds = Path(preds_dir)
    setup_global_logging(
        f'execution',
        path_to_preds,
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        disable_issues_file=True
    )
    logger = logging.getLogger('execution')
    logger.info(f"Loading eval config from {path_to_preds}")
    logger.info(f"CWD={Path().absolute().resolve()}")
    cfg = yaml.load(
        path_to_preds.joinpath('config.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )
    cfg = OmegaConf.create(
        cfg
    )
    with open_dict(cfg):
        cfg.debug = debug
        if disable_tracking:
            cfg.tracking = False

    setup_tracking_env_from_cfg(cfg)

    all_results = {}

    logger.info(f"Splits to execute: {', '.join(cfg.splits)}")
    splits = []
    for split in cfg.splits:
        if not path_to_preds.joinpath(f"{split}.jsonl").exists():
            raise FileNotFoundError(f'{path_to_preds.joinpath(f"{split}.jsonl")} does not exist')
        splits.append(path_to_preds.joinpath(f"{split}.jsonl"))

    for split_file in splits:
        split = split_file.stem
        logger.info(f"Executing code from {split_file}")
        results = evaluate_code_from_file(
            cfg.task.name,
            str(split_file),
            samples_per_problem=cfg.evaluation.seq_per_sample,
            num_workers=num_workers,
            timeout=timeout
        )
        all_results[split] = results

    save_path = path_to_preds.joinpath(f'execution_metrics.json')
    logger.info(f"Saving {save_path}")
    with save_path.open('w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=True)

    #####################################################################
    # TRACKING CODE TO REMOVE ON RELEASE                                #
    #####################################################################
    if not disable_tracking:
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
        wandb_run = wandb.init(
            job_type='code_eval',
            name=os.getenv('WANDB_RUN_NAME'),
            id=os.getenv('WANDB_RUN_ID'),
            project=os.getenv('WANDB_PROJECT'),
            group=f"{cfg.group}[execution]",
            config=get_config_for_tracking(cfg),
            entity=os.getenv('WANDB_ENTITY'),
            tags=os.getenv('WANDB_RUNS_TAGS').split(',')
        )
        metrics_to_log_dict = {}
        for split, split_dict in all_results.items():
            split_metrics = split_dict['overview']
            split_metrics['outcome_pcts'] = split_dict['outcome_pcts']
            metrics_to_log_dict[split] = split_metrics

        # WandB does not like logging things from the same step at different
        # times. Hence the ugly dict.
        wandb_run.log(flatten(metrics_to_log_dict, sep='/'), step=1)
        metric_artifact = wandb.Artifact(f"{cfg.group}.execution.{os.getenv('WANDB_RUN_NAME')}",
                                         type='execution_metrics')
        metric_artifact.add_file(str(save_path.resolve().absolute()))
        wandb_run.log_artifact(metric_artifact)
        wandb_run.finish()

    logger.info("Finished Code Eval")


if __name__ == "__main__":
    run()
