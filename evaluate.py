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


@click.command()
@click.argument('train_dir', metavar="<PATH TO THE DIR CREATED BY TRAINING>")
@click.option(
    '--debug',
    is_flag=True,
    default=False,
    help="Debug Mode"
)
@click.option(
    '--dry-run', '-dry', 'dry_run',
    is_flag=True,
    default=False,
    help="Dry run"
)
@click.option(
    '--notrack', 'disable_tracking',
    is_flag=True,
    default=False,
    help="Disable Tracking"
)
@click.option('--output-dir-name', '-o', 'output_dir_name', default=None,
              help='Name of the output dir for saving the result.')
@click.option(
    '--seq-per-sample', '-seqs', 'sequences_per_sample',
    default=None, type=int, help='Number of sequences per sample to generate.')
@click.option(
    '--batch-size', '-B', 'batch_size',
    default=None, type=int, help='Number of sequences per batch to generate.')
@click.option(
    '--debug-samples', 'debug_num_samples',
    default=None, type=int, help='Debug number of samples')
@click.option(
    '--evaluation-task-name', '-task', 'eval_task_name',
    default=None, help='Task Name for evaluation, will override '
                       'whatever is in the eval config.')
@click.option(
    '--splits-for-eval', '-splits', 'splits_for_eval',
    default=None,
    help='Comma separated list of splits for eval. Will override the list of splits provided by the task.',
    callback=lambda _ctx, _param, splits_str: splits_str.split(',') if splits_str else []
)
@click.option(
    '--evaluation-config-path', '-cfg', 'eval_cfg_path',
    default=None,
    help='The path to the eval config.')
@click.option(
    '--override-str',
    help='Bash does not like lists of variable args. so pass as seperated list of overrides, seperated by spaces.',
    default=''
)
@click.argument(
    'hydra_overrides', nargs=-1,
    type=click.UNPROCESSED, metavar="[HYDRA OVERRIDES FOR EVAL CONFIG]"
)
def eval_from_checkpoint(
        train_dir: str,
        output_dir_name,
        batch_size: int,
        splits_for_eval: str,
        sequences_per_sample: int,
        debug_num_samples: int,
        eval_task_name: Optional[str],
        override_str: str,
        hydra_overrides: List[str],
        dry_run: bool,
        disable_tracking: bool,
        eval_cfg_path: str,
        debug: bool
):
    # Need to load in the secret from the file to log to wandb
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()

    print(f"Loading Training Config from {train_dir}")
    # Loading the model from the checkpoint and the
    train_dir = Path(train_dir).resolve().absolute()
    train_config_path = train_dir.joinpath('config.yaml')
    train_cfg = yaml.load(
        train_config_path.open('r', encoding='utf-8'),
        yaml.Loader
    )
    train_cfg = OmegaConf.create(
        train_cfg
    )

    if eval_cfg_path:
        print(f"Loading config from {eval_cfg_path}")
        eval_cfg_path = Path(eval_cfg_path)
        cfg = OmegaConf.create(
            yaml.load(
                eval_cfg_path.open('r', encoding='utf-8'),
                yaml.Loader
            )
        )
        with open_dict(cfg):
            if eval_task_name:
                raise ValueError("Eval task name is not allowed to be set when "
                                 "specifying an eval config")
            if sequences_per_sample:
                cfg.seq_per_sample = sequences_per_sample

            if splits_for_eval:
                cfg.splits = splits_for_eval.split(',')
            if batch_size:
                cfg.batch_size = batch_size

            cfg.model_path = str(train_dir)

    else:
        cfg_overrides = config.create_overrides_list(
            {
                "model_path"    : train_dir,
                "task"          : eval_task_name,
                "seq_per_sample": sequences_per_sample,
                "splits"        : splits_for_eval,
                "batch_size"    : batch_size
            },
            hydra_overrides,
            override_str
        )

        print('Loading Eval Config from conf/eval_config.yaml')
        initialize(config_path="conf", job_name="evaluate")
        cfg = compose(config_name="eval_config", overrides=cfg_overrides)

    # merge_configs gives priority to the first argument, so if we are not
    # overriding the task, we need to copy the task params from the train
    # config.
    cfg = config.merge_configs(
        cfg,
        train_cfg,
        exclude_keys=[
            'preprocessors',
            'postprocessors',
            'generation'
        ]
    )
    dir_name = output_dir_name or f"{cfg.group}.{cfg.name}"

    if debug:
        dir_name = f"debug_{dir_name}"
    working_dir = PROJECT_ROOT.joinpath(
        'eval_results', cfg.task.name.upper(),
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
    logger.info(f"Working directory is {working_dir}")
    logger.info(f"Using model located at '{train_dir.resolve().absolute()}'")
    logger.debug(f"{sequences_per_sample} sequences to be generated per sample.")

    if dry_run:
        logger.warning("IN DRY RUN")

    logger.debug(f"Changing working directory to {working_dir}")
    os.chdir(working_dir.resolve().absolute())

    logger.debug("Updating config with preprocessor and postprocessors")
    with open_dict(cfg):
        for k in ['preprocessors', 'postprocessors']:
            train_processors = OmegaConf.to_object(train_cfg[k]) if k in train_cfg else []
            cfg_processors = OmegaConf.to_object(cfg[k]) if k in cfg else []
            cfg[k] = train_processors + cfg_processors

        # If we are not loading from a checkpoint, use only the name in the
        # config and nothing else.
        if not cfg.is_checkpoint:
            cfg.old_name = cfg.name

        # If we are using a different task than that with which the model was
        # trained, we need to indicate this for later saving. Thus we save the
        # old name.
        elif cfg.task.name != train_cfg.task.name and not cfg.get('force_use_group'):
            cfg.old_name = f"{cfg.group}.{cfg.name}"
            logger.info(
                f"{cfg.task.name.upper()} is a different task than training. "
                f"Saving the old name '{cfg.old_name}'"
            )
            cfg.group = cfg.task.name.upper()

        if disable_tracking:
            cfg.tracking = False
        if debug_num_samples is not None:
            cfg.debug_num_samples = debug_num_samples

    config.setup_tracking_env_from_cfg(cfg)

    # Fast tokenizers do not like to be split for parallelization.
    if (
            os.environ.get("WORLD_SIZE", '1') != '1'
            or os.environ.get('WANDB_DISABLED', 'true') != 'true'
    ):
        os.environ['DISABLE_FAST_TOK'] = 'true'

    model_cls, model = config.load_model_from_cfg(cfg, train_dir)
    evaluate(
        cfg,
        model,
        working_dir,
        dry_run
    )


if __name__ == "__main__":
    eval_from_checkpoint()
