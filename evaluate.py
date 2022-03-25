import logging
from typing import Optional, List
from hydra import compose, initialize
import yaml
from omegaconf import OmegaConf, open_dict
import os
from pathlib import Path
import click

from src import config
from src.evaluation import evaluate, make_eval_cfg_from_ctx
from src.common import setup_global_logging, PROJECT_ROOT
from src.data import NON_REGISTERED_TASKS


@click.group()
@click.option('--splits-to-use', '-splits',
              default='',
              help='Comma separated of splits to use',
              callback=lambda c, p, value: value.split(',') if value else [])
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
@click.option(
    '--seq-per-sample', '-seqs', 'sequences_per_sample',
    default=None, type=int, help='Number of sequences per sample to generate.')
@click.option(
    '--num-generate-per-step', '-genstep', 'num_generate_per_step',
    default=None, type=int, help='Number of sequences per batch to generate.')
@click.option(
    '--debug-samples', 'debug_num_samples',
    default=None, type=int, help='Debug number of samples')
@click.option(
    '--force-create-dir', '-mkdir',
    is_flag=True,
    default=False,
    help="Force creating and saving results in their own directory"
)
@click.option(
    '--num-workers', '-N',
    default=None, type=int, help='number of workers override')
@click.option(
    '--output-dir', '-o',
    default=None, type=int, help='Output directory name')
@click.pass_context
def evaluate_cli_entry(
        ctx,
        splits_to_use,
        num_generate_per_step: int,
        sequences_per_sample: int,
        debug_num_samples: int,
        dry_run: bool,
        disable_tracking: bool,
        debug: bool,
        force_create_dir,
        num_workers,
        output_dir
):
    ctx.obj = {
        "DEBUG"                : debug,
        "DRY_RUN"              : dry_run,
        "DISABLE_TRACKING"     : disable_tracking,
        "DEBUG_NUM_SAMPLES"    : debug_num_samples,
        "splits"               : splits_to_use,
        "num_generate_per_step": num_generate_per_step,
        "sequences_per_sample" : sequences_per_sample,
        "FORCE_CREATE_DIR"     : force_create_dir,
        "num_workers"          : num_workers,
        "out_dir"              : output_dir
    }


def evaluate_from_ctx_and_cfg(
        ctx,
        cfg,
):
    debug = ctx.obj['DEBUG']
    dry_run = ctx.obj['DRY_RUN']
    disable_tracking = ctx.obj['DISABLE_TRACKING']
    debug_num_samples = ctx.obj['DEBUG_NUM_SAMPLES']

    # Need to load in the secret from the file to log to wandb
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
    # if output_path is None or ctx.obj['FORCE_CREATE_DIR']:
    dir_name = cfg.name
    working_dir = [cfg.task.name.upper()]
    if 'meta' in cfg and 'card_name' in cfg.meta:
        dir_name = f"{cfg.meta.ablation}_{cfg.meta.step}"
        working_dir = [cfg.meta.card_name, cfg.task.name.upper()]

    if debug:
        dir_name = f"debug_{dir_name}"

    working_dir = PROJECT_ROOT.joinpath(
        'eval_results', *working_dir, dir_name
    )
    log_name = f'evaluate_{cfg.task.name}'
    # else:
    #     if ctx.obj['out_dir'] is None:
    #         working_dir = output_path.joinpath("eval", f"{cfg.group}_{cfg.task.name.upper()}")
    #     else:
    #         working_dir = output_path.joinpath("eval", ctx.obj['out_dir'])
    #     log_name = "evaluate"
    if not working_dir.exists():
        working_dir.mkdir(parents=True)

    setup_global_logging(
        log_name,
        working_dir,
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        debug=debug,
        disable_issues_file=True
    )
    logger = logging.getLogger(log_name)
    cfg = make_eval_cfg_from_ctx(ctx, cfg)
    logger.info("Starting Evaluate")
    logger.info(f"Working directory is {working_dir}")
    logger.info(f"Using model located at '{cfg.model_path or cfg.model}'")
    logger.debug(f"{cfg.evaluation.seq_per_sample} sequences to be generated per sample.")

    if dry_run:
        logger.warning("IN DRY RUN")

    logger.debug(f"Changing working directory to {working_dir}")
    os.chdir(working_dir.resolve().absolute())

    logger.debug("Updating config with preprocessor and postprocessors")
    with open_dict(cfg):
        cfg.name = f"{cfg.name}_{cfg.group}"
        logger.info(f"Evaluation Run Name is {cfg.name}")
        cfg.group = cfg.task.name.upper()
        cfg.debug = debug
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

    model_cls, model = config.load_model_from_cfg(cfg)
    evaluate(
        cfg,
        model,
        working_dir,
        dry_run
    )


@evaluate_cli_entry.command('chk')
@click.argument('train_dir', metavar="<PATH TO THE DIR CREATED BY TRAINING>")
@click.argument('task_name', metavar='<Name of Task to evaluate on>')
@click.option(
    '--cfg-path', '-cfg',
    default=None, help='Path to cfg to use')
@click.pass_context
def eval_from_checkpoint(
        ctx,
        train_dir: str,
        task_name: str,
        cfg_path
):
    if task_name in NON_REGISTERED_TASKS:
        raise ValueError(f"{task_name} is not compatible with evaluate")

    task_path = PROJECT_ROOT.joinpath('conf', 'task').joinpath(f"{task_name}.yaml")
    # print(f'Checking that {task_name} exists in {task_path}')
    if not task_path.exists():
        raise FileNotFoundError(f"{task_name} does not exist in {task_path}")
    print(f"Loading task {task_path}")
    task_cfg = yaml.load(
        task_path.open('r'),
        yaml.Loader
    )
    train_dir = Path(train_dir).resolve().absolute()
    if cfg_path is None:
        train_config_path = train_dir.joinpath('config.yaml')
    else:
        train_config_path = PROJECT_ROOT.joinpath(cfg_path)
    print(f"Loading Training Config from {train_config_path}")
    cfg = yaml.load(
        train_config_path.open('r', encoding='utf-8'),
        yaml.Loader
    )
    cfg = OmegaConf.create(
        cfg
    )

    if train_dir.stem != 'best_model':
        best_model_path = train_dir.joinpath('best_model')
    else:
        best_model_path = train_dir
    if not best_model_path.exists():
        raise FileNotFoundError(f"{best_model_path} does not exist")
    elif not best_model_path.joinpath('pytorch_model.bin').exists():
        raise FileNotFoundError(f"{best_model_path} does not have a model file")

    with open_dict(cfg):
        cfg.model_path = str(best_model_path.resolve().absolute())
        cfg.task = task_cfg

    evaluate_from_ctx_and_cfg(ctx, cfg)


@evaluate_cli_entry.command('cfg')
@click.argument('cfg_path', metavar='<PATH TO THE CFG>')
@click.argument('task_name', metavar='<Name of Task to evaluate on>')
@click.pass_context
def eval_from_cfg(
        ctx,
        cfg_path,
        task_name
):
    if task_name in NON_REGISTERED_TASKS:
        raise ValueError(f"{task_name} is not compatible with evaluate")

    task_path = PROJECT_ROOT.joinpath('conf', 'task').joinpath(f"{task_name}.yaml")
    # print(f'Checking that {task_name} exists in {task_path}')
    if not task_path.exists():
        raise FileNotFoundError(f"{task_name} does not exist in {task_path}")
    print(f"Loading task {task_path}")
    task_cfg = yaml.load(
        task_path.open('r'),
        yaml.Loader
    )
    print(f"Loading config from {cfg_path}")
    eval_cfg_path = Path(cfg_path)
    cfg = OmegaConf.create(
        yaml.load(
            eval_cfg_path.open('r', encoding='utf-8'),
            yaml.Loader
        )
    )
    with open_dict(cfg):
        cfg.task = task_cfg

    evaluate_from_ctx_and_cfg(ctx, cfg)


if __name__ == "__main__":
    evaluate_cli_entry()
