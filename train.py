import argparse
import logging

from hydra import compose, initialize
import torch
from omegaconf import DictConfig, open_dict, OmegaConf
from pathlib import Path
import numpy as np
import os
import random
import shutil

from tio import Task

from src.common import setup_global_logging, is_currently_distributed
from src.training import train_model
from src.config import setup_tracking_env_from_cfg, get_run_base_name_from_cfg

# Hydra Messes with the CWD, so we need to save it at the beginning.
PROJECT_ROOT = Path.cwd()


def run(name, task, config_name, force_overwrite_dir, override_str, cfg_overrides, debug):
    print(config_name)
    print(Path().resolve().absolute())
    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()

    group_name = task.upper()
    for i in cfg_overrides:
        if 'group=' in i:
            group_name = i.split('=')[-1]
            break
    print(f"Starting Train with group={group_name}, "
          f"name={name}, and task={task}")

    if not Task.is_name_registered(task):
        valid_tasks = ''
        for t in Task.list_available():
            valid_tasks += f'\t{t}\n'
        raise ValueError(f"Unknown Task '{task}'. Valid tasks are:\n{valid_tasks}")
    new_cwd = Path('outputs', group_name.lower(), name)
    if not is_currently_distributed():
        if not new_cwd.exists():
            new_cwd.mkdir(parents=True)
        else:
            shutil.rmtree(new_cwd)
            new_cwd.mkdir(parents=True)

    setup_global_logging(
        'train',
        new_cwd.joinpath('logs'),
        debug=debug,
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1))
    )

    logger = logging.getLogger('train')
    logger.info("Starting Train")
    logger.info(f"Loading the hydra config '{config_name}.yaml'")

    # We need to add the name and task (task uppercase is also the group) to the
    # hydra configs.
    cfg_overrides = [f"name={name}", f"task={task}", f"group={group_name}"] + cfg_overrides

    if override_str:
        cfg_overrides += override_str.split(" ")
    if debug:
        cfg_overrides += ['debug=True']

    initialize(config_path="conf", job_name="train")
    cfg = compose(config_name=config_name, overrides=cfg_overrides)

    os.chdir(new_cwd)
    with open('config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    setup_tracking_env_from_cfg(cfg)

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

    if os.environ.get("LOCAL_RANK", '-1') != '-1' or os.environ['WANDB_DISABLED'] != 'true':
        os.environ['DISABLE_FAST_TOK'] = 'true'

    with open_dict(cfg):
        cfg.training.local_rank = int(os.environ.get('LOCAL_RANK', '-1'))

    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    model = train_model(cfg)

    if cfg.training.local_rank <= 0:
        best_models_path = PROJECT_ROOT.joinpath('best_models', get_run_base_name_from_cfg(cfg))
        save_model = True
        if best_models_path.exists():
            if force_overwrite_dir:
                logger.info(f"Overwriting {best_models_path}")
                shutil.rmtree(best_models_path)
            else:
                save_model = False
        if save_model:
            logger.info(
                f"Saving best model to {best_models_path.absolute().resolve()}")
            model.save_pretrained(best_models_path)
            with best_models_path.joinpath('config.yaml').open('w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))
        else:
            logger.error(f"Could not save best model, {best_models_path} exists"
                         f" and force is not enabled.")
    logger.info("Training Is Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", metavar="<Name of the Run>")
    parser.add_argument("task", metavar="<Task to use>")
    parser.add_argument("--config", help="Name of the base config file to use.",
                        default='train_config')
    parser.add_argument('--force-overwrite-dir', '-force',
                        action="store_true",
                        default=False,
                        help="Force overwriting the directory if it exists.")
    parser.add_argument('-debug',
                        action="store_true",
                        default=False,
                        help="Debug Mode")

    parser.add_argument('--override-str',
                        help='Bash does not like lists of variable args. so '
                             'pass as seperated list of overrides, seperated by ' '.',
                        default=''
                        )
    # This lets us have virtually the same exact setup as the hydra decorator
    # without their annoying working directory and logging.
    parser.add_argument('--hydra-overrides', '-hydra', nargs=argparse.REMAINDER,
                        help='Everything after this argument is passed to the '
                             'hydra config creator as an override command.', default=[])

    argv = parser.parse_args()
    print(argv.config)
    run(
        argv.name,
        argv.task,
        argv.config,
        argv.force_overwrite_dir,
        argv.override_str,
        argv.hydra_overrides or [],
        argv.debug
    )
