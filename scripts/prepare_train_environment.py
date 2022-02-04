"""
Dealing with distributed training and working directories was a giant pain, so
 made this script to be ran before distributed training to handle all of that.
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT


def prep_env(config,force_overwrite_dir):
    cfg = OmegaConf.create(yaml.load(
        PROJECT_ROOT.joinpath(config).open('r'),
        yaml.Loader
    ))
    group_name = cfg.group
    name = cfg.name
    task = cfg.task.name
    print(f"Preparing train environment group={group_name}, "
          f"name={name}, and task={task}")

    new_cwd = PROJECT_ROOT.joinpath('outputs', group_name.lower(), name)
    if not new_cwd.exists():
        new_cwd.mkdir(parents=True)
    else:
        if not force_overwrite_dir:
            raise ValueError(f"{new_cwd} already exists")
        else:
            shutil.rmtree(new_cwd)
            new_cwd.mkdir(parents=True)
    new_cwd.joinpath('logs').mkdir()
    print("Train environment is prepared.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the base config file to use.")
    parser.add_argument('--force-overwrite-dir', '-force',
                        action="store_true",
                        default=False,
                        help="Force overwriting the directory if it exists.")

    argv = parser.parse_args()
    prep_env(
        argv.config,
        argv.force_overwrite_dir
    )
