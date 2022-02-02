"""
Dealing with distributed training and working directories was a giant pain, so
 made this script to be ran before distributed training to handle all of that.
"""
import argparse
import shutil
import sys
from pathlib import Path

if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT


def prep_env(name, task, config, force_overwrite_dir, override_str, cfg_overrides):
    group_name = task.upper()
    for i in cfg_overrides + override_str.split(" "):
        if 'group=' in i:
            group_name = i.split('=')[-1]
            break
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
    parser.add_argument("name", metavar="<Name of the Run>")
    parser.add_argument("task", metavar="<Task to use>")
    parser.add_argument("--config", help="Name of the base config file to use.",
                        default='train_config')
    parser.add_argument('--force-overwrite-dir', '-force',
                        action="store_true",
                        default=False,
                        help="Force overwriting the directory if it exists.")
    parser.add_argument('--override-str',
                        help='Bash does not like lists of variable args. so '
                             'pass as seperated list of overrides, seperated by ;.',
                        default=''
                        )
    # This lets us have virtually the same exact setup as the hydra decorator
    # without their annoying working directory and logging.
    parser.add_argument('--hydra-overrides', '-hydra', nargs=argparse.REMAINDER,
                        help='Everything after this argument is passed to the '
                             'hydra config creator as an override command.')

    argv = parser.parse_args()
    prep_env(
        argv.name,
        argv.task,
        argv.config,
        argv.force_overwrite_dir,
        argv.override_str,
        argv.hydra_overrides or []
    )
