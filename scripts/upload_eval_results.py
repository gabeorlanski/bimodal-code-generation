import json
import argparse
from pathlib import Path

import wandb
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src import config
from src.common import PROJECT_ROOT


def main(
        path_to_dir
):
    os.environ["WANDB_API_KEY"] = PROJECT_ROOT.joinpath('wandb_secret.txt').read_text().strip()
    path_to_dir = PROJECT_ROOT.joinpath(path_to_dir)
    metrics = json.loads(path_to_dir.joinpath('metrics.json').read_text('utf-8'))
    cfg = OmegaConf.create(
        yaml.load(path_to_dir.joinpath('config.yaml').open('r', encoding='utf-8'),
                  yaml.Loader)
    )
    config.setup_tracking_env_from_cfg(cfg)
    run_id = wandb.util.generate_id()
    with open_dict(cfg):
        cfg.run_id = run_id
    os.environ['RUN_ID'] = run_id
    run = wandb.init(
        job_type='evaluate',
        name=os.getenv('WANDB_RUN_NAME'),
        project=os.getenv('WANDB_PROJECT'),
        group=cfg.group,
        entity=os.getenv('WANDB_ENTITY'),
        config=config.get_config_for_tracking(cfg),
        id=run_id
    )
    run.config.update(config.get_config_for_tracking(cfg))
    run.log({f"eval/{k}": v for k, v in metrics.items()}, step=1)
    preds_artifact = wandb.Artifact(
        f"{config.get_run_base_name_from_cfg(cfg)}.{cfg.task.name}",
        type='predictions'
    )

    preds_artifact.add_dir(str(path_to_dir.resolve().absolute()))
    run.log_artifact(preds_artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar="<Path To Model Directory>",
                        help="Path to the model directory created by train.py")
    argv = parser.parse_args()
    main(
        argv.model_path
    )
