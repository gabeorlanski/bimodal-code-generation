import json
import argparse
import wandb
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
import os

from src import config
from src.common import PROJECT_ROOT


def main(
        path_to_dir,
        hydra_overrides
):
    os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
    path_to_dir = PROJECT_ROOT.joinpath(path_to_dir)
    metrics = json.loads(path_to_dir.joinpath('eval_metrics.json').read_text('utf-8'))

    initialize(config_path=str(path_to_dir), job_name="evaluate")
    cfg = compose(config_name="eval_config", overrides=hydra_overrides)

    run_id = wandb.util.generate_id()
    with open_dict(cfg):
        cfg.run_id = run_id
    os.environ['RUN_ID'] = run_id
    run = wandb.init(
        job_type='evaluate',
        name=cfg.name,
        project=cfg.project,
        group=cfg.group,
        config=config.get_config_for_tracking(cfg),
        id=run_id
    )
    run.config.update(config.get_config_for_tracking(cfg))
    run.log({f"eval/{k}": v for k, v in metrics.items()}, step=1)
    preds_artifact = wandb.Artifact(f"{cfg.group}.{cfg.name}",
                                    type='predictions')

    preds_artifact.add_dir(str(path_to_dir.joinpath('predictions').resolve().absolute()))
    preds_artifact.add_file(
        str(path_to_dir.joinpath(f'eval_config.yaml').resolve().absolute()))
    run.log_artifact(preds_artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar="<Path To Model Directory>",
                        help="Path to the model directory created by train.py")

    parser.add_argument('--hydra-overrides', '-hydra', nargs=argparse.REMAINDER, default=[])
    argv = parser.parse_args()
    main(
        argv.model_path,
        argv.hydra_overrides
    )
