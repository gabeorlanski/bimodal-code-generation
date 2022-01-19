import argparse
import logging
from omegaconf import OmegaConf
from pathlib import Path
import os
import yaml

from src.common import setup_global_logging
from src.evaluation.code_eval import evaluate_code
from src.config import setup_tracking_env_from_cfg


def run(path_to_dir, num_workers, disable_tracking):
    path_to_dir = Path(path_to_dir)
    if not path_to_dir.exists():
        raise FileExistsError(f"{path_to_dir.resolve().absolute()} does not exist.")
    setup_global_logging(
        'code_eval',
        path_to_dir.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1))
    )
    logger = logging.getLogger('code_eval')
    logger.info("Starting...")
    logger.info(f"Loading eval config from {path_to_dir}")
    cfg = yaml.load(
        path_to_dir.joinpath('eval_config.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )
    if disable_tracking:
        cfg['tracking'] = False
    cfg = OmegaConf.create(
        cfg
    )
    setup_tracking_env_from_cfg(cfg)
    evaluate_code(path_to_dir, num_workers, 3.0)
    logger.info("Finished Code Eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_dir", metavar="<Path to the dir with predictions>")
    parser.add_argument("workers", metavar="<Num Workers>",type=int)
    parser.add_argument('--disable-tracking', '-notrack',
                        action="store_true",
                        default=False,
                        help="Disable Tracking")

    argv = parser.parse_args()
    run(
        argv.path_to_dir,
        argv.workers,
        argv.disable_tracking
    )
