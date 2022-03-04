import argparse
import json
import logging
from omegaconf import OmegaConf, open_dict
from pathlib import Path
import os
import yaml
import wandb
from src.common import setup_global_logging
from src.evaluation.code_eval import evaluate_code_from_file, BASE_ERROR_TYPES
from src.config import setup_tracking_env_from_cfg, get_config_for_tracking
from src.common.util import flatten


def run(pred_dir, num_workers, disable_tracking, input_artifact_name, timeout):
    # I just needed a way to get the parent directory.
    path_to_dir = Path(pred_dir)
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

    # In the case this script is not called from an artifact.
    if path_to_dir.stem == 'predictions':
        path_to_cfg = path_to_dir.parent.joinpath('eval_config.yaml')
    else:
        path_to_cfg = path_to_dir.joinpath('eval_config.yaml')
    cfg = yaml.load(
        path_to_cfg.open('r', encoding='utf-8'),
        yaml.Loader
    )
    cfg = OmegaConf.create(
        cfg
    )
    setup_tracking_env_from_cfg(cfg)

    all_results = {}

    for split_file in path_to_dir.glob('*.jsonl'):
        split = split_file.stem
        logger.info(f"Executing code from {split_file}")
        results = evaluate_code_from_file(
            str(split_file),
            samples_per_problem=cfg.seq_per_sample,
            num_workers=num_workers,
            timeout=timeout
        )
        all_results[split] = results

    save_path = path_to_dir.joinpath(f'execution_metrics.json')
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
            project=cfg.project,
            group=f"{cfg.group}[execution]",
            config=get_config_for_tracking(cfg),
            entity=cfg.tracking.entity
            # id=run_id
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
        if input_artifact_name:
            wandb_run.use_artifact(
                f"{input_artifact_name}{':latest' if ':' not in input_artifact_name else ''}")

        elif cfg.get('eval_run_name'):
            wandb_run.use_artifact(f"{cfg.eval_run_name}:latest")

        wandb_run.finish()

    logger.info("Finished Code Eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", metavar="<Name of the file with the preds>")
    parser.add_argument("workers", metavar="<Num Workers>", type=int)

    parser.add_argument('--disable-tracking', '-notrack',
                        action="store_true",
                        default=False,
                        help="Disable Tracking")
    parser.add_argument('--artifact-name', default=None, help='Input artifact name for linking')
    parser.add_argument('--timeout', default=3.0, type=float, help='Timeout for executing code')
    argv, _ = parser.parse_known_args()
    run(
        argv.file_name,
        argv.workers,
        argv.disable_tracking,
        argv.artifact_name,
        argv.timeout
    )
