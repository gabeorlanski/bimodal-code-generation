import argparse
import logging
from omegaconf import OmegaConf, open_dict
from pathlib import Path
import os
import yaml
import wandb
from src.common import setup_global_logging
from src.evaluation.code_eval import evaluate_code, BASE_ERROR_TYPES
from src.config import setup_tracking_env_from_cfg, get_config_for_tracking


def run(file_name, num_workers, disable_tracking):
    # I just needed a way to get the parent directory.
    path_to_file = Path(file_name)
    path_to_dir = path_to_file.parent
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
        path_to_dir.joinpath(f'eval_config.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )
    cfg = OmegaConf.create(
        cfg
    )
    setup_tracking_env_from_cfg(cfg)

    logger.info(f"Executing code from {path_to_file}")
    results, outcome_keys, metric_path = evaluate_code(
        str(path_to_file),
        num_workers,
        out_dir=Path(), timeout=3.0,
    )
    with open_dict(cfg):
        cfg.split = path_to_file.stem
    if not disable_tracking:
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
        wandb_run = wandb.init(
            job_type='code_eval',
            name=f"{cfg.name}[{path_to_file.stem}]",
            project=cfg.project,
            group=cfg.group,
            config=get_config_for_tracking(cfg),
            entity='gabeorlanski'
            # id=run_id
        )
        overview_metrics = results['overview']
        outcome_metrics = {}
        counts_table = []
        for k in sorted(outcome_keys):
            key = k.replace(" ", "_")
            raw_val = overview_metrics.pop(key)
            raw_pct = overview_metrics.pop(f"{key}_pct")
            outcome_metrics[f"outcomes/{key}"] = raw_val
            outcome_metrics[f"outcomes_pct/{key}"] = raw_pct
            counts_table.append([key, raw_val, raw_pct])
        counts_table = wandb.Table(data=counts_table, columns=['Outcome', 'Count', 'Percent'])

        # WandB does not like logging things from the same step at different
        # times. Hence the ugly dict.
        wandb_run.log({
            'outcome_table': counts_table,
            **{f'outcomes/{k}': v for k, v in outcome_metrics.items()},
            **{f"{'eval/' if '/' not in k else ''}{k}": v for k, v in overview_metrics.items()}
        })
        metric_artifact = wandb.Artifact(f"{cfg.group}.{cfg.name}.{cfg.task.name}.{cfg.split}",
                                         type='execution_metrics')
        metric_artifact.add_file(str(metric_path.resolve().absolute()))
        wandb_run.log_artifact(metric_artifact)
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
    argv, _ = parser.parse_known_args()
    run(
        argv.file_name,
        argv.workers,
        argv.disable_tracking
    )
