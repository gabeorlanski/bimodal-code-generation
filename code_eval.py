import argparse
import logging
from omegaconf import OmegaConf
from pathlib import Path
import os
import yaml
import wandb
from src.common import setup_global_logging
from src.evaluation.code_eval import evaluate_code, BASE_ERROR_TYPES
from src.config import setup_tracking_env_from_cfg, get_config_for_tracking


def run(split_name, path_to_dir, num_workers, disable_tracking):
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
        path_to_dir.joinpath(f'eval_{split_name}.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )
    cfg = OmegaConf.create(
        cfg
    )
    setup_tracking_env_from_cfg(cfg)
    results, outcome_keys, metric_path = evaluate_code(path_to_dir, num_workers, 3.0,
                                                       out_dir=Path())

    if not disable_tracking:
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()
        wandb_run = wandb.init(
            job_type='code_eval',
            name=cfg.name,
            project=cfg.project,
            group=cfg.group,
            config=get_config_for_tracking(cfg),
            entity='gabeorlanski'
            # id=run_id
        )
        overview_metrics = results['overview']
        outcome_metrics = {}
        counts_table = []
        pct_table = []
        for k in sorted(outcome_keys):
            key = k.replace(" ", "_")
            raw_val = overview_metrics.pop(key)
            raw_pct = overview_metrics.pop(f"{key}_pct")
            outcome_metrics[f"outcomes/{key}"] = raw_val
            outcome_metrics[f"outcomes_pct/{key}"] = raw_pct
            counts_table.append([key, raw_val])
            pct_table.append([key, raw_pct])
        # wandb_run.log(outcome_metrics, step=1)
        counts_table = wandb.Table(data=counts_table, columns=['Outcome', 'Count'])
        pct_table = wandb.Table(data=pct_table, columns=['Outcome', 'Percent'])

        wandb_run.log({
            "outcome_counts_plot": wandb.plot.bar(counts_table, label="Outcome", value="Count",
                                                  title="Outcome Counts"),
            "outcome_pct_plot"   : wandb.plot.bar(pct_table, label="Outcome", value="Percent",
                                                  title="Outcome Percent")
        }, step=1)
        wandb_run.log(outcome_metrics, step=1)
        wandb_run.log({f"eval/{k}": v for k, v in overview_metrics.items()})

        metric_artifact = wandb.Artifact(f"eval_{cfg.group}.{cfg.name}.{cfg.task.name}.{cfg.split}",
                                         type='raw_metrics')
        metric_artifact.add_file(str(metric_path.resolve().absolute()))
        wandb_run.log_artifact(metric_artifact)
        wandb_run.finish()

    logger.info("Finished Code Eval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", metavar="<Split>")
    parser.add_argument("path_to_dir", metavar="<Path to the dir with predictions>")
    parser.add_argument("workers", metavar="<Num Workers>", type=int)

    parser.add_argument('--disable-tracking', '-notrack',
                        action="store_true",
                        default=False,
                        help="Disable Tracking")
    argv, _ = parser.parse_known_args()
    run(
        argv.split,
        argv.path_to_dir,
        argv.workers,
        argv.disable_tracking
    )
