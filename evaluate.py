import copy
import json
import logging
import argparse
import random
import numpy as np
import wandb
from hydra import compose, initialize
import yaml
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from pathlib import Path
from tio import Task

from src import config
from src.evaluation import evaluate_model
from src.common import setup_global_logging


def main(
        model_path,
        splits,
        zero_shot,
        seq_per_sample,
        task,
        hydra_overrides
):
    model_path = Path(model_path)

    if Path('wandb_secret.txt').exists():
        os.environ["WANDB_API_KEY"] = open('wandb_secret.txt').read().strip()

    setup_global_logging(
        'evaluate',
        model_path.joinpath('logs'),
        rank=int(os.environ.get('LOCAL_RANK', '-1')),
        world_size=int(os.environ.get("WORLD_SIZE", 1))
    )
    logger = logging.getLogger("evaluate")
    logger.info("Starting Evaluate")
    logger.info(f"Using model located at '{model_path.resolve().absolute()}'")
    logger.info(f"Loading config from '{model_path.joinpath('config.yaml')}'")
    train_cfg = OmegaConf.create(
        yaml.load(
            model_path.joinpath('config.yaml').open('r', encoding='utf-8'),
            yaml.Loader
        )
    )

    if os.environ.get("WORLD_SIZE", '1') != '1' or os.environ.get('WANDB_DISABLED',
                                                                  'true') != 'true':
        os.environ['DISABLE_FAST_TOK'] = 'true'

    if task is not None:
        use_train_task = False
    else:
        use_train_task = True
        task = train_cfg.task.name
    logger.info(f"Using split '{splits}' for task '{task}'")
    logger.debug(f"Zero shot is {'enabled' if zero_shot else 'disabled'}.")
    logger.debug(f"{seq_per_sample} sequences to be generated per sample.")
    logger.debug(f"Hydra overrides are {hydra_overrides}")

    # Yes this is not a clean solution, but for distributed running this works.
    cfg_overrides = [
        f"task={task}",
        f"is_checkpoint={not zero_shot}",
        f"model_path={str(model_path)}",
        f"seq_per_sample={seq_per_sample}",
        *hydra_overrides
    ]
    initialize(config_path="conf", job_name="evaluate")
    cfg = compose(config_name="eval_config", overrides=cfg_overrides)
    cfg = config.merge_configs(cfg, train_cfg, exclude_keys=['preprocessors', 'postprocessors'])

    with open_dict(cfg):
        for k in ['preprocessors', 'postprocessors']:
            train_processors = OmegaConf.to_object(train_cfg[k]) if k in train_cfg else []
            cfg_processors = OmegaConf.to_object(cfg[k]) if k in cfg else []
            cfg[k] = train_processors + cfg_processors

    config.setup_tracking_env_from_cfg(cfg)

    seed = cfg["seed"]
    numpy_seed = cfg["numpy_seed"]
    torch_seed = cfg["pytorch_seed"]
    logger.info(f"Seed={seed}")
    logger.info(f"NumPy Seed={numpy_seed}")
    logger.info(f"Torch Seed={torch_seed}")
    random.seed(cfg["seed"])
    np.random.seed(cfg["numpy_seed"])
    torch.manual_seed(torch_seed)

    # merge_configs gives priority to the first argument, so if we are not
    # overriding the task, we need to copy the task params from the train
    # config.
    if use_train_task:
        logger.info(
            "Task was not overridden, using the task config from training"
        )
        with open_dict(cfg):
            cfg.task = train_cfg.task

    model_cls, model = config.load_model_from_cfg(cfg)

    splits_to_use = splits.split(',')

    pred_dir = Path(cfg.get('out_path', model_path)).joinpath('predictions')
    if not pred_dir.exists():
        pred_dir.mkdir()
    all_metrics = {}
    split_paths = []
    for split in splits_to_use:
        logger.info(f"Evaluating split {split}")
        with open_dict(cfg):
            cfg.split = split
        metrics, predictions = evaluate_model(copy.deepcopy(cfg), model=model)

        all_metrics.update({f"{split}/{k}": v for k, v in metrics.items()})
        split_path = pred_dir.joinpath(f'{cfg.split}.jsonl')
        split_paths.append(split_path)
        logger.info(f"Saving predictions to '{split_path}'")
        with split_path.open("w", encoding="utf-8") as f:
            for serialized_dict in predictions:
                f.write(json.dumps(serialized_dict) + '\n')

    with model_path.joinpath('eval_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(all_metrics, f)

    run_id = wandb.util.generate_id()
    with open_dict(cfg):
        cfg.run_id = run_id
        cfg.split = splits
    with model_path.joinpath(f'eval_config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    os.environ['RUN_ID'] = run_id
    if (
            isinstance(cfg.tracking, (dict, DictConfig))
            and int(os.environ.get("LOCAL_RANK", "-1")) <= 0
    ):
        run = wandb.init(
            job_type='evaluate',
            name=cfg.name,
            project=cfg.project,
            group=cfg.group,
            config=config.get_config_for_tracking(cfg),
            id=run_id
        )

        run.config.update(config.get_config_for_tracking(cfg))
        run.log({f"eval/{k}": v for k, v in all_metrics.items()}, step=1)
        preds_artifact = wandb.Artifact(f"{cfg.group}.{cfg.name}.{cfg.task.name}",
                                        type='predictions')

        preds_artifact.add_dir(str(pred_dir.resolve().absolute()))
        preds_artifact.add_file(
            str(model_path.joinpath(f'eval_config.yaml').resolve().absolute()))
        run.log_artifact(preds_artifact)
        run.finish()

    logger.info("Finished Evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar="<Path To Model Directory>",
                        help="Path to the model directory created by train.py")
    parser.add_argument('splits', metavar="<Comma Seperated Splits>",
                        help="Name of the splits to use.")
    parser.add_argument(
        '--workers', default=1, type=int
    )
    parser.add_argument(
        '--zero-shot',
        action='store_true',
        default=False,
        help='Do not load the model from the checkpoint, instead'
             ' load the model from HF and evaluate in a '
             'zero-shot setting.'
    )
    parser.add_argument(
        '--seq-per-sample', '-seqs', type=int, default=1,
        help="Number of sequences per sample to generate"
    )
    parser.add_argument('--task', default=None,
                        help="The task to use that is "
                             "not the one specified in the training config.")
    parser.add_argument('--hydra-overrides', '-hydra', nargs=argparse.REMAINDER)
    argv = parser.parse_args()
    os.environ['WORLD_SIZE'] = str(argv.workers)
    main(
        argv.model_path,
        argv.splits,
        argv.zero_shot,
        argv.seq_per_sample,
        argv.task,
        argv.hydra_overrides
    )
