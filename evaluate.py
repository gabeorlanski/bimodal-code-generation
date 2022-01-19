import logging
import argparse
from hydra import compose, initialize
import yaml
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from pathlib import Path

from src.config import get_device_from_cfg, load_model_from_cfg, merge_configs, \
    setup_tracking_env_from_cfg
from src.evaluation import evaluate_model
from src.common import setup_global_logging


def run(model_path, split, zero_shot, seq_per_sample, task, hydra_overrides):
    model_path = Path(model_path)
    split = split

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

    if os.environ.get("LOCAL_RANK", '-1') != '-1' or os.environ.get('WANDB_DISABLED',
                                                                    'true') != 'true':
        os.environ['DISABLE_FAST_TOK'] = 'TRUE'

    if task is not None:
        use_train_task = False
    else:
        use_train_task = True
        task = train_cfg.task.name
    logger.info(f"Using split '{split}' for task '{task}'")
    logger.debug(f"Zero shot is {'enabled' if zero_shot else 'disabled'}.")
    logger.debug(f"{seq_per_sample} sequences to be generated per sample.")
    logger.debug(f"Hydra overrides are {hydra_overrides}")

    # Yes this is not a clean solution, but for distributed running this works.
    cfg_overrides = [
        f"task={task}",
        f"split={split}",
        f"is_checkpoint={not zero_shot}",
        f"model_path={str(model_path)}",
        f"seq_per_sample={seq_per_sample}",
        *hydra_overrides
    ]
    initialize(config_path="conf", job_name="evaluate")
    cfg = compose(config_name="eval_config", overrides=cfg_overrides)
    cfg = merge_configs(cfg, train_cfg, exclude_keys=['preprocessors', 'postprocessors'])

    with open_dict(cfg):
        for k in ['preprocessors', 'postprocessors']:
            train_processors = OmegaConf.to_object(train_cfg[k]) if k in train_cfg else []
            cfg_processors = OmegaConf.to_object(cfg[k]) if k in cfg else []
            cfg[k] = train_processors + cfg_processors

    setup_tracking_env_from_cfg(cfg)

    # merge_configs gives priority to the first argument, so if we are not
    # overriding the task, we need to copy the task params from the train
    # config.
    if use_train_task:
        logger.info(
            "Task was not overridden, using the task config from training"
        )
        with open_dict(cfg):
            cfg.task = train_cfg.task

    model_cls, model = load_model_from_cfg(cfg)
    model = model.to(get_device_from_cfg(cfg))
    evaluate_model(cfg, model=model)

    run_id = os.environ['RUN_ID']
    with open_dict(cfg):
        cfg.run_id = run_id

    with model_path.joinpath(f'eval_{cfg.split}_config.yaml').open('w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info("Finished Evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', metavar="<Path To Model Directory>",
                        help="Path to the model directory created by train.py")
    parser.add_argument('split', metavar="<Dataset Split>", help="Name of the split to use.")
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
    run(
        argv.model_path,
        argv.split,
        argv.zero_shot,
        argv.seq_per_sample,
        argv.task,
        argv.hydra_overrides
    )
