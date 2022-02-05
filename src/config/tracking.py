"""
Functions for tracking.
"""
import importlib
import numbers
import tempfile
from copy import copy
from pathlib import Path
from typing import Dict, Union

from transformers.integrations import TrainerCallback
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import logging
from src.common import flatten, ENV_VARS_TRUE_VALUES
from src.data.stackoverflow import StackOverflowTask

logger = logging.getLogger(__name__)

__all__ = [
    "TrackingCallback",
    "get_config_for_tracking",
    "is_tracking_enabled",
    "setup_tracking_env_from_cfg",
    "get_run_base_name_from_cfg"
]


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WAND_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    return importlib.util.find_spec("wandb") is not None


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


class TrackingCallback(TrainerCallback):
    """
    Taken from huggingface only modified slightly
    """

    def __init__(self, job_type: str, cfg: DictConfig):
        has_wandb = is_wandb_available()
        assert has_wandb, "WandbCallback requires wandb to be installed. Run `pip install wandb`."
        if has_wandb:
            import wandb

            self._wandb = wandb
        self._initialized = False
        # log outputs
        self._log_model = os.environ.get("WANDB_LOG_MODEL",
                                         "FALSE").upper() == "TRUE"
        self.cfg = cfg
        self.job_type = job_type

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also override the following environment
        variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training. Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to disable gradient logging or `"all"` to
                log gradients and parameters.
            WANDB_PROJECT (`str`, *optional*, defaults to `"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (`bool`, *optional*, defaults to `False`):
                Whether or not to disable wandb entirely. Set *WANDB_DISABLED=true* to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info(
                'Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"'
            )
            combined_dict = {**args.to_sanitized_dict()}

            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}

            combined_dict = {**get_config_for_tracking(self.cfg), **combined_dict}

            init_args = {"job_type": self.job_type, "group": self.cfg['group']}
            if self._wandb.run is None:
                self._wandb.init(
                    project=os.getenv('WANDB_PROJECT'),
                    name=os.getenv('WANDB_RUN_NAME'),
                    entity=os.getenv('WANDB_ENTITY'),
                    config=combined_dict,
                    **init_args,
                )

            # keep track of model topology and gradients, unsupported on TPU
            if os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps)
                )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model and self._initialized and state.is_world_process_zero:
            logger.info("Saving Model")
            from src.training.trainer import CustomTrainer

            fake_trainer = CustomTrainer(cfg=self.cfg, args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Do not want to save the model if it is in debug mode.
                if not self.cfg.debug:
                    fake_trainer.save_model(temp_dir)
                else:
                    with Path(temp_dir).joinpath('config.yaml').open('w', encoding='utf-8') as f:
                        f.write(OmegaConf.to_yaml(self.cfg, resolve=True))

                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss"                 : state.total_flos,
                    }
                )
                artifact = self._wandb.Artifact(name=f"model-{self._wandb.run.id}", type="model",
                                                metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                artifact.add_file(str(Path(temp_dir, 'config.yaml').resolve().absolute()))

                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)

            if isinstance(kwargs['train_dataloader'].dataset, StackOverflowTask):
                train_ds = kwargs['train_dataloader'].dataset
                logs['train/ds_epoch'] = train_ds.epoch
                logs['train/samples_seen'] = train_ds.samples_seen
            self._wandb.log(logs, step=state.global_step)


def get_config_for_tracking(cfg: Union[DictConfig, Dict]):
    out_cfg = OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else copy(cfg)
    out_cfg.pop('tracking')
    out_cfg.pop('name')
    out_cfg.pop('group')
    out_cfg.pop('project')
    return flatten(out_cfg, sep='.')


def is_tracking_enabled(cfg: DictConfig):
    return isinstance(cfg['tracking'], (DictConfig, dict))


def setup_tracking_env_from_cfg(cfg: DictConfig):
    if not is_tracking_enabled(cfg):
        logger.warning("Tracking is disabled")
        os.environ['WANDB_DISABLED'] = 'true'
        return

    logger.info("Setting up tracking")

    os.environ['WANDB_DISABLED'] = 'false'
    os.environ['WANDB_WATCH'] = cfg['tracking'].get('watch')
    project = cfg['tracking'].get('project')
    run_name = cfg["name"]

    if cfg.group != cfg.task.name.upper():
        run_name = f"{cfg.task.name.upper()}.{run_name}"

    if cfg.debug:
        project = f"debug-{project}"
        run_name = f"debug-{run_name}"

    entity = cfg['tracking'].get('entity')
    if entity and not cfg.debug:
        os.environ['WANDB_ENTITY'] = entity
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_RUN_NAME'] = run_name
    os.environ['WANDB_LOG_MODEL'] = 'true' if cfg['tracking'].get('log_model') else 'false'
    os.environ['DISABLE_FAST_TOK'] = 'true'


def get_run_base_name_from_cfg(cfg: DictConfig) -> str:
    return f"{cfg.group}.{cfg.name}"
