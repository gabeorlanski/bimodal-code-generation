"""
Functions for tracking.
"""
from typing import Dict

from transformers.integrations import WandbCallback
from omegaconf import DictConfig, OmegaConf
import os
import logging
from src.common import flatten

logger = logging.getLogger(__name__)


class TrackingCallback(WandbCallback):
    def __init__(self, job_type: str, cfg: DictConfig):
        super(TrackingCallback, self).__init__()
        self.cfg = OmegaConf.to_object(cfg)
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
                    project=os.getenv("WANDB_PROJECT", "huggingface"),
                    name=self.cfg['name'],
                    config=combined_dict,
                    **init_args,
                )
            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model, log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps)
                )


def get_config_for_tracking(cfg: Dict):
    cfg.pop('training')
    cfg.pop('tracking')
    return flatten(cfg, sep='.')


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
    os.environ['WANDB_PROJECT'] = cfg['tracking'].get('project')
    os.environ['WANDB_LOG_MODEL'] = 'true' if cfg['tracking'].get('log_model') else 'false'
