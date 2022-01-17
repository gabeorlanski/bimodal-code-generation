import math
import shutil
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Union, Dict, Callable
import torch
from torch.utils.data import Dataset as pt_Dataset
from transformers import (
    AdamW, get_scheduler, DataCollatorForSeq2Seq, PreTrainedTokenizer
)
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
import wandb
from src import config as config_util

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
            self,
            cfg: DictConfig,
            model: torch.nn.Module,
            tokenizer: PreTrainedTokenizer,
            evaluate_fn: Callable,
            path_to_use: str = None,
            metric_key_prefix: str = 'eval'
    ):
        self.local_rank = 0

        self.log_message(logging.INFO, "Initializing trainer.")
        self.cfg = cfg
        self.training_args = config_util.get_training_args_from_cfg(cfg)
        self.tokenizer = tokenizer
        self.device = config_util.get_device_from_cfg(cfg)
        self.objective = cfg.objective
        self.model = model.to(self.device)
        self.evaluate_fn = evaluate_fn
        self.metric_key_prefix = metric_key_prefix
        self.has_logged_before = {}

        self.collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding='longest',
            pad_to_multiple_of=2,
            return_tensors='pt'
        )

        self.global_step = 0

        # Initializing values used for tracking the best model.
        self.path_to_best_model = None
        self.best_metric = None
        self.checkpoints = []
        self.model_dir = Path(path_to_use) if path_to_use else Path()
        self.model_dir = self.model_dir.joinpath(self.training_args.output_dir)
        if not self.model_dir.exists():
            self.log_message(logging.DEBUG, f"Making directory at {self.model_dir}")
            self.model_dir.mkdir(parents=True)
        self.log_message(logging.INFO, f"Saving checkpoints to {self.model_dir}")

        self.run = self.setup_tracking_from_config(cfg)

    @property
    def is_world_process_zero(self):
        return self.local_rank == 0

    def setup_tracking_from_config(self, cfg):
        if not self.is_world_process_zero:
            return None

        if not config_util.is_tracking_enabled(cfg):
            self.log_message(logging.INFO, 'Tracking is disabled')
            return None

        self.log_message(logging.INFO, 'Setting up tracking')
        run_config = config_util.get_config_for_tracking(cfg)
        run_config.update({f"model.{k}": v for k, v in self.model.config.to_dict().items()})

        training_args_as_dict = {}
        training_args_keys_to_remove = {
            "do_predict", "do_train", "do_eval", "evaluation_strategy", "prediction_loss_only",
            "log_level", "log_level_replica", "log_on_each_node", "logging_dir", "logging_strategy",
            "logging_first_step", "save_strategy", "save_on_each_node", "label_names",
            "load_best_model_at_end", "ignore_data_skip"
        }
        for k, v in self.training_args.to_sanitized_dict().items():
            if k in training_args_keys_to_remove:
                continue
            training_args_as_dict[f"training.{k}"] = v
        run_config.update(training_args_as_dict)

        run = wandb.init(
            job_type='train',
            project=cfg['project'],
            group=cfg['group'],
            name=cfg['name'],
            config=run_config,
            config_exclude_keys=['tracking', 'name', 'group', 'project']
        )
        run.watch(self.model, log_freq=max(100, self.training_args.logging_steps), log='all',
                  log_graph=True)
        return run

    def __call__(
            self,
            train_dataset: pt_Dataset,
            eval_dataset: pt_Dataset
    ):

        self.log_message(logging.INFO, "Beginning training setup")

        start_time = datetime.utcnow()

        self.log_message(logging.INFO, "Training Arguments:")
        for arg_name in sorted(asdict(self.training_args)):
            self.log_message(logging.INFO,
                             f"{arg_name:>30} = {getattr(self.training_args, arg_name)}")

        self.log_message(logging.INFO, "Sorting Datasets by the 'input_ids' column")
        train_dataset = train_dataset.map(
            lambda ex, idx: {'length': len(ex['input_ids']), 'idx': idx, **ex},
            with_indices=True
        ).sort("length", reverse=True)
        train_dataset = train_dataset.remove_columns('length')

        eval_dataset = eval_dataset.map(
            lambda ex, idx: {'length': len(ex['input_ids']), 'idx': idx, **ex},
            with_indices=True
        ).sort("length", reverse=True)
        eval_dataset = eval_dataset.remove_columns('length')

        self.log_message(logging.INFO, "Setting up data loaders")
        train_loader, eval_loader = self._get_data_loaders(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        steps_per_epoch = len(train_loader)
        stop_steps = min(
            self.training_args.max_steps,
            math.ceil(steps_per_epoch * float(self.training_args.num_train_epochs))
        )
        self.training_args.max_steps = stop_steps
        self.log_message(logging.INFO, f"Stopping after {stop_steps} steps")

        self.log_message(logging.INFO, "Setting up optimizer")
        optimizer = AdamW(
            self.get_grouped_params(self.model),
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon
        )
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=stop_steps,
        )

        self.log_message(logging.INFO, "Doing zero-shot eval")
        with torch.no_grad():
            self.model.eval()
            eval_metrics = self.evaluate_fn(
                self.training_args,
                self.model,
                eval_loader,
                self.device
            )
        self.log_eval_metrics(
            epoch=0,
            metrics=eval_metrics,
            eval_metrics=eval_metrics
        )

        self.log_message(logging.INFO, "Staring Training")
        for epoch in range(1, int(self.training_args.num_train_epochs) + 1):
            self.model.train()
            epoch_start_time = datetime.utcnow()
            train_metrics, optimizer, lr_scheduler = self._train_epoch(
                epoch=epoch,
                data_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler
            )

            elapsed = datetime.utcnow() - epoch_start_time

            self.log_message(
                logging.INFO,
                f"Finished training for epoch {epoch}"
            )
            self.log_message(
                logging.DEBUG,
                f"{train_metrics.pop('updates')} Updates done in {str(elapsed)}"
            )
            with torch.no_grad():
                self.model.eval()
                eval_metrics = self.evaluate_fn(
                    self.training_args,
                    self.model,
                    eval_loader,
                    self.device
                )
            self.log_metrics_to_run(eval_metrics, 'eval', epoch)
            self.log_metrics_to_run(train_metrics, 'train', epoch)
            self.log_eval_metrics(
                epoch=epoch,
                metrics=train_metrics,
                eval_metrics=eval_metrics
            )

            self.save_model(eval_metrics)

            if self.global_step > stop_steps and epoch < self.training_args.num_train_epochs:
                self.log_message(logging.INFO, "Passed Max Steps, stopping.")
                break

            elapsed, estimated = self.get_estimated_remaining_time(
                datetime.utcnow() - start_time,
                stop_steps
            )
            self.log_message(logging.INFO,
                             f"Finished {self.global_step}/{stop_steps} Steps in {elapsed}"
                             )
            self.log_message(logging.INFO, f"Estimated time remaining: {estimated}")
        self.copy_best()
        if self.run is not None:
            self.run.finish()

    def _train_epoch(self, epoch, data_loader, optimizer, lr_scheduler):
        self.model.train()
        total_batches = len(data_loader)
        batch_iter = iter(data_loader)
        total_loss = 0
        updates = 0
        pbar = tqdm(total=total_batches, desc=f'Epoch {epoch}')
        for step in range(1, total_batches + 1):
            batch = next(batch_iter)
            local_batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                local_batch['input_ids'],
                labels=local_batch.get('labels', local_batch['input_ids']),
                use_cache=False
            )
            loss = outputs.loss
            loss /= self.training_args.gradient_accumulation_steps
            total_loss += loss.item()
            pbar.set_description(f'Epoch {epoch}: batch_loss={loss.item():0.3f} '
                                 f'loss={total_loss / step:0.3f}')

            loss.backward()
            # We make sure that even if grad accumulation is on, we still do
            # the steps if this is the last batch in the epoch.
            if (
                    step % self.training_args.gradient_accumulation_steps == 0
                    or total_batches - step == 0
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                self.model.zero_grad()
                updates += 1

            self.global_step += 1
            if self.global_step % self.training_args.logging_steps == 0:
                self.log_metrics_to_run(
                    {'batch_loss': total_loss / step, 'lr': get_lr(optimizer)},
                    prefix='train'
                )

            if self.global_step > self.training_args.max_steps:
                break
            pbar.update()
        pbar.close()
        out_metrics = {
            'loss'   : total_loss / total_batches,
            'updates': updates
        }
        return out_metrics, optimizer, lr_scheduler

    def get_grouped_params(self, model, no_decay=None):
        if no_decay is None:
            no_decay = ["bias", "LayerNorm.weight"]
        params_with_wd, params_without_wd = [], []
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": self.training_args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    def log_eval_metrics(
            self,
            epoch,
            metrics: Dict[str, float],
            eval_metrics: Dict[str, float] = None
    ):
        if eval_metrics is None:
            eval_metrics = {}

        self.log_message(logging.INFO, f"Metrics for Epoch {epoch}:")
        # self.log_message(logging.INFO,create_log_metric_message('Name', 'Train', 'Eval'))

        all_keys = set(eval_metrics).union(
            metrics
        )

        for k in all_keys:
            eval_value = eval_metrics.get(k)
            train_value = metrics.get(k)
            self.log_message(logging.INFO,
                             f"{create_log_metric_message(k, train_value, eval_value)}")

    def _get_data_loaders(
            self,
            train_dataset: pt_Dataset,
            eval_dataset: pt_Dataset,
    ):

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=self.collator,
            shuffle=False,
            batch_size=self.training_args.per_device_train_batch_size,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            collate_fn=self.collator,
            shuffle=False,
            batch_size=self.training_args.per_device_eval_batch_size,
        )
        return train_loader, eval_loader

    def get_estimated_remaining_time(self, elapsed, max_steps):
        current_rate = self.global_step / elapsed.total_seconds()
        try:
            elapsed_str, _ = str(elapsed).split(".")
        except ValueError:
            elapsed_str = str(elapsed)

        estimated_rem = timedelta(
            seconds=(max_steps - self.global_step) / current_rate
        )
        try:
            estimated_rem, _ = str(estimated_rem).split(".")
        except ValueError:
            estimated_rem = str(estimated_rem)
        return elapsed_str, estimated_rem

    def save_model(self, eval_metrics: Dict):
        self.log_message(logging.DEBUG, f"Updating the best model at step {self.global_step}")

        model_is_better = False
        metric_value = eval_metrics[self.training_args.metric_for_best_model]
        if self.best_metric is not None:
            # A best model has been set.
            self.log_message(logging.DEBUG,
                             f"Using {self.training_args.metric_for_best_model} to determine "
                             f"the best model at step {self.global_step}")
            if self.training_args.greater_is_better and metric_value > self.best_metric:
                model_is_better = True
            elif not self.training_args.greater_is_better and metric_value < self.best_metric:
                model_is_better = True
        else:
            # No Best model is set
            model_is_better = True

        checkpoint_path = self.model_dir.joinpath(
            f"model_{self.global_step}.bin")

        if len(self.checkpoints) == self.training_args.save_total_limit:
            self.log_message(logging.DEBUG, f"{len(self.checkpoints)} checkpoints saved already, "
                                            f"removing oldest.")

            to_remove = 0
            if self.checkpoints[to_remove] == self.path_to_best_model:
                self.log_message(logging.DEBUG, f"Checkpoint at index {to_remove} is the best "
                                                f"model, trying the next index.")
                to_remove = 1

            path_to_remove = self.checkpoints.pop(to_remove)
            self.log_message(logging.INFO, f"Deleting checkpoint at {path_to_remove}")
            os.remove(path_to_remove)

        self.log_message(logging.INFO, f"Saving checkpoint to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        if model_is_better:
            self.log_message(logging.INFO,
                             f"New Best Model with {self.training_args.metric_for_best_model}"
                             f" = {metric_value:.3f}")
            self.path_to_best_model = checkpoint_path
            self.best_metric = metric_value

    def copy_best(self):

        self.log_message(logging.INFO, f"Saving best model to {Path.cwd()}")
        shutil.copy2(self.path_to_best_model, Path('best_model.bin'))

    def log_message(self, level, message):
        if self.local_rank > 0:
            logger.log(level, f"Worker {self.local_rank}: {message}")
        else:
            logger.log(level, message)

    def log_metrics_to_run(self, metrics, prefix, epoch=None):
        self.log_message(logging.DEBUG, f"Logging metrics to run at step "
                                        f"{self.global_step}.")

        if self.run is None:
            return

        metrics_to_log = {f"{prefix}/{k}": v for k, v in metrics.items()}
        if epoch is not None:
            metrics_to_log['epoch'] = epoch
        self.run.log(metrics_to_log, step=self.global_step)


def create_log_metric_message(
        metric_name: str,
        train_value: Optional[Union[str, float]],
        eval_value: Optional[Union[str, float]],
        no_eval: bool = False
) -> str:
    def format_metric_msg(metric: Optional[float]):
        if metric is None:
            return f"{'N/A':>10}"
        if not isinstance(metric, str):
            return f"{metric:>10.3f}"
        return f"{metric:>10}"

    msg = f"{metric_name:>20} | "
    msg += f"{format_metric_msg(train_value)} | "
    if not no_eval:
        msg += f"{format_metric_msg(eval_value)}"
    return msg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
