from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Union, Dict, Callable
import torch
from torch import nn
from torch.utils.data import Dataset as pt_Dataset
from transformers import AdamW, get_scheduler, DataCollatorWithPadding, PreTrainedTokenizer, \
    AutoModelForCausalLM, AutoModelForSeq2SeqLM
from tqdm import tqdm
from omegaconf import DictConfig
import logging
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import os
from accelerate import Accelerator
from src import config

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    train_batch_size: int = 2
    eval_batch_size: int = 2
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    save_epochs: int = 250
    logging_steps: int = 100
    grad_accumulation_steps: int = 1
    metric_for_best_model: str = "-loss"
    max_steps: int = 1000
    max_epochs: int = 32
    steps_per_epoch: int = -1
    model_dir: str = "checkpoints"
    checkpoints_to_save: int = 5
    eval_prefix: str = "eval"

    def __post_init__(self):
        self.more_is_better = True if self.metric_for_best_model.startswith('+') else False
        self.metric_for_best_model = self.metric_for_best_model[1:]
        if self.checkpoints_to_save < 2:
            raise ValueError("Checkpoints to save must be greater than 2")


class Trainer:

    def __init__(
            self,
            cfg: DictConfig,
            device: torch.device,
            model: torch.nn.Module,
            tokenizer: PreTrainedTokenizer,
            evaluate_fn: Callable,
            data_loading_fn: Optional[Callable] = None,
            collator_fn: Optional[Callable] = None,
            path_to_use: str = None
    ):
        logger.info("Initializing trainer.")
        self.cfg = cfg
        self.args = TrainingArguments(**cfg['training'])
        self.tokenizer = tokenizer
        # self.accelerator = accelerator
        self.device = device
        self.model = model.to(self.device)
        self.evaluate_fn = evaluate_fn
        self.data_loading_fn = data_loading_fn or self._get_data_loaders
        self.collate_fn = collator_fn or DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding='longest',
            pad_to_multiple_of=2,
            return_tensors='np'
        )

        logger.info("Setting up optimizer")
        self.optimizer = AdamW(self.get_grouped_params(model), lr=self.args.learning_rate)
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=10,
            num_training_steps=1000,
        )
        self.global_step = 0
        self.path_to_best_model = None
        self.best_metric = None
        self.model_dir = Path(path_to_use) if path_to_use else Path()
        self.model_dir = self.model_dir.joinpath(self.args.model_dir)
        self.checkpoints = []
        if not self.model_dir.exists():
            logger.debug(f"Making directory at {self.model_dir}")
            self.model_dir.mkdir(parents=True)
        logger.info(f"Saving checkpoints to {self.model_dir}")

    def __call__(
            self,
            train_dataset: pt_Dataset,
            eval_dataset: pt_Dataset
    ):

        logger.info("Beginning training setup")

        start_time = datetime.utcnow()

        logger.info("Training Arguments:")
        for arg, value in asdict(self.args).items():
            logger.info(f"{arg:>24} = {value}")

        logger.info("Setting up data loaders")
        train_loader, eval_loader = self.data_loading_fn(
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        # self.model, self.optimizer, train_loader, eval_loader = self.accelerator.prepare(
        #     self.model, self.optimizer, train_loader, eval_loader
        # )

        steps_per_epoch = len(train_loader)
        stop_steps = min(
            self.args.max_steps,
            steps_per_epoch * self.args.max_epochs
        )
        logger.info(f"Stopping after {stop_steps} steps")

        logger.info("Staring Training")

        for epoch in range(1, self.args.max_epochs + 1):
            epoch_start_time = datetime.utcnow()
            train_metrics = self._train_epoch(
                data_loader=train_loader,
                epoch=epoch
            )

            elapsed = datetime.utcnow() - epoch_start_time

            logger.info(
                f"Finished training for epoch {epoch}. "
                f"{train_metrics.pop('updates')} Updates done in {str(elapsed)}"
            )

            eval_metrics = self.evaluate_fn(
                self.args,
                self.model,
                eval_loader,
                self.device
            ).items()
            eval_metrics = {f"{self.args.eval_prefix}_{k}": v for k, v in eval_metrics}
            self.log_eval_metrics(
                epoch=epoch,
                metrics=train_metrics,
                eval_metrics=eval_metrics
            )

            self.save_model(eval_metrics)

            if self.global_step >= stop_steps:
                logger.info("Passed Max Steps, stopping.")
                break

            elapsed, estimated = self.get_estimated_remaining_time(
                datetime.utcnow() - start_time,
                stop_steps
            )
            logger.info(
                f"Finished {self.global_step}/{stop_steps} Steps in {elapsed}"
            )
            logger.info(f"Estimated time remaining: {estimated}")
        logger.info(f"Loading best model from {self.path_to_best_model}")
        self._load_best()

    def _train_epoch(self, data_loader, epoch):
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
            loss /= self.args.grad_accumulation_steps
            total_loss += loss.item()
            pbar.set_description(f'Epoch {epoch}: batch_loss={loss.item():0.3f} '
                                 f'loss={total_loss / step:0.3f}')

            loss.backward()
            # We make sure that even if grad accumulation is on, we still do
            # the steps if this is the last batch in the epoch.
            if (
                    step % self.args.grad_accumulation_steps == 0
                    or total_batches - step == 0
            ):
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                updates += 1

            self.global_step += 1
            if self.global_step > self.args.max_steps:
                break
            pbar.update()
        pbar.close()
        return {
            'loss'   : total_loss / total_batches,
            'updates': updates
        }

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
            {"params": params_with_wd, "weight_decay": self.args.weight_decay},
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

        logger.info(f"Metrics for Epoch {epoch}:")
        # logger.info(create_log_metric_message('Name', 'Train', 'Eval'))

        all_keys = set("_".join(k.split("_")[1:]) for k in eval_metrics).union(
            metrics
        )

        for k in all_keys:
            eval_value = eval_metrics.get(k, eval_metrics.get(f"eval_{k}"))
            train_value = metrics.get(k)
            logger.info(f"{create_log_metric_message(k, train_value, eval_value)}")

    def _get_data_loaders(
            self,
            args: TrainingArguments,
            train_dataset: pt_Dataset,
            eval_dataset: pt_Dataset
    ):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=self.collate_fn,
            shuffle=False,
            batch_size=args.train_batch_size,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            collate_fn=self.collate_fn,
            shuffle=False,
            batch_size=args.eval_batch_size,
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
        logger.debug(f"Updating the best model at step {self.global_step}")

        model_is_better = False
        metric_value = eval_metrics[f"{self.args.eval_prefix}_{self.args.metric_for_best_model}"]
        if self.best_metric is not None:
            # A best model has been set.
            logger.debug(f"Using {self.args.metric_for_best_model} to determine "
                         f"the best model at step {self.global_step}")
            if self.args.more_is_better and metric_value > self.best_metric:
                model_is_better = True
            elif not self.args.more_is_better and metric_value < self.best_metric:
                model_is_better = True
        else:
            # No Best model is set
            model_is_better = True

        checkpoint_path = self.model_dir.joinpath(
            f"model_{self.global_step}.bin")

        if len(self.checkpoints) == self.args.checkpoints_to_save:
            logger.debug(f"{len(self.checkpoints)} checkpoints saved already, "
                         f"removing oldest.")

            to_remove = 0
            if self.checkpoints[to_remove] == self.path_to_best_model:
                logger.debug(f"Checkpoint at index {to_remove} is the best "
                             f"model, trying the next index.")
                to_remove = 1

            path_to_remove = self.checkpoints.pop(to_remove)
            logger.info(f"Deleting checkpoint at {path_to_remove}")
            os.remove(path_to_remove)

        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(self.model.state_dict(), checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        if model_is_better:
            logger.info(f"New Best Model with {self.args.metric_for_best_model}"
                        f" = {metric_value:.3f}")
            self.path_to_best_model = checkpoint_path
            self.best_metric = metric_value

    def _load_best(self):
        state_dict = torch.load(self.path_to_best_model)
        if self.cfg.get('objective') == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg['model'],
                                                               state_dict=state_dict)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg['model'],
                                                              state_dict=state_dict)


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
