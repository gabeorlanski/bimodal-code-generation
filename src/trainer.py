from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Union, Dict, Callable
import torch
from torch.utils.data import Dataset as pt_Dataset
from transformers import AdamW, get_scheduler, DataCollatorWithPadding, PreTrainedTokenizer
from tqdm import tqdm
import math
import logging

from datetime import datetime, timedelta

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

    def __post_init__(self):
        self.more_is_better = True if self.metric_for_best_model.startswith('+') else False
        self.metric_for_best_model = self.metric_for_best_model[1:]


class Trainer:

    def __init__(
            self,
            model: torch.nn.Module,
            args: TrainingArguments,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            evaluate_fn: Callable,
            data_loading_fn: Optional[Callable] = None,
            collator_fn: Optional[Callable] = None
    ):
        logger.info("Initializing trainer.")
        self.args = args
        self.tokenizer = tokenizer
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
            self.args,
            train_dataset,
            eval_dataset
        )

        steps_per_epoch = len(train_loader)
        stop_steps = min(
            self.args.max_steps,
            steps_per_epoch * self.args.max_epochs
        )
        logger.info(f"Stopping after {stop_steps} steps")

        logger.info("Staring Training")

        for epoch in range(1, self.args.max_epochs + 1):
            train_metrics = self._train_epoch(train_loader, epoch)

            logger.info(f"Finished training for epoch {epoch}")

            eval_metrics = self.evaluate_fn(self.args, self.model, eval_loader, self.device)
            self.log_eval_metrics(epoch, train_metrics, eval_metrics)

            if self.global_step >= self.args.max_steps:
                logger.info("Passed Max Steps, stopping.")
                break

            elapsed, estimated = self.get_estimated_remaining_time(
                datetime.utcnow() - start_time,
                self.args.max_steps
            )
            logger.info(
                f"Finished {self.global_step}/{stop_steps} Steps in {elapsed}"
            )
            logger.info(f"Estimated time remaining: {estimated}")

    def _train_epoch(self, data_loader, epoch):
        self.model.train()
        total_batches = len(data_loader)

        batch_iter = iter(data_loader)
        total_loss = 0

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
            # self.accelerator.backward(loss)
            loss.backward()
            if (
                    step % self.args.grad_accumulation_steps == 0
                    or total_batches - step == 0
            ):
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            self.global_step += 1
            if self.global_step > self.args.max_steps:
                break
            pbar.update()
        pbar.close()

        return {'loss': total_loss / total_batches}

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
