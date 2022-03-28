import math
import os
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List, Any
import logging

import psutil
import torch
from datasets import Dataset
from datasets import IterableDataset as HFIterableDataset
from torch.utils.data import IterableDataset
import datasets
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainerCallback, ProgressCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.integrations import WandbCallback
from tqdm import tqdm
import collections
from datetime import datetime, timedelta
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard

from src.config import TrackingCallback, is_tracking_enabled
from src.data.tensorize import TensorizedTask

logger = logging.getLogger(__name__)


class HFIterableWrapper(IterableDataset):
    def __init__(self, hf_dataset, buffer=1000):
        self.ds = hf_dataset
        self.buffer = buffer

    def __iter__(self):
        data_iter = iter(self.ds)
        lines_seen = 0
        more_examples = True
        ds_epoch = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
        else:
            raise ValueError(f"Not Compatible with multiple dataset workers")
        while more_examples:
            instances = []
            while len(instances) < self.buffer:
                try:
                    instances.append(next(data_iter))
                except StopIteration:
                    if self.infinite:
                        if worker_id == 0:
                            logger.info(f"New Dataset Epoch")
                        data_iter = iter(self.ds)
                        ds_epoch += 1
                    else:
                        more_examples = False
                        break
            os.environ['DS_EPOCH'] = f"{ds_epoch:0.5f}"
            for i in instances:
                yield i


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        # Initialize the variables to supress warnings
        self.state = None
        self.args = None
        self.control = None

        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.callback_handler.pop_callback(ProgressCallback)

        if is_tracking_enabled(cfg):
            self.callback_handler.pop_callback(WandbCallback)
            self.callback_handler.add_callback(TrackingCallback('training', cfg))

        self.cfg = cfg
        self.start_time = None
        self.last_runtime_step = 0
        self.total_time_spent_in_eval = 0

    def save_model(self, output_dir: Optional[str] = None, _internal_call=False):
        super(CustomTrainer, self).save_model(output_dir, _internal_call)
        if self.args.should_save:
            cfg_path = Path(output_dir).joinpath('config.yaml')
            with cfg_path.open('w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True))

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        print_dict = {}
        if self.start_time is None:
            if hasattr(self.state, 'start_time'):
                self.start_time = self.state.start_time
            else:
                self.start_time = datetime.utcnow()

        if self.control.should_log:
            global TENSORIZED_EPOCH
            logs: Dict[str, float] = {}
            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            current = datetime.utcnow()
            elapsed_start = (current - self.start_time).total_seconds()
            elapsed_start -= self.total_time_spent_in_eval
            logs['train_runtime_total'] = round(elapsed_start, 4)
            logs['train_steps_per_second_total'] = round(
                self.state.global_step / elapsed_start, 3
            )
            logs['train_ram_pct'] = psutil.virtual_memory()[2]

            if 'DS_EPOCH' in os.environ:
                logs['train_ds_epoch'] = float(os.getenv('DS_EPOCH', 0.0))

            self.last_runtime_step = self.state.global_step
            self.time_last_log = datetime.utcnow()
            print_dict.update(logs)

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            eval_start = datetime.utcnow()
            metrics = self.evaluate(self.eval_dataset, ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)
            self.total_time_spent_in_eval += (datetime.utcnow() - eval_start).total_seconds()
            print_dict.update(metrics)

        if print_dict:
            logger.info(f"Step {self.state.global_step}:")
            for k, v in print_dict.items():
                if k == 'learning_rate':
                    log_value_str = f"{v:.3e}"
                else:
                    log_value_str = f"{v:0.3f}"
                logger.info(f"\t{k:>32}={log_value_str}")

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.debug(f"Eval Loop is called for {len(dataloader.dataset)} samples")
        else:
            logger.debug(f"Eval loop is called")

        ram_pct = f"{psutil.virtual_memory()[2]:0.2f}%"
        logger.info(f"RAM Used={ram_pct:<6}")
        return super(CustomTrainer, self).evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        if eval_dataset is not None:
            logger.debug(f"{type(eval_dataset)=}")
            logger.debug(f"{isinstance(eval_dataset, collections.abc.Sized)=}")
        else:
            logger.debug(f"{type(self.eval_dataset)=}")
            logger.debug(f"{isinstance(self.eval_dataset, collections.abc.Sized)=}")

        return super(CustomTrainer, self).get_eval_dataloader(eval_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `self.train_dataset` does not implement `__len__`, a random sampler (adapted to
        distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if (
                isinstance(train_dataset, torch.utils.data.IterableDataset)
                or isinstance(train_dataset, HFIterableDataset)
        ):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def create_log_metric_message(
        metric_name: str,
        train_value: Optional[Union[str, float]]
) -> str:
    def format_metric_msg(metric: Optional[float]):
        if metric is None:
            return f"{'N/A':>10}"
        return f"{metric:>10.3f}"

    msg = f"{metric_name:>24} = "
    msg += f"{format_metric_msg(train_value)}"
    return msg
