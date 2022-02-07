import math
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List, Any
import logging

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import TrainerCallback, ProgressCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.integrations import WandbCallback
from tqdm import tqdm
import collections
from datetime import datetime, timedelta

from src.config import TrackingCallback, is_tracking_enabled

logger = logging.getLogger(__name__)


class BetterProgress(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.utcnow()

    def on_step_end(self, args, state, control, **kwargs):
        # if state.is_local_process_zero:
        #     self.training_bar.update(state.global_step - self.current_step)
        self.current_step = state.global_step
        if self.current_step % 100 == 0:
            elapsed = datetime.utcnow() - self.start_time
            try:
                elapsed_str, _ = str(elapsed).split(".")
            except ValueError:
                elapsed_str = str(elapsed)

            logger.info(
                f"Finished {self.current_step}/{state.max_steps} Steps in {elapsed_str}"
            )
            current_rate = self.current_step / elapsed.total_seconds()
            estimated_rem = timedelta(
                seconds=(state.max_steps - self.current_step) / current_rate
            )
            try:
                estimated_rem, _ = str(estimated_rem).split(".")
            except ValueError:
                estimated_rem = str(estimated_rem)
            logger.info(f"Estimated time remaining: {estimated_rem}")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero and isinstance(
                eval_dataloader.dataset, collections.abc.Sized
        ):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(eval_dataloader))
            self.prediction_bar.update(1)
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Metrics For {state.global_step}:")
            for k in sorted(logs):
                train_value = logs.get(k)
                logger.info(f"\t{create_log_metric_message(k, train_value)}")
        pass


class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        # Initialize the variables to supress warnings
        self.state = None
        self.args = None
        self.control = None

        super(CustomTrainer, self).__init__(*args, **kwargs)
        self.callback_handler.pop_callback(ProgressCallback)
        # self.callback_handler.add_callback(BetterProgress)

        if is_tracking_enabled(cfg):
            self.callback_handler.pop_callback(WandbCallback)
            self.callback_handler.add_callback(TrackingCallback('training', cfg))

        self.cfg = cfg
        self.start_time = None
        self.time_last_log = None
        self.last_runtime_step = 0

    def save_model(self, output_dir: Optional[str] = None):
        super(CustomTrainer, self).save_model(output_dir)
        if self.args.should_save:
            cfg_path = Path(output_dir).joinpath('config.yaml')
            with cfg_path.open('w', encoding='utf-8') as f:
                f.write(OmegaConf.to_yaml(self.cfg, resolve=True))

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        print_dict = {}
        if self.control.should_log:
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
            elapsed_last_log = (current - self.time_last_log).total_seconds()

            logs['train_runtime_last_log'] = round(elapsed_last_log, 4)
            logs['train_runtime_total'] = round(elapsed_start, 4)

            elapsed_steps = self.state.global_step - self.last_runtime_step
            logs['train_steps_per_second_last_log'] = round(
                elapsed_steps / elapsed_last_log, 3
            )
            logs['train_steps_per_second_total'] = round(
                self.state.global_step / elapsed_start, 3
            )

            self.last_runtime_step = self.state.global_step
            self.time_last_log = datetime.utcnow()
            print_dict.update(logs)

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)
            print_dict.update(metrics)

        if print_dict:
            logger.info(f"Step {self.state.global_step}:")
            for k, v in print_dict.items():
                logger.info(f"\t{k:>32}={v:0.3f}")

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def train(
            self,
            resume_from_checkpoint=None,
            trial=None,
            ignore_keys_for_eval=None,
            **kwargs,
    ):
        self.time_last_log = self.start_time = datetime.utcnow()
        super(CustomTrainer, self).train(
            resume_from_checkpoint,
            trial,
            ignore_keys_for_eval,
            **kwargs
        )

    def evaluation_loop(
            self,
            dataloader,
            description,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ):
        logger.debug(f"{type(dataloader.dataset)=}")
        logger.debug(f"{isinstance(dataloader.dataset, collections.abc.Sized)=}")
        return super(CustomTrainer, self).evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        if eval_dataset is None:
            logger.debug(f"{type(eval_dataset)=}")
            logger.debug(f"{isinstance(eval_dataset, collections.abc.Sized)=}")
        else:
            logger.debug(f"{type(self.eval_dataset)=}")
            logger.debug(f"{isinstance(self.eval_dataset, collections.abc.Sized)=}")

        return super(CustomTrainer, self).get_eval_dataloader(eval_dataset)


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
