from typing import Dict, Optional, Union
import logging
from omegaconf import DictConfig
from transformers import Seq2SeqTrainer, ProgressCallback, TrainerCallback
from transformers.integrations import WandbCallback
from overrides import overrides
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
        # self.callback_handler.pop_callback(ProgressCallback)
        # self.callback_handler.add_callback(BetterProgress)

        if is_tracking_enabled(cfg):
            self.callback_handler.pop_callback(WandbCallback)
            self.callback_handler.add_callback(TrackingCallback('training', cfg))

        self.train_stats = None


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
