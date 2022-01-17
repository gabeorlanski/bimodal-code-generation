from transformers import Seq2SeqTrainingArguments
import logging
from omegaconf import OmegaConf, DictConfig

__all__ = ["get_training_args_from_cfg"]

logger = logging.getLogger(__name__)


def get_training_args_from_cfg(cfg: DictConfig) -> Seq2SeqTrainingArguments:
    """
    Get the training arguments to create a HuggingFace training arguments
    objects from the passed in config.

    Special Keys:
        ``batch_size``: will be used for the keys ``per_device_train_batch_size``
        and ``per_device_eval_batch_size``

    Args:
        cfg (DictConfig): The OmegaConf config.

    Returns:
        TrainingArguments: The processed training arguments.
    """

    training_args = OmegaConf.to_object(cfg["training"])

    batch_size = training_args.pop("batch_size", None)
    if batch_size:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size
    return Seq2SeqTrainingArguments(**training_args)
