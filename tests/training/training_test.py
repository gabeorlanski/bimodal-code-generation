"""
Tests for the training
"""
from pathlib import Path
import pytest
from unittest.mock import patch
import torch

from omegaconf import OmegaConf
from src.config import get_device_from_cfg, get_training_args_from_cfg
from src.config.training import TrainingArguments


@pytest.fixture()
def training_args():
    yield {
        "batch_size"           : 4,
        "output_dir"           : "models",
        "group_by_length"      : True,
        "predict_with_generate": True,
        "evaluation_strategy"  : "epoch",
    }


@pytest.mark.parametrize("batch_size", [None, 4])
def test_get_training_args_from_config(training_args, batch_size):
    training_args.pop("batch_size", None)
    if batch_size is not None:
        training_args["per_device_train_batch_size"] = batch_size
        training_args["per_device_eval_batch_size"] = batch_size

    cfg = OmegaConf.create({"training": training_args})
    expected = TrainingArguments(**training_args)
    assert get_training_args_from_cfg(cfg) == expected

