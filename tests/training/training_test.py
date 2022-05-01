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
