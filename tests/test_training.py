"""
Tests for the training data.
"""
from dataclasses import asdict
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch, MagicMock

import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from omegaconf import OmegaConf
import yaml

from src.common.config import get_device_from_cfg
from src.training import train_model, get_training_args_from_config
from src.data import load_task_from_cfg


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
    expected = Seq2SeqTrainingArguments(**training_args)
    assert get_training_args_from_config(cfg) == expected


@pytest.mark.parametrize("device_val", [-1, "cuda"])
def test_train_model(tmpdir, training_args, simple_config, tiny_model_name, device_val):
    tmpdir_path = Path(tmpdir)
    simple_config['device'] = device_val
    cfg = OmegaConf.create(simple_config)

    task = load_task_from_cfg(cfg, AutoTokenizer.from_pretrained(tiny_model_name))
    task.read_data = MagicMock()
    task.read_data.side_effect = (["TRAIN_RAW", "TRAIN_TOK"], ["VAL_RAW", "VAL_TOK"])

    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))

    with patch("src.training.CustomTrainer") as mock_trainer:
        with patch(
                "src.training.AutoModelForSeq2SeqLM.from_pretrained",
                return_value=torch.zeros((1, 1), device=get_device_from_cfg(cfg)),
        ) as mock_model:
            result = train_model(cfg, tmpdir_path, task)

    assert result == mock_model.return_value
    assert task.read_data.call_count == 2

    train_args = task.read_data.call_args_list[0]
    assert train_args.args == (tmpdir_path.joinpath(cfg['task']["paths"]["train"]),)
    assert train_args.kwargs == {"set_format": "torch"}

    val_args = task.read_data.call_args_list[1]
    assert val_args.args == (tmpdir_path.joinpath(cfg['task']["paths"]["validation"]),)
    assert val_args.kwargs == {"set_format": "torch"}

    assert mock_model.call_count == 1
    assert mock_model.call_args_list[0].args == (tiny_model_name,)

    assert mock_trainer.call_count == 1

    call_kwargs = mock_trainer.call_args_list[0].kwargs
    assert call_kwargs["args"] == get_training_args_from_config(cfg)
    assert call_kwargs["train_dataset"] == "TRAIN_TOK"
    assert call_kwargs["eval_dataset"] == "VAL_TOK"
