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

from src.common import FIXTURES_ROOT, PROJECT_ROOT
from src.training import train_model, get_training_args_from_config
from src.data import MBPP, MBPPConfig


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
    training_args.pop('batch_size', None)
    if batch_size is not None:
        training_args['per_device_train_batch_size'] = batch_size
        training_args['per_device_eval_batch_size'] = batch_size

    cfg = OmegaConf.create(
        {"training": training_args}
    )
    expected = Seq2SeqTrainingArguments(**training_args)
    assert get_training_args_from_config(cfg) == expected


@pytest.mark.parametrize('device_val', [-1, 'cuda'])
def test_train_model(tmpdir, training_args, tiny_model_name, device_val):
    tmpdir_path = Path(tmpdir)
    reader_cfg = MBPPConfig()
    cfg = OmegaConf.create({
        "training": training_args,
        "model"   : tiny_model_name,
        "device"  : device_val,
        "dataset" : asdict(reader_cfg)
    })

    reader = MBPP(AutoTokenizer.from_pretrained(tiny_model_name))
    reader.read_data = MagicMock()
    reader.read_data.side_effect = (["TRAIN_RAW", "TRAIN_TOK"], ["VAL_RAW", "VAL_TOK"])

    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))

    with patch('src.training.Seq2SeqTrainer') as mock_trainer:
        with patch(
                'src.training.AutoModelForSeq2SeqLM.from_pretrained',
                return_value=torch.zeros((1, 1))
        ) as mock_model:
            train_model(cfg, tmpdir_path, reader)

    assert reader.read_data.call_count == 2

    train_args = reader.read_data.call_args_list[0]
    assert train_args.args == (tmpdir_path.joinpath(reader_cfg.train_path),)
    assert train_args.kwargs == {'set_format': 'torch'}

    val_args = reader.read_data.call_args_list[1]
    assert val_args.args == (tmpdir_path.joinpath(reader_cfg.validation_path),)
    assert val_args.kwargs == {'set_format': 'torch'}

    assert mock_model.call_count == 1
    assert mock_model.call_args_list[0].args == (tiny_model_name,)

    assert mock_trainer.call_count == 1
    assert mock_trainer.call_args_list[0].kwargs == {
        "model"        : mock_model.return_value,
        "args"         : get_training_args_from_config(cfg),
        "train_dataset": "TRAIN_TOK",
        "eval_dataset" : "VAL_TOK",
        "data_collator": DataCollatorForSeq2Seq(
            reader.tokenizer,
            mock_model.return_value,
            return_tensors='pt'
        )
    }
