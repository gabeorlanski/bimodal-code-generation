"""
Tests for the trainer
"""
from pathlib import Path
import pytest
from unittest.mock import patch
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from omegaconf import OmegaConf

from src.trainer import Trainer, TrainingArguments
from tio import Task


@pytest.fixture()
def trainer_base_objects():
    tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/t5-tiny-random')
    model = AutoModelForSeq2SeqLM.from_pretrained('patrickvonplaten/t5-tiny-random')
    task = Task.get_task('dummy', tokenizer, [], [], [])
    train_data = task.get_split('train')
    eval_data = task.get_split('validation')
    yield model, task, tokenizer, train_data, eval_data, torch.device('cpu')


class TestTrainer:

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_get_data_loaders(self, trainer_base_objects, batch_size):
        trainer_args = TrainingArguments(
            max_steps=50,
            max_epochs=5,
            train_batch_size=batch_size,
            eval_batch_size=1 if batch_size == 2 else 2
        )
        model, task, tokenizer, train_data, eval_data, device = trainer_base_objects

        def single_len_label(ex):
            ex['labels'] = [1]
            return ex

        train_data = train_data.map(single_len_label)
        eval_data = eval_data.map(single_len_label)
        trainer = Trainer(
            model,
            trainer_args,
            device,
            tokenizer,
            lambda *args, **kwargs: {'test': 1.0}
        )

        train_loader, eval_loader = trainer._get_data_loaders(trainer_args, train_data, eval_data)

        expected_train_batches, rem = divmod(len(train_data), trainer_args.train_batch_size)
        expected_train_batches += rem > 0
        expected_eval_batches, rem = divmod(len(eval_data), trainer_args.eval_batch_size)
        expected_eval_batches += rem > 0
        assert len([1 for _ in train_loader]) == expected_train_batches
        assert len([1 for _ in eval_loader]) == expected_eval_batches
