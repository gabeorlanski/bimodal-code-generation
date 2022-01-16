"""
Tests for the trainer
"""
import math
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from omegaconf import OmegaConf

from src.trainer import Trainer, TrainingArguments
from src.common.test_util import assert_mocked_correct
from tio import Task


@pytest.fixture()
def trainer_base_objects():
    tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/t5-tiny-random')
    model = AutoModelForSeq2SeqLM.from_pretrained('patrickvonplaten/t5-tiny-random')
    task = Task.get_task('dummy', tokenizer, [], [], [])
    train_data = task.get_split('train', set_format='torch')
    eval_data = task.get_split('validation', set_format='torch')
    yield model, task, tokenizer, train_data, eval_data, torch.device('cpu')


class TestTrainer:

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_get_data_loaders(self, tmpdir, trainer_base_objects, batch_size, simple_train_config):
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
            OmegaConf.create(simple_train_config),
            model,
            AutoModelForSeq2SeqLM,
            device,
            tokenizer,
            lambda *args, **kwargs: {'test': 1.0},
            path_to_use=tmpdir
        )

        train_loader, eval_loader = trainer._get_data_loaders(trainer_args, train_data, eval_data)

        expected_train_batches, rem = divmod(len(train_data), trainer_args.train_batch_size)
        expected_train_batches += rem > 0
        expected_eval_batches, rem = divmod(len(eval_data), trainer_args.eval_batch_size)
        expected_eval_batches += rem > 0
        assert len([1 for _ in train_loader]) == expected_train_batches
        assert len([1 for _ in eval_loader]) == expected_eval_batches

    def test_call(self, tmpdir, trainer_base_objects, simple_train_config):
        trainer_args = TrainingArguments(
            max_steps=50,
            max_epochs=5,
            train_batch_size=1,
            eval_batch_size=1
        )
        model, task, tokenizer, train_data, eval_data, device = trainer_base_objects
        trainer = Trainer(
            OmegaConf.create(simple_train_config),
            model,
            AutoModelForSeq2SeqLM,
            device,
            tokenizer,
            lambda *args, **kwargs: {'test': 1.0},
            path_to_use=tmpdir
        )
        trainer.args = trainer_args

        def mock_train_epoch(data_loader, epoch):
            trainer.global_step += 20
            return {"loss": 1.0}

        trainer.data_loading_fn = MagicMock(return_value=["TRAIN_LOADER", "EVAL_LOADER"])
        trainer._train_epoch = MagicMock(return_value={"loss": 1.0, "updates": 2.0},
                                         side_effect=mock_train_epoch)
        trainer.evaluate_fn = MagicMock(return_value={"em": 1.0})
        trainer.save_model = MagicMock()
        trainer._load_best = MagicMock()

        trainer(["A"], ["B"])

        assert_mocked_correct(trainer.data_loading_fn, [
            {
                'args'  : (),
                'kwargs': {'args': trainer.args, 'train_dataset': ["A"], "eval_dataset": ["B"]}
            }
        ])
        assert_mocked_correct(trainer._train_epoch, [
            {"args": (), "kwargs": {'data_loader': "TRAIN_LOADER", "epoch": i + 1}} for i in
            range(3)
        ])
        assert trainer.evaluate_fn.call_count == 3
        assert trainer.save_model.call_count == 3
        assert trainer._load_best.call_count == 3

    @pytest.mark.parametrize('higher_better', [True, False])
    def test_save_model(self, tmpdir, trainer_base_objects, simple_train_config, higher_better):
        model_path = Path(tmpdir).joinpath('checkpoints')
        model, task, tokenizer, train_data, eval_data, device = trainer_base_objects
        trainer = Trainer(
            OmegaConf.create(simple_train_config),
            model,
            AutoModelForSeq2SeqLM,
            device,
            tokenizer,
            lambda *args, **kwargs: {'test': 1.0},
            path_to_use=tmpdir
        )
        trainer.args.checkpoints_to_save = 2
        trainer.args.more_is_better = higher_better
        trainer.global_step = 1
        trainer.save_model({"eval_loss": 1.0})
        assert model_path.joinpath('model_1.bin').exists()
        assert trainer.path_to_best_model == model_path.joinpath('model_1.bin')
        assert trainer.best_metric == 1.0

        trainer.global_step += 1
        trainer.save_model({"eval_loss": 2.0})
        assert model_path.joinpath('model_2.bin').exists()
        assert model_path.joinpath('model_1.bin').exists()
        if higher_better:
            assert trainer.path_to_best_model == model_path.joinpath('model_2.bin')
            assert trainer.best_metric == 2.0
        else:
            assert trainer.path_to_best_model == model_path.joinpath('model_1.bin')
            assert trainer.best_metric == 1.0

        trainer.global_step += 1
        trainer.save_model({"eval_loss": 0.5})
        assert model_path.joinpath('model_3.bin').exists()
        if higher_better:
            assert model_path.joinpath('model_3.bin').exists()
            assert trainer.path_to_best_model == model_path.joinpath('model_2.bin')
            assert trainer.best_metric == 2.0
        else:

            assert model_path.joinpath('model_1.bin').exists()
            assert trainer.path_to_best_model == model_path.joinpath('model_3.bin')
            assert trainer.best_metric == 0.5
