import pytest
import yaml
from src.common import FIXTURES_ROOT
from .dummy_objects import *
from transformers import AutoTokenizer


@pytest.fixture()
def tiny_model_name():
    yield "patrickvonplaten/t5-tiny-random"


@pytest.fixture()
def simple_train_config():
    yield yaml.load(
        FIXTURES_ROOT.joinpath('configs', 'simple_train.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )


@pytest.fixture()
def simple_eval_config():
    yield yaml.load(
        FIXTURES_ROOT.joinpath('configs', 'simple_eval.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )


@pytest.fixture()
def tokenizer():
    yield AutoTokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")


@pytest.fixture()
def code_preds_dir():
    yield FIXTURES_ROOT.joinpath('code_predictions')

