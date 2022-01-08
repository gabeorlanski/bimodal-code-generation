import pytest
import yaml
from src.common import FIXTURES_ROOT
from .dummy_objects import *

@pytest.fixture()
def tiny_model_name():
    yield "patrickvonplaten/t5-tiny-random"


@pytest.fixture()
def simple_config():
    yield yaml.load(
        FIXTURES_ROOT.joinpath('configs', 'simple.yaml').open('r', encoding='utf-8'),
        yaml.Loader
    )
