import pytest
import yaml
from src.common import FIXTURES_ROOT


@pytest.fixture()
def experiments_dir():
    yield FIXTURES_ROOT.joinpath('experiments')


@pytest.fixture()
def experiment_cards_path(experiments_dir):
    yield experiments_dir.joinpath('experiment_card.yaml')


@pytest.fixture()
def experiment_result_files_path(experiments_dir):
    yield experiments_dir.joinpath('results')
