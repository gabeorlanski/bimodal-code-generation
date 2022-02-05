"""
Tests for the create experiments script
"""
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch

import yaml

from src.common import PROJECT_ROOT
from scripts.create_experiments import create_experiments


@pytest.mark.parametrize('overwrite', [True, False], ids=['overwrite', 'dont_overwrite'])
def test_create_experiments(
        tmpdir,
        experiments_dir,
        experiment_cards_path,
        experiment_result_files_path,
        overwrite
):
    tmpdir_path = Path(tmpdir)
    # Create an empty file for testing if overwrite clears the directory
    with tmpdir_path.joinpath('test.txt').open('w'):
        pass

    create_experiments(
        experiment_card_path=experiment_cards_path,
        config_directory=experiments_dir.joinpath('config_dir'),
        debug=True,
        output_path=tmpdir_path,
        overwrite_output_dir=overwrite
    )

    if overwrite:
        assert not tmpdir_path.joinpath('test.txt').exists()
    else:
        assert tmpdir_path.joinpath('test.txt').exists()

    for file_name in experiment_result_files_path.glob('*.yaml'):
        # Load each time so that it can be modified.
        expected_config = yaml.load(
            file_name.open('r'),
            yaml.Loader
        )
        actual_file_path = tmpdir_path.joinpath(f"{file_name.stem}.yaml")
        assert actual_file_path.exists(), file_name.stem

        actual_config = yaml.load(
            actual_file_path.open('r'),
            yaml.Loader
        )
        assert actual_config == expected_config, file_name.stem

    assert tmpdir_path.joinpath('experiments.sh').exists()
    actual_command = tmpdir_path.joinpath('experiments.sh').read_text()
    expected_command = experiment_result_files_path.joinpath('experiments.sh').read_text()
    assert actual_command == expected_command
