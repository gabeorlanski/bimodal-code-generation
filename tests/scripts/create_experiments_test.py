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

    # Setup the expected files to find, and what keys will be different from
    # the base.
    expected_files = {
        "SO.ExceptionQuestions.CodeParrot.yaml"     : {
            "model": "lvwerra/codeparrot",
            "name" : "ExceptionQuestions.CodeParrot",
        },
        "SO.ExceptionQuestions.CodeParrotSmall.yaml": {
            "model": "lvwerra/codeparrot-small",
            "name" : "ExceptionQuestions.CodeParrotSmall"
        },
    }

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

    for file, changed_keys in expected_files.items():
        # Load each time so that it can be modified.
        expected_config = yaml.load(
            experiment_result_files_path.joinpath('simple.yaml').open('r'),
            yaml.Loader
        )
        expected_config.update(changed_keys)

        actual_file_path = tmpdir_path.joinpath(file)
        assert actual_file_path.exists(), file

        actual_config = yaml.load(
            actual_file_path.open('r'),
            yaml.Loader
        )
        assert actual_config == expected_config, file
