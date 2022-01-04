"""
Tests for the MBPP dataset features
"""
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch
from transformers import AutoTokenizer
from datasets import Dataset

from src.common import FIXTURES_ROOT
from src.data import mbpp


def setup_tmpdir(tmpdir_path, copy_mbpp=True, copy_sanitized=True) -> Path:
    """
    Helper function for copying data.
    Args:
        tmpdir_path (str): path to the temp directory
        copy_mbpp (bool): copy over the MBPP.jsonl
        copy_sanitized (bool): copy over the sanitized-MBPP.json
    Returns:
        tmpdir_path (Path): path to the tmpdir.

    """
    # Copy over the fixtures to the tmp dir.
    tmpdir_path = Path(tmpdir_path)
    if copy_mbpp:
        shutil.copy(
            FIXTURES_ROOT.joinpath('MBPP', 'mbpp.jsonl'),
            tmpdir_path.joinpath('mbpp.jsonl')
        )

    if copy_sanitized:
        shutil.copy(
            FIXTURES_ROOT.joinpath('MBPP', 'sanitized-mbpp.json'),
            tmpdir_path.joinpath('sanitized-mbpp.json')
        )

    return tmpdir_path


def test_setup_mbpp_splits(tmpdir):
    tmpdir_path = setup_tmpdir(tmpdir)
    test_split_size = 4
    few_shot_size = 2
    fine_tuning_size = 5
    total = 12
    expected_mbpp = list(map(
        json.loads,
        FIXTURES_ROOT.joinpath('MBPP', 'mbpp.jsonl').read_text('utf-8').splitlines(True))
    )
    expected_edited = json.loads(
        FIXTURES_ROOT.joinpath('MBPP', 'sanitized-mbpp.json').read_text('utf-8')
    )

    expected_files = {
        "test.jsonl"      : expected_mbpp[:4],
        "few_shot.jsonl"  : expected_mbpp[4:6],
        "train.jsonl"     : expected_mbpp[6:11],
        "edited.jsonl"    : expected_edited,
        "validation.jsonl": expected_mbpp[11:],
    }

    with patch("src.data.mbpp.random.shuffle", new_callable=lambda: None):
        mbpp.setup_mbpp_splits(
            tmpdir_path,
            test_size=test_split_size,
            few_shot_size=few_shot_size,
            fine_tuning_size=fine_tuning_size
        )

    for f in expected_files:
        actual_path = tmpdir_path.joinpath(f)
        assert actual_path.exists(), f"'{f}' does not exist"
        actual_data = list(map(json.loads, actual_path.read_text('utf-8').splitlines(True)))
        assert len(actual_data) == len(expected_files[f])
        assert actual_data == expected_files[f]


@pytest.mark.parametrize("copy_mbpp", [True, False], ids=['W/ MBPP', 'No MBPP'])
@pytest.mark.parametrize("copy_sanitized", [True, False],
                         ids=['W/ Sanitized', 'No Sanitized'])
def test_setup_mbpp_splits_errors(tmpdir, copy_mbpp, copy_sanitized):
    tmpdir_path = setup_tmpdir(tmpdir, copy_mbpp, copy_sanitized)
    if copy_mbpp and copy_sanitized:
        mbpp.setup_mbpp_splits(tmpdir_path, 1)
        return

    with pytest.raises(FileExistsError) as exception_info:
        mbpp.setup_mbpp_splits(tmpdir_path, 1)

    if not copy_mbpp:
        expected = "Could not find 'MBPP.jsonl'"
    else:
        expected = "Could not find 'sanitized-MBPP.json'"
    assert str(exception_info.value) == expected


class TestMBPPTask:

    def test_read(self):
        mbpp_task = mbpp.MBPP(
            None  # Type:ignore
        )

        expected_mbpp = Dataset.from_json(str(FIXTURES_ROOT.joinpath('MBPP', 'mbpp.jsonl')))

        actual = mbpp_task._load_dataset(
            FIXTURES_ROOT.joinpath('MBPP', 'mbpp.jsonl')
        )
        assert actual.to_dict() == expected_mbpp.to_dict()

    def test_preprocess(self):
        tokenizer = AutoTokenizer.from_pretrained('patrickvonplaten/t5-tiny-random')
        mbpp_task = mbpp.MBPP(
            tokenizer
        )

        preprocessed, tokenized = mbpp_task.read_data(
            FIXTURES_ROOT.joinpath('MBPP', 'mbpp.jsonl')
        )
        for example, example_tok in zip(preprocessed, tokenized):
            expected_input_sequence = example['text'] + '\n' + '\n'.join(example['test_list'])
            expected_target = example['code']
            assert example['input_sequence'] == expected_input_sequence
            assert example['target'] == expected_target
            assert example_tok['input_ids'] == tokenizer(expected_input_sequence)['input_ids']
            assert example_tok['labels'] == tokenizer(expected_target)['input_ids']
