from pathlib import Path
from typing import List, Callable, Optional, Dict
from unittest.mock import MagicMock


def assert_list_of_files_exist(target_dir: Path, file_list: List[str]) -> None:
    """
    Assert that a list of files were created in a target directory.

    Args:
        target_dir (Path): The directory to check.
        file_list (List[str]): The list of files to check.
    """
    for file in file_list:
        file_path = target_dir.joinpath(file)
        if not file_path.exists():
            raise AssertionError(f"'{file}' does not exist in {target_dir}")


def assert_mocked_correct(mocked_function: MagicMock, expected_calls: List[Dict]):
    assert mocked_function.call_count == len(expected_calls)
    for i, expected_call_args in enumerate(expected_calls):
        expected_args = expected_call_args['args']
        expected_kwargs = expected_call_args['kwargs']
        actual_call_args = mocked_function.call_args_list[i]
        if actual_call_args.args != expected_args:
            assert actual_call_args.args == expected_args, f"Call {i}"
        if actual_call_args.kwargs != expected_kwargs:
            assert actual_call_args.kwargs == expected_kwargs, f"Call {i}"
