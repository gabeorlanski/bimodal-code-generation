from pathlib import Path
from typing import List, Callable, Optional, Dict


def assert_list_of_files_exist(
        target_dir: Path,
        file_list: List[str]
) -> None:
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
