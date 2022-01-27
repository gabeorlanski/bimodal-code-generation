from pathlib import Path
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

__all__ = ["validate_files_exist", "ENV_VARS_TRUE_VALUES"]


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
def validate_files_exist(
    target_dir: Union[Path, str], required_files: List[str]
) -> List[Path]:
    """
    Check if a list of files exist in a given directory
    Args:
        target_dir (Union[Path, str]): Directory to check.
        required_files (List[str]): List of filenames to check.
    Returns:
        List[Path]: List of paths to the required files.
    Raises:
         FileExistsError: If one of the required files is not found. An
         attribute ``file`` will be set in the exception with the name of the
         file that failed.
    """
    out = []
    for file in required_files:
        logger.info(f"Checking that '{file}' is in '{target_dir.resolve()}'")
        file_path = target_dir.joinpath(file)
        if not file_path.exists():
            exception = FileExistsError(f"Could not find '{file}'")
            exception.file = file
            raise exception

        out.append(file_path)
    return out
