import os
import pkgutil
import sys
from pathlib import Path
import logging
from typing import List, Union, Optional, Generator, T, Set
import importlib
from contextlib import contextmanager

ContextManagerFunctionReturnType = Generator[T, None, None]
PathType = Union[os.PathLike, str]

logger = logging.getLogger(__name__)
__all__ = [
    "validate_files_exist"
]


def validate_files_exist(
        target_dir: Union[Path, str],
        required_files: List[str]
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


@contextmanager
def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
     This is taken from https://github.com/allenai/allennlp/blob/8db45e87098df6f92720718eaff9bc5f605c1faf/allennlp/common/util.py#L314
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def import_module_and_submodules(package_name: str, exclude: Optional[Set[str]] = None) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.

    This is taken from https://github.com/allenai/allennlp/blob/8db45e87098df6f92720718eaff9bc5f605c1faf/allennlp/common/util.py#L332
    """
    if exclude and package_name in exclude:
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage, exclude=exclude)
