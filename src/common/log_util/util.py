import os
import logging
from pathlib import Path
from typing import Union

from .log_handlers import TQDMLoggingHandler, CompactFileHandler

__all__ = ["setup_global_logging"]


def setup_global_logging(
        name: str,
        log_path: Union[str, os.PathLike] = None,
        verbose: bool = False,
        debug: bool = False,
        rank: int = 0,
        world_size: int = 1
) -> None:
    """
    Setup the logger
    Args:
        name: Name of the logger
        log_path: Path to directory where the logs will be saved
        verbose: Enable Verbose
        debug: Enable Debug
        rank (int): The rank of this process
        world_size (int): The size of the world.

    Returns: None
    """
    # Load in the default paths for log_path
    log_path = (
        Path(log_path)
        if log_path is not None
        else Path("logs")
    )

    # Validate the path and clear the existing log file
    if not Path(log_path).exists():
        Path(log_path).mkdir(parents=True)

    if world_size <= 1:
        rank_str = ''
        normal_file = log_path.joinpath(f"{name}.log")
        error_file = log_path.joinpath("{name}.issues.log")
    else:
        rank_str = f" RANK {rank}:"
        normal_file = log_path.joinpath(f"{name}_worker_{rank}.log")
        error_file = log_path.joinpath(f"{name}_worker_{rank}.issues.log")

    # Clear the log files
    with open(normal_file, "w", encoding="utf-8") as f:
        f.write("")
    with open(error_file, "w", encoding="utf-8") as f:
        f.write("")

    # The different message formats to use
    msg_format = logging.Formatter(
        fmt=f"[%(levelname)8s]{rank_str} %(message)s"
    )
    verbose_format = logging.Formatter(
        fmt=f"[%(asctime)s - %(levelname)8s - %(name)12s]{rank_str} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_format = logging.Formatter(
        fmt=f"[%(asctime)s - %(levelname)8s - %(name)12s - %(funcName)12s]{rank_str} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create the file handler
    normal_file_handler = CompactFileHandler(str(normal_file.resolve().absolute()), logging.DEBUG,
                                             verbose_format)
    error_file_handler = CompactFileHandler(str(error_file.resolve().absolute()), logging.WARNING,
                                            error_format)

    # Setup the console handlers for normal and errors
    console_handler = TQDMLoggingHandler(
        logging.INFO if not debug else logging.DEBUG,
        fmt=msg_format if not verbose else verbose_format,
    )

    # Set the environment variable to the names of the logger for use in other parts of the
    # program
    # Create and register the two loggers
    root_logger = logging.getLogger()

    root_logger.addHandler(error_file_handler)
    root_logger.addHandler(normal_file_handler)
    if rank <= 0:
        root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.NOTSET)
