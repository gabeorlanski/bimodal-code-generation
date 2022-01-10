from typing import List, Dict, Iterable, Union
import os
from os import PathLike
import logging
from .log_handlers import TQDMLoggingHandler, CompactFileHandler
from logging import Filter
import sys

__all__ = ["setup_basic_loggers"]


def setup_basic_loggers(
    name: str, log_path: str = None, verbose: bool = False, debug: bool = False
) -> None:
    """
    Setup the logger
    Args:
        name: Name of the logger
        log_path: Path to directory where the logs will be saved
        verbose: Enable Verbose
        debug: Enable Debug

    Returns:
        The loggers
    """
    # Load in the default paths for log_path
    log_path = (
        os.path.join(log_path, "logs")
        if log_path is not None
        else os.path.join(os.getcwd(), "logs")
    )

    # Validate the path and clear the existing log file
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    normal_file = os.path.join(log_path, f"{name}.log")
    error_file = os.path.join(log_path, f"{name}.issues.log")
    with open(normal_file, "w", encoding="utf-8") as f:
        f.write("")

    # The different message formats to use
    msg_format = logging.Formatter(fmt="[%(levelname)8s] %(message)s")
    verbose_format = logging.Formatter(
        fmt="[%(asctime)s - %(levelname)8s - %(name)12s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_format = logging.Formatter(
        fmt="[%(asctime)s - %(levelname)8s - %(name)12s - %(funcName)12s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create the file handler
    normal_file_handler = CompactFileHandler(normal_file, logging.DEBUG, verbose_format)
    error_file_handler = CompactFileHandler(error_file, logging.WARNING, error_format)

    # Setup the console handlers for normal and errors
    console_handler = TQDMLoggingHandler(
        logging.INFO if not debug else logging.DEBUG,
        fmt=msg_format if not verbose else verbose_format,
    )

    # Set the environment variable to the names of the logger for use in other parts of the
    # program
    # Create and register the two loggers
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.addHandler(normal_file_handler)

    logger.addHandler(error_file_handler)
    logger.setLevel(logging.NOTSET)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)


