from typing import List, Dict, Iterable, Union
import os
from os import PathLike
import logging
from .log_handlers import TQDMLoggingHandler, CompactFileHandler
from logging import Filter
import sys

__all__ = [
    "setup_basic_loggers",
    "prepare_global_logging"
]


def setup_basic_loggers(name: str,
                        log_path: str = None,
                        verbose: bool = False,
                        debug: bool = False) -> None:
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
    log_path = os.path.join(log_path, 'logs') if log_path is not None \
        else os.path.join(os.getcwd(), 'logs')

    # Validate the path and clear the existing log file
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    normal_file = os.path.join(log_path, f'{name}.log')
    error_file = os.path.join(log_path, f'{name}.issues.log')
    with open(normal_file, 'w', encoding='utf-8') as f:
        f.write('')

    # The different message formats to use
    msg_format = logging.Formatter(fmt='[%(levelname)8s] %(message)s')
    verbose_format = logging.Formatter(
        fmt='[%(asctime)s - %(levelname)8s - %(name)12s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    error_format = logging.Formatter(
        fmt=
        '[%(asctime)s - %(levelname)8s - %(name)12s - %(funcName)12s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Create the file handler
    normal_file_handler = CompactFileHandler(normal_file, logging.DEBUG,
                                             verbose_format)
    error_file_handler = CompactFileHandler(error_file, logging.WARNING,
                                            error_format)

    # Setup the console handlers for normal and errors
    console_handler = TQDMLoggingHandler(
        logging.INFO if not debug else logging.DEBUG,
        fmt=msg_format if not verbose else verbose_format)

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


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """
    def filter(self, record):
        return record.levelno < logging.ERROR


FILE_FRIENDLY_LOGGING: bool = False


def prepare_global_logging(serialization_dir: Union[str, PathLike],
                           rank: int = 0,
                           world_size: int = 1,
                           log_name: str = "out") -> None:
    root_logger = logging.getLogger()

    normal_format = '[%(levelname)8s] %(message)s'
    error_format = '[%(asctime)s - %(levelname)8s - %(name)s] %(message)s'

    verbose_format = '[%(asctime)s - %(levelname)8s - %(name)s] %(message)s'

    # create handlers
    if world_size == 1:
        log_file = os.path.join(serialization_dir, f"{log_name}.log")
    else:
        log_file = os.path.join(serialization_dir,
                                f"{log_name}_worker{rank}.log")
        normal_format = f"{rank} | {normal_format}"
        verbose_format = f"{rank} | {verbose_format}"
        error_format = f"{rank} | {error_format}"

    normal_format = logging.Formatter(fmt=normal_format)
    error_format = logging.Formatter(fmt=error_format,
                                     datefmt='%Y-%m-%d %H:%M:%S')
    verbose_format = logging.Formatter(fmt=verbose_format,
                                       datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setFormatter(verbose_format)
    stderr_handler.setFormatter(error_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("ALLENNLP_DEBUG"):
        stdout_handler.setFormatter(verbose_format)
        LEVEL = logging.DEBUG
    else:
        stdout_handler.setFormatter(normal_format)
        level_name = os.environ.get("ALLENNLP_LOG_LEVEL", "INFO")
        LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    file_handler.setLevel(LEVEL)
    stdout_handler.setLevel(LEVEL)
    stdout_handler.addFilter(
        ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(LEVEL)

    # put all the handlers on the root logger
    root_logger.addHandler(file_handler)
    if rank == 0:
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

    # write uncaught exceptions to the logs
    def excepthook(exctype, value, traceback):
        # For a KeyboardInterrupt, call the original exception handler.
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception",
                             exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook

    # also log tqdm
    from allennlp.common.tqdm import logger as tqdm_logger

    tqdm_logger.handlers.clear()
    tqdm_logger.addHandler(file_handler)
