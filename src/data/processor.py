import logging
from typing import Dict, Callable, List, Tuple
from functools import partial

from src.common import Registrable

logger = logging.getLogger(__name__)


def load_processors(cfg) -> Tuple[List[Callable], List[Callable]]:
    logger.debug("Loading processors")
    preprocessors = []
    postprocessors = []

    if cfg.get("preprocessors") is not None:
        logger.debug("preprocessors found")
        preprocessors = [
            partial(Preprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["preprocessors"].items()
        ]
    logger.info(f"{len(preprocessors)} preprocessors found")

    if cfg.get("postprocessors") is not None:
        logger.debug("postprocessors found")
        postprocessors = [
            partial(Postprocessor.by_name(name), **func_kwargs)
            for name, func_kwargs in cfg["postprocessors"].items()
        ]
    logger.info(f"{len(postprocessors)} postprocessors found")
    return preprocessors, postprocessors


class Preprocessor(Registrable):
    """
    Just a wrapper for the registrable so that preprocessing
    functions can be registered.
    """

    pass


class Postprocessor(Registrable):
    """
    Just a wrapper for the registrable so that postprocessing functions can be
    registered.
    """

    pass


@Preprocessor.register("add_prefix")
def add_prefix(example: Dict, prefix: str):
    example["input_sequence"] = f"{prefix} {example['input_sequence']}"
    return example
