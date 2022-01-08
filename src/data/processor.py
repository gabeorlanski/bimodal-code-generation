import logging
from typing import Dict

from src.common import Registrable

logger = logging.getLogger(__name__)


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
