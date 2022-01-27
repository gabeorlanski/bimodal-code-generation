import collections
from typing import Dict, List
import numpy as np
import os

__all__ = [
    "flatten",
    "get_stats_from_list",
    "is_currently_distributed"
]

def flatten(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Flatten a dictionary.

    Args:
        d (Dict): The dict to flatten
        parent_key (str): Parent key to use.
        sep (str): The seperator to use.

    Returns:
        The flattened dictionary
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_stats_from_list(values: List[int]):
    return {
        "mean"  : np.mean(values),
        "median": np.median(values),
        "max"   : np.max(values)
    }


def is_currently_distributed() -> bool:
    return int(os.environ.get('WORLD_SIZE', "-1")) > 1

