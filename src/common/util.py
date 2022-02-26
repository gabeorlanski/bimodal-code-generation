import collections
from typing import Dict, List
import numpy as np
import os
from datetime import datetime

__all__ = [
    "flatten",
    "get_stats_from_list",
    "is_currently_distributed",
    "get_world_size",
    "get_estimated_time_remaining"
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


def get_world_size():
    return int(os.getenv('WORLD_SIZE', '-1'))


def get_estimated_time_remaining(start_time, completed, total):
    elapsed = datetime.utcnow() - start_time
    current_rate = elapsed.total_seconds() / completed
    estimated_seconds_left = (total - completed) * current_rate
    hours, rem = divmod(estimated_seconds_left, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)
