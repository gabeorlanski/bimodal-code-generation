import collections
from typing import Dict

__all__ = [
    "flatten"
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
