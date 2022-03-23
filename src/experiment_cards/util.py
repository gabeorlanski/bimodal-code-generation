import collections
from copy import deepcopy
from typing import Dict, List, Optional, Tuple


def merge_dictionaries(
        left: Dict,
        right: Dict,
        path: Optional[List[str]] = None,
        no_conflicting_leaves: bool = False
) -> Dict:
    """
    Merges the two dictionaries into each other. Will give priority to the 
    values in ``right`` (i.e. if a key is in both, the value from ``right`` 
    will overwrite the one in left). 
    
    For now, this will NOT handle lists or other non-dict iterables.
    
    Args:
        left (dict): The left dict.
        right (dict): The right dict.
        path (List[str]): The path to where we currently are.
        no_conflicting_leaves (bool): Raise an error if the two dictionaries
            share leaves.

    Returns:
        Dict: The merged dictionaries.
    """
    if path is None:
        path = []
    out = deepcopy(left)
    for key in right:
        if key in out:
            if isinstance(out[key], dict) and isinstance(out[key], dict):
                out[key] = merge_dictionaries(out[key], right[key], path + [str(key)],
                                              no_conflicting_leaves=no_conflicting_leaves)
            elif out[key] == right[key]:
                pass  # same leaf value
            else:
                if no_conflicting_leaves:
                    raise KeyError(
                        f"{'->'.join(path or [])}[{key}] is found "
                        f"in both dictionaries."
                    )
                out[key] = right[key]
        else:
            out[key] = right[key]
    return out


def set_config_at_level(cfg, path, value):
    if not path:
        return value

    current_key, *path = path
    if current_key not in cfg:
        cfg[current_key] = {}
    cfg[current_key] = set_config_at_level(cfg[current_key], path, value)
    return cfg
