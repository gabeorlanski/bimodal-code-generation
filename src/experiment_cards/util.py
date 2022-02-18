from copy import deepcopy
from typing import Dict


def merge_dictionaries(left: Dict, right: Dict, path=None) -> Dict:
    """
    Merges the two dictionaries into each other. Will give priority to the 
    values in ``right`` (i.e. if a key is in both, the value from ``right`` 
    will overwrite the one in left). 
    
    For now, this will NOT handle lists or other non-dict iterables.
    
    Args:
        left (dict): The left dict.
        right (dict): The right dict.
        path (List[str]): The path to where we currently are. 

    Returns:
        Dict: The merged dictionaries.
    """
    if path is None:
        path = []

    for key in right:
        if key in left:
            if isinstance(left[key], dict) and isinstance(left[key], dict):
                merge_dictionaries(left[key], right[key], path + [str(key)])
            elif left[key] == right[key]:
                pass  # same leaf value
            else:
                left[key] = right[key]
        else:
            left[key] = right[key]
    return left
