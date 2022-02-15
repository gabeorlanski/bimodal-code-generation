"""
Config related util functions.
"""
from typing import List, Dict
from copy import deepcopy
import logging
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "merge_configs",
    "create_overrides_list"
]


def merge_configs(
        new_cfg: DictConfig,
        old_cfg: DictConfig,
        exclude_keys: List[str] = None
) -> DictConfig:
    """
    Merge two configs without causing any changes to mutable fields. In the
    case that there are conflicting keys between the new and old configs,
    the new values are given priority.

    Args:
        new_cfg (DictConfig): The new config. Priority is given to keys in
            this config.
        old_cfg (DictConfig): The old config.
        exclude_keys (List[str]): List of keys to not merge.

    Returns:
        DictConfig: The merged configs.
    """
    exclude_keys = exclude_keys or []
    merged_cfg = deepcopy(OmegaConf.to_object(new_cfg))

    def _merge(a: Dict, b: Dict) -> Dict:
        out = deepcopy(a)
        for k, v in b.items():
            if k in exclude_keys:
                continue

            if k not in out:
                out[k] = v

            elif isinstance(out[k], dict) and isinstance(v, Dict):
                # In the case of nested dictionary configs, recurse
                out[k] = _merge(deepcopy(out[k]), v)

        return out

    return OmegaConf.create(_merge(merged_cfg, OmegaConf.to_object(old_cfg)))


def create_overrides_list(
        overrides_to_add_if_not_none: Dict,
        base_overrides_list: List[str],
        override_str: str = ""
) -> List[str]:
    cfg_overrides = []

    # Got through the dict of overrides to add if they are not None and add
    # them. The most useful place for this would be for CLI arguments where you
    # would not want to add overrides if they are None.
    for override_name, override_value in overrides_to_add_if_not_none.items():

        if override_value:
            cfg_overrides.append(f"{override_name}={override_value}")

    cfg_overrides.extend(base_overrides_list)
    if override_str:
        cfg_overrides.extend(override_str.split(' '))
    return cfg_overrides
