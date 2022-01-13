"""
Config related util functions.
"""
from typing import List, Dict
from copy import deepcopy
import logging
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "merge_configs"
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
