import pytest
import torch
from omegaconf import OmegaConf
from copy import deepcopy

from src import config


@pytest.mark.parametrize('old_cfg', [
    {},
    {"P": "Q"},
    {"A": "C", "C": "B", "E": {"F": {"Z": "H"}, "P": "Q"}, "I": "J"}
], ids=['empty', 'no-conflicts', 'all-conflicts'])
@pytest.mark.parametrize("excluded_keys", [
    [],
    ["Z", "P"]
])
def test_merge_configs(old_cfg, excluded_keys):
    cfg = {
        "A": "B",
        "C": "D",
        "E": {"F": {"G": "H"}},
        "I": ["J"]
    }

    expected = deepcopy(cfg)

    if "P" in old_cfg and "P" not in excluded_keys:
        expected['P'] = "Q"
    elif "E" in old_cfg:
        if "Z" not in excluded_keys:
            expected["E"]["F"]["Z"] = "H"
        if "P" not in excluded_keys:
            expected["E"]["P"] = "Q"

    result = OmegaConf.to_object(config.merge_configs(
        OmegaConf.create(cfg),
        OmegaConf.create(old_cfg),
        exclude_keys=excluded_keys
    ))
    assert result == expected
