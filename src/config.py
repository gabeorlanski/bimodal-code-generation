"""
Classes for Hydra's Structured Config.
"""
from typing import Dict, Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field

from src.data import READER_CONFIGS, DatasetReaderConfig


@dataclass
class BaseConfig:
    name: str = MISSING
    data_path: str = MISSING
    debug: bool = False
    disable_tracking: bool = False
    dataset: DatasetReaderConfig = MISSING
    preprocessors: Any = field(default_factory=dict)


def setup_config_store() -> ConfigStore:
    cs = ConfigStore()
    # cs.store(name="base_config", node=BaseConfig)
    # cs.store(name="base_dataset", node=DatasetReaderConfig)

    for name, cfg in READER_CONFIGS.items():
        cs.store(group='dataset', name=name, node=cfg)
    return cs
