import pytest
import torch
from omegaconf import OmegaConf

from src import config


@pytest.mark.parametrize('device', [-1, 'cuda'], ids=['cpu', 'gpu'])
def test_get_device_from_cfg(device):
    cfg = {
        "device": device
    }
    assert config.get_device_from_cfg(OmegaConf.create(cfg)) == torch.device(
        'cuda' if device == 'cuda' else 'cpu')
