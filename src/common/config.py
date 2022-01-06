"""
Config related util functions.
"""
import torch


def get_device_from_cfg(cfg) -> torch.device:
    # Cast to a string to guarantee that it will be one type rather than mixed
    # ints and strings.
    device_str = str(cfg.get('device', 'cpu'))
    if device_str == 'cpu' or device_str == "-1":
        return torch.device('cpu')
    else:
        return torch.device(f'cuda{":" + device_str if device_str != "cuda" else ""}')
