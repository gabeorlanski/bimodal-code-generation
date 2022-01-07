import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import sys

if str(Path.cwd().parents[0]) not in sys.path:
    sys.path.insert(0, str(Path.cwd().parents[0]))
from src.config import setup_config_store
import shutil

cs = setup_config_store()
ROOT = Path.cwd()


@hydra.main(config_path="../conf", config_name="train_config")
def debug_hydra(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    shutil.rmtree(ROOT.joinpath("outputs"))


if __name__ == "__main__":
    debug_hydra()
