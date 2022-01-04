from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from omegaconf import DictConfig
import logging

from src.data import DatasetReader

logger = logging.getLogger(__name__)


def train_model(data_path: Path, cfg: DictConfig, reader: DatasetReader):
    train_path = data_path.joinpath(cfg['dataset']['train_path'])
    logger.info(f"Reading training data is from '{train_path}'")
    train_raw, train_data = reader.read_data(train_path)

    validation_path = data_path.joinpath(cfg['dataset']['validation_path'])
    logger.info(f"Reading training data is from '{validation_path}'")
    validation_raw, validation_data = reader.read_data(validation_path)

    logger.info(f"{len(train_data)} training samples")
    logger.info(f"{len(validation_data)} validation samples")
