from .mbpp import MBPP, MBPPConfig
from .dataset_reader import DatasetReader, DatasetReaderConfig
from .preprocessors import *
from .processor import Preprocessor, Postprocessor

READER_CONFIGS = {
    "mbpp": MBPPConfig
}
