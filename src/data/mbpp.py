"""
Code for handling the mostly basic programming problems dataset from
https://arxiv.org/pdf/2108.07732.pdf
"""
import argparse
from pathlib import Path
import logging
import json
import random
from typing import Optional, Union, Callable, List, Tuple, Dict
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from datasets import Dataset

import sys

try:
    from src.common import Registrable, setup_basic_loggers, PROJECT_ROOT
    from src.common.file_util import validate_files_exist
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.common import setup_basic_loggers, PROJECT_ROOT
    from src.common.file_util import validate_files_exist

from .dataset_reader import DatasetReader

logger = logging.getLogger(__name__)


@DatasetReader.register('mbpp')
class MBPP(DatasetReader):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            preprocessors: List[Callable] = None,
            postprocessors: List[Callable] = None,
    ):
        preprocessors = preprocessors or []
        preprocessors = [self.mbpp_preprocess] + preprocessors

        super(MBPP, self).__init__(
            preprocessors=preprocessors,
            tokenizer=tokenizer,
            postprocessors=postprocessors
        )

        self._tokenizer = tokenizer
        self._preprocessors = preprocessors or []
        self.dataset = None
        self.raw = None

    def _load_dataset(self, data_path: Path) -> Dataset:
        # Load the data into a dict where the key is the task_id
        return Dataset.from_json(str(data_path))

    @staticmethod
    def mbpp_preprocess(example):
        example['input_sequence'] = example['text'] + '\n' + '\n'.join(example['test_list'])
        example['target'] = example['code']
        return example


def setup_mbpp_splits(
        data_path: Union[str, Path],
        test_size: int = 500,
        few_shot_size: int = 10,
        fine_tuning_size: int = 374
) -> None:
    """
    Setup the splits for the mostly basic programming problems dataset.

    The default values for the split sizes come from the paper. The size of the
     validation split is equal to total - (test+few shot+fine tuning).

    Args:
        data_path (str): Path to the data dir that has mbpp.jsonl and
            sanitized-mbpp.json in them.
        test_size (int): Size of the test set.
        few_shot_size (int): Size of the set used for few-shot prompting.
        fine_tuning_size (int): Size of the set used for fine tuning.
    """
    out_path = Path(data_path)
    data_path = Path(data_path)

    setup_mbpp_logger = logging.getLogger('setup_mbpp')
    setup_mbpp_logger.info(f"Setting up splits for MBPP files located in "
                           f"'{data_path.resolve()}'")

    setup_mbpp_logger.info("Validating directory")
    try:
        mbpp_path, sanitized_path = validate_files_exist(
            data_path, ['mbpp.jsonl', 'sanitized-mbpp.json']
        )
    except FileExistsError as e:
        setup_mbpp_logger.error(f"Missing '{e.file}' in '{data_path.resolve()}' ")
        raise e

    setup_mbpp_logger.info("Loading data from files")
    setup_mbpp_logger.debug(f"Loading json lines from '{mbpp_path.resolve()}'")
    mbpp_data = []
    for line in tqdm(mbpp_path.read_text('utf-8').splitlines(False), desc='Reading mbpp.jsonl',
                     file=sys.stdout):
        mbpp_data.append(json.loads(line))
    setup_mbpp_logger.debug(f"Loading json from '{sanitized_path.resolve()}'")
    sanitized_data = json.loads(sanitized_path.read_text('utf-8'))
    setup_mbpp_logger.info(f"{len(mbpp_data)} items in MBPP and {len(sanitized_data)} in Sanitized")

    setup_mbpp_logger.info(f"Saving sanitized to '{out_path.joinpath('edited.jsonl')}'")
    with out_path.joinpath('edited.jsonl').open('w', encoding='utf-8') as f:
        for i in sanitized_data:
            f.write(json.dumps(i) + '\n')

    validation_size = len(mbpp_data) - (test_size + few_shot_size + fine_tuning_size)
    splits = [
        ('Test', 'test.jsonl', test_size),
        ('Few-Shot', 'few_shot.jsonl', few_shot_size),
        ('Fine-Tuning', 'train.jsonl', fine_tuning_size),
        ('Validation', 'validation.jsonl', validation_size)
    ]
    progress_bar = tqdm(total=len(mbpp_data), desc='Saving Splits', file=sys.stdout)
    current = 0
    for name, out_file_name, size in splits:

        setup_mbpp_logger.info(f"Saving split {name} with {size} items to {out_file_name}")
        with out_path.joinpath(out_file_name).open('w', encoding='utf-8') as split_file:
            for i in mbpp_data[current:current + size]:
                split_file.write(json.dumps(i) + '\n')
                progress_bar.update()
            current += size

    progress_bar.close()


if __name__ == '__main__':
    setup_basic_loggers("setup_mbpp", PROJECT_ROOT)

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to the MBPP data download dir"
                                          "that contains both MBPP.jsonl and "
                                          "sanititzed-MBPP.jsonl")
    parser.add_argument("--seed", default=1, type=int,
                        help="Seed used for randomization.")
    args = parser.parse_args()
    random.seed(args.seed)
    setup_mbpp_splits(args.data_path)
