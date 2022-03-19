"""
Tests for the Tensorized Functions
"""
import json

import pytest
from transformers import AutoTokenizer
from src.data import tensorize, stackoverflow

from src.common import FIXTURES_ROOT


def test_get_dataset_info_with_processor():
    sample_dump = FIXTURES_ROOT.joinpath('so_dumps', 'python_dump.jsonl')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    processor = stackoverflow.StackOverflowProcessor(
        repeat_prompt_each_answer='full'
    )

    result = tensorize.get_dataset_info_with_processor(
        raw_data_path=sample_dump,
        output_name='testing',
        num_workers=1,
        model_name='gpt2',
        processor=processor,
        batch_size=1,
        debug_max_samples=-1
    )

    expected_cfg = tensorize.TensorizedDatasetInfo(
        'testing'
    )

    processed_gen = map(processor.__call__,
                        map(json.loads, sample_dump.open('r')))

    for processed_list in processed_gen:
        for processed in processed_list:
            input_tokens = len(tokenizer.tokenize(processed['input']))
            target_tokens = len(tokenizer.tokenize(processed['labels']))
            expected_cfg.add_instance(
                {"labels": target_tokens, "inputs": input_tokens}
            )

    assert result.name == 'testing'
    assert result.total_tokens == expected_cfg.total_tokens
