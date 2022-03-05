"""
Tests for the Tensorize
"""
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer
from src.data import tensorize


def processor(samples, tokenizer):
    tokked = tokenizer([d['text'] for d in samples])
    out = []
    for input_ids, attn in zip(tokked['input_ids'], tokked['attention_mask']):
        out.append({'input_ids': input_ids, 'attention_mask': attn, 'labels': input_ids})
    return out


def test_tensorize(tmpdir):
    tmpdir_path = Path(tmpdir)
    raw_data = tmpdir_path.joinpath('data.jsonl')
    expected = ["Hello", "World", "My", "Name", "Jeff"]
    with raw_data.open('w') as f:
        f.write('\n'.join(map(
            lambda t: json.dumps({'text': t}),
            expected
        )))
    tensorize.tensorize(
        raw_data,
        tmpdir_path,
        'result',
        1,
        model_name='gpt2',
        data_processor=processor,
        batch_size=2,
        debug_max_samples=-1
    )

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    results = list(map(json.loads, tmpdir_path.joinpath('result.jsonl').open()))
    actual = tokenizer.batch_decode([d['input_ids'] for d in results])
    assert set(actual) == set(expected)
