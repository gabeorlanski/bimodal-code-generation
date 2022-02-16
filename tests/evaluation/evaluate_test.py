"""
Tests for the eval
"""
import json
import random
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf, open_dict
from src.evaluation.evaluation import evaluate, generate_code_predictions
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_evaluate(tmpdir, simple_eval_config):
    # simple_eval_config['training']['batch_size'] = batch_size
    simple_eval_config['splits'] = ["train"]
    cfg = OmegaConf.create(simple_eval_config)
    model = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
    return_val = {
        "predictions": [
            ["A", "B", "C"],
            ["D", "E", "F"]
        ],
        "labels"     : ["A", "C"],
        "indices"    : [0, 1]
    }

    with patch('src.evaluation.evaluation.generate_code_predictions',
               return_value=return_val) as mock_gen:
        evaluate(
            cfg,
            model,
            Path(tmpdir),
            False
        )
        assert mock_gen.call_count == 1

    tmpdir_path = Path(tmpdir)
    assert tmpdir_path.joinpath('eval_config.yaml').exists()
    assert tmpdir_path.joinpath('eval_metrics.json').exists()
    assert tmpdir_path.joinpath('predictions', 'train.jsonl').exists()
    results = list(map(json.loads,
                       tmpdir_path.joinpath('predictions', 'train.jsonl').read_text().splitlines()))
    assert len(results) == 2
    assert [d['prediction'] for d in results] == return_val['predictions']


@pytest.mark.parametrize("remove_input_ids", [True, False])
def test_generate_code_predictions(remove_input_ids):
    expected = [
        ' Television mutual lined',
        ' Wheels Television Television',
        ' bravery lined lined',
        ' Boone Wheels Television',
        ' Television boils Wheels',
        ' lined boils Television'
    ]
    prompt = 'Why make trillions, when we '
    gen_kwargs = {
        'max_new_tokens': 10 - 7,
        'do_sample'     : True,
        'temperature'   : 0.2,
        'top_p'         : 0.95,
        'top_k'         : 10
    }

    data = [
        {'input_sequence': prompt, 'target': 'Hello', 'idx': 0}
    ]
    device = torch.device('cpu')
    model = AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2')
    tokenizer = AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    results = generate_code_predictions(
        model.to(device),
        dataset=data,
        tokenizer=tokenizer,
        batch_size=2,
        device=device,
        generation_kwargs=gen_kwargs,
        seq_per_sample=6,
        remove_input_ids_from_output=remove_input_ids,
    )

    if not remove_input_ids:
        expected = list(map(lambda l: prompt + l, expected))
    assert results['predictions'] == [expected]
    assert results['labels'] == ['Hello']
    assert results['indices'] == [0]
