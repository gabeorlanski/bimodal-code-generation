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
from datasets import Dataset
from omegaconf import OmegaConf, open_dict
from src.evaluation import evaluation
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
        evaluation.evaluate(
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
def test_generate_predictions(remove_input_ids):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    expected = [
        "\n    if not string:\n        return 0\n    return len(string)\n\n\n",
        "\n    return len(string)\n\n\ndef strlen_from_str(",
        "\n    return len(string)\n\n\ndef strlen(string: str)",
        "\n    if not isinstance(string, str):\n        return 0\n    return len(",
        "\n    return len(string)\n\n\ndef strlen_with_spaces(",
        "\n    return len(string)\n\n\ndef strlen_is_list(",
        "\n    return len(string)\n\ndef str_to_int(string",
        "\n    return len(string)\n\n\ndef strlen_with_default(",
        "\n    return len(string)\n\n\ndef strlen_with_empty_",
        "\n    return len(string)\n\n\ndef _check_string_type("
    ]
    prompt = '''def strlen(string: str) -> int:
    """ Return length of given string
    >>> strlen('')
    0
    >>> strlen('abc')
    3
    """'''

    tokenizer = AutoTokenizer.from_pretrained('lvwerra/codeparrot-small')
    tokenizer.pad_token = tokenizer.eos_token
    # prompt = tokenizer.eos_token + prompt
    gen_kwargs = {
        'max_new_tokens': 16,
        'do_sample'     : True,
        'temperature'   : 0.5,
        'top_p'         : 0.95,
        'top_k'         : 50
    }

    data = Dataset.from_dict({
        'input_sequence': [prompt],
        'target'        : ['Hello'],
        'idx'           : [0],
        'length'        : [len(tokenizer.tokenize(prompt))]
    })
    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained('lvwerra/codeparrot-small',
                                                 pad_token_id=tokenizer.eos_token_id)
    model.eos_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    model.bos_token_id = tokenizer.eos_token_id
    model = model.to(device)
    results = evaluation.generate_predictions(
        model,
        objective='lm',
        dataset=data,
        num_procs=1,
        tokenizer=tokenizer,
        num_generate_per_step=1,
        device=device,
        generation_kwargs=gen_kwargs,
        seq_per_sample=10,
        remove_input_ids_from_output=remove_input_ids,
        debug=True
    )

    if not remove_input_ids:
        expected = list(map(lambda l: prompt + l, expected))
    assert results['predictions'] == [expected]
    assert results['labels'] == ['Hello']
    assert results['indices'] == [0]


@pytest.mark.parametrize('inputs,expected', [
    [[10, 1, 5], 1.0],
    [[0, 1, 5], 0.5],
    [[0, 0, 0], 0.0],
], ids=['100%', '50%', '0%'])
def test_oracle(inputs, expected):
    def metric(p, t):
        return {'m': p[0] / t[0]}

    results = evaluation.oracle([inputs, 10], [metric])
    assert results == {'m': expected}
