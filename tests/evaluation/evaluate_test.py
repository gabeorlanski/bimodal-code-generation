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


def test_generate_predictions_batch_size():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    expected = [
        ["1", "1", "1.5"],
        ["2", "2", "2.5"]
    ]
    prompt = '''Test'''

    tokenizer = AutoTokenizer.from_pretrained('lvwerra/codeparrot-small')
    tokenizer.pad_token = tokenizer.eos_token
    gen_kwargs = {
        'max_new_tokens': 16,
        'do_sample'     : True,
        'temperature'   : 0.5,
        'top_p'         : 0.95,
        'top_k'         : 50
    }

    data = Dataset.from_dict({
        'input_sequence': [prompt] * 2,
        'target'        : ['Hello'] * 2,
        'idx'           : [0, 1],
        'length'        : [len(tokenizer.tokenize(prompt))] * 2
    })
    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained('lvwerra/codeparrot-small',
                                                 pad_token_id=tokenizer.eos_token_id)
    model.eos_token_id = tokenizer.eos_token_id
    model.pad_token_id = tokenizer.eos_token_id
    model.bos_token_id = tokenizer.eos_token_id

    def gen_mock(num_return_sequences, **kwargs):
        if num_return_sequences == 4:
            return tokenizer(["1", "1", "2", "2"], return_tensors='pt')['input_ids']
        return tokenizer(["1.5", "2.5"], return_tensors='pt')['input_ids']

    model.generate = MagicMock()
    model.generate.side_effect = gen_mock

    results = evaluation.generate_predictions(
        model,
        objective='lm',
        dataset=data,
        num_procs=1,
        tokenizer=tokenizer,
        num_generate_per_step=4,
        device=device,
        generation_kwargs=gen_kwargs,
        seq_per_sample=3,
        remove_input_ids_from_output=False,
        debug=True
    )

    assert results['predictions'] == expected
    assert results['labels'] == ['Hello'] * 2
    assert results['indices'] == [0, 1]