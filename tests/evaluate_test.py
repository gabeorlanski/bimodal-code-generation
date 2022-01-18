"""
Tests for the training data.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from src.config import load_task_from_cfg, merge_configs
from src.evaluation.evaluate import evaluate_model
from transformers import AutoModelForCausalLM


@pytest.mark.parametrize('seq_per_sample,num_return_seq', [[1, 1], [8, 2]])
def test_evaluate_model(tmpdir, simple_eval_config, seq_per_sample, num_return_seq):
    simple_eval_config['model_path'] = str(tmpdir)
    simple_eval_config['seq_per_sample'] = seq_per_sample
    simple_eval_config['generation']['num_return_sequences'] = num_return_seq
    simple_eval_config['preprocessors'] = []
    simple_eval_config['postprocessors'] = []
    # simple_eval_config['training']['batch_size'] = batch_size
    cfg = OmegaConf.create(simple_eval_config)
    task = load_task_from_cfg(cfg)
    expected_tok = task.get_split('train')
    raw_expected = task.preprocess('train')

    model = AutoModelForCausalLM.from_pretrained(cfg.model)
    model.generate = MagicMock()

    gen_steps = seq_per_sample // num_return_seq
    side_effect = [
        torch.Tensor([ex['input_ids'] + ex['labels'] for _ in range(num_return_seq)]).long()
        for ex in expected_tok for _ in range(gen_steps)
    ]

    model.generate.side_effect = side_effect

    results = evaluate_model(cfg, model)

    assert 'em' in results
    assert 'bleu' in results
    assert model.generate.call_count == (seq_per_sample // num_return_seq) * len(expected_tok)

    tmpdir_path = Path(tmpdir)
    pred_path = tmpdir_path.joinpath('predictions.jsonl')
    assert pred_path.exists()

    actual_preds = list(map(json.loads, pred_path.read_text('utf-8').splitlines(False)))
    assert len(actual_preds) == len(expected_tok)

    for i, pred in enumerate(actual_preds):
        raw_sample = raw_expected[pred['idx']]
        assert set(pred) == {"idx", "input_sequence", "target", "prediction", 'test'}
        assert len(pred['prediction']) == seq_per_sample
        assert all(p == raw_sample['target'] for p in pred['prediction'])
        assert pred['input_sequence'] == raw_sample['input_sequence']
        assert pred['target'] == raw_sample['target']
        assert pred['test'] == pred['idx']

    assert tmpdir_path.joinpath('eval_metrics.json').exists()
