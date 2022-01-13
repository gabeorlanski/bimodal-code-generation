"""
Tests for the training data.
"""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf, open_dict
from src.config import load_task_from_cfg, merge_configs
from src.evaluation.evaluate import evaluate_model


def test_evaluate_model(tmpdir, simple_train_config, simple_eval_config):
    tmpdir_path = Path(tmpdir)
    train_cfg = OmegaConf.create(simple_train_config)
    gen_cfg = OmegaConf.create(simple_eval_config)
    with open_dict(gen_cfg):
        gen_cfg.task.name = 'dummy'
        gen_cfg.out_path = str(tmpdir_path)

    merged_cfg = merge_configs(gen_cfg, train_cfg)
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))

    model = MagicMock()

    with patch("src.evaluation.evaluate.generate_predictions") as mock_generate:
        mock_generate.return_value = {
            "indices"    : [0, 1, 2, 3],
            "labels"     : ["D", "C", "B", "A"],
            "predictions": [["D"], ["B"], ["C"], ["A"]]
        }

        result = evaluate_model(gen_cfg, train_cfg, model)
        assert result == {
            "em": 50.00
        }
    assert mock_generate.call_count == 1
    generate_args = mock_generate.call_args.args
    generate_kwargs = mock_generate.call_args.kwargs

    assert generate_args[0] == model
    assert set(generate_kwargs) == {
        'tokenized', 'task', 'batch_size', 'device', 'generation_kwargs'
    }

    task = load_task_from_cfg(merged_cfg)
    assert generate_kwargs['tokenized'].to_dict() == task.get_split('test').to_dict()
    assert generate_kwargs['batch_size'] == train_cfg.training.batch_size
    assert generate_kwargs['generation_kwargs'] == {"max_length": 300}

    preds_path = tmpdir_path.joinpath('predictions.jsonl')
    assert preds_path.exists()

    actual_saved_data = list(map(json.loads, preds_path.read_text('utf-8').splitlines(False)))

    assert actual_saved_data == [
        {
            'idx'           : 0,
            'input_sequence': 'Generate Python: The comment section is ',
            'target'        : 'out of control.',
            "predictions"   : ["D"]
        }, {
            'idx'           : 1,
            'input_sequence': 'Generate Python: The butcher of ',
            'target'        : 'Blevkin.',
            "predictions"   : ["B"]
        }, {
            'idx'           : 2,
            'input_sequence': 'Generate Python: Get ',
            'target'        : 'Some.',
            "predictions"   : ["C"]
        }, {
            'idx'           : 3,
            'input_sequence': 'Generate Python: I hate',
            'target'        : 'tf.data',
            "predictions"   : ["A"]
        },
    ]
