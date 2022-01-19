import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from src.config import load_task_from_cfg, merge_configs
from src.evaluation.code_eval import evaluate_code, estimate_pass_at_k, BASE_ERROR_TYPES
from transformers import AutoModelForCausalLM


def test_evaluate_code(tmpdir, code_preds_dir):
    tmpdir_path = Path(tmpdir)

    results = evaluate_code(
        'split',
        code_preds_dir,
        2,
        3.0,
        tmpdir_path
    )

    results_path, *_ = tmpdir_path.joinpath('execution_metrics.json')
    assert results_path.exists()
    assert json.loads(results_path.read_text('utf-8')) == results

    expected_overview = {
        "all_invalid"          : 0,
        "tests_mean"           : 3.0,
        "preds_total"          : 8,
        "valid_syntax_mean"    : 7 / 2,
        "valid_syntax_total"   : 7,
        "valid_syntax_pct_mean": 1.75 / 2 * 100,
        "valid_syntax_pct"     : 7 / 8 * 100,
        "runtime_error_pct"    : 3 / 8 * 100,
        f"pass@1"              : estimate_pass_at_k([4, 4], [1, 1], 1).mean() * 100
    }

    for k in [5, 10, 25, 50, 100]:
        expected_overview[f"pass@{k}"] = 0.0

    expected_outcomes = {
        "SyntaxError" : 1,
        "Failed Tests": 2,
        "Correct"     : 2,
        "TypeError"   : 2,
        "KeyError"    : 1
    }

    for e in BASE_ERROR_TYPES:
        if e not in expected_outcomes:
            expected_outcomes[e] = 0
    for k, v in expected_outcomes.items():
        key = k.replace(' ', '_')
        expected_overview[key] = v
        expected_overview[f"{key}_pct"] = v / 8 * 100

    assert results['overview'] == expected_overview
    assert results['results_by_task_id'] == {
        "939": {
            "correct"          : 1,
            "total"            : 4,
            "error_types"      : {
                "SyntaxError" : 1,
                "Failed Tests": 1,
                "TypeError"   : 1
            },
            "correct_pct"      : 1 / 4 * 100,
            "runtime_error_pct": 1 / 4 * 100
        },
        "940": {
            "correct"          : 1,
            "total"            : 4,
            "error_types"      : {
                "SyntaxError" : 0,
                "Failed Tests": 1,
                "TypeError"   : 1,
                "KeyError"    : 1
            },
            "correct_pct"      : 1 / 4 * 100,
            "runtime_error_pct": 2 / 4 * 100
        },
    }
