import pytest
from transformers import AutoTokenizer
from datasets import Dataset
from itertools import product

from src.common import FIXTURES_ROOT, PROJECT_ROOT
from src.experiment_cards import experiments
from src.experiment_cards.cards import AblationGroup, AblationCombination


class TestAblationCombination:

    @pytest.mark.parametrize('input_type',
                             ['no_conflict', 'step_conflict', 'override_conflict'])
    def test_from_ablations_info(self, input_type):
        ablations_info = {
            "1": ("A", {
                "overrides"     : {"key": "A"},
                "step_overrides": {
                    "step": {"key": "B"}
                }
            })

        }
        if input_type == "no_conflict":
            ablations_info.update(
                {
                    "2": ("B",
                          {
                              "overrides"     : {"key_2": "A"},
                              "step_overrides": {
                                  "step": {"key_2": "B"}
                              }
                          })
                }
            )
            result = AblationCombination.from_ablations_info('test', ablations_info)
            assert result == AblationCombination(
                'test',
                overrides={"key": "A", "key_2": "A"},
                step_overrides={"step": {"key": "B", "key_2": "B"}},
                ablation_values={"1": "A", "2": "B"}
            )
            return
        else:
            if input_type == "step_conflict":
                ablations_info.update({
                    "2": (
                        "B",
                        {
                            "overrides"     : {"key_2": "A"},
                            "step_overrides": {
                                "step": {"key": "C"}
                            }
                        }
                    )
                })
            else:
                ablations_info.update({
                    "2": (
                        "B",
                        {
                            "overrides"     : {"key": "C"},
                            "step_overrides": {
                                "step": {"key_2": "B"}
                            }
                        }
                    )
                })

            with pytest.raises(KeyError):
                AblationCombination.from_ablations_info('test', ablations_info)
