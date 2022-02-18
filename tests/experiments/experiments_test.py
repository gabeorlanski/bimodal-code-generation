import pytest
from transformers import AutoTokenizer
from datasets import Dataset
from itertools import product

from src.common import FIXTURES_ROOT, PROJECT_ROOT
from src.experiment_cards import experiments
from src.experiment_cards.cards import AblationGroup, AblationCombination


@pytest.mark.parametrize(
    "num_ablations", [3, 1, 0],
    ids=['Multiple Ablations', '1 Ablation', 'No Ablations']
)
def test_get_all_ablation_combinations(num_ablations):
    if num_ablations == 3:
        ablation_groups = [
            AblationGroup(name="1", ablation_cards={"A": {"k": "v"}, "B": {"k": "g"}}),
            AblationGroup(name="2", ablation_cards={"C": {"z": "e"}}),
            AblationGroup(name="3", ablation_cards={"D": {"1": 2}, "E": {"1": 3}, "F": {"1": 4}}),
        ]
        all_keys = [
            ["A", "B"],
            ["C"],
            ["D", "E", "F"]
        ]
        expected = []
        for first, second, third in product(*all_keys):
            name = f"{first}.{second}.{third}"
            values = {
                "1": (first, ablation_groups[0][first]),
                "2": (second, ablation_groups[1][second]),
                "3": (third, ablation_groups[2][third]),
            }
            expected.append(
                AblationCombination.from_ablations_info(name=name, ablations_info=values))
    elif num_ablations == 1:
        ablation_groups = [
            AblationGroup(name="1", ablation_cards={"A": {"k": "v"}, "B": {"k": "g"}})]

        expected = [
            AblationCombination.from_ablations_info(name='A',
                                                    ablations_info={
                                                        '1': ("A", ablation_groups[0]['A'])
                                                    }),
            AblationCombination.from_ablations_info(name='B',
                                                    ablations_info={
                                                        '1': ("B", ablation_groups[0]['B'])
                                                    }),
        ]

    else:
        ablation_groups = [AblationGroup(name="NO_ABLATIONS_FOUND", ablation_cards={})]
        expected = [
            AblationCombination.from_ablations_info(name="NO_ABLATIONS_FOUND", ablations_info={})
        ]

    results = experiments.get_all_ablation_combinations(ablation_groups)

    assert results == expected
