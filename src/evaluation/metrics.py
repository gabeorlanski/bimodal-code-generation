from typing import Dict, List

from src.common import Registrable


class Metric(Registrable):
    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        raise NotImplementedError()


@Metric.register("exact-match")
def exact_match(predictions: List[str], targets: List[str]) -> Dict:
    return {
        'em': sum(p == t for p, t in zip(predictions, targets)) / len(targets)
    }
