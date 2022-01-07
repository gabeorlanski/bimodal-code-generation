from typing import Dict, List
import sacrebleu

from src.common import Registrable


class Metric(Registrable):
    def __call__(self, predictions: List[str], targets: List[str]) -> Dict:
        raise NotImplementedError()


@Metric.register("exact-match")
def exact_match(predictions: List[str], targets: List[str]) -> Dict:
    return {
        "em": sum(p == t for p, t in zip(predictions, targets)) / len(targets) * 100
    }


@Metric.register("bleu")
def bleu(predictions: List[str], targets: List[str]) -> Dict:
    # This came from the t5 repo
    if isinstance(targets[0], list):
        targets = [[x for x in target] for target in targets]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        targets = [targets]

    bleu_score = sacrebleu.corpus_bleu(
        predictions,
        targets,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    )
    return {"bleu": bleu_score.score}
