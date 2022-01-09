from typing import Dict, List
import json


def serialize_prediction(
        idx: int,
        input_sequence: str,
        target: str,
        predictions: List[str]
) -> str:
    return json.dumps({
        "idx"           : idx,
        "input_sequence": input_sequence,
        "target"        : target,
        "predictions"   : predictions,
    })
