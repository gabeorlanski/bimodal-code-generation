from yamrf.common import util


def test_flatten():
    actual = util.flatten({
        "A": {
            "B": "C",
            "D": {"E"}
        },
        "F": "G"
    }, sep=".")
    assert actual == {"A.B": "C", "A.D": {"E"}, "F": "G"}
