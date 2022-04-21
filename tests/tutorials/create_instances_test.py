import pytest

from src.tutorials import create_instances


def test_create_instances(tutorial_fixtures_path):
    result = create_instances.prepare_sample_with_func({
        "idx"            : 11,
        "snippet_idx"    : 2,
        "body_code"      : "x[1:3] = [10, 11]",
        "return_code"    : [
            "x", "y"
        ],
        "expected_result": [
            "array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])", "array([10, 11])"
        ],
        "start_char_idx" : 120,
        "context"        : "import numpy as np\nx = np.arange(10)\ny = x[1:3]",
        "section_path"   : "Copies and views/Indexing operations",
        "returned"       : {
            "OUT_0" : {
                "val" : "array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])",
                "type": "ndarray"
            },
            "OUT_1" : {
                "val" : "array([10, 11])",
                "type": "ndarray"
            },
            "STDOUT": "",
            "STDERR": ""
        }
    })
    assert result == {
        "idx"           : 11,
        "snippet_idx"   : 2,
        'target'        : 'def solution(x, y):\n    x[1:3] = [10, 11]\n    return x, y',
        "inputs"        : {'x', 'y'},
        "returned"      : {
            "OUT_0" : {
                "val" : "array([ 0, 10, 11,  3,  4,  5,  6,  7,  8,  9])",
                "type": "ndarray"
            },
            "OUT_1" : {
                "val" : "array([10, 11])",
                "type": "ndarray"
            },
            "STDOUT": "",
            "STDERR": ""
        },
        'context'       : "def get_inputs_from_context():\n    x = np.arange(10)\n    y = x[1:3]\n    return dict(x=x, y=y)",
        "start_char_idx": 120,
        "imports"       : ["import numpy as np"],
        "section_path"  : "Copies and views/Indexing operations",
    }

#        {
#     "idx"           : 11,
#     "snippet_idx"   : 1,
#     "target"        : "def prediction(x):\n    y = x[1:3]  # creates a view\n    return y",
#     "inputs"        : ['x'],
#     "input_values"  : {
#         "x": {'type': "ndarray", 'val': "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"},
#     },
#     "outputs"       : [
#         {'type': "ndarray", 'val': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'}
#     ],
#     "start_char_idx": 67,
#     "imports"       : ["import numpy as np"],
#     "context"       : ["x = np.arange(10)"],
#     "section_path"  : "Copies and views/Indexing operations"
# }
#        {
#     "idx"           : 11,
#     "snippet_idx"   : 0,
#     "target"        : "def prediction():\n    x = np.arange(10)\n    return x",
#     "start_char_idx": 0,
#     "inputs"        : [],
#     "input_values"  : {},
#     "outputs"       : [
#         {'type': "ndarray", 'val': 'array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])'}
#     ],
#     "imports"       : ["import numpy as np"],
#     "section_path"  : "Copies and views/Indexing operations"
# }
# assert result == {
#     "11": [
#         ,
#         {
#             "idx"           : 11,
#             "snippet_idx"   : 1,
#             "target"        : "def prediction(x):\n    y = x[1:3]  # creates a view\n    return y",
#             "inputs"        : ['x'],
#             "input_values"  : {
#                 "x": {'type': "ndarray", 'val': "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"},
#             },
#             "outputs"       : [
#                 {'type': "ndarray", 'val': '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'}
#             ],
#             "start_char_idx": 67,
#             "context"       : ["import numpy as np"],
#             "section_path"  : "Copies and views/Indexing operations"
#         },
#         {
#             "idx"           : 11,
#             "snippet_idx"   : 2,
#             "target"        : "def target(x,y):\n    x[1:3] = [10, 11]  # creates a view\n    return x, y",
#             "inputs"        : ['x', 'y'],
#             "input_values"  : {
#                 "x": {'type': "ndarray", 'val': "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"},
#                 "y": {'type': 'ndarray', 'val': "[1, 2]"}
#             },
#             "outputs"       : [
#                 {'type': "ndarray", 'val': '[0, 10, 11,  3,  4,  5,  6,  7,  8,  9]'},
#                 {'type': "ndarray", 'val': '[10, 11]'}
#             ],
#
#             "start_char_idx": 120,
#             "imports"       : ["import numpy as np"],
#             "section_path"  : "Copies and views/Indexing operations",
#         }
#
#     ]
# }
