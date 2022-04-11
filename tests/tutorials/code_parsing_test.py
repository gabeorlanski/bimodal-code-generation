import pytest
from src.tutorials import code_parsing


def test_get_snippets():
    input_code = '''>>> for i in l:\n...     print(i)\nTest'''
    expected = [
        {
            'context': [
                "out_0 = []", "for i in l:\n    out_0.append(i)"
            ],
            'code'   : ["out_0"],
            'result' : ['Test']
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_if_statement():
    input_code = '''>>> if i:\n...     print(i)\nTest'''
    expected = [
        {
            'context': [
                "out_0 = None", "if i:\n    out_0 = i"
            ],
            'code'   : ["out_0"],
            'result' : ['Test']
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_context():
    input_code = '''>>> x = 1\n>>>y = {'a':1,\n\t'b':2}\n>>> print('%d' % x)\n1'''
    expected = [
        {
            'context': [
                "x = 1", "y = {'a':1,\n\t'b':2}"
            ],
            'code'   : ["'%d' % x"],
            'result' : ['1']
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected
