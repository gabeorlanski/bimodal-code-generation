import json
from dataclasses import asdict

import astor
import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import code_parsing
from pathlib import Path


def test_get_snippets():
    input_code = '''>>> for i in l:\n...     print(i)\nTest'''
    expected = [
        {
            'context'       : [
                "out_0 = []", "for i in l:\n    out_0.append(i)"
            ],
            'code'          : ["out_0"],
            'result'        : ['Test'],
            'start_char_idx': 0
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_if_statement():
    input_code = '''>>> if i:\n...     print(i)\nTest'''
    expected = [
        {
            'context'       : [
                "out_0 = None", "if i:\n    out_0 = i"
            ],
            'code'          : ["out_0"],
            'result'        : ['Test'],
            'start_char_idx': 0
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_context():
    input_code = '''>>> x = 1\n>>>y = {'a':1,\n\t'b':2}\n>>> print('%d' % x)\n1'''
    expected = [
        {
            'context'       : [
                "x = 1", "y = {'a':1,\n\t'b':2}"
            ],
            'code'          : ["'%d' % x"],
            'result'        : ['1'],
            'start_char_idx': 0
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_multiple():
    input_code = '''>>> x\n1\n>>> y\n>>> z\n2'''
    expected = [
        {
            'context'       : [],
            'code'          : ["x"],
            'result'        : ["1"],
            'start_char_idx': 0
        },
        {
            'context'       : ["y"],
            'code'          : ["z"],
            'result'        : ["2"],
            'start_char_idx': 9
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_none():
    input_code = '''>>> x\n>>> y\n'''
    expected = [
        {
            'context'       : ["x", "y"],
            'code'          : [],
            'result'        : [],
            'start_char_idx': 0
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_code_samples_from_tutorial():
    input_file = json.loads(FIXTURES_ROOT.joinpath('tutorials', 'parsed.json').read_text())
    failed, passed, failed_tests = code_parsing.get_code_samples_from_tutorial(
        'test',
        input_file,
        global_context=['from lxml import etree'],
        fixes_by_section={'The Element class': {'overrides': [9]}}
    )

    assert dict(failed) == {
        'The Element class': [{
            'idx'        : 10,
            'error'      : 'TypeError: can only concatenate str (not "int") to str',
            'code'       : ['from lxml import etree', "'1' + 1"],
            'snippet_idx': 0
        }]
    }

    passed = [asdict(c) for c in passed]
    assert passed == [
        {
            'actual_returned': {
                'OUT_0': {'type': 'str', 'val': 'root'},
                'OUT_1': {'type': 'bool', 'val': 'True'}
            },
            'body_code'      : [],
            'context'        : ['from lxml import etree',
                                "root = etree.Element('root')"],
            'errors'         : [],
            'expected_result': ['root', 'True'],
            'idx'            : 3,
            'return_code'    : ['root.tag', 'isinstance(root.tag, str)'],
            'section_path'   : ['The Element class'],
            'snippet_idx'    : 0,
            'testing_code'   : ["assert str(root.tag) == 'root'",
                                'assert isinstance(root.tag, str) == True'],

            'start_char_idx' : 0,
        },
        {
            'actual_returned': {},
            'body_code'      : [],
            'context'        : ['from lxml import etree',
                                "root = etree.Element('root')",
                                "root.append(etree.Element('child1'))",
                                "child2 = etree.SubElement(root, 'child2')",
                                "child3 = etree.SubElement(root, 'child3')"],
            'errors'         : [],
            'expected_result': [
                '<root>\n  <child1/>\n  <child2/>\n  <child3/>\n</root>'],
            'idx'            : 9,
            'return_code'    : ['etree.tostring(root, pretty_print=True)'],
            'section_path'   : ['The Element class'],
            'snippet_idx'    : 0,
            'testing_code'   : [],
            'start_char_idx' : 0,
        }]

    failed_tests = [asdict(c) for c in failed_tests]
    assert failed_tests == [{
        'actual_returned': {'OUT_0': {'type': 'list', 'val': "['child1']"}},
        'body_code'      : ['out_0 = []', 'for i in [root[0]]:',
                            '    out_0.append(i.tag)'],
        'context'        : ['from lxml import etree',
                            "root = etree.Element('root')",
                            "root.append(etree.Element('child1'))",
                            "child2 = etree.SubElement(root, 'child2')",
                            "child3 = etree.SubElement(root, 'child3')"],
        'errors'         : [
            'SyntaxError: invalid syntax (<string>, line 9)',
            'AssertionError',
            'AssertionError'],
        'expected_result': ['This Will Fail Tests'],
        'idx'            : 1,
        'return_code'    : ['out_0'],
        'section_path'   : ['The Element class', 'Elements are lists'],
        'snippet_idx'    : 0,
        'testing_code'   : ["assert str(out_0) == 'This Will Fail Tests'"],
        'start_char_idx' : 0,
    }]


def test_variable_tracing():
    code = "from lxml import etree\nz,y=[0,0]\nz+=x\nx = 1\nx.y=1\nz=1\ny = x + 1\nx = 2\nz.x=1"
    expected_trace = [
        {'defined': [], 'used': []},
        {'defined': ['z', 'y'], 'used': []},
        {'defined': [], 'used': ['z', 'x']},
        {'defined': ['x'], 'used': []},
        {'defined': [], 'used': ['x']},
        {'defined': ['z'], 'used': []},
        {'defined': ['y'], 'used': ['x']},
        {'defined': ['x'], 'used': []},
        {'defined': [], 'used': ['z']},
    ]

    expected_context = [
        'from lxml import etree',
        'x = 1',
        'x.y = 1',
        'y = x + 1',
        'x = 2'
    ]

    visitor = code_parsing.VariableTracer()
    bodies, result, imported, import_names = visitor(code)

    assert result == expected_trace
    assert len(imported) == 1
    assert astor.to_source(imported[0]).strip() == 'from lxml import etree'
    assert import_names == ['etree']

    result = code_parsing.get_context("print(f'{y} {x}')", code)
    assert result == expected_context


def test_get_context():
    context = 'from lxml import etree\n' \
              'root = etree.Element("root")\n' \
              'build_text_list = etree.XPath("//text()")\n' \
              'etree.SubElement(root, "child").text = "Child 1"\n' \
              'etree.tostring(html, method="text")\n' \
              'etree.SubElement(root, "another").text = "Child 3"\n'
    code = "print(etree.tostring(root, pretty_print=True))"

    result = code_parsing.get_context(code, context)
    assert result == [
        'from lxml import etree',
        'root = etree.Element(\'root\')',
        'etree.SubElement(root, \'child\').text = \'Child 1\'',
        'etree.SubElement(root, \'another\').text = \'Child 3\''
    ]


def test_get_context_multi_use():
    context = [
        'from lxml import etree',
        'child2 = etree.SubElement(root, "child2")',
        'root = etree.Element(\'root\')',
        'root.append( etree.Element("child1") )',
        'child2 = etree.SubElement(root, "child2")',
        'child3 = etree.SubElement(root, "child3")',
    ]
    code = 'print(etree.tostring(root, pretty_print=True))'
    result = code_parsing.get_context(code, '\n'.join(context))
    assert result == [
        'from lxml import etree',
        'root = etree.Element(\'root\')',
        'root.append(etree.Element(\'child1\'))',
        'child2 = etree.SubElement(root, \'child2\')',
        'child3 = etree.SubElement(root, \'child3\')',
    ]
