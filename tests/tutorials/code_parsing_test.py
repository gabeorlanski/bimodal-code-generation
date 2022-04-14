import json

import astor
import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import code_parsing
from pathlib import Path


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


def test_get_snippets_multiple():
    input_code = '''>>> x\n1\n>>> y\n>>> z\n2'''
    expected = [
        {
            'context': [],
            'code'   : ["x"],
            'result' : ["1"]
        },
        {
            'context': ["y"],
            'code'   : ["z"],
            'result' : ["2"]
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_snippets_none():
    input_code = '''>>> x\n>>> y\n'''
    expected = [
        {
            'context': ["x", "y"],
            'code'   : [],
            'result' : []
        }
    ]

    result = code_parsing.get_snippets('Testing', input_code)

    assert result == expected


def test_get_code_samples_from_tutorial():
    input_file = json.loads(FIXTURES_ROOT.joinpath('tutorials', 'parsed.json').read_text())
    failed, result, passed, failed_tests = code_parsing.get_code_samples_from_tutorial(
        'test',
        input_file,
        global_context=['from lxml import etree'],
        fixes_by_section={'The Element class': {'overrides': [9]}}
    )

    assert dict(failed) == {
        'The Element class': [{
            'idx'     : 10,
            'error'   : 'can only concatenate str (not "int") to str',
            'code'    : ['from lxml import etree', "'1' + 1"],
            'snip_idx': 0
        }]
    }

    assert passed == [{
        'prior_context': [
            'from lxml import etree',
            "root = etree.Element('root')"
        ],
        'context'      : [],
        'code'         : [
            'root.tag',
            'isinstance(root.tag, str)'],
        'result'       : ['root', 'True'],
        'name'         : 'The Element class',
        'idx'          : 3,
        'snip_idx'     : 0
    }, {
        'prior_context': [
            'from lxml import etree', "root = etree.Element('root')",
            "root.append(etree.Element('child1'))",
            "child2 = etree.SubElement(root, 'child2')",
            "child3 = etree.SubElement(root, 'child3')"],
        'context'      : [],
        'code'         : [
            'etree.tostring(root, pretty_print=True)'],
        'result'       : [
            '<root>\n  <child1/>\n  <child2/>\n  <child3/>\n</root>'],
        'name'         : 'The Element class',
        'idx'          : 9,
        'snip_idx'     : 0
    }]
    assert failed_tests == [{
        'prior_context'  : [
            'from lxml import etree', "root = etree.Element('root')",
            "root.append(etree.Element('child1'))",
            "child2 = etree.SubElement(root, 'child2')",
            "child3 = etree.SubElement(root, 'child3')"],
        'context'        : [
            'out_0 = []', 'for i in [root[0]]:', '    out_0.append(i.tag)'
        ],
        'code'           : ['out_0'],
        'result'         : [
            'This Will Fail Tests'
        ],
        'name'           : 'The Element class-Elements are lists',
        'idx'            : 1,
        'snip_idx'       : 0,
        'testing_code'   : [
            "assert out_0 == 'This Will Fail Tests'"],
        'actual_returned': {
            'OUT_0': {'val': "['child1']", 'type': 'list'}
        }
    }]
    assert result == passed + failed_tests


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

