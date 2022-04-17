import json
from dataclasses import asdict

import astor
import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import code_parsing
from src.tutorials.code_sample import CodeSample
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
    parsed, faild_parse = code_parsing.get_code_samples_from_tutorial(
        'test',
        input_file,
        global_context=['from lxml import etree'],
        fixes_by_section={'The Element class': {'overrides': [9]}}
    )

    assert dict(faild_parse) == {
        'The Element class': [{
            'idx'  : 10, 'snippet_idx': 0,
            'error': 'TypeError: can only concatenate str (not "int") to str',
            'code' : ['from lxml import etree', "'1' + 1"]
        }]
    }

    parsed = [asdict(c) for c in parsed]
    assert parsed == [{
        'section_path'   : [
            'The Element class'],
        'idx'            : 3,
        'snippet_idx'    : 0,
        'body_code'      : [],
        'return_code'    : [
            'root.tag', 'isinstance(root.tag, str)'],
        'expected_result': [
            'root', 'True'],
        'start_char_idx' : 0,
        'context'        : [
            'from lxml import etree', "root = etree.Element('root')"],
        'errors'         : [],
        'testing_code'   : [],
        'actual_returned': {}
    }, {
        'section_path'   : ['The Element class'], 'idx': 9, 'snippet_idx': 0,
        'body_code'      : [],
        'return_code'    : ['etree.tostring(root, pretty_print=True)'],
        'expected_result': [
            '<root>\n  <child1/>\n  <child2/>\n  <child3/>\n</root>'],
        'start_char_idx' : 0,
        'context'        : ['from lxml import etree',
                            "root = etree.Element('root')",
                            "root.append(etree.Element('child1'))",
                            "child2 = etree.SubElement(root, 'child2')",
                            "child3 = etree.SubElement(root, 'child3')"],
        'errors'         : [], 'testing_code': [], 'actual_returned': {}
    }, {
        'section_path'   : ['The Element class', 'Elements are lists'], 'idx': 1,
        'snippet_idx'    : 0, 'body_code': ['import random'],
        'return_code'    : ['random.randint(0, 1000)'],
        'expected_result': ['This Will Fail Tests'], 'start_char_idx': 0,
        'context'        : ['from lxml import etree'], 'errors': [],
        'testing_code'   : [], 'actual_returned': {}
    }, {
        'section_path'   : ['The Element class', 'Elements are lists'], 'idx': 5,
        'snippet_idx'    : 0, 'body_code': [], 'return_code': ['None'],
        'expected_result': ['None'], 'start_char_idx': 0,
        'context'        : ['from lxml import etree', 'import random'],
        'errors'         : [], 'testing_code': [], 'actual_returned': {}
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

    result, imported = code_parsing.get_context("print(f'{y} {x}')", code)
    assert result == expected_context
    assert imported == ['from lxml import etree']


def test_get_context():
    context = 'from lxml import etree\n' \
              'root = etree.Element("root")\n' \
              'build_text_list = etree.XPath("//text()")\n' \
              'etree.SubElement(root, "child").text = "Child 1"\n' \
              'etree.tostring(html, method="text")\n' \
              'etree.SubElement(root, "another").text = "Child 3"\n'
    code = "print(etree.tostring(root, pretty_print=True))"

    result, imported = code_parsing.get_context(code, context)
    assert result == [
        'root = etree.Element(\'root\')',
        'etree.SubElement(root, \'child\').text = \'Child 1\'',
        'etree.SubElement(root, \'another\').text = \'Child 3\''
    ]
    assert imported == ['from lxml import etree']


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
    result, imported = code_parsing.get_context(code, '\n'.join(context))
    assert result == [
        'root = etree.Element(\'root\')',
        'root.append(etree.Element(\'child1\'))',
        'child2 = etree.SubElement(root, \'child2\')',
        'child3 = etree.SubElement(root, \'child3\')',
    ]
    assert imported == [
        'from lxml import etree']


def test_get_returned_values():
    code = ['x', 'y', 'z']
    context = ['x = 1', 'print(x)', 'y=2', 'z=3']

    result = code_parsing.get_returned_values(
        code,
        context
    )
    for k in result:
        if isinstance(result[k], dict):
            result[k]['type'] = result[k]['type'].__name__

    assert result == {
        'OUT_0' : {'val': '1', 'type': 'int'},
        'OUT_1' : {'val': '2', 'type': 'int'},
        'OUT_2' : {'val': '3', 'type': 'int'},
        'STDOUT': '1\n',
        'STDERR': ''
    }


def test_get_code_passes_test():
    sample = CodeSample(
        ['Testing'],
        0,
        0,
        ['x=1'],
        ['x'],
        ['1'],
        0
    )
    result = code_parsing.get_code_passes_test('test', 'testing', sample)

    assert result['file'] == 'testing'
    assert result['passed']
    assert result['sample'] == sample


@pytest.mark.parametrize(
    'body_code, return_code',
    [
        ['import random', 'random.randint(0,1000)'],
        ['x=None', 'x'],
    ],
    ids=['random', 'none']
)
def test_get_code_passes_test_fails(body_code, return_code):
    sample = CodeSample(
        ['Testing'],
        0,
        0,
        body_code.split('\n'),
        return_code.split('\n'),
        ['1'],
        0
    )
    result = code_parsing.get_code_passes_test('test', 'testing', sample)

    assert result['file'] == 'testing'
    assert not result['passed']
