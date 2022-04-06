import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import html_parsers


@pytest.fixture()
def tutorial_fixtures_path():
    yield FIXTURES_ROOT.joinpath('tutorials')


def test_lxml(tutorial_fixtures_path):
    parser = html_parsers.LXMLParser()
    result = parser(tutorial_fixtures_path.joinpath('lxml.html').read_text())

    assert len(result) == 2
    assert result[0] == [
        {
            'id'            : 2,
            'parent_id'     : 0,
            'section_id'    : 1,
            'section_str_id': 'the-element-class',
            'child_idx'     : 3,
            'section_title' : 'The Element class',
            'text'          : 'An Element is the main container object for the ElementTree API. Most of the XML tree functionality is accessed through this class. Elements are easily created through the Element factory:',
            'tag'           : 'p'
        }, {
            'id'            : 3,
            'parent_id'     : 0,
            'section_id'    : 1,
            'section_str_id': 'the-element-class',
            'child_idx'     : 5,
            'section_title' : 'The Element class',
            'text'          : '>>> root = etree.Element("root")\n\n',
            'tag'           : 'code'
        }, {
            'id'            : 5,
            'parent_id'     : 1,
            'section_id'    : 4,
            'section_str_id': 'elements-are-lists',
            'child_idx'     : 3,
            'section_title' : 'Elements are lists',
            'text'          : 'To make the access to these subelements easy and straight forward, elements mimic the behaviour of normal Python lists as closely as possible:',
            'tag'           : 'p'
        }, {
            'id'            : 6,
            'parent_id'     : 1,
            'section_id'    : 4,
            'section_str_id': 'elements-are-lists',
            'child_idx'     : 5,
            'section_title' : 'Elements are lists',
            'text'          : '>>> child = root[0]\n>>> print(child.tag)\nchild1\n\n\n',
            'tag'           : 'code'
        }, {
            'id'            : 8,
            'parent_id'     : 1,
            'section_id'    : 7,
            'section_str_id': 'elements-carry-attributes-as-a-dict',
            'child_idx'     : 3,
            'section_title' : 'Elements carry attributes as a dict',
            'text'          : 'XML elements support attributes. You can create them directly in the Element factory:',
            'tag'           : 'p'
        }, {
            'id'            : 9,
            'parent_id'     : 1,
            'section_id'    : 7,
            'section_str_id': 'elements-carry-attributes-as-a-dict',
            'child_idx'     : 5,
            'section_title' : 'Elements carry attributes as a dict',
            'text'          : '>>> root = etree.Element("root", interesting="totally")\n>>> etree.tostring(root)\nb\'<root interesting="totally"/>\'\n\n',
            'tag'           : 'code'
        }
    ]
    assert result[1] == [
        {
            'id'            : 11,
            'parent_id'     : 0,
            'section_id'    : 10,
            'child_idx'     : 3,
            'section_str_id': 'the-elementtree-class',
            'section_title' : 'The ElementTree class',
            'text'          : 'An ElementTree is mainly a document wrapper around a tree with a root node. It provides a couple of methods for serialisation and general document handling.',
            'tag'           : 'p'
        }, {
            'id'            : 12,
            'parent_id'     : 0,
            'section_id'    : 10,
            'section_str_id': 'the-elementtree-class',
            'child_idx'     : 5,
            'section_title' : 'The ElementTree class',
            'text'          : ">>> root = etree.XML(\'\'\'\\\n... <?xml version=\"1.0\"?>\n\n\n",
            'tag'           : 'code'
        }
    ]


def test_sympy(tutorial_fixtures_path):
    parser = html_parsers.SympyParser()
    result = parser(
        tutorial_fixtures_path.joinpath('sympy.html').read_text()
    )

    assert len(result) == 1
    assert result[0] == [
        {
            'id'            : 2, 'parent_id': 0, 'section_id': 1,
            'section_str_id': 'preliminaries', 'child_idx': 3,
            'section_title' : 'Preliminaries', 'text': 'P0', 'tag': 'p'
        }, {
            'id'            : 3, 'parent_id': 0, 'section_id': 1,
            'section_str_id': 'preliminaries', 'child_idx': 5,
            'section_title' : 'Preliminaries', 'text': 'P1', 'tag': 'p'
        }, {
            'id'            : 5, 'parent_id': 1, 'section_id': 4,
            'section_str_id': 'installation', 'child_idx': 5,
            'section_title' : 'Installation', 'text': 'P1', 'tag': 'p'
        }, {
            'id'            : 6, 'parent_id': 1, 'section_id': 4,
            'section_str_id': 'installation', 'child_idx': 7,
            'section_title' : 'Installation', 'text': 'P2', 'tag': 'p'
        }, {
            'id'            : 7, 'parent_id': 1, 'section_id': 4,
            'section_str_id': 'installation', 'child_idx': 9,
            'section_title' : 'Installation',
            'text'          : ">>> from sympy import *\n>>> x = symbols('x')\nResult\n",
            'tag'           : 'code'
        }, {
            'id'            : 8, 'parent_id': 1, 'section_id': 4,
            'section_str_id': 'installation', 'child_idx': 11,
            'section_title' : 'Installation', 'text': 'P3', 'tag': 'p'
        }, {
            'id'            : 9, 'parent_id': 1, 'section_id': 4,
            'section_str_id': 'installation', 'child_idx': 13,
            'section_title' : 'Installation', 'text': 'P4', 'tag': 'p'
        }, {
            'id'            : 11, 'parent_id': 1, 'section_id': 10,
            'section_str_id': 'exercises', 'child_idx': 3,
            'section_title' : 'Exercises', 'text': 'Exercises P', 'tag': 'p'
        }]
