import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import html_parsers


def test_lxml(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('lxml')()
    result = parser(tutorial_fixtures_path.joinpath('lxml.html').read_text())

    assert result == [{
        'tag'   : 'section', 'title': 'The Element class', 'id': 1,
        'id_str': 'the-element-class', 'parent': 0, 'content': [{
            'idx' : 0,
            'text': 'An Element is the main container object for the ElementTree API. Most of the XML tree functionality is accessed through this class. Elements are easily created through the Element factory:',
            'tag' : 'p'
        }, {
            'idx' : 1,
            'text': '>>> root = etree.Element("root")\n\n',
            'tag' : 'code'
        }, {
            'tag'    : 'section',
            'title'  : 'Elements are lists',
            'id'     : 2,
            'id_str' : 'elements-are-lists',
            'parent' : 1,
            'content': [{
                'idx' : 0,
                'text': 'To make the access to these subelements easy and straight forward, elements mimic the behaviour of normal Python lists as closely as possible:',
                'tag' : 'p'
            },
                {
                    'idx' : 1,
                    'text': '>>> child = root[0]\n>>> print(child.tag)\nchild1\n\n\n',
                    'tag' : 'code'
                }]
        }, {
            'tag'    : 'section',
            'title'  : 'Elements carry attributes as a dict',
            'id'     : 3,
            'id_str' : 'elements-carry-attributes-as-a-dict',
            'parent' : 1,
            'content': [{
                'idx' : 0,
                'text': 'XML elements support attributes. You can create them directly in the Element factory:',
                'tag' : 'p'
            },
                {
                    'idx' : 1,
                    'text': '>>> root = etree.Element("root", interesting="totally")\n>>> etree.tostring(root)\nb\'<root interesting="totally"/>\'\n\n',
                    'tag' : 'code'
                }]
        }]
    }, {
        'tag'   : 'section', 'title': 'The ElementTree class', 'id': 4,
        'id_str': 'the-elementtree-class', 'parent': 0, 'content': [{
            'idx' : 0,
            'text': 'An ElementTree is mainly a document wrapper around a tree with a root node. It provides a couple of methods for serialisation and general document handling.',
            'tag' : 'p'
        }, {
            'idx' : 1,
            'text': '>>> root = etree.XML(\'\'\'\\\n... <?xml version="1.0"?>\n\n\n',
            'tag' : 'code'
        }]
    }]


def test_sympy(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('sympy')()
    result = parser(
        tutorial_fixtures_path.joinpath('sympy.html').read_text()
    )

    assert result == [{
        'tag'    : 'section', 'title': 'Preliminaries', 'id': 1,
        'id_str' : 'preliminaries', 'parent': 0,
        'content': [{'idx': 0, 'text': 'P0', 'tag': 'p'},
                    {'idx': 1, 'text': 'P1', 'tag': 'p'}, {
                        'tag'    : 'section', 'title': 'Installation', 'id': 2,
                        'id_str' : 'installation', 'parent': 1,
                        'content': [{'idx': 0, 'text': 'P1', 'tag': 'p'},
                                    {'idx': 1, 'text': 'P2', 'tag': 'p'}, {
                                        'idx' : 2,
                                        'text': ">>> from sympy import *\n>>> x = symbols('x')\nResult\n",
                                        'tag' : 'code'
                                    }, {'idx': 3, 'text': 'P3', 'tag': 'p'},
                                    {'idx': 4, 'text': 'P4', 'tag': 'p'}]
                    }, {
                        'tag'    : 'section', 'title': 'Exercises', 'id': 3,
                        'id_str' : 'exercises', 'parent': 1,
                        'content': [{'idx': 0, 'text': 'Exercises P', 'tag': 'p'}]
                    }]
    }]


def test_passlib(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('passlib')()
    result = parser(
        tutorial_fixtures_path.joinpath('passlib.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': None, 'id': 1,
        'id_str': 'hashing-verifying', 'parent': 0, 'content': [{
            'tag'    : 'section',
            'title'  : 'Hashing',
            'id'     : 2,
            'id_str' : 'hashing',
            'parent' : 1,
            'content': [{
                'idx' : 0,
                'text': 'First, import the desired hash. The following example uses the pbkdf2_sha256 class (which derives from PasswordHash):',
                'tag' : 'p'
            },
                {
                    'idx' : 1,
                    'text': '>>> # import the desired hasher\n>>> from passlib.hash import pbkdf2_sha256\n',
                    'tag' : 'code'
                },
                {
                    'idx' : 2,
                    'text': 'Use PasswordHash.hash() to hash a password. This call takes care of unicode encoding, picking default rounds values, and generating a random salt:',
                    'tag' : 'p'
                },
                {
                    'idx' : 3,
                    'text': '>>> hash = pbkdf2_sha256.hash("password")\n>>> hash\n\'$pbkdf2-sha256$29000$9t7be09prfXee2/NOUeotQ$Y.RDnnq8vsezSZSKy1QNy6xhKPdoBIwc.0XDdRm9sJ8\'\n',
                    'tag' : 'code'
                },
                {
                    'idx' : 4,
                    'text': 'Note that since each call generates a new salt, the contents of the resulting hash will differ between calls (despite using the same password as input):',
                    'tag' : 'p'
                },
                {
                    'idx' : 5,
                    'text': '>>> hash2 = pbkdf2_sha256.hash("password")\n>>> hash2\n\'$pbkdf2-sha256$29000$V0rJeS.FcO4dw/h/D6E0Bg$FyLs7omUppxzXkARJQSl.ozcEOhgp3tNgNsKIAhKmp8\'\n                      ^^^^^^^^^^^^^^^^^^^^^^\n',
                    'tag' : 'code'
                }]
        }]
    }]


def test_delorean(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('delorean')()
    result = parser(
        tutorial_fixtures_path.joinpath('delorean.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'Usage', 'id': 1, 'id_str': 'usage',
        'parent': 0, 'content': [{
            'idx' : 0,
            'text': 'Delorean aims to provide you with convient ways to get significant dates and times and easy ways to move dates from state to state. ',
            'tag' : 'p'
        }, {
            'idx' : 1,
            'text': 'In order to get the most of the documentation we will define some terminology.',
            'tag' : 'p'
        }, {
            'idx' : 2,
            'text': '* naive datetime - a datetime object without a timezone.\n                        \n* localized datetime - a datetime object with a timezone.\n                        \n* localizing - associating a naive datetime object with a\n                            timezone.\n                        \n* normalizing - shifting a localized datetime object from\n                            one timezone to another, this changes both tzinfo and datetime object.\n                        ',
            'tag' : 'p'
        }, {
            'tag'    : 'section',
            'title'  : 'Making Some Time', 'id': 2,
            'id_str' : 'making-some-time', 'parent': 1,
            'content': [{
                'idx' : 0,
                'text': 'Making time with delorean is much easier than in life.',
                'tag' : 'p'
            }, {
                'idx' : 1,
                'text': 'Start with importing delorean:',
                'tag' : 'p'
            }, {
                'idx' : 2,
                'text': '\n>>> from delorean import Delorean\n\n\n',
                'tag' : 'code'
            }, {'idx': 3, 'tag': 'p', 'text': 'Note'},
                {
                    'idx' : 4,
                    'tag' : 'p',
                    'text': 'If you are comparing Delorean objects the '
                            'time since epoch will be used internally '
                            'for comparison. This allows for the '
                            'greatest accuracy when comparing Delorean '
                            'objects from different timezones!'
                }, ]
        }]
    }]


def test_arrow(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('arrow')()
    result = parser(
        tutorial_fixtures_path.joinpath('arrow.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'Arrow: Better dates & times for Python\n',
        'id'    : 1, 'id_str': 'arrow-better-dates-times-for-python',
        'parent': 0, 'content': [
            {'idx': 0, 'text': 'Release v1.2.2 (Installation) (Changelog) ', 'tag': 'p'}, {
                'idx' : 1,
                'text': 'Arrow is a Python library that offers a sensible and human-friendly approach to creating, manipulating, formatting and converting dates, times and timestamps. It implements and updates the datetime type, plugging gaps in functionality and providing an intelligent module API that supports many common creation scenarios. Simply put, it helps you work with dates and times with fewer imports and a lot less code.',
                'tag' : 'p'
            }, {
                'idx' : 2,
                'text': 'Arrow is named after the arrow of time and is heavily inspired by moment.js and requests. ',
                'tag' : 'p'
            }, {
                'tag'   : 'section', 'title': 'Why use Arrow over built-in modules?\n', 'id': 2,
                'id_str': 'why-use-arrow-over-built-in-modules', 'parent': 1, 'content': [{
                    'idx' : 0,
                    'text': "Python's standard library and some other low-level modules have near-complete date, time and timezone functionality, but don't work very well from a usability perspective:",
                    'tag' : 'p'
                }, {
                    'idx' : 1,
                    'text': '* Too many modules: datetime, time, calendar, dateutil, pytz and\n                                more\n* Too many types: date, time, datetime, tzinfo, timedelta,\n                                relativedelta, etc.\n* Timezones and timestamp conversions are verbose and\n                                unpleasant\n* Timezone naivety is the norm\n* Gaps in functionality: ISO 8601 parsing, timespans,\n                                humanization',
                    'tag' : 'p'
                }]
            }, {
                'tag'   : 'section', 'title': 'Quick Start', 'id': 3, 'id_str': 'quick-start',
                'parent': 1, 'content': [{
                    'tag'   : 'section', 'title': 'Example Usage', 'id': 4,
                    'id_str': 'example-usage', 'parent': 3, 'content': [{
                        'idx' : 0,
                        'text': "\n>>> import arrow\n>>> arrow.get('2013-05-11T21:23:58.970460+07:00')\n<Arrow [2013-05-11T21:23:58.970460+07:00]>\n\n>>> utc = arrow.utcnow()\n>>> utc\n<Arrow [2013-05-11T21:23:58.970460+00:00]>\n\n>>> utc = utc.shift(hours=-1)\n>>> utc\n<Arrow [2013-05-11T20:23:58.970460+00:00]>\n\n>>> local = utc.to('US/Pacific')\n>>> local\n<Arrow [2013-05-11T13:23:58.970460-07:00]>\n\n>>> local.timestamp()\n1368303838.970460\n\n>>> local.format()\n'2013-05-11 13:23:58 -07:00'\n\n>>> local.format('YYYY-MM-DD HH:mm:ss ZZ')\n'2013-05-11 13:23:58 -07:00'\n\n>>> local.humanize()\n'an hour ago'\n\n>>> local.humanize(locale='ko-kr')\n'hansigan jeon'\n\n\n",
                        'tag' : 'code'
                    }]
                }]
            }]
    }]


def test_theano(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('theano')()
    result = parser(
        tutorial_fixtures_path.joinpath('theano.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'Baby Steps - Algebra', 'id': 1,
        'id_str': 'baby-steps-algebra', 'parent': 0, 'content': [{
            'tag'    : 'section',
            'title'  : 'Adding two Scalars',
            'id'     : 2,
            'id_str' : 'adding-two-scalars',
            'parent' : 1,
            'content': [{
                'idx' : 0,
                'text': "To get us started with Theano and get a feel of what we're working with, let's make a simple function: add two numbers together. Here is how you do it:",
                'tag' : 'p'
            },
                {
                    'idx' : 1,
                    'text': "\n>>> import numpy\n>>> import theano.tensor as tt\n>>> from theano import function\n>>> x = tt.dscalar('x')\n>>> y = tt.dscalar('y')\n>>> z = x + y\n>>> f = function([x, y], z)\n\n\n",
                    'tag' : 'code'
                },
                {
                    'idx' : 2,
                    'text': 'The first argument to function is a list of Variables that will be provided as inputs to the function. The second argument is a single Variable or a list of Variables. For either case, the second argument is what we want to see as output when we apply the function. f may then be used like a normal Python function.',
                    'tag' : 'p'
                },
                {
                    'idx' : 3,
                    'text': 'Note',
                    'tag' : 'p'
                },
                {
                    'idx' : 4,
                    'text': "As a shortcut, you can skip step 3, and just use a variable's eval method. The eval() method is not as flexible as function() but it can do everything we've covered in the tutorial so far. It has the added benefit of not requiring you to import function() . Here is how eval() works:",
                    'tag' : 'p'
                },
                {
                    'idx' : 5,
                    'text': "\n>>> import numpy\n>>> import theano.tensor as tt\n>>> x = tt.dscalar('x')\n>>> y = tt.dscalar('y')\n>>> z = x + y\n>>> numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)\nTrue\n\n\n",
                    'tag' : 'code'
                },
                {
                    'idx' : 6,
                    'text': 'We passed eval() a dictionary mapping symbolic theano variables to the values to substitute for them, and it returned the numerical value of the expression.',
                    'tag' : 'p'
                },
                {
                    'idx' : 7,
                    'text': 'eval() will be slow the first time you call it on a variable - it needs to call function() to compile the expression behind the scenes. Subsequent calls to eval() on that same variable will be fast, because the variable caches the compiled function.',
                    'tag' : 'p'
                }]
        }]
    }]
