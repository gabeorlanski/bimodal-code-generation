import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import html_parsers


def test_lxml(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('lxml')()
    result = parser(tutorial_fixtures_path.joinpath('lxml.html').read_text())

    assert result == [{
        'tag'   : 'section', 'title': 'The Element class', 'id': 1,
        'id_str': 'the-element-class', 'parent': 0, 'idx': 0, 'content': [{
            'idx' : 1,
            'text': 'An Element is the main container object for the ElementTree API. Most of the XML tree functionality is accessed through this class. Elements are easily created through the Element factory:',
            'tag' : 'p'
        }, {
            'idx' : 2,
            'text': '>>> root = etree.Element("root")\n\n',
            'tag' : 'code'
        }, {
            'tag'    : 'section',
            'title'  : 'Elements are lists',
            'id'     : 2,
            'id_str' : 'elements-are-lists',
            'parent' : 1,
            'idx'    : 3,
            'content': [
                {
                    'idx' : 4,
                    'text': 'To make the access to these subelements easy and straight forward, elements mimic the behaviour of normal Python lists as closely as possible:',
                    'tag' : 'p'
                },
                {
                    'idx' : 5,
                    'text': '>>> child = root[0]\n>>> print(child.tag)\nchild1',
                    'tag' : 'code'
                }]
        }, {
            'tag'    : 'section',
            'title'  : 'Elements carry attributes as a dict',
            'id'     : 3,
            'id_str' : 'elements-carry-attributes-as-a-dict',
            'parent' : 1,
            'idx'    : 6,
            'content': [
                {
                    'idx' : 7,
                    'text': 'XML elements support attributes. You can create them directly in the Element factory:',
                    'tag' : 'p'
                },
                {
                    'idx' : 8,
                    'text': '>>> root = etree.Element("root", interesting="totally")\n>>> etree.tostring(root)\nb\'<root interesting="totally"/>\'\n\n',
                    'tag' : 'code'
                }]
        }]
    }, {
        'tag'   : 'section', 'title': 'The ElementTree class', 'id': 4,
        'id_str': 'the-elementtree-class', 'parent': 0, 'idx': 9, 'content': [{
            'idx' : 10,
            'text': 'An ElementTree is mainly a document wrapper around a tree with a root node. It provides a couple of methods for serialisation and general document handling.',
            'tag' : 'p'
        }, {
            'idx' : 11,
            'text': '>>> root = etree.XML(\'\'\'\\\n... <?xml version="1.0"?>',
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
        'id_str' : 'preliminaries', 'parent': 0, 'idx': 0,
        'content': [
            {'idx': 1, 'text': 'P0', 'tag': 'p'},
            {'idx': 2, 'text': 'P1', 'tag': 'p'},
            {
                'tag'    : 'section', 'title': 'Installation', 'id': 2,
                'id_str' : 'installation', 'parent': 1, 'idx': 3,
                'content': [
                    {'idx': 4, 'text': 'P1', 'tag': 'p'},
                    {'idx': 5, 'text': 'P2', 'tag': 'p'}, {
                        'idx' : 6,
                        'text': ">>> from sympy import *\n>>> x = symbols('x')\nResult\n",
                        'tag' : 'code'
                    },
                    {'idx': 7, 'text': 'P3', 'tag': 'p'},
                    {'idx': 8, 'text': 'P4', 'tag': 'p'}]
            }, {
                'tag'    : 'section', 'title': 'Exercises', 'id': 3,
                'id_str' : 'exercises', 'parent': 1, 'idx': 9,
                'content': [
                    {'idx': 10, 'text': 'Exercises', 'tag': 'p'}]
            }]
    }]


def test_passlib(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('passlib')()
    result = parser(
        tutorial_fixtures_path.joinpath('passlib.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': None, 'id': 1,
        'id_str': 'hashing-verifying', 'parent': 0, 'idx': 0, 'content': [{
            'tag'    : 'section',
            'title'  : 'Hashing',
            'id'     : 2,
            'id_str' : 'hashing',
            'parent' : 1,
            'idx'    : 1,
            'content': [
                {
                    'idx' : 2,
                    'text': 'First, import the desired hash. The following example uses the pbkdf2_sha256 class (which derives from PasswordHash):',
                    'tag' : 'p'
                },
                {
                    'idx' : 3,
                    'text': '>>> # import the desired hasher\n>>> from passlib.hash import pbkdf2_sha256\n',
                    'tag' : 'code'
                },
                {
                    'idx' : 4,
                    'text': 'Use PasswordHash.hash() to hash a password. This call takes care of unicode encoding, picking default rounds values, and generating a random salt:',
                    'tag' : 'p'
                },
                {
                    'idx' : 5,
                    'text': '>>> hash = pbkdf2_sha256.hash("password")\n>>> hash\n\'$pbkdf2-sha256$29000$9t7be09prfXee2/NOUeotQ$Y.RDnnq8vsezSZSKy1QNy6xhKPdoBIwc.0XDdRm9sJ8\'\n',
                    'tag' : 'code'
                },
                {
                    'idx' : 6,
                    'text': 'Note that since each call generates a new salt, the contents of the resulting hash will differ between calls (despite using the same password as input):',
                    'tag' : 'p'
                },
                {
                    'idx' : 7,
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
        'parent': 0, 'idx': 0, 'content': [{
            'idx' : 1,
            'text': 'Delorean aims to provide you with convient ways to get significant dates and times and easy ways to move dates from state to state. ',
            'tag' : 'p'
        }, {
            'idx' : 2,
            'text': 'In order to get the most of the documentation we will define some terminology.',
            'tag' : 'p'
        }, {
            'idx' : 3,
            'text': '* naive datetime - a datetime object without a timezone.\n                        \n* localized datetime - a datetime object with a timezone.\n                        \n* localizing - associating a naive datetime object with a\n                            timezone.\n                        \n* normalizing - shifting a localized datetime object from\n                            one timezone to another, this changes both tzinfo and datetime object.\n                        ',
            'tag' : 'p'
        }, {
            'tag'   : 'section',
            'title' : 'Making Some Time',
            'id'    : 2,
            'id_str': 'making-some-time',
            'parent': 1, 'idx': 4, 'content': [
                {
                    'idx': 5, 'text': 'Making time with delorean is much easier than in life.',
                    'tag': 'p'
                }, {
                    'idx': 6, 'text': 'Start with importing delorean:', 'tag': 'p'
                }, {
                    'idx': 7, 'text': '>>> from delorean import Delorean', 'tag': 'code'
                }, {
                    'idx': 11, 'text': 'Note', 'tag': 'p'
                }, {
                    'idx' : 12,
                    'text': 'If you are comparing Delorean objects the time since epoch will be used internally for comparison. This allows for the greatest accuracy when comparing Delorean objects from different timezones!',
                    'tag' : 'p'
                }]
        }]
    }]


def test_arrow(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('arrow')()
    result = parser(
        tutorial_fixtures_path.joinpath('arrow.html').read_text()
    )

    assert result == [{
        'tag': 'section', 'title': 'Arrow: Better dates & times for Python',
        'id' : 1, 'id_str': 'arrow-better-dates-times-for-python', 'parent': 0,
        'idx': 0, 'content': [
            {'idx': 1, 'text': 'Release v1.2.2 (Installation) (Changelog) ', 'tag': 'p'}, {
                'idx' : 2,
                'text': 'Arrow is a Python library that offers a sensible and human-friendly approach to creating, manipulating, formatting and converting dates, times and timestamps. It implements and updates the datetime type, plugging gaps in functionality and providing an intelligent module API that supports many common creation scenarios. Simply put, it helps you work with dates and times with fewer imports and a lot less code.',
                'tag' : 'p'
            }, {
                'idx' : 3,
                'text': 'Arrow is named after the arrow of time and is heavily inspired by moment.js and requests. ',
                'tag' : 'p'
            }, {
                'tag'   : 'section', 'title': 'Why use Arrow over built-in modules?', 'id': 2,
                'id_str': 'why-use-arrow-over-built-in-modules', 'parent': 1, 'idx': 4, 'content': [
                    {
                        'idx' : 5,
                        'text': "Python's standard library and some other low-level modules have near-complete date, time and timezone functionality, but don't work very well from a usability perspective:",
                        'tag' : 'p'
                    }, {
                        'idx' : 6,
                        'text': '* Too many modules: datetime, time, calendar, dateutil, pytz and\n                                more\n* Too many types: date, time, datetime, tzinfo, timedelta,\n                                relativedelta, etc.\n* Timezones and timestamp conversions are verbose and\n                                unpleasant\n* Timezone naivety is the norm\n* Gaps in functionality: ISO 8601 parsing, timespans,\n                                humanization',
                        'tag' : 'p'
                    }]
            }, {
                'tag'   : 'section', 'title': 'Quick Start', 'id': 3, 'id_str': 'quick-start',
                'parent': 1, 'idx': 7, 'content': [{
                    'tag'   : 'section',
                    'title' : 'Example Usage', 'id': 4,
                    'id_str': 'example-usage', 'parent': 3,
                    'idx'   : 8, 'content': [{
                        'idx' : 9,
                        'text': ">>> import arrow\n>>> arrow.get('2013-05-11T21:23:58.970460+07:00')\n<Arrow [2013-05-11T21:23:58.970460+07:00]>\n\n>>> utc = arrow.utcnow()\n>>> utc\n<Arrow [2013-05-11T21:23:58.970460+00:00]>\n\n>>> utc = utc.shift(hours=-1)\n>>> utc\n<Arrow [2013-05-11T20:23:58.970460+00:00]>\n\n>>> local = utc.to('US/Pacific')\n>>> local\n<Arrow [2013-05-11T13:23:58.970460-07:00]>\n\n>>> local.timestamp()\n1368303838.970460\n\n>>> local.format()\n'2013-05-11 13:23:58 -07:00'\n\n>>> local.format('YYYY-MM-DD HH:mm:ss ZZ')\n'2013-05-11 13:23:58 -07:00'\n\n>>> local.humanize()\n'an hour ago'\n\n>>> local.humanize(locale='ko-kr')\n'hansigan jeon'",
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
        'id_str': 'baby-steps-algebra', 'parent': 0, 'idx': 0, 'content': [{
            'tag'    : 'section',
            'title'  : 'Adding two Scalars',
            'id'     : 2,
            'id_str' : 'adding-two-scalars',
            'parent' : 1,
            'idx'    : 1,
            'content': [
                {
                    'idx' : 2,
                    'text': "To get us started with Theano and get a feel of what we're working with, let's make a simple function: add two numbers together. Here is how you do it:",
                    'tag' : 'p'
                },
                {
                    'idx' : 3,
                    'text': ">>> import numpy\n>>> import theano.tensor as tt\n>>> from theano import function\n>>> x = tt.dscalar('x')\n>>> y = tt.dscalar('y')\n>>> z = x + y\n>>> f = function([x, y], z)",
                    'tag' : 'code'
                },
                {
                    'idx' : 4,
                    'text': 'The first argument to function is a list of Variables that will be provided as inputs to the function. The second argument is a single Variable or a list of Variables. For either case, the second argument is what we want to see as output when we apply the function. f may then be used like a normal Python function.',
                    'tag' : 'p'
                },
                {
                    'idx' : 8,
                    'text': 'Note',
                    'tag' : 'p'
                },
                {
                    'idx' : 9,
                    'text': "As a shortcut, you can skip step 3, and just use a variable's eval method. The eval() method is not as flexible as function() but it can do everything we've covered in the tutorial so far. It has the added benefit of not requiring you to import function() . Here is how eval() works:",
                    'tag' : 'p'
                },
                {
                    'idx' : 10,
                    'text': ">>> import numpy\n>>> import theano.tensor as tt\n>>> x = tt.dscalar('x')\n>>> y = tt.dscalar('y')\n>>> z = x + y\n>>> numpy.allclose(z.eval({x : 16.3, y : 12.1}), 28.4)\nTrue",
                    'tag' : 'code'
                },
                {
                    'idx' : 11,
                    'text': 'We passed eval() a dictionary mapping symbolic theano variables to the values to substitute for them, and it returned the numerical value of the expression.',
                    'tag' : 'p'
                },
                {
                    'idx' : 12,
                    'text': 'eval() will be slow the first time you call it on a variable - it needs to call function() to compile the expression behind the scenes. Subsequent calls to eval() on that same variable will be fast, because the variable caches the compiled function.',
                    'tag' : 'p'
                }]
        }]
    }]


def test_jsonschema(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('jsonschema')()
    result = parser(
        tutorial_fixtures_path.joinpath('jsonschema.html').read_text()
    )

    assert result == [{
        'tag'    : 'section', 'title': 'Handling Validation Errors', 'id': 1,
        'id_str' : 'handling-validation-errors', 'parent': 0, 'idx': 0,
        'content': [{
            'idx' : 1,
            'text': 'When an invalid instance is encountered, a ValidationError will be raised or returned, depending on which method or function is used.',
            'tag' : 'p'
        }, {
            'idx' : 3,
            'text': 'exception jsonschema.exceptions.ValidationError(message, validator=<unset>,\n                                path=(), cause=None, context=(), validator_value=<unset>,\n                                instance=<unset>,\n                                schema=<unset>,\n                                schema_path=(),\n                                parent=None)[source]P\n',
            'tag' : 'code'
        }, {
            'idx' : 5,
            'text': 'An instance was invalid under a provided schema.',
            'tag' : 'p'
        }, {
            'idx' : 6,
            'text': 'The information carried by an error roughly breaks down into:',
            'tag' : 'p'
        }, {
            'idx' : 7,
            'text': '| What Happened | Why Did It Happen | What Was Being Validated |\n| --- | --- | --- |\n| `message` | `context`\n`cause` | `instance`\n`json\\_path`\n\n`path`\n`schema`\n`schema\\_path`\n\n`validator`\n\n`validator\\_value`\n |',
            'tag' : 'p'
        }, {'idx': 11, 'text': 'message', 'tag': 'code'}, {
            'idx' : 13,
            'text': 'A human readable message explaining the error.',
            'tag' : 'p'
        }, {
            'idx' : 14,
            'text': 'In case an invalid schema itself is encountered, a SchemaError is raised.',
            'tag' : 'p'
        }, {
            'idx' : 23,
            'text': 'exception jsonschema.exceptions.SchemaError(message, validator=<unset>,\n                                path=(), cause=None, context=(), validator_value=<unset>,\n                                instance=<unset>,\n                                schema=<unset>,\n                                schema_path=(),\n                                parent=None)[source]P\n',
            'tag' : 'code'
        }, {
            'idx' : 25,
            'text': 'A schema was invalid under its corresponding metaschema.',
            'tag' : 'p'
        }, {
            'idx' : 26,
            'text': 'The same attributes are present as for ValidationErrors.',
            'tag' : 'p'
        }, {
            'idx' : 27,
            'text': 'These attributes can be clarified with a short example:',
            'tag' : 'p'
        }, {
            'tag'    : 'section', 'title': 'ErrorTrees', 'id': 2,
            'id_str' : 'errortrees', 'parent': 1, 'idx': 28,
            'content': [{
                'idx' : 29,
                'text': 'If you want to programmatically be able to query which properties or validators failed when validating a given instance, you probably will want to do so using jsonschema.exceptions.ErrorTree objects.',
                'tag' : 'p'
            }, {
                'idx' : 31,
                'text': 'class jsonschema.exceptions.ErrorTree(errors=())[source]P\n',
                'tag' : 'code'
            }, {
                'idx' : 33,
                'text': 'ErrorTrees make it easier to check which validations failed.',
                'tag' : 'p'
            },
                {'idx': 35, 'text': 'errors', 'tag': 'code'},
                {
                    'idx' : 37,
                    'text': 'The mapping of validator names to the error objects (usually jsonschema.exceptions.ValidationErrors) at this level of the tree.',
                    'tag' : 'p'
                }, {
                    'idx' : 41,
                    'text': '__contains__(index)[source]',
                    'tag' : 'code'
                }, {
                    'idx' : 43,
                    'text': 'Check whether instance[index] has any errors.',
                    'tag' : 'p'
                }, {
                    'idx' : 49,
                    'text': '__getitem__(index)[source]',
                    'tag' : 'code'
                }, {
                    'idx' : 51,
                    'text': 'Retrieve the child tree one level down at the given index.',
                    'tag' : 'p'
                }, {
                    'idx' : 52,
                    'text': 'If the index is not in the instance that this tree corresponds to and is not known by this tree, whatever error would be raised by instance.__getitem__ will be propagated (usually this is some subclass of LookupError. ',
                    'tag' : 'p'
                }, {
                    'idx' : 61,
                    'text': '__init__(errors=())[source]',
                    'tag' : 'code'
                }, {
                    'idx': 71, 'text': '__iter__()[source]',
                    'tag': 'code'
                }, {
                    'idx' : 73,
                    'text': 'Iterate (non-recursively) over the indices in the instance with errors.',
                    'tag' : 'p'
                }, {
                    'idx': 85, 'text': '__len__()[source]',
                    'tag': 'code'
                }, {
                    'idx' : 87,
                    'text': 'Return the total_errors.',
                    'tag' : 'p'
                }, {
                    'idx': 101, 'text': '__repr__()[source]',
                    'tag': 'code'
                }, {
                    'idx': 103, 'text': 'Return repr(self).',
                    'tag': 'p'
                }, {
                    'idx' : 119,
                    'text': '__setitem__(index,\n                                            value)[source]',
                    'tag' : 'code'
                }, {
                    'idx' : 121,
                    'text': 'Add an error to the tree at the given index.',
                    'tag' : 'p'
                }, {
                    'idx' : 139,
                    'text': 'property total_errors',
                    'tag' : 'code'
                }, {
                    'idx' : 141,
                    'text': 'The total number of errors in the entire tree, including children.',
                    'tag' : 'p'
                }, {
                    'idx' : 142,
                    'text': '>>> 0 in tree\nTrue\n\n>>> 1 in tree\nFalse',
                    'tag' : 'code'
                }]
        }]
    }]


def test_natsort(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('natsort')()
    result = parser(
        tutorial_fixtures_path.joinpath('natsort.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': '1. How Does Natsort Work?', 'id': 1,
        'id_str': 'how-does-natsort-work', 'parent': 0, 'idx': 0, 'content': [{
            'idx' : 1,
            'text': ' First, How Does Natural Sorting Work At a High Level? Natsort\'s Approach Decomposing Strings Into Sub-Components Coercing Strings Containing Numbers Into Numbers TL;DR 1 - The Simple "No Special Cases" Algorithm Special Cases Everywhere! Sorting Filesystem Paths Comparing Different Types Handling NaN TL;DR 2 - Handling Crappy, Real-World Input Here Be Dragons: Adding Locale Support Basic Case Control Support Basic Unicode Support Using Locale to Compare Strings Unicode Support With Local Handling Broken Locale On OSX Handling Locale-Aware Numbers Final Thoughts ',
            'tag' : 'p'
        }, {
            'idx' : 2,
            'text': 'If you are impatient, you can skip to TL;DR 1 - The Simple "No Special Cases" Algorithm for the algorithm in the simplest case, and TL;DR 2 - Handling Crappy, Real-World Input to see what extra code is needed to handle special cases.',
            'tag' : 'p'
        }, {
            'tag'    : 'section',
            'title'  : '',
            'id'     : 2,
            'id_str' : 'first-how-does-natural-sorting-work-at-a-high-level',
            'parent' : 1,
            'idx'    : 3,
            'content': [
                {
                    'idx' : 4,
                    'text': "If I want to compare '2 ft 7 in' to '2 ft 11 in', I might do the following",
                    'tag' : 'p'
                },
                {
                    'idx' : 5,
                    'text': ">>> '2 ft 7 in' < '2 ft 11 in'\nFalse",
                    'tag' : 'code'
                },
                {
                    'idx' : 6,
                    'text': 'We as humans know that the above should be true, but why does Python think it is false? Here is how it is performing the comparison:',
                    'tag' : 'p'
                },
                {
                    'idx' : 7,
                    'text': "At its heart, natsort is simply a tool to break strings into tuples, turning numbers in strings (i.e. '79') into ints and floats as it does this.",
                    'tag' : 'p'
                }]
        }]
    }]


def test_bleach(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('bleach')()
    result = parser(
        tutorial_fixtures_path.joinpath('bleach.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'Sanitizing text fragments', 'id': 1,
        'id_str': 'sanitizing-text-fragments', 'parent': 0, 'idx': 0, 'content': [
            {
                'idx' : 1,
                'text': 'Bleach sanitizes text fragments for use in an HTML context. It provides a bleach.clean() function and a more configurable bleach.sanitizer.Cleaner class with safe defaults.',
                'tag' : 'p'
            }, {
                'idx' : 2,
                'text': 'Given a text fragment, Bleach will parse it according to the HTML5 parsing algorithm and sanitize tags, attributes, and other aspects. This also handles unescaped characters and unclosed and misnested tags. The result is text that can be used in HTML as is.',
                'tag' : 'p'
            }, {'idx': 5, 'text': 'Warning', 'tag': 'p'}, {
                'idx' : 6,
                'text': 'bleach.clean() is for sanitising HTML fragments to use in an HTML context-not for use in HTML attributes, CSS, JavaScript, JavaScript templates (mustache, handlebars, angular, jsx, etc), JSON, xhtml, SVG, or other contexts.',
                'tag' : 'p'
            }, {
                'idx' : 7,
                'text': 'For example, this is a safe use of clean output in an HTML context:',
                'tag' : 'p'
            }, {
                'idx': 8, 'text': '<p>\n  {{ bleach.clean(user_bio) }}\n</p>', 'tag': 'code'
            }, {
                'idx': 9, 'text': 'This is not a safe use of clean output in an HTML attribute:',
                'tag': 'p'
            }, {
                'idx': 10, 'text': '<body data-bio="{{ bleach.clean(user_bio) }}">', 'tag': 'code'
            }, {
                'idx' : 11,
                'text': "If you need to use the output of bleach.clean() in any other context, you need to pass it through an appropriate sanitizer/escaper for that context. For example, if you wanted to use the output in an HTML attribute value, you would need to pass it through Jinja's or Django's escape function.",
                'tag' : 'p'
            }, {
                'tag'   : 'section', 'title': 'Allowed tags (tags)', 'id': 2,
                'id_str': 'allowed-tags-tags', 'parent': 1, 'idx': 12, 'content': [{
                    'idx' : 13,
                    'text': 'The tags kwarg specifies the allowed set of HTML tags. It should be a list, tuple, or other iterable. Any HTML tags not in this list will be escaped or stripped from the text.',
                    'tag' : 'p'
                }, {
                    'idx' : 14,
                    'text': 'For example:',
                    'tag' : 'p'
                }, {
                    'idx' : 15,
                    'text': ">>> import bleach\n\n>>> bleach.clean(\n...     '<b><i>an example</i></b>',\n...     tags=['b'],\n... )\n'<b>&lt;i&gt;an example&lt;/i&gt;</b>'",
                    'tag' : 'code'
                }, {
                    'idx' : 16,
                    'text': 'The default value is a relatively conservative list found in bleach.sanitizer.ALLOWED_TAGS. ',
                    'tag' : 'p'
                }]
            }]
    }]


def test_scipy(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('scipy')()
    result = parser(
        tutorial_fixtures_path.joinpath('scipy.html').read_text()
    )

    assert result == [{
        'tag'    : 'section', 'title': 'Special functions', 'id': 1,
        'id_str' : 'special-functions-scipy-special', 'parent': 0, 'idx': 0,
        'content': [{
            'idx' : 1,
            'text': "The main feature",
            'tag' : 'p'
        }, {
            'tag'   : 'section',
            'title' : 'Bessel functions of real order',
            'id'    : 2,
            'id_str': 'bessel-functions-of-real-order-jv-jn-zeros',
            'parent': 1, 'idx': 2, 'content': [{
                'idx' : 3,
                'text': "Bessel functions are a family of solutions to Bessel's differential equation with real or complex order alpha:",
                'tag' : 'p'
            }, {
                'idx' : 4,
                'text': ' x2d2ydx2+xdydx+(x2-a2)y=0 ',
                'tag' : 'p'
            }, {
                'idx' : 5,
                'text': 'Among other uses, these functions arise in wave propagation problems, such as the vibrational modes of a thin drum head. Here is an example of a circular drum head anchored at the edge:',
                'tag' : 'p'
            }, {
                'idx' : 6,
                'text': '>>> from scipy import special\n>>> def drumhead_height(n, k, distance, angle, t):\n...    kth_zero = special.jn_zeros(n, k)[-1]\n...    return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)\n>>> theta = np.r_[0:2*np.pi:50j]\n>>> radius = np.r_[0:1:50j]\n>>> x = np.array([r * np.cos(theta) for r in radius])\n>>> y = np.array([r * np.sin(theta) for r in radius])\n>>> z = np.array([drumhead_height(1, 1, r, theta, 0.5) for r in radius])',
                'tag' : 'code'
            }]
        }]
    }]


def test_numpy(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('numpy')()
    result = parser(
        tutorial_fixtures_path.joinpath('numpy.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'NumPy quickstart', 'id': 1,
        'id_str': 'numpy-quickstart', 'parent': 0, 'idx': 0, 'content': [{
            'tag'    : 'section',
            'title'  : 'The Basics',
            'id'     : 2,
            'id_str' : 'the-basics',
            'parent' : 1,
            'idx'    : 1,
            'content': [
                {
                    'idx' : 2,
                    'text': "NumPy's main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of non-negative integers. In NumPy dimensions are called axes.",
                    'tag' : 'p'
                },
                {
                    'idx' : 3,
                    'text': '[[1., 0., 0.],\n [0., 1., 2.]]',
                    'tag' : 'code'
                },
                {
                    'idx' : 6,
                    'text': 'ndarray.ndim',
                    'tag' : 'p'
                },
                {
                    'idx' : 7,
                    'text': 'the number of axes (dimensions) of the array. ',
                    'tag' : 'p'
                },
                {
                    'idx' : 8,
                    'text': 'ndarray.shape',
                    'tag' : 'p'
                },
                {
                    'idx' : 9,
                    'text': 'the dimensions of the array. This is a tuple of integers indicating the size of the array in each dimension. For a matrix with n rows and m columns, shape will be (n,m). The length of the shape tuple is therefore the number of axes, ndim. ',
                    'tag' : 'p'
                },
                {
                    'idx' : 10,
                    'text': 'ndarray.size',
                    'tag' : 'p'
                },
                {
                    'idx' : 11,
                    'text': 'the total number of elements of the array. This is equal to the product of the elements of shape. ',
                    'tag' : 'p'
                },
                {
                    'idx' : 12,
                    'text': 'ndarray.dtype',
                    'tag' : 'p'
                },
                {
                    'idx' : 13,
                    'text': "an object describing the type of the elements in the array. One can create or specify dtype's using standard Python types. Additionally NumPy provides types of its own. numpy.int32, numpy.int16, and numpy.float64 are some examples. ",
                    'tag' : 'p'
                },
                {
                    'idx' : 14,
                    'text': 'ndarray.itemsize',
                    'tag' : 'p'
                },
                {
                    'idx' : 15,
                    'text': 'the size in bytes of each element of the array. For example, an array of elements of type float64 has itemsize 8 (=64/8), while one of type complex32 has itemsize 4 (=32/8). It is equivalent to ndarray.dtype.itemsize. ',
                    'tag' : 'p'
                },
                {
                    'idx' : 16,
                    'text': 'ndarray.data',
                    'tag' : 'p'
                },
                {
                    'idx' : 17,
                    'text': "the buffer containing the actual elements of the array. Normally, we won't need to use this attribute because we will access the elements in an array using indexing facilities. ",
                    'tag' : 'p'
                }]
        }]
    }]


def test_networkx(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('networkx')()
    result = parser(
        tutorial_fixtures_path.joinpath('networkx.html').read_text()
    )

    assert result == [{
        'tag'   : 'section', 'title': 'Tutorial', 'id': 1, 'id_str': 'tutorial',
        'parent': 0, 'idx': 0, 'content': [
            {'idx': 1, 'text': 'This guide can help you start working with NetworkX.', 'tag': 'p'},
            {
                'tag'   : 'section', 'title': 'Creating a graph', 'id': 2,
                'id_str': 'creating-a-graph', 'parent': 1, 'idx': 2, 'content': [
                {'idx': 3, 'text': 'Create an empty graph with no nodes and no edges.', 'tag': 'p'},
                {
                    'idx': 4, 'text': '>>>\n>>> import networkx as nx\n>>> G = nx.Graph()',
                    'tag': 'code'
                }, {'idx': 7, 'text': 'Note', 'tag': 'p'}, {
                    'idx' : 8,
                    'text': "Python's None object is not allowed to be used as a node. It determines whether optional function arguments have been assigned in many functions.",
                    'tag' : 'p'
                }]
            }]
    }]


def test_python(tutorial_fixtures_path):
    parser = html_parsers.TutorialHTMLParser.by_name('python')()
    result = parser(
        tutorial_fixtures_path.joinpath('python.html').read_text()
    )

    assert result == [{
        'tag'    : 'section', 'title': '3. An Informal Introduction to Python',
        'id'     : 1, 'id_str': 'an-informal-introduction-to-python', 'parent': 0,
        'idx'    : 0,
        'content': [{'idx': 1, 'text': 'In the following examples', 'tag': 'p'},
                    {'idx': 2, 'text': 'Many of the examples', 'tag': 'p'},
                    {'idx': 3, 'text': 'Some examples:', 'tag': 'p'}, {
                        'idx' : 4,
                        'text': '# this is the first comment\nspam = 1  # and this is the second comment\n          # ... and now a third!\ntext = "# This is not a comment because it\'s inside quotes."',
                        'tag' : 'code'
                    }]
    }]
