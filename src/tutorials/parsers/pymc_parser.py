from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('pymc')
class PyMCParser(TutorialHTMLParser):
    NAME = "PYMC"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'class': 'ui vertical segment'}
        )

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])
        if tag.name == 'section':
            return TagType.SECTION
        if tag.name == 'p':
            return TagType.PARAGRAPH
        if tag.name == 'div':
            return self.get_type_from_div_tag(tag)
        return TagType.UNKNOWN

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('section', recursive=False)
