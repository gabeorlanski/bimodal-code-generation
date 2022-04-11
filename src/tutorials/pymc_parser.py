from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('pymc')
class PyMCParser(TutorialHTMLParser):
    NAME = "PYMC"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'class': 'ui container'}
        )

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])
        if tag.name == 'div':
            if 'section' in tag_classes:
                return TagType.SECTION
            elif 'pre' in tag_classes:
                return TagType.CODE
            elif 'p' in tag_classes:
                return TagType.PARAGRAPH
            return self.get_type_from_div_tag(tag)

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('section', recursive=False)
