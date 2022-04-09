from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('pynacl')
class PyNaClParser(TutorialHTMLParser):
    NAME = "pynacl"

    def get_type_of_tag(self, tag: Tag) -> TagType:
        if tag.name == 'section':
            return TagType.SECTION
        elif tag.name == 'div':
            return self.get_type_from_div_tag(tag)
        elif tag.name == 'p':
            return TagType.PARAGRAPH
        elif tag.name in ['ul', 'ol', 'blockquote']:
            return TagType.LIST
        elif tag.name in ['table', 'img', 'dl', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'itemprop': 'articleBody'}
        )

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('section', recursive=False)
