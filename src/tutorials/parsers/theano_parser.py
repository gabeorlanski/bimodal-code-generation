from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('natsort')
@TutorialHTMLParser.register('pydash')
@TutorialHTMLParser.register('theano')
class TheanoParser(TutorialHTMLParser):
    NAME = "THEANO"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'itemprop': 'articleBody'}
        )

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])
        if tag.name == 'div':
            if 'section' in tag_classes:
                return TagType.SECTION
            return self.get_type_from_div_tag(tag)
        elif tag.name == 'p':
            return TagType.PARAGRAPH
        elif tag.name in ['ul', 'ol', 'blockquote']:
            return TagType.LIST
        elif tag.name in ['table', 'img', 'dl', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('div', {'class': 'section'}, recursive=False)
