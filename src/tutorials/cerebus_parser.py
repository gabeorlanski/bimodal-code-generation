from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('cerberus')
class CerberusParser(TutorialHTMLParser):
    NAME = "cerberus"


    def parse_title(self, tag: Tag) -> str:
        header_link = tag.find('a')
        if header_link is not None:
            header_link.extract()
        return self.clean_text(tag.get_text())

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

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'class': 'body'}
        )

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('div', {'class': 'section'}, recursive=False)
