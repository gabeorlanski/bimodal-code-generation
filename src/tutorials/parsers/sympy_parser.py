from typing import List, Tuple
from bs4.element import Tag
from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('sympy')
class SympyParser(TutorialHTMLParser):
    NAME = "sympy"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'class': 'body'}
        )

    def parse_code(self, tag: Tag) -> str:
        code_block = tag.find('pre')
        assert code_block is not None
        return self.clean_text(code_block.get_text()).lstrip()

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

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all('section', recursive=False)
