from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('scipy')
class ScipyParser(TutorialHTMLParser):
    NAME = "scipy"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'main'
        ).find('div')

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])
        if tag.name == 'div':
            if 'section' in tag_classes:
                return TagType.SECTION
            return self.get_type_from_div_tag(tag)
        elif tag.name in ['p', 'dt', 'dd']:
            return TagType.PARAGRAPH
        elif tag.name == 'dl':
            return TagType.NON_SECTION_EXPAND
        elif tag.name in ['ul', 'ol', 'blockquote']:
            return TagType.LIST
        elif tag.name in ['table', 'img', 'dl', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def parse_title(self, tag: Tag) -> str:
        out = super(ScipyParser, self).parse_title(tag)
        return out.split("(")[0].strip()

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:
        return [], tag.find_all('div', {'class': 'section'}, recursive=False)


@TutorialHTMLParser.register('networkx')
@TutorialHTMLParser.register('numpy')
class NumpyParser(TutorialHTMLParser):
    NAME = "numpy"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'main'
        ).find('div')

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])

        if tag.name == 'section':
            return TagType.SECTION
        elif tag.name == 'div':
            return self.get_type_from_div_tag(tag)
        elif tag.name in ['p', 'dt', 'dd']:
            return TagType.PARAGRAPH
        elif tag.name == 'dl':
            return TagType.NON_SECTION_EXPAND
        elif tag.name in ['ul', 'ol', 'blockquote']:
            return TagType.LIST
        elif tag.name in ['table', 'img', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:
        return [], tag.find_all('section', recursive=False)

