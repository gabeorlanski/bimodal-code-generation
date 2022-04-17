from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('humanize')
class HumanizeParser(TutorialHTMLParser):
    NAME = "humanize"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'div',
            {'class': 'md-content'}
        )

    def get_type_of_tag(self, tag: Tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])

        if tag.name == 'section':
            return TagType.SECTION
        if tag.name == 'dt':
            return TagType.CODE
        if tag.name in ['dl', 'dd']:
            return TagType.NON_SECTION_EXPAND
        if tag.name == 'div':
            if 'table-wrapper' in tag_classes:
                return TagType.TABLE
            if 'language-pycon' in tag_classes:
                return TagType.CODE
            return self.get_type_from_div_tag(tag)
        elif tag.name == 'p':
            return TagType.PARAGRAPH
        elif tag.name in ['img', 'dl', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all("article", recursive=False)

    def clean_text(self, text):
        out = super().clean_text(text)
        return out.lstrip()
