from typing import List, Tuple
from bs4.element import Tag

from .html_parsers import TutorialHTMLParser, TagType


@TutorialHTMLParser.register('jsonschema')
class FuroTheme(TutorialHTMLParser):
    NAME = "furo_theme"

    def get_body(self, soup) -> Tag:
        return soup.find(
            'article'
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
            return self.get_type_from_div_tag(tag)
        elif tag.name == 'p':
            return TagType.PARAGRAPH
        elif tag.name in ['img', 'dl', 'aside']:
            return TagType.IGNORED

        return TagType.UNKNOWN

    def get_header_and_sections(self, tag) -> Tuple[List[Tag], List[Tag]]:

        return [], tag.find_all("section", recursive=False)

    def clean_text(self, text):
        out = super().clean_text(text)
        return out.lstrip()
