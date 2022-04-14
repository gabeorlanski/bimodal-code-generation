from typing import List, Tuple
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
import re
from unidecode import unidecode

from enum import Enum, auto
from src.common.registrable import Registrable

import logging

logger = logging.getLogger(__name__)

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]+)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)
DOUBLE_WHITESPACE = re.compile(r'\s{2,}', flags=re.MULTILINE)
REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

__all__ = [
    "TagType",
    "TutorialHTMLParser",
]


class TagType(Enum):
    PARAGRAPH = auto()
    CODE = auto()
    SECTION = auto()
    LIST = auto()
    IGNORED = auto()
    # For blocks that are not sections but need to be recursed
    NON_SECTION_EXPAND = auto()
    UNKNOWN = auto()


class TutorialHTMLParser(Registrable):
    NAME = None

    PARAGRAPH_CLASSES = [
        'math',
        'topic',
        'versionadded',
        'versionchanged',
        'deprecated'
    ]
    CODE_CLASSES = [
        'doctest',
        'syntax',
    ]
    IGNORED_CLASSES = [
        'toctree-wrapper',
        'graphviz',
        'figure'
    ]
    IGNORED_TAGS = [
        'figure'
    ]
    SUB_CONTENT_CLASSES = [
        'admonition',
        'note'
    ]

    #####################################################################
    # These MUST be implemented by the child class                      #
    #####################################################################
    def get_body(self, soup) -> Tag:
        """
        Get the main BODY element of from the html
        Args:
            soup: The BS4 object.

        Returns:
            The body tag

        """
        raise NotImplementedError()

    def get_header_and_sections(self, tag):
        """
        Get the header and the FIRST level sections from the html
        Args:
            tag: The body tag.

        Returns: Two lists, the list of tags in the header, and the list of first section tags

        """
        raise NotImplementedError()

    def get_type_of_tag(self, tag: Tag) -> TagType:
        """
        Function to get the type of the tag that determines how it will be parsed.
        Args:
            tag (Tag): the input tag

        Returns:
            The output TagType of the tag

        """
        raise NotImplementedError()

    #####################################################################
    # These are optional to override                                    #
    #####################################################################
    # Overwrite these functions if the tutorial requires specific parsing.
    def parse_code(self, tag: Tag) -> str:
        return self.clean_text(tag.get_text())

    def parse_list(self, tag: Tag) -> str:
        list_items = []
        for line in tag.find_all('li'):
            list_items.append(f"* {line.text}")
        return self.clean_text('\n'.join(list_items))

    def parse_paragraph(self, tag: Tag) -> str:
        return DOUBLE_WHITESPACE.sub(
            ' ',
            self.clean_text(tag.get_text()).replace('\n', ' ')
        )

    def parse_title(self, tag: Tag) -> str:
        header_link = tag.find('a')
        if header_link is not None:
            header_link.extract()

        out = self.clean_text(tag.get_text()).strip()
        if out.endswith('P'):
            return out[:-1]
        return out

    @staticmethod
    def clean_text(text):
        return unidecode(text)

    def get_type_from_div_tag(self, tag) -> TagType:
        tag_classes = tag.attrs.get('class', [])
        if any('highlight-' in c for c in tag_classes):
            return TagType.CODE

        if any(c in tag_classes for c in self.CODE_CLASSES):
            return TagType.CODE

        elif any(c in tag_classes for c in self.PARAGRAPH_CLASSES):
            return TagType.PARAGRAPH
        elif any(c in tag_classes for c in self.SUB_CONTENT_CLASSES):
            return TagType.NON_SECTION_EXPAND

        elif any(c in tag_classes for c in self.IGNORED_CLASSES):
            return TagType.IGNORED
        return TagType.UNKNOWN

    #####################################################################
    # THESE ARE NOT TO BE IMPLEMENTED BY SUBCLASSES                     #
    #####################################################################

    def parse_content(
            self,
            content,
            id_counter,
            parent_id,
            section_id,
            is_section=False,
            idx_offset=0
    ):

        header_tag = [f'h{j}' for j in range(15)]
        section_title = None
        out = []
        for i, tag in enumerate(content):
            if isinstance(tag, NavigableString):
                continue
            if tag.name in header_tag:
                if not is_section:
                    out.append({
                        'idx' : len(out) + idx_offset,
                        'text': self.parse_paragraph(tag),
                        'tag' : 'p'
                    })
                else:
                    assert section_title is None
                    section_title = self.parse_title(tag)
                continue

            if tag.name in self.IGNORED_TAGS:
                logger.debug(f"Skipping {tag.name} at {i}. In Ignored")
                continue

            logger.debug(f"{self.NAME}: Parsing child {i} of {section_title} for {parent_id}")
            tag_type = self.get_type_of_tag(tag)
            if tag_type == TagType.SECTION:
                logger.debug(f"Found subsection in {section_title}")
                id_counter, child = self.parse_section(
                    section=tag,
                    parent_id=section_id,
                    id_counter=id_counter
                )
                out.append(child)

            elif not self.clean_text(tag.get_text()).strip():
                # id_counter -= 1
                continue
            else:

                if section_title is None and is_section:
                    raise ValueError('section is none')
                # id_counter += 1
                if tag_type == TagType.NON_SECTION_EXPAND:
                    id_counter, _, sub_content = self.parse_content(
                        tag,
                        id_counter,
                        parent_id=parent_id,
                        section_id=section_id,
                        is_section=False,
                        idx_offset=len(out)
                    )
                    out.extend(sub_content)
                    continue
                tag_content = {
                    'idx': len(out) + idx_offset
                }

                if tag_type == TagType.PARAGRAPH:
                    tag_content['text'] = self.parse_paragraph(tag)
                    tag_content['tag'] = 'p'
                elif tag_type == TagType.CODE:
                    tag_content['text'] = self.parse_code(tag)
                    tag_content['tag'] = 'code'
                elif tag_type == TagType.LIST:
                    tag_content['text'] = self.parse_list(tag)
                    tag_content['tag'] = 'p'

                elif tag_type == TagType.IGNORED:
                    # id_counter -= 1
                    continue
                else:
                    logger.error(tag.text)
                    logger.error(tag.attrs)
                    raise ValueError(f'Unknown tag type {tag.name}')
                out.append(tag_content)
        return id_counter, section_title, out

    def parse_section(
            self,
            section,
            parent_id,
            id_counter=0
    ):
        section_str_id = section.attrs['id']

        id_counter += 1
        section_id = id_counter

        out = {
            'tag'    : 'section',
            'title'  : None,
            'id'     : section_id,
            'id_str' : section_str_id,
            'parent' : parent_id,
            'content': [],
        }

        id_counter, out['title'], out['content'] = self.parse_content(
            section.children,
            id_counter,
            parent_id,
            section_id,
            is_section=True
        )

        return id_counter, out

    def __call__(self, raw_html):
        soup = BeautifulSoup(raw_html, 'lxml')
        body = self.get_body(soup)
        header, sections = self.get_header_and_sections(body)
        parsed_sections = []
        idx_counter = 0
        for s in sections:
            idx_counter, parsed = self.parse_section(
                s, 0, idx_counter,
            )
            parsed_sections.append(parsed)

        return parsed_sections
