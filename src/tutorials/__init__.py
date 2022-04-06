from .html_parsers import TutorialHTMLParser, SympyParser, LXMLParser
from .code_parsing import get_code_from_parsed_tutorial

def get_parser_for_domain(domain: str):
    if domain == 'lxml':
        return LXMLParser()
    elif domain == 'sympy':
        return SympyParser()

    raise ValueError(f"Unknown domain {domain}")
