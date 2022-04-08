from .html_parsers import *
from .code_parsing import get_code_from_parsed_tutorial


def get_parser_for_domain(domain: str):
    if domain == 'lxml':
        return LXMLParser()
    elif domain == 'sympy':
        return SympyParser()
    elif domain == 'passlib':
        return PassLibParser()
    elif domain == 'pynacl':
        return PyNaClParser()
    raise ValueError(f"Unknown domain {domain}")
