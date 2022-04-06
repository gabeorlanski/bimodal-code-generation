import logging
import re
from copy import deepcopy

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]+)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

logger = logging.getLogger(__name__)


def get_snippets(code_str):
    context = []
    block = []
    out = []
    output = ''

    for leading_space, snippet, output in GET_CODE_BLOCK.findall(code_str):
        num_leading_space = len(leading_space)
        code = []
        for i, line in enumerate(snippet.split('\n')):
            if i == 0:
                code.append(line[num_leading_space - 1:])
            else:
                assert line.startswith('...')
                code.append(line[num_leading_space + 3:])

        if output.strip():
            if not block and out:
                out[-1]['code'].extend(code)
                out[-1]['result'].append(output.rstrip())
            else:
                out.append({'context': block, 'code': code, 'result': [output.rstrip()]})
                block = []
        else:
            block.append('\n'.join(code))
    if block:
        out.append({
                       'context': block,
                       'code'   : None,
                       'result' : [output.rstrip()] if output.strip() else []
                   })
    return out


def get_snippets_from_sections(
        sections,
        global_context,
        sections_use_parent_ctx=None
):
    sections_use_parent_ctx = sections_use_parent_ctx or []

    section_contexts = {}
    section_snippets = []
    for i, span in enumerate(sections):

        if span['tag'] != 'code' or '>>>' not in span['text']:
            continue

        section_id = span['section_id']

        if section_id not in section_contexts:

            if span['section_title'] in sections_use_parent_ctx:
                parent_context = section_contexts.get(span['parent_id'], [])
            else:
                parent_context = []
            ctx = deepcopy(global_context) + parent_context
            section_contexts[section_id] = ctx

        for block in get_snippets(span['text']):
            block_context = block.pop('context')
            if block['result'] and block['code'] is not None:
                #                 block['context'] =
                block['code'] = block_context + [block['code']]

                section_snippets.append((i, {
                    'context': deepcopy(section_contexts[section_id]),
                    **block
                }))
            section_contexts[section_id].extend(block_context)
    return section_snippets


def get_code_from_parsed_tutorial(name, parsed_tutorial, context=None):
    total_updated = 0
    context = context or []
    for section_num, section in enumerate(parsed_tutorial):

        for i, tag in enumerate(section):
            if tag['tag'] != 'code' or '>>>' not in tag['text']:
                continue
            total_updated += 1
            section[i]['snippets'] = get_snippets(tag['text'])
    logger.debug(f"{name} had {total_updated} total code snippets")
    return total_updated, parsed_tutorial
