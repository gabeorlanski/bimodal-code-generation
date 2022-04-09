import logging
import re
from copy import deepcopy
import ast

GET_CODE_BLOCK = re.compile(
    r'>>>( *)((?:[^\n])+(?:\n\.\.\. ?[^\n]*)*)+(?:\n((?:(?!>>>)[^\n]+\n?)+)\n?)?',
    flags=re.MULTILINE
)

REMOVE_PRINT = re.compile(r'print\(([^\n]+)\)', flags=re.DOTALL)

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
            # if not block and out:
            #     out[-1]['code'].extend(code)
            #     out[-1]['result'].append(output.rstrip())
            # else:
            out.append({'context': block, 'code': code, 'result': [output.rstrip()]})
            block = []
        else:
            block.append('\n'.join(code))
    if block:
        out.append({
            'context': block,
            'code'   : [],
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

        parent_context = section_contexts.get(span['parent_id'], [])
        if section_id not in section_contexts:
            ctx = deepcopy(global_context)
            section_contexts[section_id] = ctx

        for block in get_snippets(span['text']):
            block_context = block.pop('context')
            if block['result'] and block['code']:
                #                 block['context'] =
                block['code'] = block_context + block['code']

                section_snippets.append((i, {
                    'context'       : deepcopy(section_contexts[section_id]),
                    'parent_context': parent_context,
                    **block
                }))
            section_contexts[section_id].extend(block_context)
    return section_snippets


def get_code_from_parsed_tutorial(name, parsed_tutorial, context=None):
    total_updated = 0
    context = context or []
    could_not_be_ran = []
    could_be_ran = []

    for section_num, section in enumerate(parsed_tutorial):
        section_context = []

        for i, snip in get_snippets_from_sections(
                section,
                context,
        ):
            code_to_run = []
            for l in snip['context'] + snip['code']:
                code_to_run.append(REMOVE_PRINT.sub(r'\1', l))

            try:
                for r in snip['result']:
                    ast.literal_eval(r)
            except Exception as e:
                pass
            try:

                code_to_run_str = '\n'.join(code_to_run)
                # if test_section[i]['id'] in special_fixes:
                #     code_to_run = special_fixes[section[i]['id']](code_to_run)
                exec(code_to_run_str)
                could_be_ran.append(i)
            except Exception as e:
                if snip['parent_context']:
                    try:
                        parent_clean = []
                        for l in snip['parent_context']:
                            parent_clean.append(REMOVE_PRINT.sub(r'\1', l))
                        code_to_run_str = '\n'.join(parent_clean + code_to_run)
                        exec(code_to_run_str)
                        snip['context'] = snip['parent_context'] + snip['context']
                        could_be_ran.append(i)
                    except Exception as e:
                        could_not_be_ran.append(i)
                else:
                    could_not_be_ran.append(i)
            if 'snippets' not in section[i]:
                section[i]['snippets'] = [snip]
            else:
                section[i]['snippets'].append(snip)
            total_updated += 1
        parsed_tutorial[section_num] = section

    logger.debug(f"{name} had {total_updated} total code snippets")
    return could_be_ran, could_not_be_ran, total_updated, parsed_tutorial
