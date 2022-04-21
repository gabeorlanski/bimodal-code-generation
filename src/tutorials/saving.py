import logging
from collections import defaultdict
from dataclasses import asdict
from typing import List

from .code_sample import CodeSample

logger = logging.getLogger(__name__)


def unravel_parsed(parsed, parent_idx=-1):
    for c in parsed:
        if c['tag'] == 'section':
            if c['title'] is not None:
                out = {k: v for k, v in c.items() if k in ['tag', 'idx']}
                out['text'] = c['title']
                out['parent_idx'] = parent_idx
                yield out

            for child in unravel_parsed(c['content'], c['idx']):
                yield child
        else:
            yield {'parent_idx': c['idx'], **c}


def get_context_tags_for_code(
        domain,
        name,
        parsed,
        code_samples: List[CodeSample],
        num_context_to_keep: int,
        url,
        annotation_mode
):
    code_samples_by_idx = defaultdict(list)

    tags = {}
    for t in unravel_parsed(parsed):
        if t['idx'] in tags:
            raise KeyError(t['idx'])
        tags[str(t['idx'])] = t
    # tags = list(unravel_parsed(parsed))
    logger.debug(f"Found {len(tags)} total sections for {name} in {domain}")

    for sample in code_samples:
        code_samples_by_idx[sample.idx].append(sample.to_save_dict(annotation_mode))

    out = defaultdict(dict)
    for idx in code_samples_by_idx:
        idx_key = str(idx)

        context = []
        if num_context_to_keep is not None and not annotation_mode:
            cur_idx = idx - 1
            while len(context) < num_context_to_keep and cur_idx >= 0:
                try:
                    context.append(tags[str(cur_idx)]['text'])
                except KeyError:
                    pass
                cur_idx -= 1

        out[idx_key] = {
            'samples': list(
                sorted(code_samples_by_idx[idx], key=lambda d: d['snippet_idx'])
            ),
            **tags[idx_key]
        }
        if not annotation_mode:
            out[idx_key]['context'] = context
    return {'url': url, 'domain': domain, 'name': name, 'samples': out}
