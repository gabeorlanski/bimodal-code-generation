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


def combine_code_samples_with_parsed(domain, name, parsed, code_samples: List[CodeSample]):
    tags = list(unravel_parsed(parsed))
    logger.info(f"Found {len(tags)} total sections for {name} in {domain}")
    code_samples_by_idx = defaultdict(list)

    for sample in code_samples:
        code_samples_by_idx[sample.idx].append(asdict(sample))

    for idx in code_samples_by_idx:
        code_samples_by_idx[idx] = list(
            sorted(code_samples_by_idx[idx], key=lambda d: d['snippet_idx'])
        )

    return {'domain': domain, 'name': name, 'tags': tags, 'samples': dict(code_samples_by_idx)}
