import json
import logging
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
import ujson

from src.common import PROJECT_ROOT

logger = logging.getLogger(__name__)


def create_filter_for_so_data(parsed_path, tag_filter_file, blacklist, debug, seed):
    parsed_path = PROJECT_ROOT.joinpath(parsed_path, 'question_overview.json')
    tag_filters = defaultdict(lambda: False)

    if blacklist is not None:
        blacklist = PROJECT_ROOT.joinpath(blacklist).read_text().splitlines(False)
        logger.info(f"{len(blacklist)} total questions in the blacklist")
    else:
        logger.info("No Blacklist passed")
        blacklist = []
    logger.info("Loading the parsed question overview")
    question_overview = ujson.load(parsed_path.open())

    if tag_filter_file == "RANDOM":
        logger.info("Using a random selection for the filter")
        use_random_selection = True
        total = len(question_overview)
        sample_mask = np.zeros((total,), dtype=bool)
        rng = np.random.default_rng(seed)
        sample_mask[rng.choice(total, (5000 if debug else 5000000,), replace=False)] = True
    else:
        logger.info(f"Using the filter file {tag_filter_file}")
        use_random_selection = False
        for tag in PROJECT_ROOT.joinpath(tag_filter_file).read_text().splitlines():
            tag_filters[tag] = True
        logger.info(f"{len(tag_filters)} tags in the filter")
        sample_mask = np.ones((len(question_overview),), dtype=bool)

    logger.info(f"{len(question_overview)} questions found")
    tag_files_to_get = defaultdict(list)
    questions_passing_filter = 0
    tag_file_counts = Counter()
    finished = 0
    update_freq = 1000 if debug else 100000
    for question_id, question_dict in tqdm(question_overview.items(), total=len(question_overview)):
        tag_file_counts[question_dict['tag_to_use']] += 1

        finished += 1
        tags = question_dict.get('tags', [])
        if tags and question_id not in blacklist:
            first_tag, *rest_of_tags = tags
            if use_random_selection:
                passes_filter = sample_mask[finished - 1]

            else:
                passes_filter = tag_filters[first_tag]
                for t in rest_of_tags:
                    passes_filter = passes_filter or tag_filters[t]

            if passes_filter:
                tag_files_to_get[question_dict['tag_to_use']].append(question_id)
                questions_passing_filter += 1
        if finished % update_freq == 0:
            logger.debug(f"{finished:>8}/{len(question_overview)} finished")

    logger.info(f"{len(tag_files_to_get)} tags to use.")
    logger.info(f"{questions_passing_filter} passed the filter")

    return tag_files_to_get
