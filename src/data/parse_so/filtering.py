import json
import logging
import os
from collections import defaultdict, Counter

import numpy as np
import psutil
from tqdm import tqdm
import ujson
from pathlib import Path
from src.common import PROJECT_ROOT

__all__ = [
    "create_filter_for_so_data",
    "consolidate_so_data"
]
logger = logging.getLogger(__name__)


def create_filter_for_so_data(
        parsed_path,
        tag_filter_file,
        blacklist,
        tag_blacklist_path,
        debug,
        seed
):
    tag_filters = defaultdict(lambda: False)

    logger.info("Loading the parsed question overview")
    question_overview = ujson.load(parsed_path.open())
    logger.info("Removing blacklisted questions")
    if blacklist is not None:
        blacklist = PROJECT_ROOT.joinpath(blacklist).read_text().splitlines(False)
        logger.info(f"{len(blacklist)} total questions in the blacklist")
        logger.debug(f"{len(question_overview)} prior to removing")
        blacklist_found = 0
        for qid in blacklist:
            if question_overview.pop(qid, None) is not None:
                blacklist_found += 1
        logger.debug(f"{blacklist_found} blacklisted questions found")
        logger.debug(f"{len(question_overview)} after removing blacklisted")

    else:
        logger.info("No Blacklist passed")
        blacklist = []

    if tag_blacklist_path is not None:
        logger.info(f"Loading tag blacklist from {tag_blacklist_path}")
        tag_blacklist = defaultdict(lambda: False)
        for t in PROJECT_ROOT.joinpath(tag_blacklist_path).read_text().splitlines(False):
            tag_blacklist[t] = True
        logger.info(f"Removing all questions with any of the {len(tag_blacklist)} "
                    f"tags in the tag blacklist")

        removed = 0
        # Need to cast the keys to a list so we can remove them during the
        # iteration.
        for qid in tqdm(list(question_overview), desc='Removing'):
            question = question_overview[qid]
            if any(tag_blacklist[t] for t in question['tags']):
                removed += 1
                question_overview.pop(qid)

        logger.info(f"Removed {removed} total questions")

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


def consolidate_so_data(
        name,
        filter_file,
        dump_path,
        max_buffer_size,
        seed,
        debug,
        output_path,
        max_val_size
):
    logger.info("Starting Consolidate")

    if output_path == 'data/stack_exchange':
        output_path = PROJECT_ROOT.joinpath("data", "dumps")
    else:
        output_path = Path(output_path)

    logger.info(f"Writing to {output_path}")
    if not output_path.exists():
        output_path.mkdir()

    filter_dict = json.loads(
        PROJECT_ROOT.joinpath(filter_file).read_text()
    )

    all_questions = [qid for t in filter_dict.values() for qid in t]
    logger.info(f"Total questions={len(all_questions)}")

    val_questions = min(max_val_size, int(.1 * len(all_questions)))
    logger.info(f"{val_questions} questions will be used for validation set")
    val_set_mask = {qid: False for qid in all_questions}
    logger.info("Creating mask")
    rng = np.random.default_rng(seed)
    val_question_indices = rng.choice(len(all_questions), (val_questions,), replace=False)
    for q_idx in val_question_indices:
        val_set_mask[all_questions[q_idx]] = True

    question_path = PROJECT_ROOT.joinpath(dump_path, 'questions')
    train_file = output_path.joinpath(f"{name}.jsonl").open('w')
    val_file = output_path.joinpath(f"{name}_val.jsonl").open('w')

    update_freq = 1000 if debug else 25000

    train_buffer = []

    def empty_buffer(buffer):
        logger.info(f"Emptying Buffer of size {len(buffer)}")
        rng.shuffle(buffer)
        for instance in buffer:
            train_file.write(instance.strip() + '\n')

    logger.info(f"Using buffer of {max_buffer_size}")

    for tag_name, questions in filter_dict.items():
        logger.info(f"Handling tag {tag_name}")
        line_num = 0
        found = 0

        # Use a dict to check if they exist because searching dict O(1)
        questions_looking_for = {k: True for k in questions}

        for line in tqdm(question_path.joinpath(f"{tag_name}.jsonl").open()):
            parsed = json.loads(line)
            line_num += 1
            if parsed['id'] in questions_looking_for:
                questions_looking_for.pop(parsed['id'])
                if val_set_mask[parsed['id']]:
                    val_file.write(line.strip() + "\n")
                else:
                    train_buffer.append(line)
                found += 1

                if len(train_buffer) >= max_buffer_size:
                    empty_buffer(train_buffer)
                    del train_buffer
                    train_buffer = []

            if line_num % update_freq == 0:
                logger.info(f"Finished {line_num}, found {found:>8}/{len(questions)}")
                ram_pct = f"{psutil.virtual_memory()[2]:0.2f}%"
                cpu_pct = f"{psutil.getloadavg()[-1] / os.cpu_count() * 100:0.2f}%"
                logger.info(f"RAM Used={ram_pct:<6} | CPU Used={cpu_pct:<6}")

            if not questions_looking_for:
                logger.info(f"Found all looking for")
                break
    empty_buffer(train_buffer)
    train_file.close()
    val_file.close()
