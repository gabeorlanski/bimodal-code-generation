import json
import argparse
import logging
import shutil
import threading
from collections import defaultdict, Counter
from pathlib import Path
import multiprocessing as mp
import sys
from lxml import etree
from tqdm import tqdm
from unidecode import unidecode

from src.data.parse_so.util import POST_TYPE_TO_STR, log_process

logger = logging.getLogger(__name__)

__all__ = [
    "filter_so_dump"
]


class FilterWorker(mp.Process):
    def __init__(
            self,
            worker_id,
            task_queue,
            result_queue,
            log_queue,
            tag_filter
    ):
        super().__init__()
        self.worker_id = worker_id
        self.tasks = task_queue
        self.results = result_queue
        self.logs = log_queue
        self.tag_filter = tag_filter

    def _log(self, level, message):
        self.logs.put((level, f"WORKER {self.worker_id}: {message}"))

    def run(self):

        completed = 0
        self._log(logging.INFO, "Started")
        while True:
            next_task = self.tasks.get()

            # Poison pill means shutdown.
            if next_task is None:
                self._log(logging.INFO, "Finished")
                self.logs.put(None)
                self.tasks.task_done()
                return

            self.results.put(parse_line(next_task['line_num'], next_task['line'], self.tag_filter))
            self.tasks.task_done()
            completed += 1
            if completed % 10000 == 0:
                self._log(logging.INFO, f"Finished {completed}")


def parse_line(line_number, line, tag_filter):
    result = {
        "line"  : line_number,
        "body"  : None,
        "reason": "PASS"
    }

    # Each line is its own post. If it cannot parse than it is
    # worthless to us.
    try:
        post_dict = etree.XML(line).attrib
    except Exception as e:
        result["reason"] = "PARSE_FAIL"
        return result

    try:
        post_type = int(post_dict['PostTypeId'])
    except ValueError:
        result["reason"] = "PARSE_FAIL"
        return result

    # If the post is neither a question nor an answer, skip
    if post_type not in [1, 2, 4, 5]:
        result['reason'] = "NOT_VALID_TYPE"
        return result

    # Deleted questions do not have a body, so skip them
    if not post_dict['Body']:
        result['reason'] = "NO_BODY"
        return result

    result.update(
        {
            "body"         : unidecode(post_dict['Body']),
            "type"         : post_type,
            "id"           : post_dict['Id'],
            "date"         : post_dict['CreationDate'],
            "score"        : int(post_dict['Score']),
            "comment_count": int(post_dict.get('CommentCount', 0))
        }
    )
    if post_type == 1:
        post_tags = [
            t.replace('<', '').strip()
            for t in post_dict['Tags'].split(">")
            if t.strip()
        ]
        has_a_valid_tag = any(valid_t in t for t in post_tags for valid_t in tag_filter)
        if tag_filter and (not post_tags or not has_a_valid_tag):
            if not post_tags:
                result['reason'] = "NO_VALID_TAG"
            else:
                result['reason'] = "FILTERED_OUT"
            return result
        result.update({
            'tags'           : post_tags,
            'title'          : unidecode(post_dict.get('Title')),
            'answer_count'   : int(post_dict.get('AnswerCount', 0)),
            'views'          : int(post_dict.get('ViewCount', 0)),
            'accepted_answer': post_dict.get('AcceptedAnswerId'),

        })

    else:
        result.update(
            {
                "parent_id": post_dict.get("ParentId")
            }
        )
    return result


def main_process(
        out_dir,
        line_num,
        result_queue,
        workers
):
    post_type_to_file = {}
    for k, v in POST_TYPE_TO_STR.items():
        if k == 'answers':
            continue
        post_type_to_file[k] = out_dir.joinpath(
            f"{v}.jsonl"
        ).open('w', encoding='utf-8')

    answers = defaultdict(list)
    orphaned_count = 0
    valid_questions = {}
    is_valid_question = defaultdict(lambda: True)
    failures = Counter()
    post_type_counts = Counter()
    tag_counts = Counter()
    all_answer_scores_file = out_dir.joinpath('all_answer_scores.txt').open('w')
    all_question_scores_file = out_dir.joinpath('all_question_scores.txt').open('w')
    valid_answer_scores_file = out_dir.joinpath('valid_answer_scores.txt').open('w')
    valid_question_scores_file = out_dir.joinpath('valid_question_scores.txt').open('w')
    pbar = tqdm(total=line_num, desc='Processing')
    while True:
        if all([not worker.is_alive() for worker in workers]):
            break
        if result_queue.qsize() == 0 or result_queue.empty():
            continue
        post_dict = result_queue.get(timeout=5.0)

        post_type = post_dict.get('type', None)
        pbar.update()
        if post_dict['reason'] != "PASS":
            failures[post_dict['reason']] += 1
            if post_dict['reason'] == "PARSE_FAIL":
                logger.error(f"Line {post_dict['line']} failed to parse")

            if post_type is not None and post_type == 1:
                is_valid_question[post_dict['id']] = False
                all_question_scores_file.write(str(post_dict['score']) + '\n')

            continue
        post_dict.pop('reason')
        if post_type == 1:
            valid_questions[post_dict['id']] = post_dict

            all_question_scores_file.write(str(post_dict['score']) + '\n')
            valid_question_scores_file.write(str(post_dict['score']) + '\n')

        elif post_type == 2:
            all_answer_scores_file.write(str(post_dict['score']) + '\n')
            if not is_valid_question[post_dict['parent_id']]:
                orphaned_count += 1
            else:
                answers[post_dict['parent_id']].append(post_dict)

        else:
            post_type_counts[post_type] += 1
            post_type_to_file[post_type].write(json.dumps(post_dict) + '\n')

    pbar.close()

    logger.info("Saving questions")
    for post_id, post_dict in valid_questions.items():
        post_type_counts[1] += 1
        for tag in post_dict['tags']:
            tag_counts[tag] += 1

        post_answers = {}
        for answer_dict in answers.pop(post_id, []):
            valid_answer_scores_file.write(str(answer_dict['score']) + '\n')
            post_answers[answer_dict['id']] = answer_dict
        post_dict['answers'] = post_answers
        post_type_counts[2] += len(post_dict['answers'])
        post_type_to_file[1].write(json.dumps(post_dict) + '\n')

    logger.info("Closing files")
    for k in post_type_to_file:
        post_type_to_file[k].close()

    all_answer_scores_file.close()
    all_question_scores_file.close()
    valid_answer_scores_file.close()
    valid_question_scores_file.close()

    logger.info(f"Saving list of valid question IDs to {out_dir.joinpath('valid_questions.txt')}")
    with out_dir.joinpath('valid_questions.txt').open('w') as f:
        f.write('\n'.join(map(str, valid_questions)))

    post_type_counts = {POST_TYPE_TO_STR[k]: v for k, v in post_type_counts.items()}
    orphaned_answers = sum(map(len, answers.values())) + orphaned_count
    logger.info(f"{len(valid_questions)} total questions found")
    logger.info(f"{orphaned_answers} answers left orphaned.")
    logger.info(f"{sum(failures.values())} were skipped or failed")

    logger.info("Filtered Counts")
    for post_type, c in post_type_counts.items():
        logger.info(f"\t{post_type:>16} = {c}")

    logger.info("Failure Counts")
    for fail, c in failures.items():
        logger.info(f"\t{fail:>16} = {c}")
    dump_stats = {
        'orphaned_answers': orphaned_answers,
        'failures'        : failures,
        'tags'            : tag_counts,
        'post_types'      : post_type_counts
    }

    logger.info(f"Saving dump stats to {out_dir.joinpath('stats.json')}")
    with out_dir.joinpath('stats.json').open('w') as f:
        json.dump(dump_stats, f, indent=True)
    return dump_stats


def filter_so_dump(
        posts_path: Path,
        num_workers,
        out_dir: Path,
        tag_filters,
        debug
):
    logger.info(f"{len(tag_filters)} total tag filters")
    log_queue = mp.JoinableQueue()
    task_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue()
    logger.info(f"Initializing {num_workers} workers")
    workers = [
        FilterWorker(i, task_queue, result_queue, log_queue, tag_filters)
        for i in range(num_workers)
    ]

    logger.info(f"Starting {num_workers} workers")
    for w in workers:
        w.start()

    logger.debug(f"Reading lines from {posts_path}")
    line_num = 0
    logger.info(f"Starting the logging thread")
    log_thread = threading.Thread(
        target=log_process,
        args=(log_queue, num_workers)
    )
    log_thread.start()
    with posts_path.open('r', encoding='utf-8', errors='replace') as posts_file:
        for line in posts_file:
            task_queue.put({'line_num': line_num, 'line': line})

            if (line_num + 1) % 100000 == 0:
                logger.info(f"Read {line_num + 1} lines")
            line_num += 1

            if line_num >= 2500 and debug:
                break

    logger.info(f"{line_num} total lines")

    logger.debug("Putting in poison pills")
    for _ in workers:
        task_queue.put(None)

    logger.debug("Starting result processing loop")
    try:
        dump_stats = main_process(
            out_dir,
            line_num,
            result_queue,
            workers
        )
    except Exception as e:
        logger.error("SOMETHING WENT REALLY WRONG")
        logger.error('Killing workers')
        for worker in workers:
            worker.terminate()
        # Something failed so kill all of the workers then exit
        logger.error("Poisoning the log thread")
        log_queue.put(None)
        stop_thread = True
        # log_thread.join()
        raise e
    log_queue.put('KILL')
    log_thread.join()
    for worker in workers:
        worker.terminate()

    return dump_stats
