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
    "parse_so_dump"
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


def parse_line(line_number, line):
    result = {
        "line"  : line_number,
        "body"  : None,
        "result": "PASS"
    }

    # Each line is its own post. If it cannot parse than it is
    # worthless to us.
    try:
        post_dict = etree.XML(line).attrib
    except Exception as e:
        result["result"] = "PARSE_FAIL"
        return result

    try:
        post_type = int(post_dict['PostTypeId'])
    except ValueError:
        result["result"] = "PARSE_FAIL"
        return result

    # If the post is neither a question nor an answer, skip
    if post_type not in [1, 2, 4, 5]:
        result['result'] = "NOT_VALID_TYPE"
        return result

    # Deleted questions do not have a body, so skip them
    if not post_dict['Body']:
        result['result'] = "NO_BODY"
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
        # has_a_valid_tag = any(valid_t in t for t in post_tags for valid_t in tag_filter)
        # if tag_filter and (not post_tags or not has_a_valid_tag):
        #     if not post_tags:
        #         result['reason'] = "NO_VALID_TAG"
        #     else:
        #         result['reason'] = "FILTERED_OUT"
        #     return result
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


def read_dump(dump_path: Path, debug: bool):
    line_num = 0
    with dump_path.open('r', encoding='utf-8', errors='replace') as dump_file:
        for line in dump_file:
            parsed = parse_line(line_num, line)

            if (line_num + 1) % 100000 == 0:
                logger.info(f"Read {line_num + 1} lines")
            line_num += 1

            yield parsed

            if line_num >= 2500 and debug:
                break


def initial_parse_dump(dump_path: Path, out_dir: Path, debug):
    logger.info(f"Doing initial pass on {dump_path}")
    question_overview_data = {}
    failures_counts = Counter()
    post_type_counter = Counter()
    tag_counts = Counter()
    post_type_to_file = {}
    for k, v in POST_TYPE_TO_STR.items():
        if k in ['answers', 'questions']:
            continue
        post_type_to_file[k] = out_dir.joinpath(
            f"{v}.jsonl"
        ).open('w', encoding='utf-8')
    line_number = 0
    for parsed in read_dump(dump_path, debug):
        line_number += 1
        if parsed['result'] != 'PASS':
            logger.warning(f"Line {line_number} failed to parse")
            failures_counts[parsed['result']] += 1
            continue

        post_type_counter[POST_TYPE_TO_STR[parsed['type']]] += 1
        if parsed['type'] == 1:
            question_overview_data[parsed['id']] = {
                'tags'           : parsed['tags'],
                'score'          : parsed['score'],
                'views'          : parsed['views'],
                'answer_count'   : parsed['answer_count'],
                'accepted_answer': parsed['accepted_answer'],

            }
            for t in parsed['tags']:
                tag_counts[t] += 1
        elif parsed['type'] != 2:
            post_type_to_file[parsed['type']].write(
                json.dumps(parsed) + '\n'
            )

    logger.info("Closing files")
    for k in post_type_to_file:
        post_type_to_file[k].close()

    logger.info(f"{sum(failures_counts.values())} were skipped or failed")
    logger.info("Filtered Counts")
    for post_type, c in post_type_counter.items():
        logger.info(f"\t{post_type:>16} = {c}")

    logger.info("Failure Counts")
    for fail, c in failures_counts.items():
        logger.info(f"\t{fail:>16} = {c}")

    logger.info(f"Saving Stats to {out_dir.joinpath('stats.json')}")

    logger.info(f"Saving dump stats to {out_dir.joinpath('stats.json')}")
    dump_stats = {
        'post_types': post_type_counter,
        'failures'  : failures_counts,
        'tag_counts': tag_counts
    }

    logger.info(f"Saving question overview to {out_dir.joinpath('question_overview.json')}")
    with out_dir.joinpath('question_overview.json').open('w') as f:
        json.dump(question_overview_data, f)

    return question_overview_data, tag_counts, line_number, dump_stats


def second_parse_dump(
        dump_path: Path,
        out_dir: Path,
        question_overview_data,
        tag_counts,
        total_line_count,
        debug
):
    logger.info("Starting second pass")
    question_dir = out_dir.joinpath(f'questions')
    if not question_dir.exists():
        question_dir.mkdir(parents=True)
    else:
        shutil.rmtree(question_dir)
        question_dir.mkdir()

    tag_file_descriptors = {}
    posts_per_tag = Counter()
    no_tags = 0
    for parsed in tqdm(read_dump(dump_path, debug), desc='Second Pass', total=total_line_count):
        if parsed['result'] != 'PASS' or parsed['type'] not in [1, 2]:
            continue

        if parsed['type'] == 1:
            if not parsed.get('tags', []):
                tag_to_use = 'NO_TAG'
                no_tags += 1
            else:
                tag_to_use = max(parsed['tags'], key=lambda t: tag_counts[t])

            if tag_to_use not in tag_file_descriptors:
                logger.info(f"Creating File for {tag_to_use}")
                tag_file_descriptors[tag_to_use] = question_dir.joinpath(
                    f'{tag_to_use}.jsonl').open('w')
        else:
            try:
                tags_for_answer = question_overview_data[parsed['parent_id']].get('tags', [])
            except KeyError:
                logger.error(
                    f"{parsed['id']} has a parent ({parsed['parent_id']=})that does not exist")
                continue
            if not tags_for_answer:
                tag_to_use = 'NO_TAG'
                no_tags += 1
            else:
                tag_to_use = max(tags_for_answer, key=lambda t: tag_counts[t])

        posts_per_tag[tag_to_use] += 1
        tag_file_descriptors[tag_to_use].write(json.dumps(parsed) + '\n')

    logger.info(f"{len(posts_per_tag)} total tag files created")
    logger.info(f"{no_tags} had no tags")
    logger.info(f"Breakdown of the top {min(25, len(posts_per_tag))}")
    for k, v in posts_per_tag.most_common(min(25, len(posts_per_tag))):
        logger.info(f"\t{k:>32}={v}")

    for v in tag_file_descriptors.values():
        v.close()

    return posts_per_tag


def align_tag_file(tag_file: Path):
    question_dict = {}
    orphaned_children = defaultdict(dict)
    with tag_file.open('r') as f:
        for line in map(json.loads, f):
            if line['type'] == 1:
                question_dict[line['id']] = {
                    'answers': orphaned_children[line['id']],
                    **line
                }
            else:
                orphaned_children[line['parent_id']][line['id']] = line

    orphans = 0
    for k, v in orphaned_children.items():
        if k in question_dict:
            question_dict[k]['answers'].update(v)
        else:
            orphans += len(v)

    with tag_file.open('w') as f:
        for v in question_dict.values():
            f.write(json.dumps(v) + '\n')

    return tag_file.stem, orphans


def parse_so_dump(
        posts_path: Path,
        num_workers,
        out_dir: Path,
        tag_filters,
        debug
):
    # logger.info(f"{len(tag_filters)} total tag filters")
    # log_queue = mp.JoinableQueue()
    # task_queue = mp.JoinableQueue()
    # result_queue = mp.JoinableQueue()
    # logger.info(f"Initializing {num_workers} workers")
    # workers = [
    #     FilterWorker(i, task_queue, result_queue, log_queue, tag_filters)
    #     for i in range(num_workers)
    # ]
    #
    # logger.info(f"Starting {num_workers} workers")
    # for w in workers:
    #     w.start()
    #
    # logger.debug(f"Reading lines from {posts_path}")
    # line_num = 0
    # logger.info(f"Starting the logging thread")
    # log_thread = threading.Thread(
    #     target=log_process,
    #     args=(log_queue, num_workers)
    # )
    # log_thread.start()
    # with posts_path.open('r', encoding='utf-8', errors='replace') as posts_file:
    #     for line in posts_file:
    #         task_queue.put({'line_num': line_num, 'line': line})
    #
    #         if (line_num + 1) % 100000 == 0:
    #             logger.info(f"Read {line_num + 1} lines")
    #         line_num += 1
    #
    #         if line_num >= 2500 and debug:
    #             break
    #
    # logger.info(f"{line_num} total lines")
    #
    # logger.debug("Putting in poison pills")
    # for _ in workers:
    #     task_queue.put(None)
    #
    # logger.debug("Starting result processing loop")
    # try:
    #     dump_stats = main_process(
    #         out_dir,
    #         line_num,
    #         result_queue,
    #         workers
    #     )
    # except Exception as e:
    #     logger.error("SOMETHING WENT REALLY WRONG")
    #     logger.error('Killing workers')
    #     for worker in workers:
    #         worker.terminate()
    #     # Something failed so kill all of the workers then exit
    #     logger.error("Poisoning the log thread")
    #     log_queue.put(None)
    #     stop_thread = True
    #     # log_thread.join()
    #     raise e
    # log_queue.put('KILL')
    # log_thread.join()
    # for worker in workers:
    #     worker.terminate()

    question_overview_data, tag_counts, total_line_count, dump_stats = initial_parse_dump(
        posts_path,
        out_dir=out_dir,
        debug=debug
    )

    tag_files = second_parse_dump(
        posts_path,
        out_dir,
        question_overview_data,
        tag_counts,
        total_line_count,
        debug
    )

    tag_files = [out_dir.joinpath('questions', f"{t}.jsonl") for t in tag_files]
    logger.info(f"Aligning {len(tag_files)} tag files")

    orphans_by_tag = {}

    with mp.Pool(num_workers) as pool:
        for result in tqdm(pool.imap(align_tag_file, tag_files), total=len(tag_files),
                           desc='Aligning'):
            tag, orphan = result
            orphans_by_tag[tag] = orphan

    logger.info(f"{sum(orphans_by_tag.values())} total orphaned children")
    dump_stats['orphans'] = orphans_by_tag
    with out_dir.joinpath('stats.json').open('w') as f:
        json.dump(
            dump_stats,
            f,
            indent=True
        )

    return dump_stats
