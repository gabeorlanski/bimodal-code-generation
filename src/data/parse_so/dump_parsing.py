import json
import logging
import os
import shutil
from collections import defaultdict, Counter
from functools import partial
from pathlib import Path
import multiprocessing as mp

from datetime import datetime
import psutil
import ujson
from lxml import etree
from tqdm import tqdm
from unidecode import unidecode

from src.data.parse_so.util import POST_TYPE_TO_STR
from src.common import get_estimated_time_remaining

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

        if not post_dict.get('Title'):
            result['result'] = 'NO_TITLE'
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


def read_dump(dump_path: Path, debug: bool, expected_total_lines: int):
    line_num = 0
    start_time = datetime.utcnow()
    update_freq = 2500 if debug else 250000
    with dump_path.open('r', encoding='utf-8', errors='replace') as dump_file:
        for line in dump_file:
            parsed = parse_line(line_num, line)

            yield parsed

            line_num += 1
            if line_num % update_freq == 0:
                hours, minutes, seconds = get_estimated_time_remaining(
                    start_time,
                    line_num,
                    expected_total_lines
                )

                logger.info(
                    f"Completed {line_num:>16}. "
                    f"Estimated to finish 100M in {str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}")
                ram_pct = f"{psutil.virtual_memory()[2]:0.2f}%"
                cpu_pct = f"{psutil.getloadavg()[-1] / os.cpu_count() * 100:0.2f}%"
                logger.debug(f"RAM Used={ram_pct:<6} | CPU Used={cpu_pct:<6}")


def initial_parse_dump(dump_path: Path, out_dir: Path, debug):
    logger.info(f"Doing initial pass on {dump_path}")

    question_overview_data = {}
    failures_counts = Counter()
    post_type_counter = Counter()
    tag_counts = Counter()
    post_type_to_file = {}
    for k, v in POST_TYPE_TO_STR.items():
        post_type_to_file[k] = out_dir.joinpath(
            f"{v}.jsonl"
        ).open('w', encoding='utf-8')
    line_number = 0

    for parsed in read_dump(dump_path, debug, 100000000):
        line_number += 1
        if parsed['result'] != 'PASS':
            failures_counts[parsed['result']] += 1
            continue

        post_type_counter[POST_TYPE_TO_STR[parsed['type']]] += 1
        post_type_to_file[parsed['type']].write(json.dumps(parsed) + '\n')
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

    return question_overview_data, tag_counts, dump_stats


def create_question_tag_files(
        questions_path: Path,
        out_dir: Path,
        question_overview_data,
        debug
):
    logger.info("Parsing Questions into Tag files")
    tmp_dir = out_dir.joinpath(f'tmp_questions')
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)
    else:
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()

    max_files_open = 1000 if not debug else 16
    update_freq = 1000 if not debug else 100000

    tag_file_descriptors = {}
    created_files = {}

    posts_per_tag = Counter()
    no_tags = 0
    completed = 0
    start_time = datetime.utcnow()

    for line in questions_path.open('r'):
        parsed = json.loads(line)

        if parsed['result'] != 'PASS' or parsed['type'] not in [1, 2]:
            continue

        if not parsed.get('tags', []):
            tag_to_use = 'NO_TAG'
        else:
            tag_to_use = parsed['tags'][0]

        if tag_to_use not in tag_file_descriptors:
            if tag_to_use not in created_files:
                created_files[tag_to_use] = tmp_dir.joinpath(f'{tag_to_use}.jsonl')
                tag_file_descriptors[tag_to_use] = created_files[tag_to_use].open('w')
            else:
                tag_file_descriptors[tag_to_use] = created_files[tag_to_use].open('a')

        tag_file_descriptors[tag_to_use].write(json.dumps(parsed) + '\n')
        if len(tag_file_descriptors) >= max_files_open:
            # IF we have too many files open at once, we will get an error.
            for v in tag_file_descriptors.values():
                v.close()
            tag_file_descriptors = {}

        posts_per_tag[tag_to_use] += 1
        no_tags += tag_to_use == "NO_TAG"
        question_overview_data[parsed['id']]['tag_to_use'] = tag_to_use
        completed += 1
        if completed % update_freq == 0:
            hours, minutes, seconds = get_estimated_time_remaining(
                start_time,
                completed,
                len(question_overview_data)
            )

            logger.info(
                f"Completed {completed:>10}/{len(question_overview_data)}. "
                f"Estimated to finish in {str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}")
    for v in tag_file_descriptors.values():
        v.close()

    logger.info(f"{completed} total questions")
    logger.info(f"{len(posts_per_tag)} total question files created")
    logger.info(f"{no_tags} had no tags")
    logger.info(f"Breakdown of the top {min(25, len(posts_per_tag))}")
    for k, v in posts_per_tag.most_common(min(25, len(posts_per_tag))):
        logger.info(f"\t{k:>32}={v}")

    return tmp_dir, posts_per_tag, question_overview_data


def create_answer_tag_files(
        answers_path,
        out_dir,
        question_overview_data,
        answer_count,
        debug
):
    logger.info("Parsing Answers into Tag files")
    tmp_dir = out_dir.joinpath(f'tmp_answers')
    if not tmp_dir.exists():
        tmp_dir.mkdir(parents=True)
    else:
        shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()

    max_files_open = 1000 if not debug else 16
    update_freq = 1000 if not debug else 100000

    tag_file_descriptors = {}
    created_files= {}
    posts_per_tag = Counter()
    no_tags = 0
    start_time = datetime.utcnow()
    completed = 0
    for line in answers_path.open('r'):
        parsed = json.loads(line)
        try:
            parsed['tags'] = question_overview_data[parsed['parent_id']].get('tags', [])
            tag_name_to_use = question_overview_data[parsed['parent_id']].get('tag_to_use', None)
        except KeyError:
            logger.error(
                f"{parsed['id']} has a parent ({parsed['parent_id']=})that does not exist")
            continue

        if tag_name_to_use is None:
            if not parsed.get('tags', []):
                tag_to_use = 'NO_TAG'
                no_tags += 1
            else:
                tag_to_use = parsed['tags'][0]
        else:
            tag_to_use = tag_name_to_use

        if tag_to_use not in tag_file_descriptors:
            if tag_to_use not in created_files:
                created_files[tag_to_use] = tmp_dir.joinpath(f'{tag_to_use}.jsonl')
                tag_file_descriptors[tag_to_use] = created_files[tag_to_use].open('w')
            else:
                tag_file_descriptors[tag_to_use] = created_files[tag_to_use].open('a')

        tag_file_descriptors[tag_to_use].write(json.dumps(parsed) + '\n')
        if len(tag_file_descriptors) >= max_files_open:
            # IF we have too many files open at once, we will get an error.
            for v in tag_file_descriptors.values():
                v.close()
            tag_file_descriptors = {}
        posts_per_tag[tag_to_use] += 1
        completed += 1
        if completed % update_freq == 0:
            hours, minutes, seconds = get_estimated_time_remaining(
                start_time,
                completed=completed,
                total=answer_count
            )

            logger.info(
                f"Completed {completed:>10}/{answer_count}. "
                f"Estimated to finish in "
                f"{str(hours).zfill(2)}:{str(minutes).zfill(2)}:{str(seconds).zfill(2)}")

    for v in tag_file_descriptors.values():
        v.close()
    logger.info(f"{completed} total answers")
    logger.info(f"{len(posts_per_tag)} total answer files created")
    logger.info(f"{no_tags} had no tags")
    logger.info(f"Breakdown of the top {min(25, len(posts_per_tag))}")
    for k, v in posts_per_tag.most_common(min(25, len(posts_per_tag))):
        logger.info(f"\t{k:>32}={v}")
    return tmp_dir, posts_per_tag


def align_tag_file(tag_file_name, out_dir, question_dir, answer_dir):
    question_dict = {}
    with question_dir.joinpath(tag_file_name).open('r') as f:
        for line in map(json.loads, f):
            question_dict[line['id']] = {
                'answers': {},
                **line
            }

    orphans = 0
    answer_file = answer_dir.joinpath(tag_file_name)
    if answer_file.exists():
        with answer_file.open('r') as f:
            for line in map(json.loads, f):
                if line['parent_id'] in question_dict:
                    question_dict[line['parent_id']]['answers'][line['id']] = line
                else:
                    orphans += 1

    output_file = out_dir.joinpath(tag_file_name)
    with output_file.open('w') as f:
        for v in question_dict.values():
            f.write(json.dumps(v) + '\n')

    return output_file.stem, orphans


def parse_so_dump(
        posts_path: Path,
        num_workers,
        out_dir: Path,
        debug
):
    question_overview_data, tag_counts, dump_stats = initial_parse_dump(
        posts_path,
        out_dir=out_dir,
        debug=debug
    )

    if posts_path.parent.joinpath('Tags.xml').exists():
        tag_counts = {}
        for line in posts_path.parent.joinpath('Tags.xml').open('r'):
            try:
                post_dict = etree.XML(line).attrib
            except Exception as e:
                continue

            tag_counts[post_dict['TagName']] = post_dict['Count']

    tmp_question_dir, question_posts_per_tag, question_overview_data = create_question_tag_files(
        out_dir.joinpath('questions.jsonl'),
        out_dir,
        question_overview_data,
        debug
    )
    tmp_answer_dir, answer_posts_per_tag = create_answer_tag_files(
        out_dir.joinpath('answers.jsonl'),
        out_dir,
        question_overview_data,
        answer_count=dump_stats['post_types']['answers'],
        debug=debug
    )

    for k, v in answer_posts_per_tag.items():
        if k not in question_posts_per_tag:
            logger.error(f"Found unknown tag file {k}")
        else:
            question_posts_per_tag[k] += v

    logger.info(f"Saving question overview to {out_dir.joinpath('question_overview.json')}")
    with out_dir.joinpath('question_overview.json').open('w') as f:
        ujson.dump(question_overview_data, f)

    os.remove(out_dir.joinpath('questions.jsonl'))
    os.remove(out_dir.joinpath('answers.jsonl'))
    if out_dir.joinpath('questions').exists():
        shutil.rmtree(out_dir.joinpath('questions'))

    out_dir.joinpath('questions').mkdir()
    align_fn = partial(
        align_tag_file,
        out_dir=out_dir.joinpath('questions'),
        question_dir=tmp_question_dir,
        answer_dir=tmp_answer_dir
    )
    tag_files = [
        f"{t}.jsonl"
        for t in question_posts_per_tag
    ]
    logger.info(f"Aligning {len(tag_files)} tag files")

    orphans_by_tag = {}

    with mp.Pool(num_workers) as pool:
        for result in tqdm(
                pool.imap(align_fn, tag_files), total=len(tag_files),
                desc='Aligning'
        ):
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

    logger.info(f"Cleaning temporary files")
    shutil.rmtree(tmp_question_dir)
    shutil.rmtree(tmp_answer_dir)

    return dump_stats
