import json
import argparse
import logging
import shutil
import threading
from collections import defaultdict
from pathlib import Path
import multiprocessing as mp
import sys
from lxml import etree
from tqdm import tqdm
from unidecode import unidecode

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src import config
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist


class PostParser(mp.Process):
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
            if completed % 100 == 0:
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
    if post_type not in [1, 2]:
        result['reason'] = "NOT_QA"
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

        if not has_a_valid_tag and tag_filter:
            result['reason'] = "NO_VALID_TAG"
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


def log_process(log_queue, worker_count):
    logger = logging.getLogger('parse_so.log_thread')
    finished = 0
    while True:
        message = log_queue.get()
        if message is None:
            finished += 1
            logger.debug(f'Finished is at {finished}')
            if finished >= worker_count:
                logger.info("Log Thread is done.")
                return
            continue

        level, message = message
        logger.log(level, message)


def process_file(logger, posts_path, num_workers, tag_filters, debug):
    logger.info(f"{len(tag_filters)} total tag filters")
    log_queue = mp.Queue()
    task_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue()
    logger.info(f"Initializing {num_workers} workers")
    workers = [
        PostParser(i, task_queue, result_queue, log_queue, tag_filters)
        for i in range(num_workers)
    ]

    logger.info(f"Starting {num_workers} workers")
    for w in workers:
        w.start()

    logger.info(f"Starting the logging thread")
    log_thread = threading.Thread(target=log_process, args=(log_queue, num_workers))
    log_thread.start()

    logger.debug(f"Reading lines from {posts_path}")
    total_lines = 0
    for line_num, line in enumerate(posts_path.open('r', encoding='utf-8').readlines()):
        task_queue.put({'line_num': line_num, 'line': line})
        if (line_num + 1) % 1000 == 0:
            logger.info(f"Read {line_num + 1} lines")
        total_lines += 1
        if total_lines >= 2500 and debug:
            break
    logger.info(f"{total_lines} total lines")

    logger.debug("Putting in poison pills")
    for _ in workers:
        task_queue.put(None)

    logger.debug("Starting result processing loop")
    answers = defaultdict(dict)
    questions = {}
    failures = defaultdict(list)
    failure_count = 0
    pbar = tqdm(total=total_lines, desc='Processing')
    while True:
        if all([not worker.is_alive() for worker in workers]):
            break
        if result_queue.qsize() == 0 or result_queue.empty():
            continue
        post_dict = result_queue.get(timeout=5.0)

        pbar.update()
        if post_dict['reason'] != "PASS":
            failures[post_dict['reason']].append(post_dict['line'])
            failure_count += 1
            if post_dict['reason'] == "PARSE_FAIL":
                logger.error(f"Line {post_dict['line']} failed to parse")

            continue
        post_dict.pop('reason')
        post_type = post_dict.pop('type')
        post_id = post_dict['id']
        if post_type == 1:
            questions[post_id] = post_dict
        else:
            answers[post_dict['parent_id']][post_id] = post_dict

    pbar.close()
    log_thread.join()
    for worker in workers:
        worker.terminate()

    logger.info("Joining Answers with Questions")
    for qid in tqdm(questions):
        try:
            questions[qid]['answers'] = answers.pop(qid)
        except KeyError:
            logger.debug(f"Question {qid} had no answers")
            questions[qid]['answers'] = {}

        # Debug may cutoff answers b/c of the total line cutoff
        if not debug:
            assert len(questions[qid]['answers']) == questions[qid]['answer_count']
            assert len(answers[qid]) == 0
    logger.info(f"{len(questions)} total questions found")
    logger.info(f"{sum(map(len, answers.values()))} answers left orphaned.")
    logger.info(f"{failure_count} were skipped or failed")
    logger.info("Failure Counts")
    for fail, c in failures.items():
        logger.info(f"\t{fail:>16} = {len(c)}")

    return questions, failures


def main(
        path_to_dump,
        num_workers,
        tag_filter_file,
        output_path,
        debug
):
    setup_global_logging("parse_so", str(PROJECT_ROOT.joinpath('logs')), debug=debug)
    logger = logging.getLogger('parse_so')
    output_path = PROJECT_ROOT.joinpath(output_path)
    path_to_dump = PROJECT_ROOT.joinpath(path_to_dump)
    logger.info(f"Starting parse_so with inputs {path_to_dump} "
                f"and outputting to {output_path}")
    try:
        posts_path, *_ = validate_files_exist(
            path_to_dump, ["Posts.xml"]
        )
    except FileExistsError as e:
        logger.error(f"Missing '{e.file}' in '{path_to_dump.resolve()}' ")
        raise e

    logger.info(f"Reading tags filters from {tag_filter_file}")
    tag_filters = PROJECT_ROOT.joinpath(tag_filter_file).read_text('utf-8').splitlines(False)
    questions, failures = process_file(logger, posts_path, num_workers, tag_filters, debug)

    dump_name = path_to_dump.stem.split(".")[0]
    logger.info(f"Saving '{dump_name}' to {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with output_path.joinpath(f'{dump_name}_failures.json').open('w', encoding='utf-8') as f:
        json.dump(failures, f, indent=True)
    with output_path.joinpath(f'{dump_name}.jsonl').open('w', encoding='utf-8') as f:
        for question in questions.values():
            f.write(json.dumps(question) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_dump', metavar="<Path to the SO Dump>",
                        help="Path to the dump with the XML files. Must have "
                             "the Post.xml file.")
    parser.add_argument('workers', metavar="<Number of workers>", type=int)
    parser.add_argument('tag_filter_file', metavar="<Text file with filter strings>")
    parser.add_argument('--output-path', '-out', default='data/stack_exchange',
                        help='Path where the outputs will be saved. '
                             'Defaults to /data/stack_exchange')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Debug mode.')
    argv = parser.parse_args()
    main(
        argv.path_to_dump,
        argv.workers,
        argv.tag_filter_file,
        argv.output_path,
        argv.debug
    )
