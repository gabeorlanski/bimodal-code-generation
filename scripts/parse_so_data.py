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

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src import config
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.parse_so import FilterWorker


def log_process(log_queue, worker_count, stop_func):
    logger = logging.getLogger('parse_so.log_thread')
    finished = 0
    while True:
        try:
            message = log_queue.get(timeout=2.0)
        except Exception:
            continue
        if message is None or stop_func():
            finished += 1
            logger.debug(f'Finished is at {finished}')
            if finished >= worker_count:
                logger.info("Log Thread is done.")
                return
            continue

        level, message = message
        logger.log(level, message)


def main_process(
        logger,
        out_dir,
        line_num,
        result_queue,
        workers
):
    post_type_to_str = {
        1: "questions",
        2: "answers",
        4: "wiki_excerpts",
        5: "wiki"
    }
    post_type_to_file = {}
    for k, v in post_type_to_str.items():
        post_type_to_file[k] = out_dir.joinpath(
            f"{v}.jsonl"
        ).open('w', encoding='utf-8')

    answers = defaultdict(list)
    valid_questions = {}
    failures = Counter()
    post_type_counts = {k: 0 for k in post_type_to_str}
    tag_counts = Counter()
    pbar = tqdm(total=line_num, desc='Processing')
    while True:
        if all([not worker.is_alive() for worker in workers]):
            break
        if result_queue.qsize() == 0 or result_queue.empty():
            continue
        post_dict = result_queue.get(timeout=5.0)

        pbar.update()
        if post_dict['reason'] != "PASS":
            failures[post_dict['reason']] += 1
            if post_dict['reason'] == "PARSE_FAIL":
                logger.error(f"Line {post_dict['line']} failed to parse")

            continue
        post_dict.pop('reason')
        post_type = post_dict['type']
        if post_type == 1:
            post_type_counts[post_type] += 1
            post_type_to_file[post_type].write(json.dumps(post_dict) + '\n')
            valid_questions[post_dict['id']] = True
            q_answers = answers.pop(post_dict['id'], [])
            for t in post_dict['tags']:
                tag_counts[t] += 1
            for answer in q_answers:
                post_type_counts['answers'] += 1
                post_type_to_file[2].write(json.dumps(answer) + '\n')
        elif post_type == 2:
            if post_dict['parent_id'] not in valid_questions:
                answers[post_dict['parent_id']].append(post_dict)
            else:
                post_type_counts[post_type] += 1
                post_type_to_file[post_type].write(json.dumps(post_dict) + '\n')

        else:
            post_type_counts[post_type] += 1
            post_type_to_file[post_type].write(json.dumps(post_dict) + '\n')

    pbar.close()

    for k in post_type_to_file:
        post_type_to_file[k].close()

    logger.info(f"Saving list of valid question IDs to {out_dir.joinpath('valid_questions.txt')}")
    with out_dir.joinpath('valid_questions.txt').open('w') as f:
        f.write('\n'.join(map(str, valid_questions)))

    post_type_counts = {post_type_to_str[k]: v for k, v in post_type_counts.items()}
    orphaned_answers = sum(map(len, answers.values()))
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


def process_file(
        logger,
        posts_path: Path,
        num_workers,
        out_dir: Path,
        tag_filters,
        debug
):
    logger.info(f"{len(tag_filters)} total tag filters")
    log_queue = mp.Queue()
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
    stop_thread = False
    log_thread = threading.Thread(
        target=log_process,
        args=(log_queue, num_workers, lambda: stop_thread)
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
            logger,
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

    log_thread.join()
    for worker in workers:
        worker.terminate()

    return dump_stats


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
    dump_name = path_to_dump.stem.split(".")[0]
    output_path = output_path.joinpath(dump_name)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    failures = process_file(
        logger,
        posts_path,
        num_workers,
        output_path,
        tag_filters,
        debug
    )

    logger.info(f"Saving stats for '{dump_name}' to {output_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    with output_path.joinpath(f'{dump_name}_failures.json').open('w', encoding='utf-8') as f:
        json.dump(failures, f, indent=True)


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
