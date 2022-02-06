import json
import logging
import random
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Dict, Callable
import multiprocessing as mp
from tqdm import tqdm
from bs4 import BeautifulSoup
from src.data.parse_so.util import POST_TYPE_TO_STR, log_process

logger = logging.getLogger(__name__)
__all__ = [
    "QuestionFilter",
    "filter_and_parse_so_posts"
]


class QuestionFilterException(Exception):

    def __init__(self, msg):
        super(QuestionFilterException, self).__init__(msg)


@dataclass()
class QuestionFilter:
    maximum_score: float = float('inf')
    minimum_score: float = float('-inf')
    must_have_answer: bool = False
    use_question_score: bool = False
    only_question_body: bool = False
    word_whitelist: List[str] = field(default_factory=list)

    def str_contains_whitelist_words(self, sequence):
        return any([w in sequence for w in self.word_whitelist])

    def __call__(self, post_dict: Dict):

        post_type = post_dict['type']
        if isinstance(post_type, int):
            # Saved in the original w/ an int value because it was. Takes a LONG
            # time to rerun the original so just leave it as is.
            try:
                post_type = POST_TYPE_TO_STR[post_type]
            except KeyError:
                raise QuestionFilterException(
                    f'{post_dict["id"]} has an invalid type of {post_dict["type"]}'
                )

        sequences_to_test = [post_dict['title'], post_dict['body']]
        if not self.only_question_body:
            sequences_to_test.extend([d['body'] for d in post_dict['answers'].values()])

        if self.word_whitelist:
            found = False
            for seq in sequences_to_test:
                if self.str_contains_whitelist_words(seq.lower()):
                    found = True
                    break
            if not found:
                return False

        # Only questions, answers, and comments have scores. IFF they are not
        # one of those three and they pass the previous filters, they must be
        # valid.
        if post_type not in ["questions", "answers"]:
            return False

        if self.must_have_answer and len(post_dict['answers']) == 0:
            return False

        if self.use_question_score:
            return self.minimum_score <= post_dict['score'] <= self.maximum_score

        answer_scores_pass_range = [
            self.minimum_score <= d['score'] <= self.maximum_score
            for d in post_dict['answers'].values()
        ]

        # Already have a check above for if children are present, and
        # any([])=False, but we will allow it.
        return any(answer_scores_pass_range)


class CleaningProcessor(mp.Process):
    def __init__(
            self,
            worker_id,
            task_queue,
            result_queue,
            log_queue,
            clean_fn,
            filter_fn
    ):
        super().__init__()
        self.worker_id = worker_id
        self.tasks = task_queue
        self.results = result_queue
        self.logs = log_queue
        self.clean_fn = clean_fn
        self.filter_fn = filter_fn

    def _log(self, level, message):
        self.logs.put((level, f"WORKER {self.worker_id}: {message}"))

    def run(self):
        completed = 0

        self._log(logging.INFO, "Started")
        while True:
            next_task = self.tasks.get()

            # Poison pill means shutdown.
            if next_task is None:
                self.tasks.task_done()
                self._log(logging.INFO, "Finished")
                return

            line_num, question_dict = next_task

            if not self.filter_fn(question_dict):
                self.results.put((False, question_dict))
            else:
                self.results.put((True, self.clean_fn(question_dict)))

            self.tasks.task_done()
            completed += 1
            if completed % 5000 == 0:
                self._log(logging.INFO, f"Finished {completed}")


def basic_cleaning(ex: Dict) -> Dict:
    soup = BeautifulSoup(ex['body'], 'lxml')
    ex['body'] = soup.text

    for k in ex['answers'].keys():
        soup = BeautifulSoup(ex['answers'][k]['body'], 'lxml')
        ex['answers'][k]['body'] = soup.text

    return ex


CLEANING_NAME_TO_FN = {
    "BASE": basic_cleaning
}


def main_process(questions_path, workers, task_queue, result_queue):
    line_num = 0
    with questions_path.open('r', encoding='utf-8', errors='replace') as questions_file:
        for line in questions_file:
            task_queue.put((line_num, json.loads(line.strip())))

            if (line_num + 1) % 25000 == 0:
                logger.info(f"Read {line_num + 1} lines")
            line_num += 1

    for _ in workers:
        task_queue.put(None)
    logger.debug(f"{result_queue.qsize()=}")
    logger.info(f"Processing {line_num} lines")

    results_processed = 0
    questions_filtered_out = []
    questions_saved = 0

    to_save = []
    pbar = tqdm(total=line_num, desc='Processing')
    while True:
        if all([not worker.is_alive() for worker in workers]):
            break
        if result_queue.qsize() == 0 or result_queue.empty():
            continue
        passed_filter, result_dict = result_queue.get(timeout=5.0)

        if not passed_filter:
            questions_filtered_out.append(result_dict['id'])
        else:
            to_save.append(result_dict)
            questions_saved += 1
        results_processed += 1
        if results_processed % 25000 == 0:
            logger.info(
                f"Processed {results_processed}. "
                f"{questions_saved} Saved and "
                f"{len(questions_filtered_out)} filtered out."
            )
        pbar.update()
    pbar.close()

    logger.info(f"Saved {questions_saved} questions")
    logger.info(f"Filtered out {len(questions_filtered_out)} questions")
    return to_save


def filter_and_parse_so_posts(
        path_to_posts: Path,
        out_file: Path,
        validation_path: Path,
        num_workers: int,
        clean_fn_name: str,
        question_filter: QuestionFilter,
        validation_pct: int
):
    questions_path = Path(path_to_posts).joinpath('questions.jsonl')
    logger.info(f"Parsing posts from {path_to_posts}")
    # Setup the queues
    log_queue = mp.Queue()
    task_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue()

    logger.info(f"Getting cleaner {clean_fn_name}")
    clean_fn = CLEANING_NAME_TO_FN[clean_fn_name]

    # Setup processors
    processor_init_fn = partial(
        CleaningProcessor,
        task_queue=task_queue,
        result_queue=result_queue,
        log_queue=log_queue,
        clean_fn=clean_fn,
        filter_fn=question_filter
    )
    logger.info(f"Creating {num_workers} workers")
    workers = [processor_init_fn(i) for i in range(num_workers)]
    log_thread = threading.Thread(
        target=log_process,
        args=(log_queue, num_workers)
    )
    log_thread.start()
    for w in workers:
        w.start()
    try:
        to_save = main_process(
            questions_path,
            workers,
            task_queue,
            result_queue
        )
    except Exception as e:
        # Making sure we cleanup
        for w in workers:
            w.terminate()
        log_queue.put('KILL')
        log_thread.join()
        raise e

    log_queue.put('KILL')
    log_thread.join()
    for w in workers:
        w.terminate()

    validation_size = min(int(validation_pct * len(to_save)), 500)
    logger.info(f"Saving {validation_size} to {validation_path}")
    logger.info(f"Saving {len(to_save) - validation_size} to {out_file}")
    validation_indices = random.sample(range(len(to_save)), validation_size)

    output_fd = out_file.open('w', encoding='utf-8')
    val_fd = validation_path.open('w', encoding='utf-8')
    for i, v in enumerate(to_save):
        if i in validation_indices:
            val_fd.write(json.dumps(v) + '\n')
        else:
            output_fd.write(json.dumps(v) + '\n')
    output_fd.close()
    val_fd.close()
