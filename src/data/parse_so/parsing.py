import logging
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import multiprocessing as mp
import enum
from src.common import PathType, validate_files_exist
from src.data.parse_so.util import POST_TYPE_TO_STR, NAME_TO_POST_TYPE, log_process

logger = logging.getLogger(__name__)
__all__ = [
    "QuestionFilter",
    "filter_and_parse_so_posts"
]


class QuestionState(enum.Enum):
    FAILED_FILTER = enum.auto()
    PASSED_FILTER = enum.auto()
    FINISHED = enum.auto()


class QuestionFilterException(Exception):

    def __init__(self, msg):
        super(QuestionFilterException, self).__init__(msg)


@dataclass()
class QuestionFilter:
    maximum_score: float = float('inf')
    minimum_score: float = float('-inf')
    word_whitelist: List[str] = field(default_factory=list)

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

        # Only questions, answers, and comments have scores. IFF they are not
        # one of those three and they pass the previous filters, they must be
        # valid.
        if post_type not in ["questions", "answers"]:
            return True

        return self.minimum_score <= post_dict['score'] <= self.maximum_score


class ResultProcessor(mp.Process):
    def __init__(self, result_queue, log_queue, output_path):
        super().__init__()
        self.results = result_queue
        self.logs = log_queue
        self.output_path = output_path

    def _log(self, level, message):
        self.logs.put((level, f"RESULT PROCESSOR: {message}"))

    def run(self) -> None:
        self._log(logging.INFO, 'Starting')
        results_processed = 0
        while True:
            next_result = self.results.get(timeout=3.0)
            if next_result is None:
                self._log(logging.INFO, 'Finished')
                self.results.task_done()
                break
            self.results.task_done()
            results_processed += 1
            if results_processed % 1000 == 0:
                self._log(logging.INFO, f"Processed {results_processed}")


def filter_and_parse_so_posts(
        path_to_posts: PathType,
        out_path: PathType,
        num_workers: int,
        clean_fn: Callable,
        question_filter: QuestionFilter,
):
    post_type_whitelist = ['questions', 'answers']
    logger.debug(f"Validating that {post_type_whitelist} files are present at {path_to_posts}")
    questions_path, answers_path = validate_files_exist(
        path_to_posts,
        [f"{p}.jsonl" for p in post_type_whitelist]
    )

    # Setup the queues
    log_queue = mp.JoinableQueue()
    results_queue = mp.JoinableQueue()

    # Setup processors
    result_processor = ResultProcessor(
        log_queue=log_queue,
        result_queue=results_queue,
        output_path=out_path
    )

    result_processor.start()
    stop_thread = False
    log_thread = threading.Thread(
        target=log_process,
        args=(log_queue, num_workers, lambda: stop_thread)
    )
    log_thread.start()
    line_num = 0
    with questions_path.open('r', encoding='utf-8', errors='replace') as questions_file:
        for line in questions_file:
            results_queue.put({'line_num': line_num, 'line': line})

            if (line_num + 1) % 1000 == 0:
                logger.info(f"Read {line_num + 1} lines")
            line_num += 1
    results_queue.put(None)
    logger.debug(f"{results_queue.qsize()=}")
    results_queue.join()
    result_processor.terminate()
    stop_thread = True
    log_queue.put("KILL")
    log_queue.join()
