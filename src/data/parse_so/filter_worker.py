import logging
import multiprocessing as mp
from lxml import etree
from unidecode import unidecode


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
        if not post_tags or not has_a_valid_tag:
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
