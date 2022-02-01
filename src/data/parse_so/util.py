import logging

logger = logging.getLogger(__name__)
__all__ = [
    "POST_TYPE_TO_STR",
    "NAME_TO_POST_TYPE",
    "log_process"
]

# Taken from https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede/2678#2678
POST_TYPE_TO_STR = {
    1: "questions",
    2: "answers",
    3: "orphaned_wiki",
    4: "wiki_excerpts",
    5: "wiki",
    6: "moderation_nomination",
    7: "wiki_placeholder",
    8: "privilege_wiki"
}
NAME_TO_POST_TYPE = {
    v: k for k, v in POST_TYPE_TO_STR.items()
}


def log_process(log_queue, worker_count):
    finished = 0
    while True:
        try:
            message = log_queue.get(timeout=2.0)
        except Exception:
            continue
        if message is not None and message != "KILL":

            level, message = message
            logger.log(level, message)
        else:
            finished += 1
            logger.debug(f'Finished is at {finished}')
            if finished >= worker_count or message == "KILL":
                logger.info("Log Thread is done.")
                return
            continue
