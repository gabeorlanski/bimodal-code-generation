import logging

logger = logging.getLogger(__name__)
__all__ = [
    "POST_TYPE_TO_STR",
    "NAME_TO_POST_TYPE"
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