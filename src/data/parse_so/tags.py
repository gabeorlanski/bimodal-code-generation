import json
import argparse
import logging
import random
from collections import defaultdict, Counter
from pathlib import Path
import sys
from dataclasses import asdict
from lxml import etree

import click
import numpy as np
from tqdm import tqdm
import csv

import ujson
from src.common import PROJECT_ROOT

__all__ = [
    "create_tag_info_file",
    "create_tag_knowledge_graph"
]

logger = logging.getLogger(__name__)


def create_tag_info_file(parsed_path, dump_path, tag_synonyms_path):
    logger.info(f"Making tag info from {parsed_path}")
    parsed_path = PROJECT_ROOT.joinpath(parsed_path)
    dump_path = PROJECT_ROOT.joinpath(dump_path)
    tag_synonyms_path = PROJECT_ROOT.joinpath(tag_synonyms_path)
    logger.info(f"{parsed_path=}")
    logger.info(f"{dump_path=}")
    logger.info(f"{tag_synonyms_path=}")

    logger.info("Loading Synonyms")
    tag_synonyms = defaultdict(list)
    synonyms_found = 0
    with tag_synonyms_path.open() as f:
        col = []
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                col = row
                continue

            row_dict = {col[j]: v for j, v in enumerate(row)}
            tag_synonyms[row_dict['TargetTagName']].append(row_dict['SourceTagName'])
            synonyms_found += 1
    logger.info(f"{synonyms_found} synonyms found for {len(tag_synonyms)} tags")

    logger.info("Reading tag data")
    tag_information = {}
    excerpt_id_mapping = {}
    wiki_id_mapping = {}
    tag_id_to_name = {}
    tag_xml_path = dump_path.joinpath('Tags.xml')
    for line in tqdm(tag_xml_path.open('r')):
        try:
            tag_dict = etree.XML(line).attrib
        except Exception as e:
            continue

        tag_name = tag_dict['TagName']
        if tag_name in tag_information:
            raise KeyError(f"{tag_dict['TagName']} is duplicated")
        tag_id_to_name[tag_dict['Id']] = tag_name
        tag_information[tag_name] = {
            'count'       : int(tag_dict['Count']),
            'id'          : tag_dict['Id'],
            'wiki'        : None,
            'wiki_date'   : None,
            'wiki_id'     : tag_dict.get('WikiPostId'),
            'excerpt'     : None,
            'excerpt_date': None,
            'excerpt_id'  : tag_dict.get('ExcerptPostId'),
            'synonyms'    : tag_synonyms[tag_name]
        }

        if 'ExcerptPostId' in tag_dict:
            if tag_dict['ExcerptPostId'] in excerpt_id_mapping:
                logger.info(f"Duplicate excerpts found {tag_dict['ExcerptPostId']}")
            excerpt_id_mapping[tag_dict['ExcerptPostId']] = tag_name

        if 'WikiPostId' in tag_dict:
            if tag_dict['WikiPostId'] in excerpt_id_mapping:
                logger.info(f"Duplicate wiki found {tag_dict['WikiPostId']}")
            wiki_id_mapping[tag_dict['WikiPostId']] = tag_name

    logger.info(f"{len(tag_information)} unique tags found")
    logger.info(f"{len(excerpt_id_mapping)} excerpts to get")
    logger.info(f"{len(wiki_id_mapping)} wiki items to get")

    excerpt_file = parsed_path.joinpath('wiki_excerpts.jsonl')
    logger.info(f"Reading excerpts file {excerpt_file}")

    if excerpt_file.exists():
        orphaned_excerpts = 0
        with excerpt_file.open() as f:
            for excerpt in tqdm(map(json.loads, f.readlines())):
                try:
                    tag_name = excerpt_id_mapping[excerpt['id']]
                except KeyError:
                    orphaned_excerpts += 1
                    continue

                tag_information[tag_name]['excerpt'] = excerpt['body']
                tag_information[tag_name]['excerpt_date'] = excerpt['date']
        logger.info(f"{orphaned_excerpts} orphaned excerpts")

    else:
        logger.info(f"Could not find {excerpt_file}")

    wiki_file = parsed_path.joinpath('wiki.jsonl')
    logger.info(f"Reading wiki file {wiki_file}")

    if wiki_file.exists():
        orphaned_wiki = 0
        with wiki_file.open() as f:
            for wiki in tqdm(map(json.loads, f.readlines())):
                try:
                    tag_name = wiki_id_mapping[wiki['id']]
                except KeyError:
                    orphaned_wiki += 1
                    continue

                tag_information[tag_name]['wiki'] = wiki['body']
                tag_information[tag_name]['wiki_body'] = wiki['date']
        logger.info(f"{orphaned_wiki} orphaned wiki pages")

    else:
        logger.info(f"Could not find {wiki_file}")

    save_path = parsed_path.joinpath('tags.json')
    logger.info(f"Saving to {save_path}")
    with save_path.open('w') as f:
        json.dump(tag_information, f, indent=True)


def create_tag_knowledge_graph(parsed_path):
    parsed_path = PROJECT_ROOT.joinpath(parsed_path)

    tag_info_path = parsed_path.joinpath('tags.json')
    if not tag_info_path.exists():
        raise ValueError(f'Tag info does not exist in {parsed_path}. Run '
                         f'parse_so_data.py make_tag_info to create it')
    logger.info(f"Loading tag info from {tag_info_path}")
    tag_info = json.loads(tag_info_path.read_text())

    logger.info(f"Creating tag synonym mapping for {len(tag_info)}")
    synonym_to_tag_map = {}
    tag_counts = {}

    for tag, tag_dict in tag_info.items():
        for synonym in tag_dict['synonyms']:
            synonym_to_tag_map[synonym] = tag
        tag_counts[tag] = tag_dict['count']
    logger.info(f"{len(synonym_to_tag_map)} total synonyms")

    question_path = parsed_path.joinpath('question_overview.json')
    logger.info(f"Loading the parsed question overview from {question_path}")
    question_overview = ujson.load(question_path.open())

    logger.info(f"{len(question_overview)} questions found")
    logger.info(f"Creating knowledge Graph for {len(question_overview)} questions")

    # Each question has at most 5 tags, and we have a ranking function of so we
    # need to count the number of questions where a give tag is ranked as the
    # top tag
    highest_count_num_questions = Counter()
    first_tag_num_questions = Counter()

    highest_count_kg = defaultdict(Counter)
    first_tag_kg = defaultdict(Counter)
    co_occurrences_kg = Counter()

    # Make a separate dict for the edges b/c of how naming keys works in json.
    co_occurrences_edges = {
        "start": defaultdict(set),
        "end"  : defaultdict(set)
    }
    for question_id, question_dict in tqdm(question_overview.items(), total=len(question_overview)):

        tags = question_dict.get('tags', [])
        if not tags:
            continue

        # Replace any tags that have a synonym. If the tag is a synonym, the
        # get function will return the correct tag, otherwise there is no
        # synonym and return the original tag.
        tags = [synonym_to_tag_map.get(t, t) for t in tags]

        # Make 1 KG that just stores the number of times two tags co-occur with
        # each other. The edge name represents the respective ordering.
        for i, t1 in enumerate(tags):
            for t2 in tags[i + 1:]:
                edge_name = f"({t1},{t2})"
                co_occurrences_kg[edge_name] += 1
                co_occurrences_edges['start'][t1].add(edge_name)
                co_occurrences_edges['end'][t2].add(edge_name)

        # Make another KG that stores the number of times a tag is first in the
        # list of tags and what other tags are included.
        first_tag, *remaining = tags
        first_tag_num_questions[first_tag] += 1
        for tag in remaining:
            first_tag_kg[first_tag][tag] += 1

        # Make another KG that stores the number of times a tag has the highest
        # count in the list of tags and what other tags are included.
        first_tag, *remaining = sorted(
            tags,
            key=lambda t: tag_counts[t],
            reverse=True
        )
        highest_count_num_questions[first_tag] += 1
        for tag in remaining:
            highest_count_kg[first_tag][tag] += 1

    kg_path = PROJECT_ROOT.joinpath('data', 'knowledge_graph')
    logger.info(f"KGs will be saved to {kg_path}")
    if not kg_path.exists():
        kg_path.mkdir()
    logger.info(f"Saving co-occurrence KG with {len(co_occurrences_kg)} edges ")
    with kg_path.joinpath(f"{parsed_path.stem}_occurrence.json").open('w') as f:
        json.dump({
            "edges"   : co_occurrences_kg,
            "vertices": {
                "start": {k: list(v) for k, v in co_occurrences_edges['start'].items()},
                "end"  : {k: list(v) for k, v in co_occurrences_edges['end'].items()},
            }
        }, f)

    logger.info(f"Saving highest count KG with {len(highest_count_kg)} elements")
    with kg_path.joinpath(f"{parsed_path.stem}_highest.json").open('w') as f:
        json.dump({
            "question_count": highest_count_num_questions,
            "edges"         : highest_count_kg
        }, f)
    logger.info(f"Saving first tag KG with {len(first_tag_kg)} elements")
    with kg_path.joinpath(f"{parsed_path.stem}_first.json").open('w') as f:
        json.dump({
            "question_count": first_tag_num_questions,
            "edges"         : first_tag_kg
        }, f)
