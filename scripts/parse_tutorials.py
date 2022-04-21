import json
import argparse
import logging
import random
import shutil
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path
import sys
from dataclasses import asdict
from urllib.parse import urlparse, urljoin

import psutil
import ujson
import yaml
from lxml import etree
import multiprocessing as mp

from bs4 import BeautifulSoup
import click
import numpy as np
from tqdm import tqdm
import csv
import tldextract
from urllib.parse import urljoin
import requests

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging
from src.tutorials import TutorialHTMLParser, parse_domain_path


def parse_tutorials(debug, input_path, output_path):
    setup_global_logging(
        "parse_tutorials",
        PROJECT_ROOT.joinpath('logs'),
        debug=debug,
        disable_issues_file=True
    )

    logger = logging.getLogger('parse_tutorials')
    logger.info(f"Parsing tutorials from {input_path}")
    logger.info(f"Saving to {output_path}")

    domains = list(PROJECT_ROOT.joinpath(input_path).glob('*.yaml'))
    logger.info(f"{len(domains)} domain configs found")
    domain_to_urls = {}
    cfg = {}
    for domain_path in domains:
        logger.info(f"Loading config from {domain_path}")

        try:
            TutorialHTMLParser.by_name(domain_path.stem)
        except KeyError:
            logger.error(f"Skipping {domain_path.stem}, no parser found")
            continue
        loaded_cfg = yaml.load(
            domain_path.open(),
            yaml.Loader
        )
        cfg[domain_path.stem] = loaded_cfg['groups']
        try:
            domain_to_urls[domain_path.stem] = loaded_cfg['url']
        except KeyError:
            raise KeyError(f"{domain_path.stem} missing 'url'")

    maps_path = PROJECT_ROOT.joinpath('data', 'crawled_maps')
    crawled_path = PROJECT_ROOT.joinpath('data', 'crawled_tutorials')

    if not output_path.exists():
        output_path.mkdir(parents=True)
    logger.info(f"{len(cfg)} total unique domains to parse")

    total_found = {}
    parsed_file_to_url = defaultdict(dict)
    for domain, groups in cfg.items():
        logger.info(f"Looking for {len(groups)} group(s) for {domain}")
        path_to_name = {}

        for g, paths in groups.items():
            path_to_use = paths['path'] + '/' if paths['path'] != '/' else '/'
            for n, f in paths['pages'].items():
                path_to_name[f"{path_to_use}{f}"] = f"{g}_{n}"

        json_map = json.loads(maps_path.joinpath(f'{domain}.json').read_text())

        full_url = domain_to_urls[domain]
        to_parse = {}
        found = []
        for d in json_map:
            url_path = urlparse(d['url']).path
            if url_path in path_to_name:
                if path_to_name[url_path] in found:
                    raise ValueError("!!!DUPLICATES!!!")
                found.append(path_to_name[url_path])
                to_parse[d['cleaned_name']] = {
                    'name': path_to_name[url_path],
                    'url' : urljoin(full_url, url_path)
                }

        logger.info(f"{len(found)}/{len(path_to_name)} found")

        total_found[domain] = (len(found), len(path_to_name))
        parser = TutorialHTMLParser.by_name(domain)()

        files = {crawled_path.joinpath(domain, p): v for p, v in to_parse.items()}

        domain_out = output_path.joinpath(domain)
        if domain_out.exists():
            logger.warning(f"Removing existing dir at {domain_out}")
            shutil.rmtree(domain_out)
        logger.info(f"Saving to {domain_out}")
        domain_out.mkdir()

        for file, file_dict in tqdm(files.items(), desc='parsing'):
            out_name = file_dict['name']
            url = file_dict['url']

            try:
                parsed = parser(file.read_text())
            except Exception as e:
                logger.error(f"{file} failed to parse")
                raise e

            with domain_out.joinpath(f'{out_name}.json').open('w') as f:
                json.dump(parsed, f, indent=True)
            parsed_file_to_url[domain][out_name] = url
    with PROJECT_ROOT.joinpath('data/tutorials/parsed_to_url.json').open('w') as f:
        json.dump(parsed_file_to_url, f)
    logger.info("Found:")
    num_found = 0
    for k, v in sorted(total_found.items(), key=lambda e: e[0]):
        logger.info(f"\t{k:>16} = {v[0]:>5}/{v[1]:<5}")
        num_found += v[0]
    logger.info(f"{num_found} total found")


@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.pass_context
def main(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@main.command('parse')
@click.argument('input_path',
                callback=lambda c, p, v: PROJECT_ROOT.joinpath(v))
@click.option('--output-path', '-o', default='data/tutorials/parsed', help='Output path for saving',
              callback=lambda c, p, v: PROJECT_ROOT.joinpath(v))
@click.pass_context
def parse_tutorials_cli(ctx, input_path, output_path):
    parse_tutorials(ctx.obj['DEBUG'], input_path, output_path)


@main.command('get_code')
@click.option('--num-ctx', '-c', type=int, default=None)
@click.option('-annotations', is_flag=True, default=False, help='Annotation formats')
@click.pass_context
def parse_code_samples(ctx, num_ctx, annotations):
    setup_global_logging(
        "parse_code_samples",
        PROJECT_ROOT.joinpath('logs'),
        debug=ctx.obj['DEBUG'],
        disable_issues_file=True
    )
    logger = logging.getLogger('parse_code_samples')
    logger.info("Parsing out code samples")

    parsed_path = PROJECT_ROOT.joinpath('data/tutorials/parsed')

    global_context = {
        'lxml'      : {
            'global': [
                'from lxml import etree'
            ],
            'files' : {
                'developing_with_lxml_parsing'   : ['from io import StringIO, BytesIO'],
                'developing_with_lxml_validation': ['from io import StringIO, BytesIO'],
            }
        },
        'passlib'   : {
            'global': [
                'import os',
                'def TEMP_URANDOM(x):\n    raise NotImplementedError()',
                'os.urandom=TEMP_URANDOM'
            ],
            'files' : {
                'tutorial_totp': [
                    'from passlib.totp import TOTP',
                    'TotpFactory = TOTP.using(issuer="myapp.example.org")',
                    'totp = TotpFactory.new()',
                ]
            }
        },
        'jsonschema': {
            'global': [
                'from jsonschema import Draft3Validator, Draft7Validator, ErrorTree, validate'],
            'files' : {
                'tutorials_errors': [
                    'schema = {"type" : "array","items" : {"type" : "number", "enum" : [1, 2, 3]},"minItems" : 3,}',
                    'instance = ["spam", 2]',
                    'v = Draft3Validator(schema)',
                    'tree = ErrorTree(v.iter_errors(instance))'
                ]
            }
        },
        'cerberus'  : {
            'global': ['from cerberus import Validator', 'v = Validator()'],
            'files' : {
            }
        },
        'theano'    : {
            'global': [],
            'files' : {
                'basics_broadcasting': ['import theano.tensor as tt']
            }
        },
        'arrow'     : {
            'global': ['from datetime import datetime'],
            'files' : {}
        },
        'numpy'     : {
            'global': ['import numpy as np'],
            'files' : {}
        }
    }
    fail_dir = PROJECT_ROOT.joinpath('data/tutorials/fails')
    out_dir = PROJECT_ROOT.joinpath('data/tutorials/code')
    if annotations:
        out_dir = PROJECT_ROOT.joinpath('data/tutorials/raw_annotations')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if fail_dir.exists():
        shutil.rmtree(fail_dir)
    fail_dir.mkdir(parents=True)
    fixes_by_section = yaml.load(
        PROJECT_ROOT.joinpath('data', 'tutorial_code_fixes.yaml').open(),
        yaml.Loader
    )

    directories = [f for f in parsed_path.glob('*') if f.is_dir()]
    logger.info(f"Found {len(directories)} directories")
    total_runnable_code = Counter()
    total_fails = Counter()
    total_passed = Counter()
    parsed_file_idx = 0
    parsed_to_url = json.loads(
        PROJECT_ROOT.joinpath('data/tutorials/parsed_to_url.json').read_text())

    for domain_path in directories:
        logger.info(f"Parsing {domain_path}")
        parsed_files, num_runnable, passed, num_fail = parse_domain_path(
            domain_path,
            global_context,
            fixes_by_section,
            fail_dir,
            parsed_file_idx,
            num_ctx,
            parsed_to_url[domain_path.stem],
            annotations
        )
        parsed_file_idx += len(parsed_files)
        # all_parsed[domain_path.stem] = parsed_files
        total_passed[domain_path.stem] = passed
        total_runnable_code[domain_path.stem] = num_runnable
        total_fails[domain_path.stem] = num_fail
        with out_dir.joinpath(f'{domain_path.stem}.json').open('w') as f:
            json.dump(parsed_files, f, indent=True)

    num_found = sum(total_runnable_code.values())
    logger.info(
        f"{num_found}/{sum(total_fails.values()) + num_found} are runnable"
    )
    logger.info(f"{sum(total_passed.values())}/{num_found} returned the expected value")
    logger.info("Results in the form Passed Test | Runnable | Total:")
    with_one_passed = 0
    for k, v in sorted(total_runnable_code.items(), key=lambda e: e[0]):
        total = v + total_fails[k]
        if total > 0:
            pct_ver = total_passed[k] / total
        else:
            pct_ver = 0

        with_one_passed += total_passed[k] > 0
        logger.info(
            f"\t{k:>16} = {total_passed[k]:>11} | {v:>8} | {total:>7}     "
            f"{pct_ver:>7.2%} verified.")
    logger.info(f"{with_one_passed} had more than one pass")
    # logger.info(f"Saving to {out_dir}")
    #
    # with out_dir.joinpath('parsed.json').open('w') as f:
    #     json.dump(all_parsed, f, indent=True)


if __name__ == '__main__':
    main()
