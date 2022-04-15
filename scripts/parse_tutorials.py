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
from urllib.parse import urlparse

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

    cfg = {}
    for domain_path in domains:
        logger.info(f"Loading config from {domain_path}")

        try:
            TutorialHTMLParser.by_name(domain_path.stem)
        except KeyError:
            logger.error(f"Skipping {domain_path.stem}, no parser found")
            continue

        cfg[domain_path.stem] = yaml.load(
            domain_path.open(),
            yaml.Loader
        )['groups']

    maps_path = PROJECT_ROOT.joinpath('data', 'crawled_maps')
    crawled_path = PROJECT_ROOT.joinpath('data', 'crawled_tutorials')

    if not output_path.exists():
        output_path.mkdir(parents=True)
    logger.info(f"{len(cfg)} total unique domains to parse")

    domain_run = defaultdict(dict)
    for domain, groups in cfg.items():
        logger.info(f"Looking for {len(groups)} group(s) for {domain}")
        path_to_name = {}

        for g, paths in groups.items():
            path_to_use = paths['path'] + '/' if paths['path'] != '/' else '/'
            for n, f in paths['pages'].items():
                path_to_name[f"{path_to_use}{f}"] = f"{g}_{n}"

        json_map = json.loads(maps_path.joinpath(f'{domain}.json').read_text())

        to_parse = {}
        found = []
        for d in json_map:
            url_path = urlparse(d['url']).path
            if url_path in path_to_name:
                if path_to_name[url_path] in found:
                    raise ValueError("!!!DUPLICATES!!!")
                found.append(path_to_name[url_path])
                to_parse[d['cleaned_name']] = path_to_name[url_path]

        logger.info(f"{len(found)}/{len(path_to_name)} found")
        parser = TutorialHTMLParser.by_name(domain)()

        files = {crawled_path.joinpath(domain, p): v for p, v in to_parse.items()}

        domain_out = output_path.joinpath(domain)
        if domain_out.exists():
            logger.warning(f"Removing existing dir at {domain_out}")
            shutil.rmtree(domain_out)
        logger.info(f"Saving to {domain_out}")
        domain_out.mkdir()

        for file, out_name in tqdm(files.items(), desc='parsing'):
            try:
                parsed = parser(file.read_text())
            except Exception as e:
                logger.error(f"{file} failed to parse")
                raise e

            with domain_out.joinpath(f'{out_name}.json').open('w') as f:
                json.dump(parsed, f, indent=True)


@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable Debug Mode')
@click.pass_context
def main(ctx, debug):
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug


@main.command('parse')
@click.argument('input_path',
                callback=lambda c, p, v: PROJECT_ROOT.joinpath(v))
@click.option('--output-path', '-o', default='data/parsed_tutorials', help='Output path for saving',
              callback=lambda c, p, v: PROJECT_ROOT.joinpath(v))
@click.pass_context
def parse_tutorials_cli(ctx, input_path, output_path):
    parse_tutorials(ctx.obj['DEBUG'], input_path, output_path)


@main.command('get_code')
@click.pass_context
def make_samples(ctx):
    setup_global_logging(
        "make_samples_from_tutorials",
        PROJECT_ROOT.joinpath('logs'),
        debug=ctx.obj['DEBUG'],
        disable_issues_file=True
    )
    logger = logging.getLogger('make_samples_from_tutorials')
    logger.info("Making the samples from the parsed tutorials")

    parsed_path = PROJECT_ROOT.joinpath('data/parsed_tutorials')

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
            'global': ['from jsonschema import validate'],
            'files' : {
                'tutorials_errors': [
                    'from jsonschema import Draft3Validator, Draft7Validator',
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
        }
    }
    out_dir = PROJECT_ROOT.joinpath('data/tutorial_code')
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    fixes_by_section = yaml.load(
        PROJECT_ROOT.joinpath('data', 'tutorial_code_fixes.yaml').open(),
        yaml.Loader
    )

    directories = list(parsed_path.glob('*'))
    logger.info(f"Found {len(directories)} directories")
    all_passed = {}
    total_runnable_code = Counter()
    total_fails = Counter()
    total_passed = Counter()
    parsed_file_idx = 0
    all_parsed = []
    for domain_path in directories:
        parsed_files, passed, num_runnable, num_fail = parse_domain_path(
            domain_path,
            global_context,
            fixes_by_section,
            out_dir,
            parsed_file_idx
        )
        parsed_file_idx += len(parsed_files)
        all_parsed.extend(parsed_files)
        all_passed[domain_path.stem] = passed
        total_passed[domain_path.stem] += sum(map(len, passed.values()))
        total_runnable_code[domain_path.stem] = num_runnable
        total_fails[domain_path.stem] = num_fail

    num_found = sum(total_runnable_code.values())
    logger.info(
        f"{num_found}/{sum(total_fails.values()) + num_found} are runnable")
    logger.info(f"{sum(total_passed.values())}/{num_found} returned the expected value")
    logger.info("Results in the form Passed Test | Runnable | Total:")
    for k, v in total_runnable_code.items():
        total = v + total_fails[k]
        logger.info(
            f"\t{k:>16} = {total_passed[k]:>11} | {v:>8} | {total:>7}     "
            f"{total_passed[k] / total:>7.2%} verified.")

    logger.info(f"Saving to {out_dir}")

    with out_dir.joinpath('parsed.json').open('w') as f:
        json.dump(all_parsed, f, indent=True)


@main.command('download')
@click.argument('download_cfg')
@click.pass_context
def download(ctx, download_cfg):
    out_path = PROJECT_ROOT.joinpath('data', 'tutorials')
    print(f"Reading cfg from {download_cfg}")
    cfg = yaml.load(
        PROJECT_ROOT.joinpath(download_cfg).open('r'),
        yaml.Loader
    )
    print(f"{len(cfg)} total sites to download from")
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True)

    for name, site_cfg in cfg.items():
        print(f"Downloading from {name}")

        site_path = out_path.joinpath(name)
        site_path.mkdir()

        print(f"{len(site_cfg)} total section(s)")

        for section_name, section_cfg in site_cfg.items():
            base_url = section_cfg['url']

            print(f"Section {section_name} has {len(section_cfg['pages'])} total files to download")
            for file_name, path in tqdm(section_cfg['pages'].items(), desc='Downloading'):
                r = requests.get(urljoin(base_url, path))
                with site_path.joinpath(f'{section_name}_{file_name}.html').open('w') as f:
                    f.write(r.text)


if __name__ == '__main__':
    main()
