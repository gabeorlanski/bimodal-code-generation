"""
Scripts to parse StackExchange dumps
"""

import json
import argparse
import logging
import random
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path
import sys
from dataclasses import asdict
from urllib.parse import urlparse

import psutil
import ujson
from lxml import etree
import multiprocessing as mp

from bs4 import BeautifulSoup
import click
import numpy as np
from tqdm import tqdm
import csv
import tldextract

# If this file is called by itself (for creating the splits) then it will
# have import issues.
if str(Path(__file__).parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import PROJECT_ROOT, setup_global_logging
from src.common.file_util import validate_files_exist
from src.data.parse_so import *


# Here just to allow the grouping.
@click.command()
@click.argument('dump_name')
@click.argument('out_name')
def main(dump_name, out_name):
    pass


if __name__ == "__main__":
    main()
