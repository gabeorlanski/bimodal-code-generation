import json
from dataclasses import asdict

import astor
import pytest

from src.common import FIXTURES_ROOT
from src.tutorials import node_visitors
from pathlib import Path


