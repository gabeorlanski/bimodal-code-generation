import logging
from pathlib import Path
import pytest
import json
import shutil
from unittest.mock import patch

from src.common import FIXTURES_ROOT
from src.data.parse_so import parse_so_dump


