from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard.writer import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from adapter.models import build_mapper  # noqa: E402
from utils.logger import (logger, log_experiment_header, log_parameter_table,
                          setup_logging)  # noqa: E402
from utils.scheduler import FakeLR, WarmUpCosineAnnealingLR  # noqa: E402

