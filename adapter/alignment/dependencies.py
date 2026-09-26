#!/usr/bin/env python3
"""Analyze Adapter hidden-neuron alignment and save reproducible reports.

The script reads compact 26-tensor Adapter state files from a task data tree.
A reference is built from train tasks only
using iterative in-memory barycenters.  Train/val/test states are then matched
to that frozen reference, permuted in memory, and checked for exact functional
equivalence.  By default only metrics, permutation indices, logs, and
paper-ready plots are written; pass ``--save-aligned`` to write the final
permuted states beside each source Adapter.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from queue import Queue
import sys
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import torch
import torch.nn.functional as F


SUBMIT_ROOT = Path(__file__).resolve().parents[2]
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))
sys.dont_write_bytecode = True

from adapter.functional import functional_adapter


def expected_parameter_shapes() -> OrderedDict[str, tuple[int, ...]]:
    shapes: OrderedDict[str, tuple[int, ...]] = OrderedDict([
        ("alignment_weight", (512, 512)),
        ("alignment_bias", (512,)),
    ])
    for block in range(4):
        prefix = f"blocks.{block}."
        shapes[prefix + "norm.weight"] = (512,)
        shapes[prefix + "norm.bias"] = (512,)
        shapes[prefix + "net.0.weight"] = (512, 512)
        shapes[prefix + "net.0.bias"] = (512,)
        shapes[prefix + "net.3.weight"] = (512, 512)
        shapes[prefix + "net.3.bias"] = (512,)
    return shapes


EXPECTED_SHAPES = expected_parameter_shapes()


DATA_ROOT = Path("data")
DEFAULT_DATASET = "COST2100"
DEFAULT_SCENARIO = "in"
DEFAULT_TASK_ROOT = DATA_ROOT / DEFAULT_DATASET / DEFAULT_SCENARIO
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
SPLIT_ORDER = ("train", "val", "test")

