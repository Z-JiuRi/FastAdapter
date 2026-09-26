#!/usr/bin/env python3
"""Compatibility entry point for standalone Adapter training."""

import sys
from pathlib import Path


SUBMIT_ROOT = Path(__file__).resolve().parents[1]
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))

from adapter.training.data import *  # noqa: E402,F401,F403
from adapter.training.model_io import *  # noqa: E402,F401,F403
from adapter.training.optimization import *  # noqa: E402,F401,F403
from adapter.training.losses import *  # noqa: E402,F401,F403
from adapter.training.engine import *  # noqa: E402,F401,F403
from adapter.training.cli import parse_args  # noqa: E402
from adapter.training.pipeline import run_training  # noqa: E402


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
