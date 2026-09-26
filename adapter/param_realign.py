#!/usr/bin/env python3
"""Compatibility entry point for Adapter parameter realignment."""

import sys
from pathlib import Path


SUBMIT_ROOT = Path(__file__).resolve().parents[1]
if str(SUBMIT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMIT_ROOT))

from adapter.alignment.io import *  # noqa: E402,F401,F403
from adapter.alignment.features import *  # noqa: E402,F401,F403
from adapter.alignment.matching import *  # noqa: E402,F401,F403
from adapter.alignment.reporting import *  # noqa: E402,F401,F403
from adapter.alignment.cli import *  # noqa: E402,F401,F403
from adapter.alignment.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
