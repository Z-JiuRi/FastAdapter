from __future__ import annotations

import os

import torch


def configure_torch_threads() -> dict[str, int]:
    """See README.md for English documentation."""
    torch_threads = int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "0")) or 0)
    interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "0") or 0)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
    if interop_threads > 0:
        torch.set_num_interop_threads(interop_threads)
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
