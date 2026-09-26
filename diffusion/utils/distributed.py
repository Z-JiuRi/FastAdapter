from __future__ import annotations

import os

import torch
import torch.distributed as dist


def distributed_requested() -> bool:
    """See README.md for English documentation."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> dict[str, int | bool]:
    """See README.md for English documentation."""
    if not distributed_requested():
        return {"enabled": False, "rank": 0, "local_rank": 0, "world_size": 1}
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return {
        "enabled": True,
        "rank": int(dist.get_rank()),
        "local_rank": local_rank,
        "world_size": int(dist.get_world_size()),
    }


def is_main_process() -> bool:
    """See README.md for English documentation."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    """See README.md for English documentation."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    """See README.md for English documentation."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
