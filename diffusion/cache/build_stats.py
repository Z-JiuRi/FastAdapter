from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.adapter_tokenizer import build_stats_payload


def discover_train_adapters(task_root: Path, filename: str) -> list[Path]:
    return sorted(path for path in (task_root / "train").glob(f"*/*/{filename}") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Adapter normalization statistics from training data")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "cache" / "stats.pt")
    parser.add_argument("--alignment", choices=("aligned", "raw"), default="aligned")
    parser.add_argument("--token-size", type=int, default=512)
    args = parser.parse_args()

    filename = "aligned_adapter.pth" if args.alignment == "aligned" else "adapter.pth"
    paths = discover_train_adapters(args.task_root, filename)
    payload = build_stats_payload(paths, args.token_size, args.alignment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"Saved {args.output}: tasks={payload['num_train_tasks']}, "
        f"tokens={payload['manifest']['num_tokens']}, alignment={args.alignment}"
    )


if __name__ == "__main__":
    main()
