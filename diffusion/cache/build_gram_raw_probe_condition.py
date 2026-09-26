"""Sample codewords independently per task and save matched Gram--Raw inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build Gram--Raw Probe Condition files")
    parser.add_argument("--task-root", required=True, help="root containing train/val/test task folders")
    parser.add_argument("--source-file", default="train.pt", help="per-task codeword matrix, shape (N, C)")
    parser.add_argument("--indices-output-file", default="support_indices.pt",
                        help="per-task filename for reproducible support indices")
    parser.add_argument("--raw-output-file", default="raw_probe_K128.pt")
    parser.add_argument("--gram-output-file", default="gram_K128.pt")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_matrix(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{path}: expected a two-dimensional codeword tensor")
    value = value.detach().float().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return value


def discover(root: Path, splits: list[str], source_file: str) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        paths.extend(sorted((root / split).glob(f"*/*/{source_file}")))
    if not paths:
        raise FileNotFoundError(f"No */*/{source_file} found below {root}")
    return paths


def task_indices(path: Path, rows: int, k: int, seed: int, task_id: str,
                 overwrite: bool = False) -> torch.Tensor:
    if rows < k or k <= 0:
        raise ValueError(f"{path}: require 0 < K={k} <= N={rows}")
    digest = hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).digest()
    task_seed = int.from_bytes(digest[:8], "little") % (2**63)
    indices = torch.randperm(rows, generator=torch.Generator().manual_seed(task_seed))[:k]
    if path.is_file() and not overwrite:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(saved) or saved.ndim != 1:
            raise ValueError(f"{path}: expected a one-dimensional index tensor")
        if not torch.equal(saved.long(), indices):
            raise ValueError(f"{path}: saved indices differ from task seed; use --overwrite to rebuild")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(indices, path)
    if indices.numel() != k or indices.unique().numel() != k:
        raise ValueError(f"{path}: expected {k} distinct indices")
    if indices.min().item() < 0 or indices.max().item() >= rows:
        raise ValueError(f"{path}: indices are outside [0, {rows})")
    return indices


def main() -> int:
    args = parse_args()
    root = Path(args.task_root)
    source_paths = discover(root, list(args.splits), args.source_file)
    output_names = (args.indices_output_file, args.raw_output_file, args.gram_output_file)
    if any(Path(name).name != name for name in output_names) or len(set(output_names)) != 3:
        raise ValueError("output filenames must be distinct basenames inside each task directory")
    for source_path in source_paths:
        task_dir = source_path.parent
        if source_path.name in output_names:
            raise ValueError(f"{source_path}: source file cannot be overwritten")
        if not args.overwrite and any((task_dir / name).exists() for name in output_names[1:]):
            raise FileExistsError(f"{task_dir}: condition exists; use --overwrite to rebuild it")
    for source_path in source_paths:
        matrix = load_matrix(source_path)
        task_dir = source_path.parent
        task_id = task_dir.relative_to(root).as_posix()
        indices = task_indices(task_dir / args.indices_output_file,
                               matrix.shape[0], args.k, args.seed, task_id,
                               overwrite=args.overwrite)
        raw = matrix.index_select(0, indices).contiguous()
        gram = raw @ raw.transpose(0, 1)
        raw_path = task_dir / args.raw_output_file
        gram_path = task_dir / args.gram_output_file
        torch.save(raw, raw_path)
        torch.save(gram, gram_path)
    print(f"built independently sampled Gram--Raw conditions for {len(source_paths)} tasks; K={args.k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
