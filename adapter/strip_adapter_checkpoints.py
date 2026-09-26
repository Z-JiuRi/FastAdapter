#!/usr/bin/env python3
"""Replace Adapter training checkpoints with compact EMA parameter states.

For every recursively discovered Adapter checkpoint under a selected task
root, this script:

1. extracts the 26 generated tensors from ``checkpoint['ema']['shadow']``;
2. excludes the runtime-only ``_delta_ratio`` buffer;
3. writes the original checkpoint metadata to ``adapter_meta.json``; and
4. atomically replaces ``adapter.pth`` with the compact ``OrderedDict``.

An Adapter is skipped only when it is already a compact state and its metadata
file contains the matching state hash.  Ambiguous or partially processed data
is rejected instead of being overwritten.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch


DATA_ROOT = Path("data")
ADAPTER_NAME = "adapter.pth"
META_NAME = "adapter_meta.json"
FORMAT_VERSION = 1
STATE_FORMAT = "adapter_parameter_fm.ema_shadow.v1"


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


def compact_state(value: Mapping[str, Any], source: Path) -> OrderedDict[str, torch.Tensor]:
    keys = set(value)
    expected = set(EXPECTED_SHAPES)
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected - {"_delta_ratio"})
    if missing or unexpected:
        raise ValueError(
            f"{source}: invalid EMA shadow keys; missing={missing}, "
            f"unexpected={unexpected}"
        )

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, shape in EXPECTED_SHAPES.items():
        tensor = value[key]
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError(f"{source}: {key} must be a floating tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{source}: {key} has shape={tuple(tensor.shape)}, expected={shape}"
            )
        tensor = tensor.detach().float().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{source}: {key} contains NaN or Inf")
        state[key] = tensor
    return state


def is_compact_state(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(EXPECTED_SHAPES)
        and all(torch.is_tensor(value[key]) for key in EXPECTED_SHAPES)
    )


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in EXPECTED_SHAPES:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o664)
    os.replace(temporary, path)


def atomic_torch_replace(payload: Any, path: Path) -> None:
    original_mode = path.stat().st_mode & 0o777
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if not is_compact_state(reloaded):
            raise RuntimeError(f"temporary compact checkpoint validation failed: {temporary}")
        reloaded_state = compact_state(reloaded, temporary)
        if state_sha256(reloaded_state) != state_sha256(payload):
            raise RuntimeError(f"temporary compact checkpoint hash mismatch: {temporary}")
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_existing_meta(meta_path: Path, digest: str) -> None:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid metadata file {meta_path}: {error}") from error
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{meta_path}: format_version={meta.get('format_version')!r}, "
            f"expected={FORMAT_VERSION}"
        )
    if meta.get("state_format") != STATE_FORMAT:
        raise ValueError(
            f"{meta_path}: state_format={meta.get('state_format')!r}, "
            f"expected={STATE_FORMAT!r}"
        )
    if meta.get("state_sha256") != digest:
        raise ValueError(
            f"{meta_path}: state hash does not match compact {ADAPTER_NAME}"
        )


def build_metadata(
    checkpoint: Mapping[str, Any],
    adapter_path: Path,
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    ema = checkpoint.get("ema", {})
    ema_metadata = {
        key: json_value(value)
        for key, value in ema.items()
        if key != "shadow"
    }
    return {
        "format_version": FORMAT_VERSION,
        "state_format": STATE_FORMAT,
        "task": adapter_path.parent.relative_to(DATA_ROOT).as_posix(),
        "source_checkpoint": ADAPTER_NAME,
        "source_checkpoint_size_bytes": adapter_path.stat().st_size,
        "stripped_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensor_count": len(state),
        "parameter_count": sum(tensor.numel() for tensor in state.values()),
        "parameter_dtype": "torch.float32",
        "state_sha256": state_sha256(state),
        "excluded_shadow_keys": ["_delta_ratio"] if "_delta_ratio" in ema["shadow"] else [],
        "epoch": json_value(checkpoint.get("epoch")),
        "args": json_value(checkpoint.get("args", {})),
        "metrics": json_value(checkpoint.get("metrics", {})),
        "ema": ema_metadata,
    }


def process_adapter(adapter_path: Path, dry_run: bool) -> str:
    meta_path = adapter_path.with_name(META_NAME)
    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)

    if is_compact_state(checkpoint):
        state = compact_state(checkpoint, adapter_path)
        if not meta_path.is_file():
            raise ValueError(
                f"{adapter_path}: compact state exists without {META_NAME}; "
                "original checkpoint metadata cannot be reconstructed"
            )
        validate_existing_meta(meta_path, state_sha256(state))
        return "skip"

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{adapter_path}: checkpoint root must be a mapping")
    ema = checkpoint.get("ema")
    if not isinstance(ema, Mapping) or not isinstance(ema.get("shadow"), Mapping):
        raise KeyError(f"{adapter_path}: full checkpoint must contain ema.shadow")

    state = compact_state(ema["shadow"], adapter_path)
    metadata = build_metadata(checkpoint, adapter_path, state)
    if dry_run:
        return "strip"

    # Metadata is committed first.  If the process stops between the two
    # replacements, rerunning still has the full checkpoint and can recover.
    atomic_write_json(metadata, meta_path)
    atomic_torch_replace(state, adapter_path)
    return "strip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strip every task Adapter checkpoint to its "
            "26-tensor EMA Adapter state and preserve metadata as JSON"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="task root recursively containing Adapter checkpoints",
    )
    parser.add_argument(
        "--adapter-name",
        default="adapter.pth",
        help="Adapter checkpoint filename inside each task directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and report actions without writing files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N sorted Adapter files; 0 means all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "number of checkpoint I/O worker threads (default: 4); "
            "larger values use more RAM and may not improve NFS throughput"
        ),
    )
    return parser.parse_args()


def parallel_process(
    paths: list[Path],
    *,
    dry_run: bool,
    workers: int,
) -> list[tuple[Path, str | None, Exception | None]]:
    """Process independent task files concurrently and collect all outcomes."""
    outcomes: list[tuple[Path, str | None, Exception | None]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="adapter-strip") as pool:
        futures = {
            pool.submit(process_adapter, path, dry_run): path
            for path in paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                action = future.result()
            except Exception as error:
                outcomes.append((path, None, error))
            else:
                outcomes.append((path, action, None))
    return sorted(outcomes, key=lambda item: str(item[0]))


def main() -> None:
    global DATA_ROOT, ADAPTER_NAME
    args = parse_args()
    DATA_ROOT = args.data_root.resolve()
    ADAPTER_NAME = args.adapter_name
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"data root does not exist: {DATA_ROOT}")
    paths = sorted(DATA_ROOT.rglob(ADAPTER_NAME))
    if args.limit > 0:
        paths = paths[:args.limit]
    if not paths:
        raise RuntimeError(f"no {ADAPTER_NAME} files found under {DATA_ROOT}")

    print(
        f"data_root={DATA_ROOT} adapters={len(paths)} "
        f"dry_run={args.dry_run} workers={args.workers}"
    )

    planned: list[tuple[Path, str]] = []
    preflight_failures = 0
    preflight = parallel_process(paths, dry_run=True, workers=args.workers)
    for index, (path, action, error) in enumerate(preflight, start=1):
        relative = path.relative_to(DATA_ROOT)
        if error is not None:
            preflight_failures += 1
            print(f"[{index}/{len(paths)}] FAIL  {relative}: {error}")
            continue
        assert action is not None
        planned.append((path, action))
        label = "WOULD_STRIP" if action == "strip" else "SKIP"
        print(f"[{index}/{len(paths)}] {label:<11} {relative}")

    planned_strip = sum(action == "strip" for _, action in planned)
    planned_skip = sum(action == "skip" for _, action in planned)
    print(
        f"preflight strip={planned_strip} skip={planned_skip} "
        f"failed={preflight_failures}"
    )
    if preflight_failures:
        print("preflight failed; no files were modified")
        raise SystemExit(1)
    if args.dry_run:
        return

    strip_paths = [path for path, action in planned if action == "strip"]
    applied_outcomes = parallel_process(
        strip_paths, dry_run=False, workers=args.workers
    )
    completed = 0
    apply_failures = 0
    for index, (path, applied, error) in enumerate(applied_outcomes, start=1):
        relative = path.relative_to(DATA_ROOT)
        if error is not None:
            apply_failures += 1
            print(f"[{index}/{len(applied_outcomes)}] FAIL        {relative}: {error}")
            continue
        if applied != "strip":
            apply_failures += 1
            print(
                f"[{index}/{len(applied_outcomes)}] FAIL        {relative}: "
                f"changed between preflight and apply; got {applied!r}"
            )
            continue
        completed += 1
        print(f"[{index}/{len(applied_outcomes)}] STRIPPED    {relative}")
    print(
        f"apply completed stripped={completed} skipped={planned_skip} "
        f"failed={apply_failures}"
    )
    if apply_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
