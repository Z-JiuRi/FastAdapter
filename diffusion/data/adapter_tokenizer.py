from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


CANONICAL_ADAPTER_KEYS = ["alignment_weight", "alignment_bias"]
for _block in range(4):
    CANONICAL_ADAPTER_KEYS.extend([
        f"blocks.{_block}.norm.weight",
        f"blocks.{_block}.norm.bias",
        f"blocks.{_block}.net.0.weight",
        f"blocks.{_block}.net.0.bias",
        f"blocks.{_block}.net.3.weight",
        f"blocks.{_block}.net.3.bias",
    ])

ROLE_NAMES = ("alignment", "norm", "ffn_up", "ffn_down")
KIND_NAMES = ("weight", "bias")


def load_adapter_state(path: str | Path) -> OrderedDict[str, torch.Tensor]:
    path = Path(path)
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}: expected an Adapter state mapping")
    if tuple(value) != tuple(CANONICAL_ADAPTER_KEYS):
        raise ValueError(f"{path}: Adapter tensors do not follow the canonical 26-key order")

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, tensor in value.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            raise TypeError(f"{path}: {key} must be a floating tensor")
        tensor = tensor.detach().float().cpu().contiguous()
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")
        state[str(key)] = tensor
    return state


def _semantic_ids(key: str) -> tuple[int, int, int]:
    if key.startswith("alignment_"):
        block_id, role = 0, "alignment"
    else:
        parts = key.split(".")
        block_id = int(parts[1]) + 1
        if parts[2] == "norm":
            role = "norm"
        elif parts[2:4] == ["net", "0"]:
            role = "ffn_up"
        elif parts[2:4] == ["net", "3"]:
            role = "ffn_down"
        else:
            raise ValueError(f"Unknown Adapter parameter role: {key}")
    kind = "weight" if key.endswith("weight") else "bias"
    return block_id, ROLE_NAMES.index(role), KIND_NAMES.index(kind)


def build_manifest(state: Mapping[str, torch.Tensor], token_size: int) -> dict[str, Any]:
    if token_size <= 0:
        raise ValueError("token_size must be positive")
    tensors = []
    token_mask = []
    tensor_ids = []
    token_ids = []
    block_ids = []
    role_ids = []
    kind_ids = []
    token_start = 0

    for tensor_id, key in enumerate(CANONICAL_ADAPTER_KEYS):
        value = state[key]
        if value.ndim not in (1, 2):
            raise ValueError(f"{key}: only 1D/2D Adapter tensors are supported")
        width = value.shape[-1]
        chunks_per_row = (width + token_size - 1) // token_size
        rows = value.shape[0] if value.ndim == 2 else 1
        token_count = rows * chunks_per_row
        block_id, role_id, kind_id = _semantic_ids(key)
        for local_id in range(token_count):
            chunk_id = local_id % chunks_per_row
            valid = min(token_size, width - chunk_id * token_size)
            token_mask.append([1.0] * valid + [0.0] * (token_size - valid))
            tensor_ids.append(tensor_id)
            token_ids.append(local_id)
            block_ids.append(block_id)
            role_ids.append(role_id)
            kind_ids.append(kind_id)
        tensors.append({
            "key": key,
            "shape": list(value.shape),
            "numel": value.numel(),
            "token_start": token_start,
            "token_count": token_count,
            "chunks_per_row": chunks_per_row,
            "layout": "output_row_chunks" if value.ndim == 2 else "vector_chunks",
            "tensor_id": tensor_id,
            "block_id": block_id,
            "role_id": role_id,
            "kind_id": kind_id,
        })
        token_start += token_count

    return {
        "version": 1,
        "token_size": token_size,
        "num_tokens": token_start,
        "num_tensors": len(tensors),
        "num_block_ids": max(block_ids) + 1,
        "num_role_ids": len(ROLE_NAMES),
        "num_kind_ids": len(KIND_NAMES),
        "max_token_id": max(token_ids),
        "role_names": list(ROLE_NAMES),
        "kind_names": list(KIND_NAMES),
        "tensors": tensors,
        "token_mask": torch.tensor(token_mask, dtype=torch.bool),
        "tensor_ids": torch.tensor(tensor_ids, dtype=torch.long),
        "token_ids": torch.tensor(token_ids, dtype=torch.long),
        "block_ids": torch.tensor(block_ids, dtype=torch.long),
        "role_ids": torch.tensor(role_ids, dtype=torch.long),
        "kind_ids": torch.tensor(kind_ids, dtype=torch.long),
    }


def validate_state_against_manifest(state: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]) -> None:
    for spec in manifest["tensors"]:
        key = spec["key"]
        if key not in state or list(state[key].shape) != list(spec["shape"]):
            actual = None if key not in state else list(state[key].shape)
            raise ValueError(f"{key}: shape {actual} does not match manifest {spec['shape']}")


def tokenize_adapter(
    state: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    parameter_stats: Mapping[str, Mapping[str, Any]],
) -> torch.Tensor:
    validate_state_against_manifest(state, manifest)
    tokens = torch.zeros(manifest["num_tokens"], manifest["token_size"], dtype=torch.float32)
    for spec in manifest["tensors"]:
        key = spec["key"]
        stats = parameter_stats[key]
        value = (state[key].float() - float(stats["mean"])) / float(stats["std"])
        rows = value.reshape(-1, value.shape[-1]) if value.ndim == 2 else value.reshape(1, -1)
        start = spec["token_start"]
        for row_id, row in enumerate(rows):
            for chunk_id in range(spec["chunks_per_row"]):
                source = row[chunk_id * manifest["token_size"]:(chunk_id + 1) * manifest["token_size"]]
                token_id = start + row_id * spec["chunks_per_row"] + chunk_id
                tokens[token_id, :source.numel()] = source
    return tokens


def _reconstruct_adapter_state(
    tokens: torch.Tensor,
    manifest: Mapping[str, Any],
    parameter_stats: Mapping[str, Mapping[str, Any]],
) -> OrderedDict[str, torch.Tensor]:
    if tuple(tokens.shape) != (manifest["num_tokens"], manifest["token_size"]):
        raise ValueError(
            f"tokens shape {tuple(tokens.shape)} != "
            f"({manifest['num_tokens']}, {manifest['token_size']})"
        )
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for spec in manifest["tensors"]:
        start = spec["token_start"]
        count = spec["token_count"]
        rows = tokens[start:start + count].reshape(-1, spec["chunks_per_row"], manifest["token_size"])
        width = spec["shape"][-1]
        normalized = rows.reshape(rows.shape[0], -1)[:, :width]
        value = normalized * float(parameter_stats[spec["key"]]["std"])
        value = value + float(parameter_stats[spec["key"]]["mean"])
        state[spec["key"]] = value.reshape(spec["shape"]).detach().cpu().contiguous()
    return state


def detokenize_adapter_batch(
    tokens: torch.Tensor,
    manifest: Mapping[str, Any],
    parameter_stats: Mapping[str, Mapping[str, Any]],
) -> OrderedDict[str, torch.Tensor]:
    """See README.md for English documentation."""
    squeeze = tokens.ndim == 2
    if squeeze:
        tokens = tokens.unsqueeze(0)
    expected = (manifest["num_tokens"], manifest["token_size"])
    if tuple(tokens.shape[1:]) != expected:
        raise ValueError(f"tokens shape {tuple(tokens.shape[1:])} != {expected}")
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for spec in manifest["tensors"]:
        start = spec["token_start"]
        count = spec["token_count"]
        rows = tokens[:, start:start + count]
        rows = rows.reshape(tokens.shape[0], -1, spec["chunks_per_row"], manifest["token_size"])
        width = spec["shape"][-1]
        normalized = rows.reshape(tokens.shape[0], rows.shape[1], -1)[..., :width]
        mean = torch.as_tensor(
            parameter_stats[spec["key"]]["mean"], device=tokens.device, dtype=tokens.dtype
        )
        std = torch.as_tensor(
            parameter_stats[spec["key"]]["std"], device=tokens.device, dtype=tokens.dtype
        )
        value = normalized * std + mean
        state[spec["key"]] = value.reshape(tokens.shape[0], *spec["shape"])
        if squeeze:
            state[spec["key"]] = state[spec["key"]][0]
    return state


def build_stats_payload(
    adapter_paths: Iterable[str | Path], token_size: int, alignment: str
) -> dict[str, Any]:
    paths = [Path(path) for path in adapter_paths]
    if not paths:
        raise ValueError("No training Adapter files were found")
    first = load_adapter_state(paths[0])
    manifest = build_manifest(first, token_size)
    sums = {key: torch.zeros((), dtype=torch.float64) for key in CANONICAL_ADAPTER_KEYS}
    sum_squares = {key: torch.zeros((), dtype=torch.float64) for key in CANONICAL_ADAPTER_KEYS}
    counts = {key: 0 for key in CANONICAL_ADAPTER_KEYS}
    fingerprint = sha256()
    fingerprint.update(f"alignment={alignment};token_size={token_size}".encode())
    for spec in manifest["tensors"]:
        fingerprint.update(f"{spec['key']}:{spec['shape']}".encode())
    source_files = []

    for path in paths:
        state = load_adapter_state(path)
        validate_state_against_manifest(state, manifest)
        stat = path.stat()
        fingerprint.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        source_files.append({
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
        for key, value in state.items():
            value64 = value.double()
            sums[key] += value64.sum()
            sum_squares[key] += value64.square().sum()
            counts[key] += value.numel()

    parameter_stats = OrderedDict()
    for key in CANONICAL_ADAPTER_KEYS:
        mean = sums[key] / counts[key]
        variance = sum_squares[key] / counts[key] - mean.square()
        std = variance.clamp_min(0).sqrt()
        if not torch.isfinite(std) or std <= 0:
            raise ValueError(f"{key}: training statistics have zero or invalid std")
        parameter_stats[key] = {"mean": mean.float(), "std": std.float(), "count": counts[key]}

    return {
        "version": 1,
        "alignment": alignment,
        "num_train_tasks": len(paths),
        "fingerprint": fingerprint.hexdigest(),
        "config": {
            "alignment": alignment,
            "token_size": token_size,
            "canonical_keys": list(CANONICAL_ADAPTER_KEYS),
        },
        "source_files": source_files,
        "manifest": manifest,
        "parameter_stats": parameter_stats,
    }


def validate_stats_fingerprint(stats: Mapping[str, Any], verify_sources: bool = True) -> None:
    fingerprint = sha256()
    config = stats["config"]
    fingerprint.update(
        f"alignment={config['alignment']};token_size={config['token_size']}".encode()
    )
    for spec in stats["manifest"]["tensors"]:
        fingerprint.update(f"{spec['key']}:{spec['shape']}".encode())
    for source in stats["source_files"]:
        path = Path(source["path"])
        if verify_sources:
            if not path.is_file():
                raise FileNotFoundError(f"Stats source file no longer exists: {path}")
            current = path.stat()
            if current.st_size != source["size"] or current.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(f"Stats source file changed after cache creation: {path}")
        fingerprint.update(f"{path}:{source['size']}:{source['mtime_ns']}".encode())
    if fingerprint.hexdigest() != stats["fingerprint"]:
        raise ValueError("stats.pt fingerprint is inconsistent with its manifest or source files")


def structure_from_manifest(manifest: Mapping[str, Any], batch_size: int = 1) -> dict[str, torch.Tensor]:
    names = ("tensor_ids", "token_ids", "block_ids", "role_ids", "kind_ids")
    return {name: manifest[name].unsqueeze(0).expand(batch_size, -1) for name in names}
