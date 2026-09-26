from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from data.adapter_tokenizer import load_adapter_state, tokenize_adapter, validate_stats_fingerprint


logger = logging.getLogger(__name__)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise TypeError(f"{path}: expected a tensor")
    value = value.detach().float().cpu().contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{path}: contains NaN or Inf")
    return value


def discover_tasks(task_root: str | Path, split: str, codeword_file: str = "gram_K128.pt") -> list[Path]:
    split_root = Path(task_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Task split does not exist: {split_root}")
    tasks = sorted(path.parent for path in split_root.glob(f"*/*/{codeword_file}"))
    if not tasks:
        raise ValueError(f"No tasks found under {split_root}/<encoder>/<seed>")
    return tasks


class CodewordAdapterDataset(Dataset[dict[str, Any]]):
    def __init__(self, cfg, split: str, stats: dict[str, Any] | None = None):
        self.cfg = cfg
        self.split = split
        self.tasks = discover_tasks(cfg.data.task_root, split, str(cfg.data.codeword_file))
        self.stats = stats or torch.load(cfg.cache.stats_path, map_location="cpu", weights_only=True)
        self.manifest = self.stats["manifest"]
        self.parameter_stats = self.stats["parameter_stats"]
        self.alignment = str(cfg.ablation.alignment)
        if self.alignment not in ("aligned", "raw"):
            raise ValueError("ablation.alignment must be 'aligned' or 'raw'")
        if self.stats.get("alignment") != self.alignment:
            raise ValueError(
                f"stats alignment={self.stats.get('alignment')!r} does not match "
                f"ablation.alignment={self.alignment!r}"
            )
        if int(cfg.data.token_size) != int(self.manifest["token_size"]):
            raise ValueError("data.token_size does not match cache manifest")
        self.adapter_filename = (
            str(cfg.data.adapter_file_aligned)
            if self.alignment == "aligned"
            else str(cfg.data.adapter_file_raw)
        )
        self.loading = str(cfg.data.get("loading", "lazy"))
        if self.loading not in ("lazy", "memory"):
            raise ValueError("data.loading must be lazy or memory")
        if str(cfg.ablation.get("code_normalization", "none")) != "none":
            raise NotImplementedError(
                "Codeword normalization is reserved for ablation; use none")
        self.preload_workers = int(cfg.data.get("preload_workers", 1))
        if self.preload_workers < 1:
            raise ValueError("data.preload_workers must be at least 1")
        self._memory = self._preload_tasks() if self.loading == "memory" else None

    def __len__(self) -> int:
        return len(self.tasks)

    def _preload_tasks(self) -> list[dict[str, Any]]:
        start_time = time.time()
        if self.preload_workers == 1:
            memory = [self._load_task(path) for path in self.tasks]
        else:
            memory: list[dict[str, Any] | None] = [None] * len(self.tasks)
            with ThreadPoolExecutor(max_workers=self.preload_workers) as executor:
                futures = {
                    executor.submit(self._load_task, path): index
                    for index, path in enumerate(self.tasks)
                }
                for future in as_completed(futures):
                    memory[futures[future]] = future.result()
            memory = [item for item in memory if item is not None]
        logger.info(
            "=> Preloaded split=%s tasks=%d workers=%d time=%.2fs",
            self.split, len(memory), self.preload_workers, time.time() - start_time,
        )
        return memory

    def _load_task(self, task_dir: Path) -> dict[str, Any]:
        cond = _load_tensor(task_dir / str(self.cfg.data.codeword_file))
        expected_shape = tuple(int(value) for value in self.cfg.data.cond_shape)
        if tuple(cond.shape) != expected_shape:
            raise ValueError(f"{task_dir}: condition shape {tuple(cond.shape)} != {expected_shape}")
        raw_file = str(self.cfg.data.get("raw_codeword_file", ""))
        raw_cond = None
        if raw_file:
            raw_cond = _load_tensor(task_dir / raw_file)
            raw_shape = tuple(int(value) for value in self.cfg.data.raw_cond_shape)
            if tuple(raw_cond.shape) != raw_shape:
                raise ValueError(
                    f"{task_dir}: raw probe shape {tuple(raw_cond.shape)} != {raw_shape}"
                )
        state = load_adapter_state(task_dir / self.adapter_filename)
        tokens = tokenize_adapter(state, self.manifest, self.parameter_stats)
        meta_path = task_dir / "adapter_meta.json"
        adapter_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        result: dict[str, Any] = {
            "cond": cond,
            "tokens": tokens,
            "token_mask": self.manifest["token_mask"].clone(),
            "meta": {
                "task_dir": str(task_dir),
                "split": self.split,
                "encoder": task_dir.parent.name,
                "seed": task_dir.name,
                "decoder_nmse": float(adapter_meta.get("decoder_nmse", float("nan"))),
            },
        }
        if raw_cond is not None:
            result["raw_cond"] = raw_cond
        for name in ("tensor_ids", "token_ids", "block_ids", "role_ids", "kind_ids"):
            result[name] = self.manifest[name].clone()
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._memory[index] if self._memory is not None else self._load_task(self.tasks[index])


class SharedCodewordAdapterDataset(Dataset[dict[str, Any]]):
    """See README.md for English documentation."""

    def __init__(self, split_payload: dict[str, Any], stats: dict[str, Any], split: str):
        self.split = split
        self.cond = split_payload["cond"]
        self.tokens = split_payload["tokens"]
        self.raw_cond = split_payload.get("raw_cond")
        self.meta = split_payload["meta"]
        self.manifest = stats["manifest"]
        if len(self.meta) != self.cond.shape[0] or len(self.meta) != self.tokens.shape[0]:
            raise ValueError(
                f"shared bundle split={split} has inconsistent counts: "
                f"meta={len(self.meta)} cond={self.cond.shape[0]} tokens={self.tokens.shape[0]}"
            )

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "cond": self.cond[index],
            "tokens": self.tokens[index],
            "token_mask": self.manifest["token_mask"],
            "meta": dict(self.meta[index]),
        }
        if self.raw_cond is not None:
            result["raw_cond"] = self.raw_cond[index]
        for name in ("tensor_ids", "token_ids", "block_ids", "role_ids", "kind_ids"):
            result[name] = self.manifest[name]
        return result


def _as_shared_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous().share_memory_()


def _lazy_cfg(cfg):
    cloned = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cloned.data.loading = "lazy"
    return cloned


def build_shared_data_bundle(
    cfg,
    stats: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """See README.md for English documentation."""

    started = time.time()
    bundle: dict[str, Any] = {"splits": {}, "alignment": str(cfg.ablation.alignment)}
    for split in (str(cfg.data.train_split), str(cfg.data.val_split)):
        dataset = CodewordAdapterDataset(_lazy_cfg(cfg), split, stats)
        workers = int(cfg.data.get("preload_workers", 1))
        if progress is not None:
            progress(f"preload split={split} tasks={len(dataset.tasks)} workers={workers}")
        if workers == 1:
            items = []
            for index, path in enumerate(dataset.tasks, start=1):
                items.append(dataset._load_task(path))
                if progress is not None and (index % 50 == 0 or index == len(dataset.tasks)):
                    progress(f"preload split={split} progress={index}/{len(dataset.tasks)}")
        else:
            items: list[dict[str, Any] | None] = [None] * len(dataset.tasks)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(dataset._load_task, path): index
                    for index, path in enumerate(dataset.tasks)
                }
                completed = 0
                for future in as_completed(futures):
                    items[futures[future]] = future.result()
                    completed += 1
                    if progress is not None and (completed % 50 == 0 or completed == len(dataset.tasks)):
                        progress(f"preload split={split} progress={completed}/{len(dataset.tasks)}")
            items = [item for item in items if item is not None]
        if progress is not None:
            progress(f"stack/share split={split} tasks={len(items)}")
        split_payload = {
            "cond": _as_shared_tensor(torch.stack([item["cond"] for item in items], dim=0)),
            "tokens": _as_shared_tensor(torch.stack([item["tokens"] for item in items], dim=0)),
            "meta": [dict(item["meta"]) for item in items],
        }
        if "raw_cond" in items[0]:
            split_payload["raw_cond"] = _as_shared_tensor(
                torch.stack([item["raw_cond"] for item in items], dim=0)
            )
        bundle["splits"][split] = split_payload
        logger.info(
            "=> Built shared split=%s tasks=%d cond=%s tokens=%s",
            split, len(items), tuple(bundle["splits"][split]["cond"].shape),
            tuple(bundle["splits"][split]["tokens"].shape),
        )
    logger.info("=> Shared data bundle ready: time=%.2fs", time.time() - started)
    return bundle


def get_dataloaders(
    cfg,
    stats: dict[str, Any] | None = None,
    shared_bundle: dict[str, Any] | None = None,
    distributed: bool = False,
):
    stats_path = Path(str(cfg.cache.stats_path))
    if stats is None:
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Offline statistics file not found: {stats_path}. "
                "Run cache/build_stats.py before training."
            )
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        if bool(cfg.cache.get("strict_fingerprint", True)):
            validate_stats_fingerprint(stats)
    if shared_bundle is None:
        train_dataset = CodewordAdapterDataset(cfg, str(cfg.data.train_split), stats)
        val_dataset = CodewordAdapterDataset(cfg, str(cfg.data.val_split), stats)
    else:
        train_dataset = SharedCodewordAdapterDataset(
            shared_bundle["splits"][str(cfg.data.train_split)], stats, str(cfg.data.train_split)
        )
        val_dataset = SharedCodewordAdapterDataset(
            shared_bundle["splits"][str(cfg.data.val_split)], stats, str(cfg.data.val_split)
        )
    common = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.get("pin_memory", False)),
    }
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.train.get("val_batch_size", cfg.train.batch_size)),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader
