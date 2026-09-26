from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) in sys.path:
    sys.path.remove(str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT))

from alignment.losses import (
    position_residual_cosine_loss,
    task_level_alignment_loss,
)
from alignment.models import TokenContextAlignmentModel, GramRowAlignmentModel
from core.model_factory import load_stats
from data.dataloader import get_dataloaders
from utils.distributed import cleanup_distributed, init_distributed, is_main_process


SUBMIT_ROOT = Path(__file__).resolve().parents[2]
STRUCTURE_FIELDS = ("tensor_ids", "token_ids", "block_ids", "role_ids", "kind_ids")


def parse_args():
    parser = argparse.ArgumentParser("Train token_context to adapter-token alignment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def _load_config_recursive(config_path: Path):
    cfg = OmegaConf.load(config_path)
    base_path = cfg.pop("_base_", None)
    if not base_path:
        return cfg
    base_path = Path(str(base_path))
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    return OmegaConf.merge(_load_config_recursive(base_path.resolve()), cfg)


def load_config(path: str, overrides: list[str]):
    cfg = _load_config_recursive(Path(path).resolve())
    if overrides:
        invalid = [item for item in overrides if "=" not in item]
        if invalid:
            raise ValueError("--override entries must be KEY=VALUE: " + ", ".join(invalid))
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    for field in ("data.task_root", "cache.stats_path", "alignment.exp_dir", "alignment.resume"):
        value = OmegaConf.select(cfg, field)
        if value in (None, ""):
            continue
        path_value = Path(str(value))
        if not path_value.is_absolute():
            OmegaConf.update(
                cfg, field, str((SUBMIT_ROOT / path_value).resolve()),
                merge=False)
    return cfg


def setup(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(exp_dir: Path, rank: int):
    exp_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if rank == 0:
        log_name = time.strftime("train_%Y%m%d_%H%M%S.log")
        handlers.append(logging.FileHandler(exp_dir / log_name, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def move_batch(batch: dict[str, Any], device: torch.device):
    structure = {name: batch[name].to(device, non_blocking=True) for name in STRUCTURE_FIELDS}
    raw_cond = batch.get("raw_cond")
    return (
        batch["cond"].to(device, non_blocking=True),
        raw_cond.to(device, non_blocking=True) if raw_cond is not None else None,
        batch["tokens"].to(device, non_blocking=True),
        batch["token_mask"].to(device, non_blocking=True),
        structure,
    )


def ddp_wrap(model: torch.nn.Module, distributed: bool, device: torch.device):
    model = model.to(device)
    if not distributed:
        return model
    return DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.cpu())


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best: float, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val_loss": best,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("epoch", 0)), float(payload.get("best_val_loss", float("inf")))


def build_scheduler(optimizer, cfg):
    scheduler_type = str(cfg.train.get("scheduler", "warmup_cosine"))
    if scheduler_type == "none":
        return None
    if scheduler_type != "warmup_cosine":
        raise ValueError("train.scheduler must be warmup_cosine or none")
    epochs = max(1, int(cfg.train.epochs))
    warmup_epochs = max(0, int(cfg.train.get("warmup_epochs", 0)))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.01))

    def lr_lambda(epoch_index: int) -> float:
        current = epoch_index + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return current / warmup_epochs
        progress = (current - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(model, loader, optimizer, cfg, device, train: bool):
    model.train(train)
    totals = {"loss": 0.0, "contrastive": 0.0, "cosine": 0.0, "top1": 0.0, "top5": 0.0,
              "pos": 0.0, "neg": 0.0, "pos_resid": 0.0}
    count = 0
    level = str(cfg.alignment.get("level", "task"))
    if level != "task":
        raise ValueError("alignment.level must be 'task' for the paper objective")
    if "position_residual_weight" not in cfg.alignment:
        raise ValueError("alignment.position_residual_weight must be set explicitly")
    position_residual_weight = float(cfg.alignment.position_residual_weight)
    if position_residual_weight <= 0:
        raise ValueError("alignment.position_residual_weight must be positive for the paper objective")
    if float(cfg.alignment.task_cosine_weight) <= 0:
        raise ValueError("alignment.task_cosine_weight must be positive for the paper objective")
    max_batches = int(cfg.train.get("max_train_batches" if train else "max_val_batches", 0))
    for batch_index, batch in enumerate(loader, start=1):
        cond, raw_cond, tokens, token_mask, structure = move_batch(batch, device)
        if cond.shape[0] < 2:
            continue  # InfoNCE has no negative task in a singleton final batch.
        position_residual = torch.zeros((), device=device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                output = model(cond, tokens, token_mask, structure, raw_cond=raw_cond)
                loss_output = task_level_alignment_loss(
                    output["context_global"],
                    output["target_global"],
                    temperature=float(cfg.alignment.task_temperature),
                    cosine_weight=float(cfg.alignment.task_cosine_weight),
                )
                if position_residual_weight > 0.0:
                    position_residual = position_residual_cosine_loss(
                        output["token_context"],
                        output["parameter_embedding"],
                        output["valid"],
                    )
            loss = loss_output.loss + position_residual_weight * position_residual
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
        batch_count = cond.shape[0]
        count += batch_count
        totals["loss"] += float(loss.detach()) * batch_count
        totals["contrastive"] += float(loss_output.contrastive.detach()) * batch_count
        totals["cosine"] += float(loss_output.cosine.detach()) * batch_count
        totals["top1"] += float(loss_output.top1.detach()) * batch_count
        totals["top5"] += float(loss_output.top5.detach()) * batch_count
        totals["pos"] += float(loss_output.positive_logit.detach()) * batch_count
        totals["neg"] += float(loss_output.negative_logit.detach()) * batch_count
        totals["pos_resid"] += float(position_residual.detach()) * batch_count
        if max_batches > 0 and batch_index >= max_batches:
            break
    if count == 0:
        raise ValueError("task-level alignment requires at least one batch of two encoder tasks")
    metrics = {key: reduce_mean(value / max(count, 1), device) for key, value in totals.items()}
    metrics["tasks"] = reduce_mean(float(count), device)
    return metrics


def main():
    args = parse_args()
    dist_state = init_distributed()
    cfg = load_config(args.config, args.override)
    if dist_state["enabled"]:
        OmegaConf.update(cfg, "data.device", f"cuda:{dist_state['local_rank']}", merge=False)
    device = torch.device(str(cfg.data.device) if torch.cuda.is_available() else "cpu")
    setup(int(cfg.train.seed) + int(dist_state["rank"]))
    exp_dir = Path(str(cfg.alignment.exp_dir))
    setup_logging(exp_dir, int(dist_state["rank"]))
    if is_main_process():
        (exp_dir / "config.resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        logging.info("token-context alignment | device=%s distributed=%s", device, dist_state)
    stats = load_stats(cfg)
    train_loader, val_loader = get_dataloaders(cfg, stats=stats, distributed=bool(dist_state["enabled"]))
    model_type = str(cfg.alignment.get("model_type", "token_context"))
    if model_type == "token_context":
        model = TokenContextAlignmentModel(cfg, stats["manifest"])
    elif model_type == "gram_row":
        model = GramRowAlignmentModel(cfg, stats["manifest"])
    else:
        raise ValueError("alignment.model_type must be 'token_context' or 'gram_row'")
    model.detach_parameter_embedding = bool(cfg.alignment.get("detach_parameter_embedding", True))
    model = ddp_wrap(model, bool(dist_state["enabled"]), device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = build_scheduler(optimizer, cfg)
    start_epoch = 0
    best = float("inf")
    resume = str(cfg.alignment.get("resume", ""))
    if resume:
        start_epoch, best = load_checkpoint(resume, model, optimizer, scheduler)
        logging.info("resumed from %s epoch=%d best=%.6f", resume, start_epoch, best)

    stale_epochs = 0
    try:
        for epoch in range(start_epoch + 1, int(cfg.train.epochs) + 1):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            train_metrics = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
            if scheduler is not None:
                scheduler.step()
            val_metrics = run_epoch(model, val_loader, optimizer, cfg, device, train=False)
            metric_name = str(cfg.alignment.get("checkpoint_metric", "top1"))
            if metric_name not in ("loss", "top1"):
                raise ValueError("alignment.checkpoint_metric must be 'loss' or 'top1'")
            current = val_metrics["loss"] if metric_name == "loss" else -val_metrics["top1"]
            improved = current < best
            if improved:
                best = current
                stale_epochs = 0
            else:
                stale_epochs += 1
            if is_main_process():
                tc_gen = getattr(unwrap(model), "token_context_generator", None)
                gates = getattr(tc_gen, "last_condition_gates", {}) if tc_gen is not None else {}
                logging.info(
                    "epoch=%d lr=%.6g train=%s val=%s gates=%s",
                    epoch, optimizer.param_groups[0]["lr"], train_metrics, val_metrics, gates,
                )
                save_checkpoint(exp_dir / "checkpoints" / "last.pth", model, optimizer, scheduler, epoch, best, cfg)
                if improved:
                    save_checkpoint(exp_dir / "checkpoints" / "best.pth", model, optimizer, scheduler, epoch, best, cfg)
            patience = int(cfg.train.get("early_stop_patience", 0))
            if patience > 0 and stale_epochs >= patience:
                if is_main_process():
                    logging.info(
                        "early stop at epoch=%d metric=%s best=%.6f", epoch, metric_name, best
                    )
                break
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
