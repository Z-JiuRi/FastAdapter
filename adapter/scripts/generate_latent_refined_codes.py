#!/usr/bin/env python3
"""Offline per-sample code refinement through a frozen decoder.

For each (source_code, csi):
  z0 = source @ W + b   (ridge affine to teacher codes)
  z  = Adam-min_z ||D_frozen(z) - csi||^2 for `steps` iterations
Save z as distillation targets for a feedforward adapter.

This is train-time only; inference still uses a frozen E + trained adapter + frozen D.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "base"
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from models import universal_csi  # noqa: E402


def fit_affine(source: torch.Tensor, target: torch.Tensor, ridge: float = 0.0):
    # Match adapter/train_adapter.py::fit_affine: z = source @ W + b.
    dim = source.size(1)
    src = source.to(torch.float64)
    tgt = target.to(torch.float64)
    ones = torch.ones(src.size(0), 1, dtype=src.dtype)
    aug = torch.cat([src, ones], dim=1)
    reg = ridge * torch.eye(dim + 1, dtype=src.dtype)
    reg[-1, -1] = 0.0
    solution = torch.linalg.solve(aug.t().matmul(aug) + reg, aug.t().matmul(tgt))
    return solution[:-1].float().contiguous(), solution[-1].float().contiguous()


def load_model(exp: Path, device: torch.device):
    cfg = json.loads((exp / "args.json").read_text())
    model = universal_csi(
        encoder_name=cfg["encoder"],
        decoder_name=cfg["decoder"],
        reduction=cfg["cr"],
        d_model=cfg["d_model"],
        channel=cfg["channel"],
        nt=cfg["nt"],
        nc=cfg["nc"],
        dim_feedforward=cfg["dim_feedforward"],
    )
    ckpt = torch.load(
        exp / "checkpoints/best_nmse.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    ckpt = {
        k: v
        for k, v in ckpt.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }
    model.load_state_dict(ckpt)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


@torch.no_grad()
def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    err = (pred - target).double().square().sum()
    power = target.double().square().sum().clamp_min(1e-12)
    return float((10.0 * torch.log10(err / power)).cpu())


def refine_split(
    target_decoder,
    source: torch.Tensor,
    csi: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    report_ks=None,
    init_mode: str = "reencode",
    loss_target: str = "source_recon",
    source_decoder=None,
    target_encoder=None,
):
    """Refine codes through frozen target decoder.

    init_mode:
      - affine: z0 = source @ W + b  (weak basin for GD)
      - reencode: z0 = E_target(D_source(source))  (strong basin; needs D_s/E_t offline)
    loss_target:
      - gt: match ground-truth CSI
      - source_recon: match D_source(source)  (matches prior oracle that reached ~-26dB)
    """
    if report_ks is None:
        report_ks = sorted({0, 1, 2, 5, 10, steps})
        report_ks = [k for k in report_ks if k <= steps]
    if init_mode not in ("affine", "reencode"):
        raise ValueError(init_mode)
    if loss_target not in ("gt", "source_recon"):
        raise ValueError(loss_target)
    if init_mode == "reencode" or loss_target == "source_recon":
        if source_decoder is None:
            raise ValueError("source_decoder required for reencode/source_recon")
    if init_mode == "reencode" and target_encoder is None:
        raise ValueError("target_encoder required for reencode init")

    refined = torch.empty_like(source)
    err = {k: torch.zeros((), dtype=torch.float64, device=device) for k in report_ks}
    power = torch.zeros((), dtype=torch.float64, device=device)
    n = source.size(0)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        zs = source[start:end].to(device, non_blocking=True)
        xb = csi[start:end].to(device, non_blocking=True)
        with torch.no_grad():
            if source_decoder is not None:
                xs = source_decoder(zs)
            else:
                xs = None
            if init_mode == "affine":
                z0 = zs.matmul(weight) + bias
            else:
                z0 = target_encoder(xs)
            if loss_target == "gt":
                y = xb
            else:
                y = xs

        z = z0.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)

        for k in range(steps + 1):
            if k in err:
                with torch.no_grad():
                    pred = target_decoder(z.detach())
                    err[k] += (pred - xb).double().square().sum()
            if k == steps:
                break
            pred = target_decoder(z)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        refined[start:end] = z.detach().cpu()
        power += xb.double().square().sum()

    metrics = {
        str(k): float((10.0 * torch.log10(err[k] / power.clamp_min(1e-12))).cpu())
        for k in report_ks
    }
    return refined, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_exp", required=True)
    p.add_argument("--target_exp", required=True)
    p.add_argument("--train_csi", required=True)
    p.add_argument("--val_csi", required=True)
    p.add_argument("--test_csi", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--align_ridge", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_eval", type=int, default=0)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    p.add_argument(
        "--init_mode",
        default="reencode",
        choices=["affine", "reencode"],
        help="Code init for Adam refine. reencode matches the strong oracle.",
    )
    p.add_argument(
        "--loss_target",
        default="source_recon",
        choices=["gt", "source_recon"],
        help="What D_target(z) is pulled toward during refine.",
    )
    return p.parse_args()


def run_refinement(
    args,
    *,
    preloaded_csi=None,
    preloaded_source_codes=None,
    preloaded_target_codes=None,
):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    source_exp = Path(args.source_exp)
    target_exp = Path(args.target_exp)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_model, _ = load_model(target_exp, device)
    source_model, _ = load_model(source_exp, device)
    target_decoder = target_model.decoder
    source_decoder = source_model.decoder
    target_encoder = target_model.encoder

    def load_code(split: str, max_n: int):
        if preloaded_source_codes is None:
            code = torch.load(
                source_exp / "codewords" / f"{split}_code.pt",
                weights_only=True,
                map_location="cpu",
            ).float()
        else:
            code = preloaded_source_codes[split]
        if preloaded_target_codes is None:
            teacher = torch.load(
                target_exp / "codewords" / f"{split}_code.pt",
                weights_only=True,
                map_location="cpu",
            ).float()
        else:
            teacher = preloaded_target_codes[split]
        if max_n > 0:
            code = code[:max_n]
            teacher = teacher[:max_n]
        return code, teacher

    def load_csi(split: str, path: str, max_n: int):
        if preloaded_csi is None:
            x = torch.load(path, weights_only=True, map_location="cpu").float()
        else:
            x = preloaded_csi[split]
        if max_n > 0:
            x = x[:max_n]
        return x

    train_src, train_tgt = load_code(
        "train", args.max_train if "train" in args.splits else 4096)
    weight, bias = fit_affine(train_src, train_tgt, ridge=args.align_ridge)
    torch.save({"weight": weight, "bias": bias}, out_dir / "affine_alignment.pt")

    csi_paths = {
        "train": args.train_csi,
        "val": args.val_csi,
        "test": args.test_csi,
    }
    all_metrics = {
        "steps": args.steps,
        "lr": args.lr,
        "align_ridge": args.align_ridge,
        "init_mode": args.init_mode,
        "loss_target": args.loss_target,
        "source_exp": str(source_exp),
        "target_exp": str(target_exp),
    }

    for split in args.splits:
        max_n = args.max_train if split == "train" else args.max_eval
        src, _ = load_code(split, max_n)
        csi = load_csi(split, csi_paths[split], max_n)
        assert src.size(0) == csi.size(0), (split, src.shape, csi.shape)
        refined, metrics = refine_split(
            target_decoder,
            src,
            csi,
            weight.to(device),
            bias.to(device),
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            init_mode=args.init_mode,
            loss_target=args.loss_target,
            source_decoder=source_decoder,
            target_encoder=target_encoder,
        )
        torch.save(refined, out_dir / f"{split}_refined_code.pt")
        all_metrics[split] = {
            "n": int(src.size(0)),
            "refine_nmse_vs_gt": metrics,
            "init_nmse": metrics.get("0"),
            "final_nmse": metrics.get(str(args.steps)),
        }
        print(json.dumps({split: all_metrics[split]}, indent=2), flush=True)

    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(json.dumps(all_metrics, indent=2), flush=True)


def main():
    run_refinement(parse_args())


if __name__ == "__main__":
    main()
