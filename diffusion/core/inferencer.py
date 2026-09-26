from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from core.model_factory import build_generation_models, load_stats
from data.adapter_tokenizer import _reconstruct_adapter_state, structure_from_manifest
from utils.init import seed_everything
from utils.tools import log_runtime_context, setup_logger


BASE_ROOT = Path(__file__).resolve().parents[2] / "base"
logger = logging.getLogger(__name__)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value):
        raise TypeError(f"{path}: expected a tensor")
    return value.detach().float().cpu()


def _load_root_model_package():
    package_name = "adapter_csi_root_models"
    if package_name in sys.modules:
        return sys.modules[package_name]
    package_dir = BASE_ROOT / "models"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load root models package from {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def _clean_checkpoint_state(checkpoint: Mapping) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if torch.is_tensor(value) and not str(key).endswith(("total_ops", "total_params"))
    }


def load_frozen_decoder(args_path: str | Path, checkpoint_path: str | Path, device: torch.device):
    args = json.loads(Path(args_path).read_text(encoding="utf-8"))
    models_package = _load_root_model_package()
    decoder = models_package.build_decoder(
        args["decoder"], args["cr"], args.get("d_model", 64),
        args.get("channel", 2), args.get("nt", 32), args.get("nc", 32),
        args.get("dim_feedforward", 2048), args.get("hidden", 16),
        args.get("num_blocks", 2),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _clean_checkpoint_state(checkpoint)
    decoder_state = {
        key.removeprefix("decoder."): value for key, value in state.items()
        if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = state
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder.to(device).eval().requires_grad_(False), args


def functional_adapter(codewords: torch.Tensor, params: Mapping[str, torch.Tensor], residual_scale: float):
    weight = params["alignment_weight"]
    batched_params = weight.ndim == 3
    squeeze = codewords.ndim == 2 and batched_params
    if squeeze:
        codewords = codewords.unsqueeze(0)
    if batched_params:
        if codewords.ndim != 3 or codewords.shape[0] != weight.shape[0]:
            raise ValueError(
                "Batched Adapter parameters require codewords shaped (B,N,C)")
        value = torch.matmul(codewords, weight) + params["alignment_bias"][:, None]
    else:
        value = codewords.matmul(weight) + params["alignment_bias"]
    for block in range(4):
        prefix = f"blocks.{block}."
        if batched_params:
            mean = value.mean(dim=-1, keepdim=True)
            variance = value.var(dim=-1, unbiased=False, keepdim=True)
            hidden = (value - mean) * torch.rsqrt(variance + 1e-5)
            hidden = hidden * params[prefix + "norm.weight"][:, None]
            hidden = hidden + params[prefix + "norm.bias"][:, None]
            hidden = torch.matmul(
                hidden, params[prefix + "net.0.weight"].transpose(-1, -2)
            ) + params[prefix + "net.0.bias"][:, None]
            hidden = torch.nn.functional.gelu(hidden)
            hidden = torch.matmul(
                hidden, params[prefix + "net.3.weight"].transpose(-1, -2)
            ) + params[prefix + "net.3.bias"][:, None]
        else:
            hidden = torch.nn.functional.layer_norm(
                value, (value.shape[-1],), params[prefix + "norm.weight"],
                params[prefix + "norm.bias"], 1e-5,
            )
            hidden = torch.nn.functional.gelu(torch.nn.functional.linear(
                hidden, params[prefix + "net.0.weight"], params[prefix + "net.0.bias"]
            ))
            hidden = torch.nn.functional.linear(
                hidden, params[prefix + "net.3.weight"], params[prefix + "net.3.bias"]
            )
        value = value + residual_scale * hidden
    return value[0] if squeeze else value


class Inferencer:
    def __init__(self, cfg, force_eval: bool = False):
        self.cfg = cfg
        self.device = torch.device(str(cfg.data.device))
        self.output_dir = Path(str(cfg.inference.output_dir))
        setup_logger(self.output_dir)
        log_runtime_context(logger, cfg, self.output_dir)
        seed_everything(int(cfg.inference.seed))
        self.eval_enabled = force_eval or bool(cfg.eval.get("enabled", False))
        self.stats = load_stats(cfg, verify_sources=False)
        self.manifest = self.stats["manifest"]
        self.token_context_generator, self.denoiser, self.diffusion = build_generation_models(
            cfg, self.stats, self.device
        )
        self.null_context = nn.Parameter(torch.zeros(
            1,
            int(self.manifest["num_tokens"]),
            int(cfg.token_context.hidden_dim),
            device=self.device,
        ), requires_grad=False)
        self._load_checkpoint(Path(str(cfg.inference.checkpoint_path)))
        self.token_context_generator.eval()
        self.denoiser.eval()
        self.diffusion.eval()
        self._eval_decoder = None
        self._eval_decoder_args = None
        self._eval_csi = None
        logger.info(
            "=> Inference initialized: checkpoint=%s device=%s eval=%s split=%s",
            cfg.inference.checkpoint_path, self.device, self.eval_enabled,
            cfg.inference.cond_path,
        )

    def _load_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("stats_fingerprint") not in (None, self.stats.get("fingerprint")):
            raise ValueError("Checkpoint and stats.pt fingerprints differ")
        use_ema = bool(self.cfg.inference.use_ema) and "ema_shadow" in checkpoint
        if use_ema:
            shadow = checkpoint["ema_shadow"]
            for prefix, module in (("token_context_generator", self.token_context_generator), ("denoiser", self.denoiser)):
                state = module.state_dict()
                for name in state:
                    ema_name = f"{prefix}.{name}"
                    if ema_name in shadow:
                        state[name] = shadow[ema_name]
                module.load_state_dict(state)
            if "null_context_wrapper.0" in shadow:
                self.null_context.data.copy_(shadow["null_context_wrapper.0"].to(self.device))
            logger.info("=> Loaded EMA weights from checkpoint")
        else:
            if bool(self.cfg.inference.use_ema):
                logger.info("=> EMA requested but checkpoint has no ema_shadow; loaded raw weights")
            self.token_context_generator.load_state_dict(checkpoint["token_context_generator"])
            self.denoiser.load_state_dict(checkpoint["denoiser"])
            if "null_context" in checkpoint:
                self.null_context.data.copy_(checkpoint["null_context"].to(self.device))

    def _condition_tasks(self) -> list[tuple[Path, Path]]:
        source = Path(str(self.cfg.inference.cond_path))
        codeword_file = str(self.cfg.data.codeword_file)
        if source.is_file():
            return [(source.parent, source)]
        if (source / codeword_file).is_file():
            return [(source, source / codeword_file)]
        files = sorted(source.glob(f"**/{codeword_file}"))
        if not files:
            raise FileNotFoundError(f"No {codeword_file} found under {source}")
        return [(path.parent, path) for path in files]

    def _output_path(self, task_dir: Path) -> Path:
        source = Path(str(self.cfg.inference.cond_path))
        base = source if source.is_dir() else source.parent
        try:
            relative = task_dir.relative_to(base)
        except ValueError:
            relative = Path(task_dir.parent.name) / task_dir.name
        return self.output_dir / relative / "generated_adapter.pth"

    @torch.no_grad()
    def _generate_single(self, cond_path: Path, raw_cond_path: Path | None = None):
        cond = _load_tensor(cond_path)
        expected = tuple(int(value) for value in self.cfg.data.cond_shape)
        if tuple(cond.shape) != expected:
            raise ValueError(f"{cond_path}: condition shape {tuple(cond.shape)} != {expected}")
        raw_cond = None
        if raw_cond_path is not None:
            raw_cond = _load_tensor(raw_cond_path)
            expected_raw = tuple(int(value) for value in self.cfg.data.raw_cond_shape)
            if tuple(raw_cond.shape) != expected_raw:
                raise ValueError(
                    f"{raw_cond_path}: raw condition shape {tuple(raw_cond.shape)} != {expected_raw}"
                )
        structure = {
            key: value.to(self.device)
            for key, value in structure_from_manifest(self.manifest).items()
        }
        token_context = self.token_context_generator(
            cond.unsqueeze(0).to(self.device), structure,
            raw_cond.unsqueeze(0).to(self.device) if raw_cond is not None else None,
        )
        samples = self.diffusion.sample(
            cond=token_context,
            shape=(1, int(self.manifest["num_tokens"]), int(self.manifest["token_size"])),
            use_ddim=bool(self.cfg.inference.use_ddim),
            ddim_steps=int(self.cfg.inference.ddim_steps),
            eta=float(self.cfg.inference.eta),
            structure=structure,
            cfg_scale=float(self.cfg.inference.cfg_scale),
            uncond_cond=self.null_context,
        )
        return self._reconstruct_adapter_state(samples[0])

    def _reconstruct_adapter_state(self, tokens: torch.Tensor):
        return _reconstruct_adapter_state(
            tokens, self.manifest, self.stats["parameter_stats"]
        )

    @torch.no_grad()
    def inference(self) -> list[dict]:
        records = []
        tasks = self._condition_tasks()
        logger.info("=> Generation started: tasks=%d", len(tasks))
        for task_index, (task_dir, cond_path) in enumerate(tasks, start=1):
            raw_name = str(self.cfg.data.get("raw_codeword_file", ""))
            raw_cond_path = task_dir / raw_name if raw_name else None
            if raw_cond_path is not None and not raw_cond_path.is_file():
                raise FileNotFoundError(f"Missing raw probe condition for {cond_path}: {raw_cond_path}")
            state = self._generate_single(cond_path, raw_cond_path)
            output_path = self._output_path(task_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, output_path)
            generation_meta = {
                "condition_path": str(cond_path),
                "raw_condition_path": str(raw_cond_path) if raw_cond_path is not None else None,
                "condition_type": str(self.cfg.condition_encoder.get("type", "legacy")),
                "checkpoint_path": str(self.cfg.inference.checkpoint_path),
                "stats_fingerprint": self.stats.get("fingerprint"),
                "cfg_scale": float(self.cfg.inference.cfg_scale),
                "use_ddim": bool(self.cfg.inference.use_ddim),
                "sampling_steps": (
                    int(self.cfg.inference.ddim_steps)
                    if bool(self.cfg.inference.use_ddim)
                    else int(self.cfg.diffusion.timesteps)
                ),
                "eta": float(self.cfg.inference.eta),
                "residual_scale": float(self.cfg.adapter.residual_scale),
                "alignment": str(self.cfg.ablation.alignment),
                "position_2d": str(self.cfg.structure_embedding.position_2d.type),
                "token_context_implementation": str(self.cfg.token_context.get("implementation", "transformer")),
                "condition_injection": str(self.cfg.token_context.get("condition_injection", "cross_attention")),
                "condition_branches": list(self.cfg.condition_encoder.branches),
                "denoiser_type": str(self.cfg.denoiser.type),
            }
            output_path.with_suffix(".meta.json").write_text(
                json.dumps(generation_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            record = {"task_dir": str(task_dir), "output_path": str(output_path)}
            logger.info(
                "=> Generated Adapter: [%d/%d] task=%s output=%s",
                task_index, len(tasks), task_dir, output_path,
            )
            if self.eval_enabled:
                metrics = self.evaluate_adapter(task_dir, state)
                record.update(metrics)
                logger.info(
                    "=> Decoder NMSE: [%d/%d] task=%s generated=%.4f dB "
                    "meta=%.4f dB gap=%+.4f dB",
                    task_index, len(tasks), task_dir,
                    metrics["generated_nmse_db"], metrics["meta_decoder_nmse_db"],
                    metrics["gap_db"],
                )
            records.append(record)
        if self.eval_enabled:
            self._write_eval_summary(records)
        logger.info("=> Generation complete: tasks=%d", len(records))
        return records

    def _load_eval_context(self):
        """See README.md for English documentation."""
        if self._eval_decoder is not None:
            return self._eval_decoder, self._eval_decoder_args, self._eval_csi
        decoder, decoder_args = load_frozen_decoder(
            self.cfg.eval.decoder_args, self.cfg.eval.decoder_checkpoint, self.device
        )
        csi_path = (self.cfg.eval.get("csi_path") or
                    self.cfg.eval.get("csi_val_path") or decoder_args.get("val_path"))
        if not csi_path:
            raise ValueError(
                "eval.csi_path is empty and decoder args have no val_path")
        csi_path = Path(str(csi_path))
        if not csi_path.is_file():
            raise FileNotFoundError(
                f"Evaluation CSI does not exist: {csi_path}. Set a valid "
                "path through eval.csi_path."
            )
        self._eval_decoder = decoder
        self._eval_decoder_args = decoder_args
        self._eval_csi = _load_tensor(csi_path)
        logger.info(
            "=> Evaluation assets loaded: decoder=%s checkpoint=%s csi=%s samples=%d",
            decoder_args.get("decoder"), self.cfg.eval.decoder_checkpoint,
            csi_path, self._eval_csi.shape[0],
        )
        return self._eval_decoder, self._eval_decoder_args, self._eval_csi

    @torch.no_grad()
    def evaluate_adapter(self, task_dir: Path, state: Mapping[str, torch.Tensor]) -> dict:
        decoder, decoder_args, csi = self._load_eval_context()
        codewords = _load_tensor(task_dir / str(self.cfg.data.query_codeword_file))
        if codewords.shape[0] != csi.shape[0]:
            raise ValueError(
                f"Evaluation codeword count {codewords.shape[0]} does not "
                f"match CSI count {csi.shape[0]}"
            )
        batch_size = int(self.cfg.eval.batch_size)

        def calculate_nmse(adapter_state: Mapping[str, torch.Tensor]) -> float:
            state_device = {key: value.to(self.device) for key, value in adapter_state.items()}
            error_energy = torch.zeros((), dtype=torch.float64, device=self.device)
            signal_energy = torch.zeros((), dtype=torch.float64, device=self.device)
            for start in range(0, codewords.shape[0], batch_size):
                codes = codewords[start:start + batch_size].to(self.device)
                target = csi[start:start + batch_size].to(self.device)
                mapped = functional_adapter(codes, state_device, float(self.cfg.adapter.residual_scale))
                prediction = decoder(mapped)
                if target.shape != prediction.shape:
                    target = target.reshape_as(prediction)
                error_energy += (prediction.double() - target.double()).square().sum()
                signal_energy += target.double().square().sum()
            value = 10 * torch.log10(
                error_energy / signal_energy.clamp_min(torch.finfo(torch.float64).tiny)
            )
            return float(value.cpu())

        generated_nmse = calculate_nmse(state)
        meta_path = task_dir / "adapter_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if "decoder_nmse" not in meta:
            raise KeyError(f"{meta_path}: decoder_nmse is required for evaluation")
        meta_nmse = float(meta["decoder_nmse"])
        result = {
            "generated_nmse_db": generated_nmse,
            "meta_decoder_nmse_db": meta_nmse,
            "gap_db": generated_nmse - meta_nmse,
            "decoder": str(decoder_args.get("decoder")),
        }
        if bool(self.cfg.eval.get("recompute_teacher_nmse", False)):
            teacher_name = (
                str(self.cfg.data.adapter_file_aligned)
                if str(self.cfg.ablation.alignment) == "aligned"
                else str(self.cfg.data.adapter_file_raw)
            )
            teacher = torch.load(task_dir / teacher_name, map_location="cpu", weights_only=True)
            teacher_nmse = calculate_nmse(teacher)
            result["recomputed_teacher_nmse_db"] = teacher_nmse
            result["teacher_meta_gap_db"] = teacher_nmse - meta_nmse
        return result

    def _write_eval_summary(self, records: list[dict]) -> None:
        generated = torch.tensor([item["generated_nmse_db"] for item in records], dtype=torch.float64)
        meta = torch.tensor([item["meta_decoder_nmse_db"] for item in records], dtype=torch.float64)
        summary = {
            "num_tasks": len(records),
            "mean_generated_nmse_db": float(generated.mean()),
            "mean_meta_decoder_nmse_db": float(meta.mean()),
            "mean_gap_db": float((generated - meta).mean()),
            "median_gap_db": float(torch.quantile(generated - meta, 0.5)),
            "tasks": records,
        }
        if all("recomputed_teacher_nmse_db" in item for item in records):
            teacher = torch.tensor(
                [item["recomputed_teacher_nmse_db"] for item in records], dtype=torch.float64
            )
            summary["mean_recomputed_teacher_nmse_db"] = float(teacher.mean())
            summary["mean_teacher_meta_gap_db"] = float((teacher - meta).mean())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "eval_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "=> Evaluation summary: tasks=%d generated_nmse=%.4f dB "
            "meta_nmse=%.4f dB mean_gap=%+.4f dB median_gap=%+.4f dB",
            summary["num_tasks"], summary["mean_generated_nmse_db"],
            summary["mean_meta_decoder_nmse_db"], summary["mean_gap_db"],
            summary["median_gap_db"],
        )
        logger.info("=> Evaluation results saved: %s", self.output_dir / "eval_results.json")
