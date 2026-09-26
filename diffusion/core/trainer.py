from __future__ import annotations

import logging
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from core.model_factory import build_generation_models, load_stats
from core.inferencer import _load_tensor, functional_adapter, load_frozen_decoder
from data.adapter_tokenizer import _reconstruct_adapter_state, detokenize_adapter_batch
from data.dataloader import get_dataloaders
from utils.distributed import is_main_process
from utils.init import line_seg, seed_everything, show_parameter
from utils.scheduler import get_lr_scheduler
from utils.tools import EMA, create_exp_dirs, log_runtime_context, setup_logger


logger = logging.getLogger(__name__)
STRUCTURE_FIELDS = ("tensor_ids", "token_ids", "block_ids", "role_ids", "kind_ids")


def _parameter_count(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return total, trainable


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _weighted_mean(value: torch.Tensor, sample_weight: torch.Tensor | None) -> torch.Tensor:
    if sample_weight is None:
        return value.mean()
    weight = sample_weight.to(device=value.device, dtype=value.dtype)
    return ((value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-12)).mean()


def _relative_log1p_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    dims = tuple(range(2, prediction.ndim))
    error = (prediction - target).float()
    signal = target.float()
    error_rms = error.square().mean(dim=dims).clamp_min(eps).sqrt()
    signal_rms = signal.square().mean(dim=dims).clamp_min(eps).sqrt()
    value = torch.log1p((error_rms / signal_rms).square())
    return _weighted_mean(value, sample_weight)


def _functional_output_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if kind == "mse":
        dims = tuple(range(2, prediction.ndim))
        return _weighted_mean((prediction - target).float().square().mean(dim=dims), sample_weight)
    if kind == "relative_log1p":
        return _relative_log1p_loss(prediction, target, sample_weight=sample_weight, eps=eps)
    raise ValueError(f"Unknown functional loss type: {kind}")


def _reconstruction_nmse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    error = (prediction - target).float().square()
    signal = target.float().square()
    if sample_weight is not None:
        weight = sample_weight.to(device=error.device, dtype=error.dtype)
        weight = weight.reshape(*weight.shape, *((1,) * (prediction.ndim - weight.ndim)))
        error = error * weight
        signal = signal * weight
    return error.sum() / signal.sum().clamp_min(eps)


def _decoder_forward_batched(decoder: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim == 2:
        return decoder(codes)
    batch, samples, dim = codes.shape
    output = decoder(codes.reshape(batch * samples, dim))
    return output.reshape(batch, samples, *output.shape[1:])


class Trainer:
    def __init__(self, cfg, shared_bundle: dict | None = None, distributed: bool = False):
        self.cfg = cfg
        self.distributed = bool(distributed)
        self.rank = dist.get_rank() if self.distributed and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if self.distributed and dist.is_initialized() else 1
        self.local_rank = int(torch.cuda.current_device()) if self.distributed and torch.cuda.is_available() else 0
        self.is_main = is_main_process()
        self.device = torch.device(
            f"cuda:{self.local_rank}" if self.distributed and torch.cuda.is_available() else str(cfg.data.device)
        )
        self.exp_dir = create_exp_dirs(str(cfg.exp_dir))
        if self.is_main:
            setup_logger(self.exp_dir, append=bool(cfg.train.get("resume", False)))
            log_runtime_context(logger, cfg, self.exp_dir)
        seed_everything(int(cfg.train.seed))
        if self.is_main:
            OmegaConf.save(cfg, self.exp_dir / "logs" / "config.yaml")
            self.writer = SummaryWriter(self.exp_dir / "tensorboard")
        else:
            self.writer = None

        self.stats = load_stats(cfg)
        self.train_loader, self.val_loader = get_dataloaders(
            cfg, self.stats, shared_bundle=shared_bundle, distributed=self.distributed
        )
        if self.is_main:
            logger.info(
                "=> Dataset sizes: train=%d, val=%d | tokens=%d token_size=%d tensors=%d",
                len(self.train_loader.dataset), len(self.val_loader.dataset),
                int(self.stats["manifest"]["num_tokens"]),
                int(self.stats["manifest"]["token_size"]),
                int(self.stats["manifest"]["num_tensors"]),
            )
            logger.info(
                "=> Loader config: batch_size=%d val_batch_size=%d workers=%d loading=%s preload_workers=%d pin_memory=%s distributed=%s world_size=%d",
                int(cfg.train.batch_size), int(cfg.train.get("val_batch_size", cfg.train.batch_size)),
                int(cfg.data.num_workers), "shared_memory" if shared_bundle is not None else str(cfg.data.loading),
                int(cfg.data.get("preload_workers", 1)), bool(cfg.data.pin_memory),
                self.distributed, self.world_size,
            )

        self.token_context_generator, self.denoiser, self.diffusion = build_generation_models(
            cfg, self.stats, self.device
        )
        self.freeze_context_generator = bool(
            cfg.train.get("freeze_context_generator", False)
        )
        if self.freeze_context_generator:
            for parameter in self.token_context_generator.parameters():
                parameter.requires_grad_(False)
        if self.distributed:
            device_ids = [self.local_rank] if self.device.type == "cuda" else None
            if not self.freeze_context_generator:
                self.token_context_generator = DistributedDataParallel(
                    self.token_context_generator,
                    device_ids=device_ids,
                    broadcast_buffers=False,
                    find_unused_parameters=True,
                )
            self.denoiser = DistributedDataParallel(
                self.denoiser, device_ids=device_ids, broadcast_buffers=False
            )
            self.diffusion.denoiser = self.denoiser
        self.null_context = nn.Parameter(torch.zeros(
            1,
            int(self.stats["manifest"]["num_tokens"]),
            int(cfg.token_context.hidden_dim),
            device=self.device,
        ))
        self.trainable_params = (
            [parameter for parameter in self.token_context_generator.parameters() if parameter.requires_grad]
            + [parameter for parameter in self.denoiser.parameters() if parameter.requires_grad]
            + [self.null_context]
        )
        self.optimizer = optim.AdamW(
            self.trainable_params,
            lr=float(cfg.lr_scheduler.max_lr),
            weight_decay=float(cfg.train.weight_decay),
        )
        self.grad_accum_steps = int(cfg.train.grad_accum_steps)
        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        scheduler_cfg = dict(cfg.lr_scheduler)
        scheduler_cfg["max_steps"] = max(1, steps_per_epoch * int(cfg.train.epochs))
        self.scheduler = get_lr_scheduler(self.optimizer, **scheduler_cfg)
        self.ema_model = nn.ModuleDict({
            "token_context_generator": self._unwrap(self.token_context_generator),
            "denoiser": self._unwrap(self.denoiser),
            "null_context_wrapper": nn.ParameterList([self.null_context]),
        })
        self.ema_rate = float(cfg.train.get("ema_rate", 0.999))
        self.use_ema = self.ema_rate > 0.0
        self.ema = EMA(self.ema_model, decay=self.ema_rate) if self.use_ema else None
        self.start_epoch = 1
        self.step = 0
        self._eval_decoder = None
        self._eval_decoder_args = None
        self._eval_csi = None
        self._functional_code_cache: dict[str, torch.Tensor] = {}

        if self.is_main:
            logger.info(f"\n{line_seg}\n{self.ema_model}\n{line_seg}\n")
            show_parameter(self.ema_model, logger)

        context_params = _parameter_count(self._unwrap(self.token_context_generator))
        denoiser_params = _parameter_count(self._unwrap(self.denoiser))
        if self.is_main:
            logger.info(
                "=> Model: token_context=%s impl=%s injection=%s branches=%s params=%s | denoiser=%s type=%s hidden=%d params=%s",
                type(self._unwrap(self.token_context_generator)).__name__,
                str(cfg.token_context.get("implementation", "transformer")),
                str(cfg.token_context.get("condition_injection", "cross_attention")),
                ",".join(str(name) for name in cfg.condition_encoder.branches),
                f"{context_params[0]:,}",
                type(self._unwrap(self.denoiser)).__name__, str(cfg.denoiser.type),
                int(cfg.token_context.hidden_dim), f"{denoiser_params[0]:,}",
            )
            if hasattr(self._unwrap(self.denoiser), "residual_gate_init"):
                logger.info(
                    "=> Residual gate init: %.3e | trainable=%s",
                    float(self._unwrap(self.denoiser).residual_gate_init),
                    f"{context_params[1] + denoiser_params[1] + self.null_context.numel():,}",
                )
            logger.info(
                "=> Optimizer: AdamW lr=%.4e weight_decay=%.4e accumulation=%d per_gpu_batch=%d global_batch=%d",
                float(cfg.lr_scheduler.max_lr), float(cfg.train.weight_decay), self.grad_accum_steps,
                int(cfg.train.batch_size), int(cfg.train.batch_size) * self.grad_accum_steps * self.world_size,
            )
            logger.info(
                "=> EMA: %s%s",
                "enabled" if self.use_ema else "disabled",
                f" decay={self.ema_rate:.6f}" if self.use_ema else " (train.ema_rate <= 0)",
            )
        pretrained = cfg.train.get("pretrained")
        if pretrained:
            self.load_checkpoint(str(pretrained), resume=bool(cfg.train.get("resume", False)))

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        return module.module if isinstance(module, DistributedDataParallel) else module

    def _prepare_batch(self, batch):
        structure = {name: batch[name].to(self.device) for name in STRUCTURE_FIELDS}
        raw_cond = batch.get("raw_cond")
        return (
            batch["cond"].to(self.device),
            raw_cond.to(self.device) if raw_cond is not None else None,
            batch["tokens"].to(self.device),
            batch["token_mask"].to(self.device),
            structure,
        )

    def _sample_token_indices(self, token_mask: torch.Tensor) -> torch.Tensor | None:
        """See README.md for English documentation."""
        cfg = self.cfg.train.get("token_subsample", {})
        if not bool(cfg.get("enabled", False)):
            return None
        num_tokens = int(cfg.get("num_tokens", 0))
        total_tokens = token_mask.shape[1]
        if num_tokens <= 0 or num_tokens >= total_tokens:
            return None
        valid_tokens = token_mask[0].any(dim=-1)
        valid_indices = torch.nonzero(valid_tokens, as_tuple=False).flatten()
        if num_tokens >= valid_indices.numel():
            return valid_indices
        strategy = str(cfg.get("strategy", "uniform"))
        generator = None
        if bool(cfg.get("deterministic", False)):
            generator = torch.Generator(device=token_mask.device)
            generator.manual_seed(int(self.cfg.train.seed) + self.step)
        if strategy == "uniform":
            order = torch.randperm(valid_indices.numel(), device=token_mask.device, generator=generator)
            return valid_indices[order[:num_tokens]].sort().values
        if strategy != "tensor_balanced":
            raise ValueError(
                f"Unknown train.token_subsample.strategy: {strategy}")

        tensor_ids = self.stats["manifest"]["tensor_ids"].to(token_mask.device)
        chosen = []
        tensor_values = torch.unique(tensor_ids[valid_indices], sorted=True)
        base = max(1, num_tokens // max(int(tensor_values.numel()), 1))
        for tensor_id in tensor_values:
            candidates = valid_indices[tensor_ids[valid_indices] == tensor_id]
            take = min(base, int(candidates.numel()))
            if take <= 0:
                continue
            order = torch.randperm(candidates.numel(), device=token_mask.device, generator=generator)
            chosen.append(candidates[order[:take]])
        selected = torch.cat(chosen) if chosen else valid_indices[:0]
        if selected.numel() < num_tokens:
            selected_mask = torch.zeros(total_tokens, dtype=torch.bool, device=token_mask.device)
            selected_mask[selected] = True
            remaining = valid_indices[~selected_mask[valid_indices]]
            take = min(num_tokens - int(selected.numel()), int(remaining.numel()))
            if take > 0:
                order = torch.randperm(remaining.numel(), device=token_mask.device, generator=generator)
                selected = torch.cat((selected, remaining[order[:take]]))
        return selected[:num_tokens].sort().values

    def _subsample_tokens(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        token_context: torch.Tensor,
        structure: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        indices = self._sample_token_indices(token_mask)
        if indices is None:
            return tokens, token_mask, token_context, structure, {
                "token_subsample_count": float(tokens.shape[1]),
                "token_subsample_ratio": 1.0,
            }
        sampled_structure = {
            name: value.index_select(1, indices) for name, value in structure.items()
        }
        return (
            tokens.index_select(1, indices),
            token_mask.index_select(1, indices),
            token_context.index_select(1, indices),
            sampled_structure,
            {
                "token_subsample_count": float(indices.numel()),
                "token_subsample_ratio": float(indices.numel()) / max(float(tokens.shape[1]), 1.0),
            },
        )

    @staticmethod
    def _grad_norm(parameters) -> float:
        total = torch.zeros((), dtype=torch.float64)
        found = False
        for parameter in parameters:
            if parameter.grad is not None:
                total += parameter.grad.detach().double().square().sum().cpu()
                found = True
        return float(total.sqrt()) if found else 0.0

    def _gradient_metrics(self) -> dict[str, float]:
        blocks = list(getattr(self._unwrap(self.denoiser), "blocks", []))
        if not blocks:
            return {}
        residual_branch_params = []
        residual_gate_square = torch.zeros((), dtype=torch.float64)
        for block in blocks:
            if hasattr(block, "mlp"):
                residual_branch_params.extend(block.mlp.parameters())
            if not hasattr(block, "adaLN_modulation") or not hasattr(block, "hidden_dim"):
                continue
            projection = block.adaLN_modulation[-1]
            gate_count = projection.out_features // block.hidden_dim
            if gate_count != 3:
                continue
            if projection.weight.grad is not None:
                gradient = projection.weight.grad.detach().view(gate_count, block.hidden_dim, -1)[2]
                residual_gate_square += gradient.double().square().sum().cpu()
            if projection.bias.grad is not None:
                gradient = projection.bias.grad.detach().view(gate_count, block.hidden_dim)[2]
                residual_gate_square += gradient.double().square().sum().cpu()
        return {
            "grad_residual_branch": self._grad_norm(residual_branch_params),
            "grad_residual_gate": float(residual_gate_square.sqrt()),
        }

    def _denoiser_metrics(self) -> dict[str, float]:
        metrics_fn = getattr(self._unwrap(self.denoiser), "gate_metrics", None)
        if metrics_fn is None:
            return {}
        return {name: float(value) for name, value in metrics_fn().items()}

    def _token_context_metrics(self, token_context: torch.Tensor) -> dict[str, float]:
        null_context = self.null_context.expand(token_context.shape[0], -1, -1)
        context_rms = _rms(token_context)
        delta_rms = _rms(token_context - null_context)
        metrics = {
            "token_context_rms": context_rms,
            "token_context_null_delta_rms": delta_rms,
            "token_context_null_delta_ratio": delta_rms / max(context_rms, 1e-12),
            "token_context_token_std": float(token_context.detach().float().std(dim=1, unbiased=False).mean()),
            "token_context_task_std": (
                float(token_context.detach().float().std(dim=0, unbiased=False).mean())
                if token_context.shape[0] > 1 else 0.0
            ),
        }
        metrics.update(getattr(self._unwrap(self.token_context_generator), "last_condition_gates", {}))
        return metrics

    def _gpu_memory_gib(self) -> tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        scale = 1024 ** 3
        return (
            torch.cuda.max_memory_allocated(self.device) / scale,
            torch.cuda.max_memory_reserved(self.device) / scale,
        )

    def _sync_null_context_grad(self) -> None:
        """See README.md for English documentation."""
        if not self.distributed:
            return
        if self.null_context.grad is None:
            self.null_context.grad = torch.zeros_like(self.null_context)
        dist.all_reduce(self.null_context.grad, op=dist.ReduceOp.SUM)
        self.null_context.grad.div_(self.world_size)

    def _write_step_metrics(self, values: dict[str, float]) -> None:
        if self.writer is None:
            return
        for key, value in values.items():
            self.writer.add_scalar(f"train_step/{key}", value, self.step)

    def _distributed_mean_dict(self, values: dict[str, float]) -> dict[str, float]:
        """See README.md for English documentation."""
        if not self.distributed:
            return values
        keys = sorted(values)
        if not keys:
            return values
        tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(self.world_size)
        return {key: float(tensor[index].cpu()) for index, key in enumerate(keys)}

    def train(self):
        best_loss = float("inf")
        log_every = max(1, int(self.cfg.train.get("log_every_updates", 10)))
        total_epochs = int(self.cfg.train.epochs)
        if self.is_main:
            logger.info("=> Training started: epochs=%d updates_per_epoch=%d", total_epochs,
                        math.ceil(len(self.train_loader) / self.grad_accum_steps))
        for epoch in range(self.start_epoch, total_epochs + 1):
            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            epoch_started = time.perf_counter()
            interval_started = epoch_started
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.diffusion.train()
            if self.freeze_context_generator:
                self.token_context_generator.eval()
            else:
                self.token_context_generator.train()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            totals: defaultdict[str, float] = defaultdict(float)
            recorded_batches = 0
            last_log_batch = 0
            last_grad_metrics = {}

            for batch_index, batch in enumerate(self.train_loader, start=1):
                update = accumulated + 1 == self.grad_accum_steps or batch_index == len(self.train_loader)
                cond, raw_cond, tokens, token_mask, structure = self._prepare_batch(batch)
                token_context = self.token_context_generator(cond, structure, raw_cond)
                noise_factor = float(self.cfg.train.get("cond_noise_factor", 0.0))
                if noise_factor:
                    token_context = token_context + torch.randn_like(token_context) * noise_factor
                condition_dropped = bool(
                    torch.rand((), device=self.device) < float(self.cfg.train.cfg_drop_rate)
                )
                current_context = (
                    self.null_context.expand(token_context.shape[0], -1, -1)
                    if condition_dropped else token_context
                )
                loss_tokens, loss_mask, loss_context, loss_structure, subsample_metrics = self._subsample_tokens(
                    tokens, token_mask, current_context, structure
                )
                function_active, _ = self._auxiliary_loss_weight("function", epoch)
                reconstruction_active, _ = self._auxiliary_loss_weight("reconstruction", epoch)
                auxiliary_active = update and (function_active or reconstruction_active)
                auxiliary_task_limit = self._auxiliary_task_limit(
                    function_active, reconstruction_active, tokens.shape[0]
                )
                need_full_auxiliary_forward = auxiliary_active and loss_tokens.shape[1] != tokens.shape[1]
                losses = self.diffusion(
                    loss_tokens,
                    loss_context,
                    structure=loss_structure,
                    token_mask=loss_mask,
                    return_pred=auxiliary_active and not need_full_auxiliary_forward,
                )
                # English documentation is provided in README.md.
                contrastive_active, contrastive_weight = self._auxiliary_loss_weight("contrastive", epoch)
                contrastive_loss_total = losses["loss"] * 0.0
                contrastive_metrics = {
                    "contrastive_loss": 0.0, "contrastive_weight": 0.0,
                    "contrastive_gap": 0.0, "contrastive_correct": 0.0,
                    "contrastive_wrong": 0.0, "contrastive_zero": 0.0,
                }
                if contrastive_active and not condition_dropped and token_context.shape[0] >= 2:
                    contrastive_margin = float(self.cfg.loss.contrastive.get("margin", 1.0))
                    use_derangement = bool(self.cfg.loss.contrastive.get("use_derangement", True))
                    stop_grad_wrong = bool(self.cfg.loss.contrastive.get("stop_grad_wrong", True))
                    stop_grad_zero = bool(self.cfg.loss.contrastive.get("stop_grad_zero", True))
                    n_contrastive = min(
                        int(self.cfg.loss.contrastive.get("num_pairs", 6)),
                        token_context.shape[0],
                    )
                    n_contrastive = max(2, n_contrastive)
                    tok_sub = tokens[:n_contrastive]
                    mask_sub = token_mask[:n_contrastive]
                    struct_sub = {k: v[:n_contrastive] for k, v in structure.items()}

                    # English documentation is provided in README.md.
                    device = tok_sub.device
                    shared_t = torch.randint(
                        0, self.diffusion.timesteps, (n_contrastive,),
                        device=device, dtype=torch.long,
                    )
                    shared_noise = torch.randn_like(tok_sub)

                    correct_ctx = token_context[:n_contrastive]

                    # L_correct
                    losses_c = self.diffusion.loss_fn(
                        tok_sub, shared_t, correct_ctx, noise=shared_noise,
                        structure=struct_sub, token_mask=mask_sub,
                    )

                    # English documentation is provided in README.md.
                    if use_derangement:
                        indices = torch.randperm(n_contrastive, device=device)
                        if n_contrastive >= 2:
                            safety = 0
                            while (indices == torch.arange(n_contrastive, device=device)).any():
                                indices = torch.randperm(n_contrastive, device=device)
                                safety += 1
                                if safety > 100:
                                    # English documentation is provided in README.md.
                                    tmp = indices[0].clone()
                                    indices[0] = indices[1]
                                    indices[1] = tmp
                                    break
                        wrong_ctx = correct_ctx[indices]
                        wrong_ctx_input = wrong_ctx.detach() if stop_grad_wrong else wrong_ctx
                        losses_wrong = self.diffusion.loss_fn(
                            tok_sub, shared_t, wrong_ctx_input, noise=shared_noise,
                            structure=struct_sub, token_mask=mask_sub,
                        )
                    else:
                        losses_wrong = losses_c  # English documentation is provided in README.md.

                    # English documentation is provided in README.md.
                    zero_ctx = torch.zeros_like(correct_ctx)
                    zero_ctx_input = zero_ctx.detach() if stop_grad_zero else zero_ctx
                    losses_zero = self.diffusion.loss_fn(
                        tok_sub, shared_t, zero_ctx_input, noise=shared_noise,
                        structure=struct_sub, token_mask=mask_sub,
                    )

                    # English documentation is provided in README.md.
                    if use_derangement:
                        penalty = (
                            F.relu(contrastive_margin + losses_c["loss"] - losses_wrong["loss"])
                            + F.relu(contrastive_margin + losses_c["loss"] - losses_zero["loss"])
                        )
                    else:
                        penalty = F.relu(contrastive_margin + losses_c["loss"] - losses_zero["loss"])

                    contrastive_loss_total = contrastive_weight * penalty
                    contrastive_metrics = {
                        "contrastive_loss": float(penalty.detach()),
                        "contrastive_weight": contrastive_weight,
                        "contrastive_gap": float((losses_zero["loss"] - losses_c["loss"]).detach()),
                        "contrastive_correct": float(losses_c["loss"].detach()),
                        "contrastive_wrong": float(losses_wrong["loss"].detach()) if use_derangement else 0.0,
                        "contrastive_zero": float(losses_zero["loss"].detach()),
                    }
                # ------------------------------------------------------------------------
                full_pred_tokens = losses.get("pred_x0")
                if need_full_auxiliary_forward:
                    limited_structure = {
                        name: value[:auxiliary_task_limit] for name, value in structure.items()
                    }
                    full_losses = self.diffusion(
                        tokens[:auxiliary_task_limit],
                        current_context[:auxiliary_task_limit],
                        structure=limited_structure,
                        token_mask=token_mask[:auxiliary_task_limit],
                        return_pred=True,
                    )
                    full_pred_tokens = full_losses["pred_x0"]
                if auxiliary_active:
                    auxiliary_loss, auxiliary_metrics = self._calculate_training_functional_losses(
                        batch, full_pred_tokens, tokens[:auxiliary_task_limit], epoch
                    )
                else:
                    auxiliary_loss = losses["loss"] * 0.0
                    auxiliary_metrics = {
                        "function_loss": 0.0,
                        "function_weight": 0.0,
                        "reconstruction_loss": 0.0,
                        "reconstruction_weight": 0.0,
                        "functional_tasks": 0.0,
                        "functional_queries": 0.0,
                    }
                total_loss = losses["loss"] + auxiliary_loss + contrastive_loss_total

                # English documentation is provided in README.md.
                loss_finite = torch.isfinite(total_loss).item()
                if not loss_finite:
                    logger.error(
                        "NaN/Inf detected in total_loss at step %s epoch %s. "
                        "diffusion=%.6f auxiliary=%.6f contrastive=%.6f. "
                        "Skipping step.",
                        self.step, epoch,
                        float(losses["loss"].detach()),
                        float(auxiliary_loss.detach()),
                        float(contrastive_loss_total.detach()),
                    )
                    dump_dir = Path(self.cfg.exp_dir) / "nan_dumps"
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    dump_path = dump_dir / f"step_{self.step}_epoch_{epoch}.pt"
                    dump_data = {
                        "step": self.step,
                        "epoch": epoch,
                        "batch_meta": [(item["meta"]["task_dir"], item["meta"]["seed"]) for item in batch],
                        "pred_tokens": (full_pred_tokens.detach().cpu() if full_pred_tokens is not None else None),
                        "target_tokens": tokens.detach().cpu(),
                        "diffusion_loss": float(losses["loss"].detach()),
                        "auxiliary_loss": float(auxiliary_loss.detach()),
                        "contrastive_loss": float(contrastive_loss_total.detach()),
                        "loss_components": {k: float(v.detach()) if torch.is_tensor(v) else v
                                           for k, v in losses.items() if k != "pred" and k != "target" and k != "pred_x0"},
                    }
                    torch.save(dump_data, dump_path)
                    logger.error("Offending batch saved to %s", dump_path)
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    # English documentation is provided in README.md.
                    continue

                total_loss.backward()
                accumulated += 1
                grad_norm = 0.0
                if update:
                    self._sync_null_context_grad()
                    for parameter in self.trainable_params:
                        if parameter.grad is not None:
                            parameter.grad.div_(accumulated)

                    # ---- Gradient finite guard ----
                    grad_finite = all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in self.trainable_params
                    )
                    if not grad_finite:
                        bad_indices = [
                            i for i, param in enumerate(self.trainable_params)
                            if param.grad is not None and not torch.isfinite(param.grad).all()
                        ]
                        logger.error(
                            "NaN/Inf gradient in trainable_params indices: %s (total=%d). Skipping step %s.",
                            bad_indices[:10], len(self.trainable_params), self.step,
                        )
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated = 0
                        continue

                    last_grad_metrics = self._gradient_metrics()
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, float(self.cfg.train.grad_clip)
                    )
                    grad_norm = float(grad_norm_tensor)
                    self.optimizer.step()
                    self.scheduler.step()

                    # ---- Parameter finite guard (post-optimizer) ----
                    param_finite = all(
                        torch.isfinite(parameter).all()
                        for parameter in self.trainable_params
                    )
                    if not param_finite:
                        bad_indices = [
                            i for i, param in enumerate(self.trainable_params)
                            if not torch.isfinite(param).all()
                        ]
                        logger.error(
                            "NaN/Inf in parameters after optimizer step, indices: %s (total=%d). "
                            "Consider reducing learning rate or grad_clip. Step %s.",
                            bad_indices[:10], len(self.trainable_params), self.step,
                        )
                        # English documentation is provided in README.md.
                        if self.use_ema:
                            logger.warning("Attempting to restore parameters from EMA shadow.")
                            self.ema.copy_to(self.trainable_params)
                        else:
                            logger.error("No EMA available — parameters are corrupted.")

                    if self.use_ema:
                        self.ema.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    self.step += 1

                values = {
                    "loss": float(total_loss.detach()),
                    "diffusion_loss": float(losses["loss"].detach()),
                    "element_loss": float(losses["element_loss"].detach()),
                    "tensor_loss": float(losses["tensor_loss"].detach()),
                    "cos_sim": float(losses["cos_sim"].detach()),
                    "norm_ratio": float(losses["norm_sim"].detach()),
                    "prediction_rms": float(losses["prediction_rms"]),
                    "prediction_abs_max": float(losses["prediction_abs_max"]),
                    "target_rms": float(losses["target_rms"]),
                    "timestep_mean": float(losses["timestep_mean"]),
                    "condition_drop": float(condition_dropped),
                }
                values.update(auxiliary_metrics)
                values.update(contrastive_metrics)
                values.update(subsample_metrics)
                values.update(self._token_context_metrics(token_context))
                values.update(self._denoiser_metrics())
                for key, value in values.items():
                    totals[key] += value
                recorded_batches += 1

                if update:
                    allocated_gib, reserved_gib = self._gpu_memory_gib()
                    step_values = {
                        **values, **last_grad_metrics,
                        "grad_norm": grad_norm,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "gpu_peak_allocated_gib": allocated_gib,
                        "gpu_peak_reserved_gib": reserved_gib,
                    }
                    self._write_step_metrics(step_values)
                    if self.is_main and (self.step % log_every == 0 or batch_index == len(self.train_loader)):
                        elapsed = time.perf_counter() - interval_started
                        batch_time = elapsed / max(batch_index - last_log_batch, 1)
                        gate_text = ""
                        if "residual_gate_rms" in values:
                            gate_text = (
                                f" | residual_gate: {values['residual_gate_rms']:.2e} "
                                f"max: {values['residual_gate_abs_max']:.2e}"
                            )
                        grad_text = ""
                        if "grad_residual_branch" in last_grad_metrics:
                            grad_text = (
                                f" residual_branch_grad: {last_grad_metrics['grad_residual_branch']:.2e} "
                                f"residual_gate_grad: {last_grad_metrics['grad_residual_gate']:.2e}"
                            )
                        logger.info(
                            "Epoch: [%d/%d][%d/%d] lr: %.4e | loss: %.4e element: %.4e "
                            "tensor: %.4e func: %.4e(%.2e) recon: %.4e(%.2e) | "
                            "cos: %.3f norm: %.3f | grad: %.3e%s%s | "
                            "pred_rms: %.2e context_rms: %.2e ctx_delta: %.2e "
                            "ctx_token_std: %.2e ctx_task_std: %.2e | "
                            "ctr_gap: %.2f ctr_loss: %.3e | tokens: %.0f(%.2f) | time: %.3f",
                            epoch, total_epochs, batch_index, len(self.train_loader),
                            self.optimizer.param_groups[0]["lr"], values["loss"],
                            values["element_loss"], values["tensor_loss"],
                            values["function_loss"], values["function_weight"],
                            values["reconstruction_loss"], values["reconstruction_weight"],
                            values["cos_sim"],
                            values["norm_ratio"], grad_norm, grad_text, gate_text,
                            values["prediction_rms"], values["token_context_rms"],
                            values["token_context_null_delta_ratio"],
                            values["token_context_token_std"],
                            values["token_context_task_std"],
                            values.get("contrastive_gap", 0.0),
                            values.get("contrastive_loss", 0.0),
                            values["token_subsample_count"], values["token_subsample_ratio"],
                            batch_time,
                        )
                        interval_started = time.perf_counter()
                        last_log_batch = batch_index

            summary = {key: value / recorded_batches for key, value in totals.items()}
            summary = self._distributed_mean_dict(summary)
            allocated_gib, reserved_gib = self._gpu_memory_gib()
            summary.update(last_grad_metrics)
            summary["gpu_peak_allocated_gib"] = allocated_gib
            summary["gpu_peak_reserved_gib"] = reserved_gib
            summary["epoch_time_seconds"] = time.perf_counter() - epoch_started
            if self.is_main:
                logger.info(
                    "=> Train Loss: %.4e | Diffusion: %.4e | Element: %.4e | Tensor: %.4e | "
                    "Function: %.4e(%.2e) | Reconstruction: %.4e(%.2e) | "
                    "Contrastive: %.4e(%.2e) gap=%.4f | Cos: %.4f | Norm: %.4f",
                    summary["loss"], summary["diffusion_loss"], summary["element_loss"], summary["tensor_loss"],
                    summary["function_loss"], summary["function_weight"],
                    summary["reconstruction_loss"], summary["reconstruction_weight"],
                    summary.get("contrastive_loss", 0.0), summary.get("contrastive_weight", 0.0),
                    summary.get("contrastive_gap", 0.0),
                    summary["cos_sim"], summary["norm_ratio"],
                )
            gate_text = ""
            if "residual_gate_rms" in summary:
                gate_text = (
                    f"residual_gate={summary['residual_gate_rms']:.3e} "
                    f"max={summary['residual_gate_abs_max']:.3e} | "
                )
            grad_text = ""
            if "grad_residual_branch" in summary:
                grad_text = (
                    f"residual_branch_grad={summary['grad_residual_branch']:.3e} "
                    f"residual_gate_grad={summary['grad_residual_gate']:.3e} | "
                )
            condition_gate_text = " ".join(
                f"{key.removeprefix('condition_gate_')}={summary[key]:.3f}"
                for key in sorted(summary)
                if key.startswith("condition_gate_")
            )
            if condition_gate_text:
                condition_gate_text = f" | condition_gates: {condition_gate_text}"
            if self.is_main:
                logger.info(
                    "=> Diagnostics: %s%spred_rms=%.3e target_rms=%.3e cfg_drop=%.3f | "
                    "context_rms=%.3e ctx_delta=%.3e token_std=%.3e task_std=%.3e%s | "
                    "tokens=%.0f(%.3f) | gpu_peak=%.2f/%.2f GiB | epoch_time=%.2fs",
                    gate_text, grad_text, summary["prediction_rms"], summary["target_rms"], summary["condition_drop"],
                    summary["token_context_rms"], summary["token_context_null_delta_ratio"],
                    summary["token_context_token_std"], summary["token_context_task_std"], condition_gate_text,
                    summary["token_subsample_count"], summary["token_subsample_ratio"],
                    allocated_gib, reserved_gib, summary["epoch_time_seconds"],
                )
                for key, value in summary.items():
                    self.writer.add_scalar(f"train_epoch/{key}", value, epoch)

            if epoch % int(self.cfg.train.val_interval) == 0:
                val_summary = self.validate(epoch)
                if self.is_main and val_summary["loss"] < best_loss:
                    best_loss = val_summary["loss"]
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info("=! Best Val Loss: %.4e (epoch=%d)", best_loss, epoch)
                if self.distributed:
                    dist.barrier()
            if self.is_main and epoch % int(self.cfg.train.save_interval) == 0:
                self.save_checkpoint(epoch)
            if self.distributed and epoch % int(self.cfg.train.save_interval) == 0:
                dist.barrier()
            if self.writer is not None:
                self.writer.flush()
        if self.writer is not None:
            self.writer.close()
        if self.is_main:
            logger.info("=> Training complete: best_val_loss=%.4e", best_loss)
        return best_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        started = time.perf_counter()
        if self.use_ema:
            self.ema.apply_shadow()
        self.diffusion.eval()
        totals: defaultdict[str, float] = defaultdict(float)
        count = 0
        functional_summary = None
        try:
            for batch in self.val_loader:
                cond, raw_cond, tokens, token_mask, structure = self._prepare_batch(batch)
                token_context = self.token_context_generator(cond, structure, raw_cond)
                losses = self.diffusion(
                    tokens, token_context, structure=structure, token_mask=token_mask
                )
                values = {
                    "loss": float(losses["loss"]),
                    "element_loss": float(losses["element_loss"]),
                    "tensor_loss": float(losses["tensor_loss"]),
                    "cos_sim": float(losses["cos_sim"]),
                    "norm_ratio": float(losses["norm_sim"]),
                    "prediction_rms": float(losses["prediction_rms"]),
                }
                values.update(self._token_context_metrics(token_context))
                values.update(self._denoiser_metrics())
                for key, value in values.items():
                    totals[key] += value
                count += 1
            summary = {key: value / count for key, value in totals.items()}
            summary = self._distributed_mean_dict(summary)
            if (
                self.is_main
                and
                bool(self.cfg.eval.get("enabled", False))
                and bool(self.cfg.eval.get("during_training", True))
            ):
                functional_summary = self._validate_decoder_nmse(epoch)
        finally:
            if self.use_ema:
                self.ema.restore()
        if functional_summary is not None:
            summary.update(functional_summary)
        summary["time_seconds"] = time.perf_counter() - started
        gate_text = ""
        if "residual_gate_rms" in summary:
            gate_text = (
                f" | residual_gate={summary['residual_gate_rms']:.3e} "
                f"max={summary['residual_gate_abs_max']:.3e}"
            )
        condition_gate_text = " ".join(
            f"{key.removeprefix('condition_gate_')}={summary[key]:.3f}"
            for key in sorted(summary)
            if key.startswith("condition_gate_")
        )
        if condition_gate_text:
            condition_gate_text = f" | condition_gates: {condition_gate_text}"
        if self.is_main:
            logger.info(
                "=> Val Loss: %.4e | Element: %.4e | Tensor: %.4e | Cos: %.4f | Norm: %.4f%s | "
                "context_rms=%.3e ctx_delta=%.3e token_std=%.3e task_std=%.3e%s | time: %.2fs",
                summary["loss"], summary["element_loss"], summary["tensor_loss"],
                summary["cos_sim"], summary["norm_ratio"], gate_text,
                summary["token_context_rms"], summary["token_context_null_delta_ratio"],
                summary["token_context_token_std"], summary["token_context_task_std"], condition_gate_text,
                summary["time_seconds"],
            )
            for key, value in summary.items():
                self.writer.add_scalar(f"validation/{key}", value, epoch)
        return summary

    def _load_training_eval_context(self):
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
            "=> Training evaluation assets loaded: decoder=%s checkpoint=%s csi=%s samples=%d",
            decoder_args.get("decoder"), self.cfg.eval.decoder_checkpoint,
            csi_path, self._eval_csi.shape[0],
        )
        return self._eval_decoder, self._eval_decoder_args, self._eval_csi

    def _load_functional_codewords(self, task_dir: str | Path) -> torch.Tensor:
        cache_size = int(self.cfg.loss.get("codeword_cache_size", 0))
        path = str((Path(task_dir) / str(self.cfg.data.query_codeword_file)).resolve())
        if cache_size > 0 and path in self._functional_code_cache:
            return self._functional_code_cache[path]
        value = _load_tensor(Path(path))
        if cache_size > 0:
            if len(self._functional_code_cache) >= cache_size:
                self._functional_code_cache.pop(next(iter(self._functional_code_cache)))
            self._functional_code_cache[path] = value
        return value

    def _sample_functional_batch(self, batch: dict, max_tasks: int, query_samples: int):
        """See README.md for English documentation."""
        _, _, csi_cpu = self._load_training_eval_context()
        task_dirs = list(batch["meta"]["task_dir"])
        task_count = min(max(1, max_tasks), len(task_dirs))
        code_batches = []
        csi_batches = []
        weights = []
        for task_dir in task_dirs[:task_count]:
            codewords = self._load_functional_codewords(task_dir)
            if codewords.shape[0] != csi_cpu.shape[0]:
                raise ValueError(
                    f"{task_dir}: {self.cfg.data.query_codeword_file} count "
                    f"{codewords.shape[0]} does not match CSI count "
                    f"{csi_cpu.shape[0]}"
                )
            count = codewords.shape[0] if query_samples <= 0 else min(query_samples, codewords.shape[0])
            indices = torch.randperm(codewords.shape[0])[:count]
            code_batches.append(codewords.index_select(0, indices))
            csi_batches.append(csi_cpu.index_select(0, indices))
            weights.append(torch.ones(count))
        return (
            torch.stack(code_batches).to(self.device, non_blocking=True),
            torch.stack(csi_batches).to(self.device, non_blocking=True),
            torch.stack(weights).to(self.device, non_blocking=True),
            task_count,
        )

    def _auxiliary_loss_weight(self, name: str, epoch: int) -> tuple[bool, float]:
        cfg = self.cfg.loss.get(name, {})
        enabled = bool(cfg.get("enabled", False)) and float(cfg.get("weight", 0.0)) != 0.0
        if not enabled:
            return False, 0.0
        every_steps = max(1, int(cfg.get("every_steps", 1)))
        next_step = self.step + 1
        if next_step % every_steps != 0:
            return False, 0.0
        warmup_epochs = max(0, int(cfg.get("warmup_epochs", 0)))
        scale = 1.0 if warmup_epochs <= 0 else min(1.0, max(float(epoch), 0.0) / warmup_epochs)
        return True, float(cfg.get("weight", 0.0)) * scale

    def _auxiliary_task_limit(self, function_active: bool, reconstruction_active: bool, batch_size: int) -> int:
        if not function_active and not reconstruction_active:
            return 0
        function_cfg = self.cfg.loss.get("function", {})
        reconstruction_cfg = self.cfg.loss.get("reconstruction", {})
        task_limit = max(
            int(function_cfg.get("max_tasks", 1)) if function_active else 1,
            int(reconstruction_cfg.get("max_tasks", 1)) if reconstruction_active else 1,
        )
        return min(max(1, task_limit), batch_size)

    def _calculate_training_functional_losses(
        self,
        batch: dict,
        pred_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        function_active, function_weight = self._auxiliary_loss_weight("function", epoch)
        reconstruction_active, reconstruction_weight = self._auxiliary_loss_weight("reconstruction", epoch)
        zero = pred_tokens.sum() * 0.0
        if not function_active and not reconstruction_active:
            return zero, {
                "function_loss": 0.0,
                "function_weight": 0.0,
                "reconstruction_loss": 0.0,
                "reconstruction_weight": 0.0,
                "functional_tasks": 0.0,
                "functional_queries": 0.0,
            }

        function_cfg = self.cfg.loss.get("function", {})
        reconstruction_cfg = self.cfg.loss.get("reconstruction", {})
        max_tasks = self._auxiliary_task_limit(function_active, reconstruction_active, pred_tokens.shape[0])
        query_samples = max(
            int(function_cfg.get("query_samples", 0)) if function_active else 0,
            int(reconstruction_cfg.get("query_samples", 0)) if reconstruction_active else 0,
        )
        decoder, _, _ = self._load_training_eval_context()
        query_codes, csi, query_weight, task_count = self._sample_functional_batch(
            batch, max_tasks=max_tasks, query_samples=query_samples
        )
        pred_params = detokenize_adapter_batch(
            pred_tokens[:task_count], self.stats["manifest"], self.stats["parameter_stats"]
        )
        with torch.no_grad():
            teacher_params = detokenize_adapter_batch(
                target_tokens[:task_count].detach(), self.stats["manifest"], self.stats["parameter_stats"]
            )
            teacher_code = functional_adapter(
                query_codes.float(), teacher_params, float(self.cfg.adapter.residual_scale)
            )
        generated_code = functional_adapter(
            query_codes.float(), pred_params, float(self.cfg.adapter.residual_scale)
        )

        function_loss = zero
        if function_active:
            function_loss = _functional_output_loss(
                generated_code,
                teacher_code,
                kind=str(function_cfg.get("type", "relative_log1p")),
                sample_weight=query_weight,
                eps=float(function_cfg.get("eps", 1e-6)),
            )

        reconstruction_loss = zero
        if reconstruction_active:
            generated_csi = _decoder_forward_batched(decoder, generated_code)
            if generated_csi.shape != csi.shape:
                csi = csi.reshape_as(generated_csi)
            reconstruction_loss = _reconstruction_nmse_loss(
                generated_csi, csi.float(), sample_weight=query_weight
            )

        total = function_weight * function_loss + reconstruction_weight * reconstruction_loss
        return total, {
            "function_loss": float(function_loss.detach()),
            "function_weight": float(function_weight),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "reconstruction_weight": float(reconstruction_weight),
            "functional_tasks": float(task_count),
            "functional_queries": float(query_codes.shape[1]),
        }

    @torch.no_grad()
    def _calculate_adapter_nmse(self, task_dir: Path, adapter_state) -> float:
        decoder, _, csi = self._load_training_eval_context()
        codewords = _load_tensor(task_dir / str(self.cfg.data.query_codeword_file))
        if codewords.shape[0] != csi.shape[0]:
            raise ValueError(
                f"{task_dir}: validation codeword count {codewords.shape[0]} "
                f"does not match CSI count {csi.shape[0]}"
            )
        state_device = {key: value.to(self.device) for key, value in adapter_state.items()}
        error_energy = torch.zeros((), dtype=torch.float64, device=self.device)
        signal_energy = torch.zeros((), dtype=torch.float64, device=self.device)
        batch_size = int(self.cfg.eval.batch_size)
        for start in range(0, codewords.shape[0], batch_size):
            codes = codewords[start:start + batch_size].to(self.device)
            target = csi[start:start + batch_size].to(self.device)
            mapped = functional_adapter(
                codes, state_device, float(self.cfg.adapter.residual_scale)
            )
            prediction = decoder(mapped)
            if target.shape != prediction.shape:
                target = target.reshape_as(prediction)
            error_energy += (prediction.double() - target.double()).square().sum()
            signal_energy += target.double().square().sum()
        nmse = 10 * torch.log10(
            error_energy / signal_energy.clamp_min(torch.finfo(torch.float64).tiny)
        )
        return float(nmse.cpu())

    def _functional_eval_max_tasks(self) -> int | None:
        value = self.cfg.eval.get("max_tasks", 1)
        if value is None:
            return None
        return max(1, int(value))

    def _collect_functional_eval_items(self, target_count: int) -> list[dict]:
        items = []
        for batch in self.val_loader:
            batch_size = batch["cond"].shape[0]
            for index in range(batch_size):
                items.append({
                    "cond": batch["cond"][index].detach().cpu(),
                    "raw_cond": (
                        batch["raw_cond"][index].detach().cpu()
                        if "raw_cond" in batch else None
                    ),
                    "structure": {
                        name: batch[name][index].detach().cpu() for name in STRUCTURE_FIELDS
                    },
                    "task_dir": Path(batch["meta"]["task_dir"][index]),
                    "decoder_nmse": float(batch["meta"]["decoder_nmse"][index]),
                })
                if len(items) >= target_count:
                    return items
        return items

    @torch.no_grad()
    def _run_functional_eval_items(
        self,
        epoch: int,
        items: list[dict],
        target_count: int,
        mismatch: bool,
    ) -> tuple[dict[str, float], list[dict]]:
        label = "Mismatched Functional Eval" if mismatch else "Functional Eval"
        generated_values = []
        meta_values = []
        records = []
        log_every = max(1, int(self.cfg.eval.get("log_every_tasks", 20)))
        for index, item in enumerate(items[:target_count]):
            source = items[(index + 1) % len(items)] if mismatch else item
            cond = source["cond"].unsqueeze(0).to(self.device)
            raw_cond = source["raw_cond"]
            if raw_cond is not None:
                raw_cond = raw_cond.unsqueeze(0).to(self.device)
            structure = {
                name: value.unsqueeze(0).to(self.device)
                for name, value in source["structure"].items()
            }
            token_context = self._unwrap(self.token_context_generator)(cond, structure, raw_cond)
            previous_denoiser = self.diffusion.denoiser
            self.diffusion.denoiser = self._unwrap(self.denoiser)
            try:
                samples = self.diffusion.sample(
                    cond=token_context,
                    shape=(
                        1,
                        int(self.stats["manifest"]["num_tokens"]),
                        int(self.stats["manifest"]["token_size"]),
                    ),
                    use_ddim=bool(self.cfg.inference.use_ddim),
                    ddim_steps=int(self.cfg.inference.ddim_steps),
                    eta=float(self.cfg.inference.eta),
                    structure=structure,
                    cfg_scale=float(self.cfg.inference.cfg_scale),
                    uncond_cond=self.null_context,
                )
            finally:
                self.diffusion.denoiser = previous_denoiser
            state = _reconstruct_adapter_state(
                samples[0], self.stats["manifest"], self.stats["parameter_stats"]
            )
            generated_nmse = self._calculate_adapter_nmse(item["task_dir"], state)
            meta_nmse = item["decoder_nmse"]
            gap = generated_nmse - meta_nmse
            generated_values.append(generated_nmse)
            meta_values.append(meta_nmse)
            record = {
                "task_dir": str(item["task_dir"]),
                "condition_task_dir": str(source["task_dir"]),
                "generated_nmse_db": generated_nmse,
                "meta_decoder_nmse_db": meta_nmse,
                "gap_db": gap,
            }
            records.append(record)
            completed = index + 1
            if completed % log_every == 0 or completed == target_count:
                logger.info(
                    "=> %s: epoch=%d [%d/%d] task=%s condition=%s "
                    "generated=%.4f dB meta=%.4f dB gap=%+.4f dB",
                    label, epoch, completed, target_count, item["task_dir"],
                    source["task_dir"], generated_nmse, meta_nmse, gap,
                )

        generated = torch.tensor(generated_values, dtype=torch.float64)
        meta = torch.tensor(meta_values, dtype=torch.float64)
        gaps = generated - meta
        prefix = "mismatch_decoder" if mismatch else "decoder"
        summary = {
            f"{prefix}_generated_nmse_db": float(generated.mean()),
            f"{prefix}_meta_nmse_db": float(meta.mean()),
            f"{prefix}_mean_gap_db": float(gaps.mean()),
            f"{prefix}_median_gap_db": float(torch.quantile(gaps, 0.5)),
        }
        return summary, records

    @torch.no_grad()
    def _validate_decoder_nmse(self, epoch: int) -> dict[str, float]:
        """See README.md for English documentation."""
        started = time.perf_counter()
        self._load_training_eval_context()
        max_tasks = self._functional_eval_max_tasks()
        task_total = len(self.val_loader.dataset) if max_tasks is None else min(max_tasks, len(self.val_loader.dataset))
        collect_count = task_total if len(self.val_loader.dataset) < 2 else max(task_total, 2)
        items = self._collect_functional_eval_items(collect_count)
        if not items:
            raise RuntimeError(
                "val_loader is empty; training-time NMSE evaluation is unavailable")
        task_total = min(task_total, len(items))
        random_devices = [] if self.device.type != "cuda" else [self.device]
        with torch.random.fork_rng(devices=random_devices):
            torch.manual_seed(int(self.cfg.inference.seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(self.cfg.inference.seed))
            summary, records = self._run_functional_eval_items(epoch, items, task_total, mismatch=False)
            if len(items) >= 2:
                mismatch_summary, mismatch_records = self._run_functional_eval_items(
                    epoch, items, task_total, mismatch=True
                )
                summary.update(mismatch_summary)
            else:
                mismatch_records = []
                logger.warning(
                    "=> Mismatched Functional Eval skipped: fewer than 2 "
                    "validation tasks")
        summary["decoder_eval_time_seconds"] = time.perf_counter() - started
        if "mismatch_decoder_mean_gap_db" in summary:
            summary["mismatch_decoder_gap_delta_db"] = (
                summary["mismatch_decoder_mean_gap_db"] - summary["decoder_mean_gap_db"]
            )
        result = {
            "epoch": epoch,
            "max_tasks": max_tasks,
            "num_tasks": len(records),
            **summary,
            "tasks": records,
            "mismatch_tasks": mismatch_records,
        }
        result_path = self.exp_dir / "results" / f"functional_eval_epoch_{epoch:04d}.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "=> Functional Eval Summary: epoch=%d tasks=%d generated_nmse=%.4f dB "
            "meta_nmse=%.4f dB mean_gap=%+.4f dB median_gap=%+.4f dB time=%.2fs",
            epoch, len(records), summary["decoder_generated_nmse_db"],
            summary["decoder_meta_nmse_db"], summary["decoder_mean_gap_db"],
            summary["decoder_median_gap_db"], summary["decoder_eval_time_seconds"],
        )
        if "mismatch_decoder_mean_gap_db" in summary:
            logger.info(
                "=> Mismatched Functional Eval Summary: epoch=%d tasks=%d "
                "generated_nmse=%.4f dB meta_nmse=%.4f dB mean_gap=%+.4f dB "
                "median_gap=%+.4f dB gap_delta=%+.4f dB",
                epoch, len(mismatch_records), summary["mismatch_decoder_generated_nmse_db"],
                summary["mismatch_decoder_meta_nmse_db"], summary["mismatch_decoder_mean_gap_db"],
                summary["mismatch_decoder_median_gap_db"], summary["mismatch_decoder_gap_delta_db"],
            )
        logger.info("=> Functional evaluation results saved: %s", result_path)
        return summary

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        if not self.is_main:
            return
        path = self.exp_dir / "ckpts" / ("best.pth" if is_best else f"{epoch}.pth")
        checkpoint = {
            "token_context_generator": self._unwrap(self.token_context_generator).state_dict(),
            "denoiser": self._unwrap(self.denoiser).state_dict(),
            "null_context": self.null_context.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "stats_fingerprint": self.stats.get("fingerprint"),
        }
        if self.use_ema:
            checkpoint["ema_shadow"] = self.ema.shadow
        torch.save(checkpoint, path)
        logger.info("=> Saved checkpoint: %s", path)

    def load_checkpoint(self, path: str, resume: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_fingerprint = checkpoint.get("stats_fingerprint")
        if ckpt_fingerprint is not None and ckpt_fingerprint != self.stats.get("fingerprint"):
            raise ValueError("Checkpoint and stats.pt fingerprints differ")
        if ckpt_fingerprint is None and resume:
            logger.warning("Checkpoint has no stats_fingerprint; skipping fingerprint validation")
        # English documentation is provided in README.md.
        # English documentation is provided in README.md.
        # English documentation is provided in README.md.
        if "model" in checkpoint and "token_context_generator" not in checkpoint:
            tcg_state = {}
            for key, value in checkpoint["model"].items():
                if key.startswith("token_context_generator."):
                    tcg_state[key[len("token_context_generator."):]] = value
            if not tcg_state:
                raise KeyError(
                    "token_context_generator weights were not found in the "
                    "alignment checkpoint"
                )
            self._unwrap(self.token_context_generator).load_state_dict(tcg_state, strict=False)
            skipped = [k for k in tcg_state if k not in self._unwrap(self.token_context_generator).state_dict()]
            if skipped:
                logger.warning("=> Pretrained keys skipped (arch mismatch): %s", skipped)
            logger.info(
                "=> Loaded alignment pretrained generator from %s "
                "(%d params, denoiser remains randomly initialized)",
                path, len(tcg_state)
            )
        else:
            self._unwrap(self.token_context_generator).load_state_dict(
                checkpoint["token_context_generator"]
            )
        if "denoiser" in checkpoint:
            self._unwrap(self.denoiser).load_state_dict(checkpoint["denoiser"])
        if "null_context" in checkpoint:
            self.null_context.data.copy_(checkpoint["null_context"].to(self.device))
        if resume:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            if self.use_ema:
                if "ema_shadow" in checkpoint:
                    self.ema.shadow = checkpoint["ema_shadow"]
                else:
                    self.ema.register()
                    logger.warning("=> Checkpoint has no EMA shadow; initialized EMA from loaded weights")
            self.start_epoch = int(checkpoint["epoch"]) + 1
            logger.info("=> Resumed checkpoint: %s (start_epoch=%d)", path, self.start_epoch)
        else:
            if self.use_ema:
                self.ema.register()
            logger.info("=> Loaded pretrained checkpoint: %s", path)
