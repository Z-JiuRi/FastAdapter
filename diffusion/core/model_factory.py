from __future__ import annotations

from pathlib import Path

import torch

from models.ddpm import GaussianDiffusion
from models.condition_context import TokenContextGenerator
from models.denoiser import ResFFNDenoiser, TokenUNetDenoiser
from data.adapter_tokenizer import validate_stats_fingerprint


def load_stats(cfg, verify_sources: bool = True) -> dict:
    path = Path(str(cfg.cache.stats_path))
    if not path.is_file():
        raise FileNotFoundError(f"Offline stats not found: {path}")
    stats = torch.load(path, map_location="cpu", weights_only=True)
    if stats.get("alignment") != str(cfg.ablation.alignment):
        raise ValueError("stats alignment does not match ablation.alignment")
    if bool(cfg.cache.get("strict_fingerprint", True)):
        validate_stats_fingerprint(stats, verify_sources=verify_sources)
    return stats


def build_generation_models(cfg, stats: dict, device: torch.device):
    manifest = stats["manifest"]
    hidden_dim = int(cfg.token_context.hidden_dim)
    structure_config = dict(cfg.structure_embedding)
    structure_config["disable_structure"] = bool(cfg.ablation.get("disable_structure", False))
    token_context_generator = TokenContextGenerator(
        input_dim=int(cfg.data.cond_shape[1]),
        cond_len=int(cfg.data.cond_shape[0]),
        raw_input_dim=(
            int(cfg.data.raw_cond_shape[1])
            if str(cfg.data.get("raw_codeword_file", "")) else None
        ),
        raw_cond_len=(
            int(cfg.data.raw_cond_shape[0])
            if str(cfg.data.get("raw_codeword_file", "")) else None
        ),
        hidden_dim=hidden_dim,
        condition_config=dict(cfg.condition_encoder),
        structure_config=structure_config,
        structure_sizes=manifest,
        num_layers=int(cfg.token_context.get("num_layers", 2)),
        num_heads=int(cfg.token_context.get("num_heads", 8)),
        dropout=float(cfg.token_context.get("dropout", 0.0)),
        implementation=str(cfg.token_context.get("implementation", "transformer")),
        condition_injection=str(cfg.token_context.get("condition_injection", "cross_attention")),
        mamba_d_state=int(cfg.token_context.get("mamba", {}).get("d_state", 16)),
        mamba_d_conv=int(cfg.token_context.get("mamba", {}).get("d_conv", 4)),
        mamba_expand=int(cfg.token_context.get("mamba", {}).get("expand", 2)),
        output_norm=bool(cfg.token_context.get("output_norm", True)),
    ).to(device)
    denoiser_type = str(cfg.denoiser.type)
    if denoiser_type == "res_ffn":
        res_ffn_cfg = cfg.denoiser.res_ffn
        denoiser = ResFFNDenoiser(
            input_dim=int(cfg.data.token_size),
            hidden_dim=hidden_dim,
            depth=int(res_ffn_cfg.depth),
            mlp_ratio=float(res_ffn_cfg.mlp_ratio),
            dropout=float(res_ffn_cfg.dropout),
            gate_init=float(res_ffn_cfg.gate_init),
            context_injection=str(res_ffn_cfg.get("context_injection", "per_token")),
        ).to(device)
    elif denoiser_type == "unet":
        unet_cfg = cfg.denoiser.unet
        denoiser = TokenUNetDenoiser(
            input_dim=int(cfg.data.token_size),
            hidden_dim=hidden_dim,
            layer_channels=[int(value) for value in unet_cfg.layer_channels],
            kernel_size=int(unet_cfg.kernel_size),
            time_embed_dim=int(unet_cfg.get("time_embed_dim", hidden_dim)),
            context_injection=str(unet_cfg.get("context_injection", "none")),
        ).to(device)
    else:
        raise ValueError("denoiser.type must be res_ffn or unet")
    tensor_balance = 0.0 if bool(cfg.ablation.get("disable_tensor_balance", False)) else float(cfg.loss.tensor_balance)
    tensor_ranges = [
        (int(spec["token_start"]), int(spec["token_count"])) for spec in manifest["tensors"]
    ]
    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        timesteps=int(cfg.diffusion.timesteps),
        beta_kwargs=dict(cfg.diffusion.betas),
        prediction_type=str(cfg.diffusion.prediction_type),
        snr_gamma=cfg.diffusion.get("snr_gamma"),
        tensor_balance=tensor_balance,
        tensor_ranges=tensor_ranges,
    ).to(device)
    return token_context_generator, denoiser, diffusion
