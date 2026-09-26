from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


BASE_ROOT = Path(__file__).resolve().parents[1] / "base"
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))


def _block_indices(params: Mapping[str, torch.Tensor]) -> list[int]:
    return sorted({int(key.split(".")[1]) for key in params if key.startswith("blocks.")})


def functional_adapter(
    codewords: torch.Tensor,
    params: Mapping[str, torch.Tensor],
    residual_scale: float = 0.4,
) -> torch.Tensor:
    """Differentiable Adapter forward for one state or a batch of states.

    Shapes are inferred from the selected dataset.  Unbatched parameters map
    ``(P,C) -> (P,C)``; batched parameters map ``(B,P,C) -> (B,P,C)``.
    """
    weight = params["alignment_weight"]
    batched_params = weight.ndim == 3
    squeeze_batch = codewords.ndim == 2 and batched_params
    if squeeze_batch:
        codewords = codewords.unsqueeze(0)
    if batched_params:
        if codewords.ndim != 3 or codewords.shape[0] != weight.shape[0]:
            raise ValueError("Batched Adapter parameters require codewords shaped (B,P,C)")
        value = torch.matmul(codewords, weight) + params["alignment_bias"][:, None]
    else:
        value = codewords.matmul(weight) + params["alignment_bias"]
    for block in _block_indices(params):
        prefix = f"blocks.{block}."
        if batched_params:
            mean = value.mean(-1, keepdim=True)
            variance = value.var(-1, unbiased=False, keepdim=True)
            hidden = (value - mean) * torch.rsqrt(variance + 1e-5)
            hidden = hidden * params[prefix + "norm.weight"][:, None]
            hidden = hidden + params[prefix + "norm.bias"][:, None]
            hidden = torch.matmul(
                hidden, params[prefix + "net.0.weight"].transpose(-1, -2)
            ) + params[prefix + "net.0.bias"][:, None]
            hidden = F.gelu(hidden)
            hidden = torch.matmul(
                hidden, params[prefix + "net.3.weight"].transpose(-1, -2)
            ) + params[prefix + "net.3.bias"][:, None]
        else:
            hidden = F.layer_norm(
                value, (value.shape[-1],), params[prefix + "norm.weight"],
                params[prefix + "norm.bias"], 1e-5,
            )
            hidden = F.gelu(F.linear(
                hidden, params[prefix + "net.0.weight"], params[prefix + "net.0.bias"]
            ))
            hidden = F.linear(
                hidden, params[prefix + "net.3.weight"], params[prefix + "net.3.bias"]
            )
        value = value + residual_scale * hidden
    return value[0] if squeeze_batch else value


def _clean_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    clean = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.endswith("total_ops") or key.endswith("total_params"):
            continue
        clean[key] = value
    return clean


def load_frozen_decoder(spec: dict[str, Any], device: torch.device) -> nn.Module:
    args = json.loads(Path(spec["args_json"]).read_text(encoding="utf-8"))
    from models.UniversalCSI import build_decoder
    decoder = build_decoder(
        args["decoder"], args["cr"], args.get("d_model", 64),
        args.get("channel", 2), args.get("nt", 32), args.get("nc", 32),
        args.get("dim_feedforward", 2048), args.get("hidden", 16),
        args.get("num_blocks", 2),
    )
    checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = _clean_state_dict(checkpoint)
    decoder_state = {
        key[len("decoder."):]: value for key, value in state.items() if key.startswith("decoder.")
    }
    if not decoder_state:
        decoder_state = state
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Decoder mismatch: missing={missing}, unexpected={unexpected}")
    decoder.to(device).eval().requires_grad_(False)
    if spec.get("compile", False) and hasattr(torch, "compile"):
        decoder = torch.compile(decoder)
    return decoder


def decoder_forward(decoder: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim == 2:
        return decoder(codes)
    batch, samples, dim = codes.shape
    output = decoder(codes.reshape(batch * samples, dim))
    return output.reshape(batch, samples, *output.shape[1:])
