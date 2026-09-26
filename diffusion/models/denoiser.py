from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # English documentation is provided in README.md.
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def sinusoidal_embedding(ids: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return ids.float().unsqueeze(-1)
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=ids.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = ids.float().unsqueeze(-1) * frequencies
    result = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


class ResFFNBlock(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float, context_injection: str = 'per_token'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_injection = context_injection
        self.last_gate_metrics: dict[str, float] = {}
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        inner_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim), nn.Dropout(dropout),
        )
        # English documentation is provided in README.md.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 3 * hidden_dim))
        # English documentation is provided in README.md.
        self._has_ctx = context_injection in ('adaln', 'per_token', 'cross_attn')
        if self._has_ctx:
            self.ctx_modulation = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 3 * hidden_dim),
            )
        # English documentation is provided in README.md.
        if context_injection == 'cross_attn':
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=8, batch_first=True, dropout=dropout,
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, time: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        # Time modulation: unsqueeze (B, D) to (B, 1, D) for broadcasting.
        shift, scale, gate = self.adaLN_modulation(time).chunk(3, dim=-1)  # (B, D) each
        shift = shift.unsqueeze(1)    # (B, 1, D)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)

        if self._has_ctx and context is not None:
            if self.context_injection == 'adaln':
                # English documentation is provided in README.md.
                ctx_in = context.mean(dim=1)                       # (B, D)
                shift_c, scale_c, gate_c = self.ctx_modulation(ctx_in).chunk(3, dim=-1)  # (B, D)
                shift_c = shift_c.unsqueeze(1)  # (B, 1, D)
                scale_c = scale_c.unsqueeze(1)
                gate_c = gate_c.unsqueeze(1)
            else:
                # Per-token or cross-attention context maps from (B,N,D)
                # through modulation (B,N,3*D) and returns (B,N,D).
                shift_c, scale_c, gate_c = self.ctx_modulation(context).chunk(3, dim=-1)
            shift = shift + shift_c
            scale = scale + scale_c
            gate = gate + gate_c

        # English documentation is provided in README.md.
        if self.context_injection == 'cross_attn' and context is not None:
            x = x + self.cross_attn(
                self.cross_attn_norm(x), context, context, need_weights=False,
            )[0]

        with torch.no_grad():
            gate_rms = float(gate.float().square().mean().sqrt())
            self.last_gate_metrics = {
                "residual_gate_rms": gate_rms,
                "residual_gate_abs_max": float(gate.float().abs().max()),
            }
            if self._has_ctx and context is not None:
                time_rms = float(self.adaLN_modulation(time).float().square().mean().sqrt())
                ctx_rms = float(self.ctx_modulation(context).float().square().mean().sqrt())
                total_rms = time_rms + ctx_rms + 1e-12
                self.last_gate_metrics["ctx_mod_ratio"] = ctx_rms / total_rms

        # The gate is (B,N,D) or (B,1,D), so it broadcasts with x.
        return x + gate * self.mlp(modulate(self.norm(x), shift, scale))


class ResFFNDenoiser(nn.Module):
    """See README.md for English documentation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gate_init: float = 0.01,
        context_injection: str = 'per_token',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.residual_gate_init = float(gate_init)
        self.context_injection = context_injection
        self.x_embedder = nn.Linear(input_dim, hidden_dim)
        self.t_embedder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            ResFFNBlock(hidden_dim, mlp_ratio, dropout, context_injection=context_injection)
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-8)
        self.final_linear = nn.Linear(hidden_dim, input_dim)
        # English documentation is provided in README.md.
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self._has_ctx = context_injection in ('adaln', 'per_token', 'cross_attn')
        if self._has_ctx:
            self.ctx_final = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 2 * hidden_dim),
            )
        # Cross-attention in final layer
        if context_injection == 'cross_attn':
            self.final_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=8, batch_first=True, dropout=dropout,
            )
            self.final_cross_attn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        for block in self.blocks:
            # English documentation is provided in README.md.
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            bias = block.adaLN_modulation[-1].bias.view(3, self.hidden_dim)
            with torch.no_grad():
                bias[2].fill_(self.residual_gate_init)
            # English documentation is provided in README.md.
            if block._has_ctx:
                nn.init.xavier_uniform_(block.ctx_modulation[0].weight)
                nn.init.xavier_uniform_(block.ctx_modulation[-1].weight)
                nn.init.zeros_(block.ctx_modulation[-1].bias)
        # final layer init
        nn.init.zeros_(self.adaLN_final[-1].weight)
        nn.init.zeros_(self.adaLN_final[-1].bias)
        if self._has_ctx:
            nn.init.xavier_uniform_(self.ctx_final[0].weight)
            nn.init.xavier_uniform_(self.ctx_final[-1].weight)
            nn.init.zeros_(self.ctx_final[-1].bias)

    def gate_metrics(self) -> dict[str, float]:
        """See README.md for English documentation."""
        available = [block.last_gate_metrics for block in self.blocks if block.last_gate_metrics]
        if not available:
            return {"residual_gate_rms": 0.0, "residual_gate_abs_max": 0.0}
        result = {}
        for name in ("residual_gate_rms", "residual_gate_abs_max", "ctx_mod_ratio"):
            if name not in available[0]:
                continue
            values = [metrics[name] for metrics in available if name in metrics]
            if name == "residual_gate_abs_max":
                result[name] = max(values)
            else:
                result[name] = sum(values) / len(values) if values else 0.0
        return result

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        token_context: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        hidden = self.x_embedder(x)
        if token_context is None:
            raise ValueError("res_ffn denoiser requires token_context")
        if token_context.shape != hidden.shape:
            raise ValueError(
                f"token_context shape {tuple(token_context.shape)} != hidden shape {tuple(hidden.shape)}"
            )
        if self.context_injection == 'input':
            # English documentation is provided in README.md.
            hidden = hidden + token_context
        time = self.t_embedder(sinusoidal_embedding(t, self.hidden_dim))
        for block in self.blocks:
            hidden = block(hidden, time, token_context)
        # English documentation is provided in README.md.
        shift, scale = self.adaLN_final(time).chunk(2, dim=-1)         # (B, D) each
        shift = shift.unsqueeze(1)   # (B, 1, D)
        scale = scale.unsqueeze(1)
        if self._has_ctx:
            if self.context_injection == 'adaln':
                ctx_in = token_context.mean(dim=1)                            # (B, D)
                shift_c, scale_c = self.ctx_final(ctx_in).chunk(2, dim=-1)   # (B, D)
                shift_c = shift_c.unsqueeze(1)
                scale_c = scale_c.unsqueeze(1)
            else:
                ctx_in = token_context                                         # (B, N, D)
                shift_c, scale_c = self.ctx_final(ctx_in).chunk(2, dim=-1)   # (B, N, D)
            shift = shift + shift_c
            scale = scale + scale_c
        # Cross-attention in final layer
        if self.context_injection == 'cross_attn':
            hidden = hidden + self.final_cross_attn(
                self.final_cross_attn_norm(hidden), token_context, token_context,
                need_weights=False,
            )[0]
        return self.final_linear(modulate(self.final_norm(hidden), shift, scale))


class TokenUNetDenoiser(nn.Module):
    """See README.md for English documentation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_channels: list[int],
        kernel_size: int = 3,
        time_embed_dim: int | None = None,
        context_injection: str = 'none',
    ):
        super().__init__()
        if len(layer_channels) < 3:
            raise ValueError(
                "denoiser.unet.layer_channels requires at least 3 entries")
        if layer_channels[0] != 1 or layer_channels[-1] != 1:
            raise ValueError(
                "denoiser.unet.layer_channels must start and end with 1")
        if kernel_size % 2 == 0:
            raise ValueError(
                "denoiser.unet.kernel_size must be odd to preserve token_size")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_injection = context_injection
        self.time_embed_dim = int(time_embed_dim or hidden_dim)
        self.time_embedder = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, input_dim)
        )
        self.context_embedder = nn.Linear(hidden_dim, input_dim)
        split = len(layer_channels) // 2
        self.encoder = nn.ModuleList()
        for index in range(split):
            self.encoder.append(self._conv_block(layer_channels[index], layer_channels[index + 1], kernel_size))
        self.middle = self._conv_block(layer_channels[split], layer_channels[split + 1], kernel_size)
        self.decoder = nn.ModuleList()
        for index in range(split + 1, len(layer_channels) - 1):
            self.decoder.append(self._conv_block(
                layer_channels[index], layer_channels[index + 1], kernel_size,
                final=layer_channels[index + 1] == 1,
            ))
        # English documentation is provided in README.md.
        if context_injection == 'adaln':
            all_channels = list(layer_channels)
            # English documentation is provided in README.md.
            self.ctx_biases = nn.ModuleList()
            for oc in all_channels[1:-1]:  # skip input channel=1 and final channel=1
                self.ctx_biases.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                    nn.Linear(hidden_dim, oc),
                ))

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, kernel_size: int, final: bool = False) -> nn.Sequential:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        ]
        if not final:
            layers.extend([nn.BatchNorm1d(out_channels), nn.ELU()])
        return nn.Sequential(*layers)

    def _apply_context_bias(self, value: torch.Tensor, ctx_bias: torch.Tensor, batch: int, num_tokens: int) -> torch.Tensor:
        """See README.md for English documentation."""
        expanded = ctx_bias.unsqueeze(-1).unsqueeze(1).expand(batch, num_tokens, -1, 1)
        expanded = expanded.reshape(batch * num_tokens, -1, 1)
        return value + expanded

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        token_context: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if token_context is None:
            raise ValueError("unet denoiser requires token_context")
        if x.shape[:2] != token_context.shape[:2] or token_context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x shape {tuple(x.shape)} does not match token_context "
                f"shape {tuple(token_context.shape)}"
            )
        batch, num_tokens, token_size = x.shape
        if token_size != self.input_dim:
            raise ValueError(f"token_size={token_size} != input_dim={self.input_dim}")
        flat_x = x.reshape(batch * num_tokens, token_size)
        flat_context = token_context.reshape(batch * num_tokens, self.hidden_dim)
        flat_t = t[:, None].expand(batch, num_tokens).reshape(batch * num_tokens)
        time = self.time_embedder(sinusoidal_embedding(flat_t, self.time_embed_dim))
        context = self.context_embedder(flat_context)
        value = ((flat_x + context) * (1 + time)).unsqueeze(1)

        # Per-layer context injection
        ctx_pooled = None
        if self.context_injection == 'adaln':
            ctx_pooled = token_context.mean(dim=1)  # (B, hidden_dim)

        skips = []
        ctx_idx = 0
        for index, block in enumerate(self.encoder):
            value = block(value)
            if ctx_pooled is not None:
                value = self._apply_context_bias(value, self.ctx_biases[ctx_idx](ctx_pooled), batch, num_tokens)
                ctx_idx += 1
            if index < len(self.encoder) - 1:
                skips.append(value)
        value = self.middle(value)
        if ctx_pooled is not None:
            value = self._apply_context_bias(value, self.ctx_biases[ctx_idx](ctx_pooled), batch, num_tokens)
            ctx_idx += 1
        for index, block in enumerate(self.decoder):
            if index < len(skips):
                value = value + skips[-index - 1]
            value = block(value)
            if ctx_pooled is not None and index < len(self.decoder) - 1:
                value = self._apply_context_bias(value, self.ctx_biases[ctx_idx](ctx_pooled), batch, num_tokens)
                ctx_idx += 1
        return value.squeeze(1).reshape(batch, num_tokens, token_size)
