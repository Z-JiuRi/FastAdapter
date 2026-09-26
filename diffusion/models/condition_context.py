from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

def _mlp(sizes: list[int], dropout: float = 0.0, final_zero: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(sizes) - 2:
            layers.extend([nn.SiLU(), nn.Dropout(dropout)])
    if final_zero:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


def _load_mamba_class():
    try:
        from mamba_ssm import Mamba
    except Exception as exc:
        raise ImportError(
            "token_context.implementation=mamba requires mamba_ssm. "
            "Import failure usually means the C compiler or Triton/CUDA "
            "runtime is incomplete. Use gru or transformer when mamba is "
            "not required."
        ) from exc
    return Mamba


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


@dataclass
class ConditionState:
    global_condition: torch.Tensor
    detailed_conditions: torch.Tensor
    gates: torch.Tensor


class MarginalBranch(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, code_dim: int, hidden_dim: int, statistics: list[str], dropout: float):
        super().__init__()
        self.statistics = list(statistics)
        self.mlp = _mlp([len(self.statistics) * code_dim, 2 * hidden_dim, hidden_dim], dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        features = {
            "mean": cond.mean(dim=1),
            "log_std": cond.std(dim=1, unbiased=False).clamp_min(1e-6).log(),
            "mean_abs": cond.abs().mean(dim=1),
        }
        quantile_names = [name for name in self.statistics if name.startswith("q")]
        if quantile_names:
            quantiles = torch.tensor(
                [int(name[1:]) / 100 for name in quantile_names],
                device=cond.device,
                dtype=torch.float32,
            )
            values = torch.quantile(cond.float(), quantiles, dim=1).permute(1, 0, 2)
            for index, name in enumerate(quantile_names):
                features[name] = values[:, index].to(cond.dtype)
        return self.mlp(torch.cat([features[name] for name in self.statistics], dim=-1))[:, None]


class GeometryBranch(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, code_dim: int, hidden_dim: int, cfg: dict[str, Any], dropout: float):
        super().__init__()
        projection_dim = min(code_dim, int(cfg["projection_dim"]))
        generator = torch.Generator().manual_seed(int(cfg.get("projection_seed", 2026)))
        raw = torch.randn(code_dim, projection_dim, generator=generator)
        projection, _ = torch.linalg.qr(raw, mode="reduced")
        self.register_buffer("projection", projection.float())
        self.statistics = list(cfg.get("statistics", ["mean", "log_std", "covariance"]))
        valid_statistics = {"mean", "log_std", "covariance"}
        unknown = sorted(set(self.statistics) - valid_statistics)
        if unknown:
            raise ValueError(
                f"Unsupported geometry.statistics: {unknown}; "
                f"valid values are {sorted(valid_statistics)}")
        covariance_dim = projection_dim * (projection_dim + 1) // 2
        input_dim = 0
        for name in self.statistics:
            input_dim += covariance_dim if name == "covariance" else projection_dim
        self.mlp = _mlp([input_dim, 2 * hidden_dim, hidden_dim], dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        value = cond.matmul(self.projection.to(cond.device, cond.dtype))
        mean = value.mean(dim=1)
        centered = value - mean[:, None]
        covariance = centered.transpose(1, 2).matmul(centered) / max(1, value.shape[1])
        indices = torch.triu_indices(value.shape[-1], value.shape[-1], device=value.device)
        features = {
            "mean": mean,
            "log_std": value.std(dim=1, unbiased=False).clamp_min(1e-6).log(),
            "covariance": covariance[:, indices[0], indices[1]],
        }
        return self.mlp(torch.cat([features[name] for name in self.statistics], dim=-1))[:, None]


class DeepSetsBranch(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, code_dim: int, hidden_dim: int, cfg: dict[str, Any], dropout: float):
        super().__init__()
        inner_dim = int(cfg.get("hidden_dim", hidden_dim))
        self.pooling = list(cfg.get("pooling", ["mean", "max"]))
        valid_pooling = {"mean", "max"}
        unknown = sorted(set(self.pooling) - valid_pooling)
        if unknown:
            raise ValueError(
                f"Unsupported interaction.pooling: {unknown}; "
                f"valid values are {sorted(valid_pooling)}")
        self.phi = _mlp([code_dim, inner_dim, hidden_dim], dropout)
        self.rho = _mlp([len(self.pooling) * hidden_dim, 2 * hidden_dim, hidden_dim], dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        point = self.phi(cond)
        features = {"mean": point.mean(dim=1), "max": point.amax(dim=1)}
        return self.rho(torch.cat([features[name] for name in self.pooling], dim=-1))[:, None]


class SetTransformerBranch(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, code_dim: int, hidden_dim: int, cfg: dict[str, Any], dropout: float):
        super().__init__()
        self.input = nn.Linear(code_dim, hidden_dim)
        self.inducing = nn.Parameter(torch.randn(int(cfg.get("num_inducing", 32)), hidden_dim) * 0.02)
        heads = int(cfg.get("num_heads", 8))
        depth = int(cfg.get("depth", 2))
        self.cross = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2 * depth)])

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        source = self.input(cond)
        latent = self.inducing[None].expand(cond.shape[0], -1, -1)
        for index, (cross, self_attn) in enumerate(zip(self.cross, self.self_attn)):
            latent = latent + cross(self.norms[2 * index](latent), source, source, need_weights=False)[0]
            latent = latent + self_attn(
                self.norms[2 * index + 1](latent), latent, latent, need_weights=False
            )[0]
        return latent


class GramRowBranch(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, num_rows: int, code_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.num_rows = int(num_rows)
        self.row_projection = nn.Linear(code_dim, output_dim)      # English documentation is provided in README.md.
        self.row_position = nn.Embedding(self.num_rows, output_dim)  # English documentation is provided in README.md.
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.shape[1] != self.num_rows:
            raise ValueError(
                f"GramRowBranch expects {self.num_rows} rows, got "
                f"gram shape={tuple(cond.shape)}"
            )
        rows = self.row_projection(cond)                            # (B, num_rows, output_dim)
        positions = self.row_position.weight.unsqueeze(0)           # (1, num_rows, output_dim)
        return self.dropout(self.norm(rows + positions))


class RawProbeRowBranch(nn.Module):
    """Encode coordinate-sensitive codewords from the fixed CSI probes.

    Unlike a Gram matrix, raw probe codewords retain the codeword coordinate
    system.  The amplitude bypass is deliberate: normalizing each input row
    would reintroduce an avoidable invariance and discard potentially useful
    task information.
    """

    def __init__(self, num_rows: int, code_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.num_rows = int(num_rows)
        self.row_projection = nn.Linear(code_dim, output_dim)
        self.row_position = nn.Embedding(self.num_rows, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.amplitude = _mlp([2, output_dim, output_dim], dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 3 or cond.shape[1] != self.num_rows:
            raise ValueError(
                f"RawProbeRowBranch expected (B, {self.num_rows}, C), got {tuple(cond.shape)}"
            )
        rows = self.row_projection(cond)
        positions = self.row_position.weight.unsqueeze(0)
        # Keep per-probe scale and mean as an explicit residual path around
        # LayerNorm; the raw coordinates themselves are projected before this.
        amplitude = torch.stack((
            cond.square().mean(dim=-1).clamp_min(1e-12).sqrt().log(),
            cond.mean(dim=-1),
        ), dim=-1)
        return self.dropout(self.norm(rows + positions) + self.amplitude(amplitude))


class BilinearPool(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, num_rows: int, code_dim: int, output_dim: int, num_queries: int, dropout: float):
        super().__init__()
        self.num_queries = int(num_queries)
        self.query = nn.Parameter(torch.randn(self.num_queries, num_rows) * (num_rows ** -0.5))
        self.mlp = _mlp([self.num_queries * code_dim, 2 * output_dim, output_dim], dropout)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        pooled = torch.einsum("mn,bnd->bmd", self.query, cond)      # (B, m, code_dim)
        return self.mlp(pooled.flatten(1))                          # (B, output_dim)


class GramRawProbeConditionEncoder(nn.Module):
    """Fuse invariant Gram geometry with coordinate-sensitive raw probe rows."""

    names = ("gram_row", "raw_probe")

    def __init__(
        self,
        gram_len: int,
        gram_dim: int,
        raw_len: int,
        raw_dim: int,
        cfg: Mapping[str, Any],
    ):
        super().__init__()
        output_dim = int(cfg.get("output_dim", cfg.get("branch_dim", 512)))
        dropout = float(cfg.get("dropout", 0.0))
        row_cfg = dict(cfg.get("gram_row", cfg.get("row_token", {})))
        raw_cfg = dict(cfg.get("raw_probe", {}))
        if int(row_cfg.get("num_rows", gram_len)) != gram_len:
            raise ValueError("gram_row.num_rows must equal data.cond_shape[0]")
        if int(raw_cfg.get("num_rows", raw_len)) != raw_len:
            raise ValueError("raw_probe.num_rows must equal data.raw_cond_shape[0]")
        self.gram_branch = GramRowBranch(gram_len, gram_dim, output_dim, dropout)
        self.raw_branch = RawProbeRowBranch(raw_len, raw_dim, output_dim, dropout)
        gram_pool_cfg = dict(cfg.get("gram_global_pool", cfg.get("global_pool", {})))
        raw_pool_cfg = dict(cfg.get("raw_global_pool", gram_pool_cfg))
        self.use_gram_pool = bool(gram_pool_cfg.get("enabled", True))
        self.use_raw_pool = bool(raw_pool_cfg.get("enabled", True))
        self.gram_pool = BilinearPool(
            gram_len, gram_dim, output_dim,
            int(gram_pool_cfg.get("num_queries", 8)), dropout,
        )
        self.raw_pool = BilinearPool(
            raw_len, raw_dim, output_dim,
            int(raw_pool_cfg.get("num_queries", 8)), dropout,
        )
        self.global_fusion = _mlp([2 * output_dim, 2 * output_dim, output_dim], dropout)
        self.branch_type = nn.Parameter(torch.randn(2, output_dim) * 0.02)

    def forward(self, gram: torch.Tensor, raw_probe: torch.Tensor) -> ConditionState:
        if gram.ndim != 3 or gram.shape[1] != self.gram_branch.num_rows:
            raise ValueError(f"invalid Gram condition shape: {tuple(gram.shape)}")
        if raw_probe.ndim != 3 or raw_probe.shape[1] != self.raw_branch.num_rows:
            raise ValueError(f"invalid raw-probe condition shape: {tuple(raw_probe.shape)}")
        if gram.shape[0] != raw_probe.shape[0]:
            raise ValueError("Gram and raw-probe conditions must have the same batch size")
        gram_detail = self.gram_branch(gram) + self.branch_type[0]
        raw_detail = self.raw_branch(raw_probe) + self.branch_type[1]
        gram_global = self.gram_pool(gram) if self.use_gram_pool else gram_detail.mean(dim=1)
        raw_global = self.raw_pool(raw_probe) if self.use_raw_pool else raw_detail.mean(dim=1)
        global_condition = self.global_fusion(torch.cat((gram_global, raw_global), dim=-1))
        gates = torch.ones(gram.shape[0], 2, device=gram.device, dtype=gram.dtype)
        return ConditionState(global_condition, torch.cat((gram_detail, raw_detail), dim=1), gates)


class ConditionEncoder(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, code_dim: int, cfg: dict[str, Any]):
        super().__init__()
        self.names = list(cfg.get("branches", ["marginal", "geometry", "interaction"]))
        if not self.names:
            raise ValueError("condition_encoder.branches cannot be empty")
        # English documentation is provided in README.md.
        self.row_token_mode = self.names == ["row_token"]
        if "row_token" in self.names and not self.row_token_mode:
            raise ValueError(
                "row_token must be used alone: branches=[row_token]")
        if self.row_token_mode:
            output_dim = int(cfg.get("output_dim", cfg.get("branch_dim", 512)))
            dropout = float(cfg.get("dropout", 0.0))
            row_cfg = dict(cfg.get("row_token", {}))
            num_rows = int(row_cfg.get("num_rows", cfg.get("_cond_len", code_dim)))
            pool_cfg = dict(cfg.get("global_pool", {}))
            self.row_branch = GramRowBranch(num_rows, code_dim, output_dim, dropout)
            self.global_pool = BilinearPool(
                num_rows, code_dim, output_dim,
                num_queries=int(pool_cfg.get("num_queries", 8)),
                dropout=dropout,
            )
            self.use_global_pool = bool(pool_cfg.get("enabled", True))
            if not self.use_global_pool:
                for parameter in self.global_pool.parameters():
                    parameter.requires_grad_(False)
            return
        hidden_dim = int(cfg.get("branch_dim", cfg.get("output_dim", 512)))
        output_dim = int(cfg.get("output_dim", hidden_dim))
        dropout = float(cfg.get("dropout", 0.0))
        modules: dict[str, nn.Module] = {}
        for name in self.names:
            if name == "marginal":
                modules[name] = MarginalBranch(
                    code_dim, hidden_dim,
                    list(cfg.get(name, {}).get("statistics", ["mean", "log_std", "mean_abs"])),
                    dropout,
                )
            elif name == "geometry":
                modules[name] = GeometryBranch(code_dim, hidden_dim, dict(cfg.get(name, {})), dropout)
            elif name == "interaction":
                interaction_cfg = dict(cfg.get(name, {}))
                if str(interaction_cfg.get("type", "deepsets")) == "set_transformer":
                    modules[name] = SetTransformerBranch(code_dim, hidden_dim, interaction_cfg, dropout)
                else:
                    modules[name] = DeepSetsBranch(code_dim, hidden_dim, interaction_cfg, dropout)
            else:
                raise ValueError(f"Unknown condition_encoder branch: {name}")
        self.branches = nn.ModuleDict(modules)
        self.norms = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in self.names})
        self.gate_type = str(cfg.get("gate_type", "softmax"))
        valid_gate_types = {"softmax", "sigmoid", "uniform"}
        if self.gate_type not in valid_gate_types:
            raise ValueError(
                f"Unsupported condition_encoder.gate_type: {self.gate_type}; "
                f"valid values are {sorted(valid_gate_types)}")
        self.gate = _mlp([len(self.names) * hidden_dim, hidden_dim, len(self.names)], dropout, final_zero=True)
        if self.gate_type == "uniform":
            for parameter in self.gate.parameters():
                parameter.requires_grad_(False)
            for norm in self.norms.values():
                for parameter in norm.parameters():
                    parameter.requires_grad_(False)
        self.global_fusion = _mlp([len(self.names) * hidden_dim, 2 * output_dim, output_dim], dropout)
        self.detail_projection = nn.ModuleDict({
            name: nn.Linear(hidden_dim, output_dim) for name in self.names
        })
        self.use_branch_type = "branch_type" in list(cfg.get("embeddings", []))
        self.branch_type = nn.Parameter(torch.randn(len(self.names), output_dim) * 0.02)

    def forward(self, cond: torch.Tensor) -> ConditionState:
        if cond.ndim != 3:
            raise ValueError(
                f"condition must be a 3D tensor, got {tuple(cond.shape)}")
        if self.row_token_mode:
            detailed = self.row_branch(cond)                        # (B, num_rows, D)
            if self.use_global_pool:
                global_condition = self.global_pool(cond)           # (B, D)
            else:
                global_condition = detailed.mean(dim=1)
            gates = torch.ones(cond.shape[0], 1, device=cond.device, dtype=detailed.dtype)
            return ConditionState(global_condition, detailed, gates)
        branch_sequences = [self.branches[name](cond) for name in self.names]
        summaries = [sequence.mean(dim=1) for sequence in branch_sequences]
        normalized = [self.norms[name](value) for name, value in zip(self.names, summaries)]
        gate_logits = self.gate(torch.cat(normalized, dim=-1))
        if self.gate_type == "softmax":
            gates = gate_logits.softmax(dim=-1)
        elif self.gate_type == "sigmoid":
            gates = gate_logits.sigmoid()
        else:
            gates = torch.ones_like(gate_logits)
        weighted = torch.cat([
            value * gates[:, index:index + 1] for index, value in enumerate(summaries)
        ], dim=-1)
        global_condition = self.global_fusion(weighted)
        detailed = []
        for index, (name, sequence) in enumerate(zip(self.names, branch_sequences)):
            value = self.detail_projection[name](sequence)
            if self.use_branch_type:
                value = value + self.branch_type[index]
            detailed.append(value)
        return ConditionState(global_condition, torch.cat(detailed, dim=1), gates)


class TokenContextGenerator(nn.Module):
    """Translate per-task Gram--Raw support conditions into token contexts."""

    def __init__(
        self,
        input_dim: int,
        cond_len: int,
        raw_input_dim: int | None,
        raw_cond_len: int | None,
        hidden_dim: int,
        condition_config: Mapping,
        structure_config: Mapping,
        structure_sizes: Mapping,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        implementation: str = "transformer",
        condition_injection: str = "cross_attention",
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        output_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_len = cond_len
        self.raw_input_dim = raw_input_dim
        self.raw_cond_len = raw_cond_len
        self.hidden_dim = hidden_dim
        self.implementation = str(implementation)
        self.condition_injection = str(condition_injection)
        self.last_condition_gates: dict[str, float] = {}
        valid_implementations = {"transformer", "gru", "mamba", "none"}
        valid_injections = {"cross_attention", "prefix"}
        if self.implementation not in valid_implementations:
            raise ValueError(
                f"Unknown token_context.implementation: {implementation}")
        if self.condition_injection not in valid_injections:
            raise ValueError(
                "Unknown token_context.condition_injection: "
                f"{condition_injection}")

        condition_cfg = dict(condition_config)
        condition_cfg["output_dim"] = hidden_dim
        condition_cfg["_cond_len"] = cond_len  # English documentation is provided in README.md.
        self.condition_type = str(condition_cfg.get("type", "gram_raw_probe"))
        if self.condition_type == "gram_raw_probe":
            if raw_input_dim is None or raw_cond_len is None:
                raise ValueError("Gram--Raw Probe Condition requires raw_input_dim and raw_cond_len")
            self.condition_encoder = GramRawProbeConditionEncoder(
                cond_len, input_dim, raw_cond_len, raw_input_dim, condition_cfg
            )
        elif self.condition_type == "legacy":
            self.condition_encoder = ConditionEncoder(input_dim, condition_cfg)
        else:
            raise ValueError("condition_encoder.type must be 'gram_raw_probe' or 'legacy'")

        config = dict(structure_config)
        sizes = dict(structure_sizes)
        self.disable_structure = bool(config.get("disable_structure", False))
        position_cfg = dict(config.get("position_2d", {}))
        self.position_type = str(position_cfg.get("type", "learned"))
        valid_position_types = {"learned", "separate", "sinusoidal", "none"}
        if self.position_type not in valid_position_types:
            raise ValueError(
                "Unknown structure_embedding.position_2d.type: "
                f"{self.position_type}")

        num_tensors = int(sizes.get("num_tensors", 26))
        self.num_token_positions = int(sizes.get("max_token_id", 0)) + 1
        if self.position_type == "learned":
            self.position_2d_embed = nn.Embedding(num_tensors * self.num_token_positions, hidden_dim)
            self.tensor_position_embed = None
            self.token_position_embed = None
        elif self.position_type == "separate":
            self.position_2d_embed = None
            self.tensor_position_embed = nn.Embedding(num_tensors, hidden_dim)
            self.token_position_embed = nn.Embedding(self.num_token_positions, hidden_dim)
        elif self.position_type == "sinusoidal":
            self.position_2d_embed = None
            self.tensor_position_embed = None
            self.token_position_embed = None
            self.position_2d_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.position_2d_embed = None
            self.tensor_position_embed = None
            self.token_position_embed = None
            self.position_2d_projection = None
        self.block_embed = (
            nn.Embedding(int(sizes.get("num_block_ids", 5)), hidden_dim)
            if bool(config.get("use_block_id", True)) else None
        )
        self.role_embed = (
            nn.Embedding(int(sizes.get("num_role_ids", 4)), hidden_dim)
            if bool(config.get("use_role_id", True)) else None
        )
        self.kind_embed = (
            nn.Embedding(int(sizes.get("num_kind_ids", 2)), hidden_dim)
            if bool(config.get("use_kind_id", True)) else None
        )

        self.global_projection = nn.Linear(hidden_dim, hidden_dim)
        self.condition_modulation = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.condition_modulation[-1].weight)
        nn.init.zeros_(self.condition_modulation[-1].bias)
        self.detail_projection = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.sequence_type_embed = nn.Embedding(2, hidden_dim)
        # This embedding is only consumed by the prefix-injection branch.  Keep
        # it in the module/state dict for checkpoint compatibility, but exclude
        # it from DDP reduction when cross-attention is selected.
        if self.condition_injection == "cross_attention":
            self.sequence_type_embed.weight.requires_grad_(False)

        self.transformer = None
        self.gru = None
        self.gru_output = None
        self.mamba_blocks = None
        self.mamba_norms = None
        if self.implementation == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif self.implementation == "gru":
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.gru_output = nn.Linear(2 * hidden_dim, hidden_dim)
        elif self.implementation == "mamba":
            Mamba = _load_mamba_class()
            self.mamba_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
            self.mamba_blocks = nn.ModuleList([
                Mamba(
                    d_model=hidden_dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                )
                for _ in range(num_layers)
            ])
            self.mamba_dropout = nn.Dropout(dropout)
        elif self.implementation == "none":
            pass  # English documentation is provided in README.md.
        self.output_norm = nn.LayerNorm(hidden_dim) if output_norm else nn.Identity()

    def _position_2d(self, tensor_ids: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if self.position_type == "learned":
            return self.position_2d_embed(tensor_ids * self.num_token_positions + token_ids)
        if self.position_type == "separate":
            return self.tensor_position_embed(tensor_ids) + self.token_position_embed(token_ids)
        if self.position_type == "none":
            return torch.zeros((*tensor_ids.shape, self.hidden_dim), device=tensor_ids.device)
        left_dim = self.hidden_dim // 2
        value = torch.cat(
            (
                sinusoidal_embedding(tensor_ids, left_dim),
                sinusoidal_embedding(token_ids, self.hidden_dim - left_dim),
            ),
            dim=-1,
        )
        return self.position_2d_projection(value.to(dtype=self.position_2d_projection.weight.dtype))

    def _structure_embedding(self, structure: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensor_ids = structure["tensor_ids"]
        token_ids = structure["token_ids"]
        result = torch.zeros((*tensor_ids.shape, self.hidden_dim), device=tensor_ids.device)
        if self.disable_structure:
            return result
        result = result + self._position_2d(tensor_ids, token_ids)
        for name, embedding in (
            ("block_ids", self.block_embed),
            ("role_ids", self.role_embed),
            ("kind_ids", self.kind_embed),
        ):
            if embedding is not None:
                result = result + embedding(structure[name])
        return result

    def _run_sequence_model(self, sequence: torch.Tensor) -> torch.Tensor:
        if self.implementation == "transformer":
            return self.transformer(sequence)
        if self.implementation == "gru":
            mixed, _ = self.gru(sequence)
            return self.gru_output(mixed)
        if self.implementation == "mamba":
            if not sequence.is_cuda:
                raise RuntimeError(
                    "token_context implementation=mamba requires CUDA")
            for norm, block in zip(self.mamba_norms, self.mamba_blocks):
                sequence = sequence + self.mamba_dropout(block(norm(sequence)))
            return sequence
        if self.implementation == "none":
            return sequence  # English documentation is provided in README.md.
        raise RuntimeError(
            f"Unreachable token_context.implementation: {self.implementation}")

    def forward(
        self,
        cond: torch.Tensor,
        structure: Mapping[str, torch.Tensor],
        raw_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond.ndim != 3 or cond.shape[1:] != (self.cond_len, self.input_dim):
            raise ValueError(
                f"condition shape {tuple(cond.shape)} != (B, {self.cond_len}, {self.input_dim})"
            )
        if self.condition_type == "gram_raw_probe":
            if raw_cond is None:
                raise ValueError("Gram--Raw Probe Condition requires raw_cond")
            if raw_cond.ndim != 3 or raw_cond.shape[1:] != (self.raw_cond_len, self.raw_input_dim):
                raise ValueError(
                    f"raw condition shape {tuple(raw_cond.shape)} != "
                    f"(B, {self.raw_cond_len}, {self.raw_input_dim})"
                )
            condition = self.condition_encoder(cond, raw_cond)
        else:
            condition = self.condition_encoder(cond)
        with torch.no_grad():
            mean_gates = condition.gates.detach().float().mean(dim=0)
            self.last_condition_gates = {
                f"condition_gate_{name}": float(mean_gates[index])
                for index, name in enumerate(self.condition_encoder.names)
            }
        structure_value = self._structure_embedding(structure)
        global_value = self.global_projection(condition.global_condition).unsqueeze(1)
        modulation = self.condition_modulation(
            torch.cat((structure_value, global_value.expand_as(structure_value)), dim=-1)
        )
        scale, shift = modulation.chunk(2, dim=-1)
        value = structure_value * (1.0 + torch.tanh(scale)) + shift
        detailed = self.detail_projection(condition.detailed_conditions)
        if self.condition_injection == "cross_attention":
            attended = self.cross_attention(
                value, detailed, detailed, need_weights=False
            )[0]
            token_context = self._run_sequence_model(attended)
        else:
            detail_type = self.sequence_type_embed.weight[0].view(1, 1, -1)
            token_type = self.sequence_type_embed.weight[1].view(1, 1, -1)
            sequence = torch.cat((detailed + detail_type, value + token_type), dim=1)
            token_context = self._run_sequence_model(sequence)[:, detailed.shape[1]:]
        return self.output_norm(token_context)
