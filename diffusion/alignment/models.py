from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) in sys.path:
    sys.path.remove(str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT))

from models.condition_context import TokenContextGenerator, sinusoidal_embedding, GramRowBranch, BilinearPool


class AttentionPool(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * (dim ** -0.5))
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # English documentation is provided in README.md.
        query = self.query.expand(tokens.shape[0], -1, -1)             # (B, 1, D)
        key_padding_mask = ~valid.bool()                               # English documentation is provided in README.md.
        pooled, _ = self.attention(
            query, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(pooled.squeeze(1))                            # (B, D)


class GramRowAlignmentModel(nn.Module):
    """See README.md for English documentation."""

    def __init__(self, cfg, manifest: Mapping[str, Any]):
        super().__init__()
        hidden_dim = int(cfg.token_context.hidden_dim)
        embedding_dim = int(cfg.alignment.embedding_dim)
        if hidden_dim != embedding_dim:
            raise ValueError("hidden_dim must equal embedding_dim")
        cond_cfg = dict(cfg.condition_encoder)
        output_dim = int(cond_cfg.get("output_dim", hidden_dim))
        row_cfg = dict(cond_cfg.get("row_token", {}))
        pool_cfg = dict(cond_cfg.get("global_pool", {}))
        num_rows = int(row_cfg.get("num_rows", 128))
        dropout = float(cond_cfg.get("dropout", 0.0))
        # English documentation is provided in README.md.
        self.row_branch = GramRowBranch(num_rows, int(cfg.data.cond_shape[1]), output_dim, dropout)
        self.use_global_pool = bool(pool_cfg.get("enabled", True))
        if self.use_global_pool:
            self.global_pool = BilinearPool(num_rows, int(cfg.data.cond_shape[1]), output_dim,
                                            int(pool_cfg.get("num_queries", 8)), dropout)
        # English documentation is provided in README.md.
        row_layers = int(cond_cfg.get("row_transformer_layers", 2))
        if row_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=output_dim, nhead=int(cfg.token_context.get("num_heads", 8)),
                dim_feedforward=4 * output_dim, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.row_transformer = nn.TransformerEncoder(layer, num_layers=row_layers)
        else:
            self.row_transformer = None
        self.output_norm = nn.LayerNorm(output_dim)
        # English documentation is provided in README.md.
        self.parameter_projection = ParameterTokenProjector(
            int(manifest["token_size"]), embedding_dim, manifest,
            dict(cfg.structure_embedding), dropout=0.0, use_structure=False,
        )
        if bool(cfg.alignment.train_parameter_projection):
            raise ValueError("train_parameter_projection must be false")
        for p in self.parameter_projection.parameters():
            p.requires_grad_(False)

    def forward(self, cond, tokens, token_mask, structure, raw_cond=None):
        """Map a (B,128,128) Gram tensor to a (B,512) task vector."""
        row_tokens = self.row_branch(cond)                            # (B, 128, 512)
        if self.row_transformer is not None:
            row_tokens = self.row_transformer(row_tokens)            # (B, 128, 512)
        # English documentation is provided in README.md.
        context_global = self.output_norm(row_tokens.mean(dim=1))    # (B, 512)
        if self.use_global_pool:
            global_val = self.global_pool(cond)                       # (B, 512)
            context_global = context_global + global_val              # English documentation is provided in README.md.
        # English documentation is provided in README.md.
        parameter_embedding = self.parameter_projection(tokens, token_mask, structure).detach()
        if token_mask is not None:
            valid = token_mask.any(dim=-1) if token_mask.ndim == 3 else token_mask.bool()
            mask = valid.to(parameter_embedding.dtype).unsqueeze(-1)
            target_global = (parameter_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        else:
            valid = torch.ones(parameter_embedding.shape[:2], dtype=torch.bool,
                               device=parameter_embedding.device)
            target_global = parameter_embedding.mean(dim=1)
        broadcast_context = row_tokens.mean(dim=1, keepdim=True).expand_as(parameter_embedding)
        return {
            "context_global": context_global,
            "target_global": target_global,
            "token_context": broadcast_context,
            "context_embedding": broadcast_context,
            "parameter_embedding": parameter_embedding,
            "valid": valid,
        }


class ParameterTokenProjector(nn.Module):
    """Fixed or trainable adapter-token projection used only for alignment loss."""

    def __init__(
        self,
        token_size: int,
        embedding_dim: int,
        structure_sizes: Mapping[str, Any],
        structure_config: Mapping[str, Any],
        dropout: float = 0.0,
        use_structure: bool = False,
    ):
        super().__init__()
        self.use_structure = use_structure
        self.token_projection = nn.Linear(token_size, embedding_dim, bias=False)
        if token_size == embedding_dim:
            nn.init.eye_(self.token_projection.weight)
        else:
            nn.init.orthogonal_(self.token_projection.weight)
        config = dict(structure_config)
        position_cfg = dict(config.get("position_2d", {}))
        self.position_type = str(position_cfg.get("type", "learned"))
        valid_position_types = {"learned", "separate", "sinusoidal", "none"}
        if self.position_type not in valid_position_types:
            raise ValueError(
                "Unknown structure_embedding.position_2d.type: "
                f"{self.position_type}")
        num_tensors = int(structure_sizes["num_tensors"])
        self.num_token_positions = int(structure_sizes["max_token_id"]) + 1
        if self.position_type == "learned":
            self.position_2d_embed = nn.Embedding(num_tensors * self.num_token_positions, embedding_dim)
            self.tensor_position_embed = None
            self.token_position_embed = None
            self.position_2d_projection = None
        elif self.position_type == "separate":
            self.position_2d_embed = None
            self.tensor_position_embed = nn.Embedding(num_tensors, embedding_dim)
            self.token_position_embed = nn.Embedding(self.num_token_positions, embedding_dim)
            self.position_2d_projection = None
        elif self.position_type == "sinusoidal":
            self.position_2d_embed = None
            self.tensor_position_embed = None
            self.token_position_embed = None
            self.position_2d_projection = nn.Linear(embedding_dim, embedding_dim)
        else:
            self.position_2d_embed = None
            self.tensor_position_embed = None
            self.token_position_embed = None
            self.position_2d_projection = None
        self.block_embed = (
            nn.Embedding(int(structure_sizes["num_block_ids"]), embedding_dim)
            if bool(config.get("use_block_id", True)) else None
        )
        self.role_embed = (
            nn.Embedding(int(structure_sizes["num_role_ids"]), embedding_dim)
            if bool(config.get("use_role_id", True)) else None
        )
        self.kind_embed = (
            nn.Embedding(int(structure_sizes["num_kind_ids"]), embedding_dim)
            if bool(config.get("use_kind_id", True)) else None
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def _position_2d(self, tensor_ids: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if self.position_type == "learned":
            return self.position_2d_embed(tensor_ids * self.num_token_positions + token_ids)
        if self.position_type == "separate":
            return self.tensor_position_embed(tensor_ids) + self.token_position_embed(token_ids)
        if self.position_type == "none":
            return torch.zeros((*tensor_ids.shape, self.norm.normalized_shape[0]), device=tensor_ids.device)
        left_dim = self.norm.normalized_shape[0] // 2
        value = torch.cat(
            (
                sinusoidal_embedding(tensor_ids, left_dim),
                sinusoidal_embedding(token_ids, self.norm.normalized_shape[0] - left_dim),
            ),
            dim=-1,
        )
        return self.position_2d_projection(value.to(dtype=self.position_2d_projection.weight.dtype))

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        structure: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        value = self.token_projection(tokens * token_mask.to(tokens.dtype))
        if self.use_structure:
            value = value + self._position_2d(structure["tensor_ids"], structure["token_ids"])
            for name, embedding in (
                ("block_ids", self.block_embed),
                ("role_ids", self.role_embed),
                ("kind_ids", self.kind_embed),
            ):
                if embedding is not None:
                    value = value + embedding(structure[name])
        return self.dropout(self.norm(value))


class TokenContextAlignmentModel(nn.Module):
    def __init__(self, cfg, manifest: Mapping[str, Any]):
        super().__init__()
        hidden_dim = int(cfg.token_context.hidden_dim)
        embedding_dim = int(cfg.alignment.embedding_dim)
        self.token_context_generator = TokenContextGenerator(
            input_dim=int(cfg.data.cond_shape[1]),
            cond_len=int(cfg.data.cond_shape[0]),
            raw_input_dim=(int(cfg.data.raw_cond_shape[1]) if str(cfg.data.get("raw_codeword_file", "")) else None),
            raw_cond_len=(int(cfg.data.raw_cond_shape[0]) if str(cfg.data.get("raw_codeword_file", "")) else None),
            hidden_dim=hidden_dim,
            condition_config=dict(cfg.condition_encoder),
            structure_config=dict(cfg.structure_embedding),
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
        )
        if str(cfg.token_context.get("condition_injection", "cross_attention")) == "cross_attention":
            self.token_context_generator.sequence_type_embed.requires_grad_(False)
        else:
            for parameter in self.token_context_generator.cross_attention.parameters():
                parameter.requires_grad_(False)
        if hidden_dim != embedding_dim:
            raise ValueError(
                "token_context.hidden_dim must equal alignment.embedding_dim: "
                f"got {hidden_dim} and {embedding_dim}"
            )
        self.context_projection = nn.Identity()
        self.parameter_projection = ParameterTokenProjector(
            int(manifest["token_size"]),
            embedding_dim,
            manifest,
            dict(cfg.structure_embedding),
            dropout=0.0,
            use_structure=False,
        )
        if bool(cfg.alignment.train_parameter_projection):
            raise ValueError("alignment.train_parameter_projection must be false: the adapter target is fixed")
        for parameter in self.parameter_projection.parameters():
            parameter.requires_grad_(False)
        # English documentation is provided in README.md.
        self.context_pool_method = str(cfg.alignment.get("context_pool", "attention"))
        if self.context_pool_method not in ("attention", "mean"):
            raise ValueError("alignment.context_pool must be 'attention' or 'mean'")
        if self.context_pool_method == "attention":
            self.context_pool = AttentionPool(embedding_dim, num_heads=int(cfg.token_context.get("num_heads", 8)))
        else:
            self.context_pool = None

    def forward(
        self,
        cond: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        structure: Mapping[str, torch.Tensor],
        raw_cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        token_context = self.token_context_generator(cond, structure, raw_cond)
        context_embedding = self.context_projection(token_context)
        parameter_embedding = self.parameter_projection(tokens, token_mask, structure)
        if bool(getattr(self, "detach_parameter_embedding", True)):
            parameter_embedding = parameter_embedding.detach()

        # token_mask may be (B, T) or (B, T, token_size); reduce to (B, T).
        if token_mask is not None and token_mask.any():
            if token_mask.ndim == 3:
                valid = token_mask.any(dim=-1)  # (B, T) bool
            else:
                valid = token_mask.bool()
        else:
            valid = torch.ones(context_embedding.shape[:2], dtype=torch.bool, device=context_embedding.device)
        pool_mask = valid.to(context_embedding.dtype)
        mask = pool_mask.unsqueeze(-1)  # (B, T, 1)

        # English documentation is provided in README.md.
        if self.context_pool is not None:
            context_global = self.context_pool(context_embedding, valid)
        else:
            context_global = (context_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        # English documentation is provided in README.md.
        target_global = (parameter_embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        return {
            "token_context": token_context,
            "context_embedding": context_embedding,
            "parameter_embedding": parameter_embedding,
            "context_global": context_global,        # (B, D)
            "target_global": target_global,          # (B, D)
            "valid": valid,                          # (B, T)
        }
