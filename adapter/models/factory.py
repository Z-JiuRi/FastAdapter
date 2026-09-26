import torch
import torch.nn as nn
import torch.nn.functional as F

from .paper import *  # noqa: F401,F403
from .attention import *  # noqa: F401,F403
from .structured import *  # noqa: F401,F403
from .iterative import *  # noqa: F401,F403

def build_mapper(mapper_type, weight, bias, hidden_dim=1024, num_blocks=4,
                 dropout=0.0, residual_scale=0.1, use_block_norm=True,
                 use_final_norm=False, train_affine=False,
                 learnable_residual_gate=False, gate_max=0.5,
                 lowrank_rank=64, gate_mode="block",
                 final_gate_max=1.0, final_gate_init=1.0,
                 adaptive_gate_hidden=128, bottleneck_dim=128,
                 num_groups=16, group_hidden=64, gate_hidden=64,
                 gate_init=0.5, num_tokens=16, token_hidden=64,
                 channel_hidden=64, num_heads=2,
                 transformer_ffn_dim=128, num_experts=4,
                 attention_dim=32, attention_heads=4,
                 attention_dropout=0.0, attention_scale=0.1,
                 attention_input="value_delta",
                 attention_use_position=True,
                 flow_hidden_dim=128, whole_mlp_dims=None,
                 whole_mlp_activation="gelu", use_affine_alignment=True):
    mapper_type = mapper_type.lower()
    if mapper_type == "affine_residual_mlp":
        return AffineResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden)
    if mapper_type == "affine_residual_mlp_attention":
        return AffineResidualMLPAttentionMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max, gate_mode=gate_mode,
            final_gate_max=final_gate_max,
            final_gate_init=final_gate_init,
            adaptive_gate_hidden=adaptive_gate_hidden,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            attention_scale=attention_scale,
            attention_input=attention_input,
            attention_use_position=attention_use_position)
    if mapper_type == "affine_iterative_residual":
        # num_blocks is reused as the number of refine iterations.
        return AffineIterativeResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_iters=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, share_weights=True,
            use_step_embed=True)
    if mapper_type == "affine_iterative_residual_unshared":
        return AffineIterativeResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_iters=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, share_weights=False,
            use_step_embed=True)
    if mapper_type == "affine_sens_weighted_residual":
        return AffineSensWeightedResidualMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_film_residual_mlp":
        return AffineFiLMResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_multiscale_residual_mlp":
        return AffineMultiScaleResidualMLPMapper(
            weight, bias, hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_lowrank_residual":
        return AffineLowRankResidualMapper(
            weight, bias, rank=lowrank_rank, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine,
            learnable_residual_gate=learnable_residual_gate,
            gate_max=gate_max)
    if mapper_type == "affine_bottleneck_residual":
        return AffineBottleneckResidualMapper(
            weight, bias, bottleneck_dim=bottleneck_dim,
            num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_group_gated":
        return AffineGroupGatedMapper(
            weight, bias, num_groups=num_groups, group_hidden=group_hidden,
            gate_hidden=gate_hidden, num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, gate_init=gate_init,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine)
    if mapper_type == "affine_token_mixer":
        return AffineTokenMixerMapper(
            weight, bias, num_tokens=num_tokens, token_hidden=token_hidden,
            channel_hidden=channel_hidden, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_tiny_transformer":
        return AffineTinyTransformerMapper(
            weight, bias, num_tokens=num_tokens, num_heads=num_heads,
            transformer_ffn_dim=transformer_ffn_dim, num_blocks=num_blocks,
            dropout=dropout, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_moe_bottleneck":
        return AffineMoEBottleneckMapper(
            weight, bias, bottleneck_dim=bottleneck_dim,
            num_experts=num_experts, gate_hidden=gate_hidden,
            num_blocks=num_blocks, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_coupling_flow":
        return AffineCouplingFlowMapper(
            weight, bias, flow_hidden_dim=flow_hidden_dim,
            num_blocks=num_blocks, residual_scale=residual_scale,
            use_final_norm=use_final_norm, train_affine=train_affine)
    if mapper_type == "affine_whole_residual_mlp":
        return AffineWholeResidualMLPMapper(
            weight, bias, hidden_dims=whole_mlp_dims, dropout=dropout,
            residual_scale=residual_scale, use_block_norm=use_block_norm,
            use_final_norm=use_final_norm, train_affine=train_affine,
            activation=whole_mlp_activation)
    if mapper_type == "affine_whole_direct_mlp":
        return AffineWholeDirectMLPMapper(
            weight, bias, hidden_dims=whole_mlp_dims, dropout=dropout,
            use_block_norm=use_block_norm, use_final_norm=use_final_norm,
            train_affine=train_affine, activation=whole_mlp_activation)
    if mapper_type == "legacy_mlp_adapter":
        return LegacyMLPAdapterMapper(
            weight, bias, hidden_dim=hidden_dim, dropout=dropout,
            residual_scale=residual_scale,
            use_affine_alignment=use_affine_alignment,
            train_affine=train_affine)
    if mapper_type == "affine_linear":
        return AffineLinearStackMapper(
            weight, bias, num_layers=num_blocks, train_affine=train_affine)
    if mapper_type == "direct_mlp":
        return DirectMLPMapper(
            weight.size(0), hidden_dim=hidden_dim, num_layers=num_blocks,
            dropout=dropout)
    raise ValueError(f"Unknown mapper_type: {mapper_type}")

