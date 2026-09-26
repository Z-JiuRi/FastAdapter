from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AlignmentOutput:
    loss: torch.Tensor
    contrastive: torch.Tensor
    cosine: torch.Tensor
    top1: torch.Tensor
    top5: torch.Tensor
    positive_logit: torch.Tensor
    negative_logit: torch.Tensor
    tokens: int


def task_level_alignment_loss(
    context: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    cosine_weight: float,
) -> AlignmentOutput:
    """Task/adapter-level bidirectional InfoNCE.

    Pools each task to a single vector and contrasts across the batch.
    One B-way retrieval problem instead of T independent problems.

    Args:
        context:  (B, D)  global-pooled token_context per task.
        target:   (B, D)  global adapter representation per task.
        temperature:  InfoNCE temperature.
        cosine_weight:  weight for the (1 - cosine) auxiliary term.

    Returns:
        AlignmentOutput with token count = B (one "token" per task).
    """
    if context.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"task_level_alignment_loss expects (B, D) tensors, "
            f"got context {tuple(context.shape)} target {tuple(target.shape)}"
        )
    if context.shape != target.shape:
        raise ValueError(
            f"context/target shape mismatch: {tuple(context.shape)} vs {tuple(target.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    B = context.shape[0]
    if B < 2:
        raise ValueError("task-level InfoNCE requires at least two encoder tasks per batch")

    context_norm = F.normalize(context.float(), dim=-1, eps=1e-6)
    target_norm = F.normalize(target.float(), dim=-1, eps=1e-6)

    # (B, B) logits: context[i] retrieves target[j]
    logits = (context_norm @ target_norm.T) / temperature
    reverse_logits = (target_norm @ context_norm.T) / temperature
    labels = torch.arange(B, device=context.device)

    contrastive = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(reverse_logits, labels)
    )
    cosine = (context_norm * target_norm).sum(dim=-1).mean()
    loss = contrastive + float(cosine_weight) * (1.0 - cosine)

    with torch.no_grad():
        top1 = 0.5 * (
            (logits.argmax(dim=-1) == labels).float().mean()
            + (reverse_logits.argmax(dim=-1) == labels).float().mean()
        )
        top_k = min(5, B)
        top5 = 0.5 * (
            logits.topk(top_k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean()
            + reverse_logits.topk(top_k, dim=-1).indices.eq(labels.unsqueeze(-1)).any(dim=-1).float().mean()
        )
        positive = logits.diag().mean()
        negative_mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
        negative = logits[negative_mask].mean() if B > 1 else torch.zeros((), device=logits.device, dtype=logits.dtype)

    return AlignmentOutput(
        loss=loss,
        contrastive=contrastive.detach(),
        cosine=cosine.detach(),
        top1=top1,
        top5=top5,
        positive_logit=positive,
        negative_logit=negative,
        tokens=B,
    )


def position_residual_cosine_loss(
    token_context: torch.Tensor,
    target_embedding: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """See README.md for English documentation."""
    if token_context.shape != target_embedding.shape or token_context.ndim != 3:
        raise ValueError(
            f"expected matching (B, T, D) tensors, got {tuple(token_context.shape)} "
            f"and {tuple(target_embedding.shape)}"
        )
    valid_float = valid.to(token_context.dtype)                        # (B, T)
    mask = valid_float.unsqueeze(-1)                                   # (B, T, 1)
    # English documentation is provided in README.md.
    position_count = mask.sum(dim=0).clamp_min(1.0)                    # (T, 1)
    position_mean = (target_embedding * mask).sum(dim=0) / position_count   # (T, D)
    resid = (target_embedding - position_mean.unsqueeze(0)).detach()  # (B, T, D)
    cos = F.cosine_similarity(token_context.float(), resid.float(), dim=-1)  # (B, T)
    weighted = (cos * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    return 1.0 - weighted
