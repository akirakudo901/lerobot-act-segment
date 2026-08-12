# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Shared segment-label train helpers (MP/L-separated CE + pad masks)."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def label_targets(batch: dict[str, Tensor], label_feature_key: str) -> Tensor:
    labels = batch[label_feature_key]
    if labels.ndim == 3:
        labels = labels.squeeze(-1)
    return labels.long()


def label_valid_mask(batch: dict[str, Tensor], label_feature_key: str) -> Tensor:
    pad_key = f"{label_feature_key}_is_pad"
    if pad_key in batch:
        return ~batch[pad_key]
    return ~batch["action_is_pad"]


def masked_ce_mean(per_step_ce: Tensor, mask: Tensor) -> Tensor:
    """Mean CE over masked steps; empty mask contributes 0."""
    return (per_step_ce * mask).sum() / mask.sum().clamp_min(1)


def mp_l_label_masks(labels: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    """MP/L masks over valid label steps (ground-truth frame type)."""
    from dataset.core.frame_labels import FrameLabelEnum

    mp_mask = FrameLabelEnum.is_mp_frame_label_array(labels)
    l_mask = FrameLabelEnum.is_l_frame_label_array(labels)
    return valid_mask & mp_mask, valid_mask & l_mask


def mp_l_action_masks(
    batch: dict[str, Tensor],
    action_valid_mask: Tensor,
    *,
    label_feature_key: str,
) -> tuple[Tensor, Tensor]:
    """MP mask = MP-labeled valid steps; L mask = L-labeled valid steps.

    ``action_valid_mask`` may be ``[B, T]`` or ``[B, T, 1]``; returned masks match
    the broadcast shape of a per-dim action loss (unsqueezed to ``[B, T, 1]`` when
    the input had a trailing singleton dim).
    """
    label_mask = label_valid_mask(batch, label_feature_key)
    squeeze_trailing = action_valid_mask.ndim == 3 and action_valid_mask.shape[-1] == 1
    step_action = action_valid_mask.squeeze(-1) if squeeze_trailing else action_valid_mask
    step_valid = step_action & label_mask
    labels = label_targets(batch, label_feature_key)
    mp_step, l_step = mp_l_label_masks(labels, step_valid)
    if squeeze_trailing:
        return mp_step.unsqueeze(-1), l_step.unsqueeze(-1)
    return mp_step, l_step


def masked_action_loss_mean(per_elem_loss: Tensor, mask: Tensor) -> Tensor:
    """Mean of a per-element action loss over a boolean mask (empty → 0).

    ``per_elem_loss`` and ``mask`` share leading dims; the last dim of the loss is
    treated as the action feature axis (e.g. L1 abs error ``[B, T, act_dim]``).
    """
    num_valid = mask.sum() * per_elem_loss.shape[-1]
    return (per_elem_loss * mask).sum() / num_valid.clamp_min(1)


def segment_label_ce(
    logits: Tensor,
    labels: Tensor,
    valid_mask: Tensor,
    *,
    mp_ce_weight: float = 1.0,
    l_ce_weight: float = 1.0,
    num_label_classes: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """MP/L-separated label cross-entropy over a predicted chunk.

    Computes per-step CE, takes separate means over GT MP vs L steps (empty mask → 0),
    then ``weighted = mp_ce_weight * mp_ce + l_ce_weight * l_ce``.

    Args:
        logits: ``[B, T, C]`` label logits.
        labels: ``[B, T]`` integer BIO labels (``frame_label_int``).
        valid_mask: ``[B, T]`` bool pad mask (True = valid).
        mp_ce_weight / l_ce_weight: per-type multipliers.
        num_label_classes: optional check against ``logits.shape[-1]``.

    Returns:
        ``(weighted_label_ce, loss_dict)`` where ``loss_dict`` has
        ``mp_ce_loss``, ``l_ce_loss``, ``weighted_label_ce_loss``, and
        ``label_accuracy`` (over all valid steps).
    """
    if num_label_classes is not None and int(logits.shape[-1]) != int(num_label_classes):
        raise ValueError(
            f"logits last dim {logits.shape[-1]} != num_label_classes={num_label_classes}"
        )

    per_step_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view(logits.shape[0], logits.shape[1])

    mp_mask, l_mask = mp_l_label_masks(labels, valid_mask)
    mp_ce_loss = masked_ce_mean(per_step_ce, mp_mask)
    l_ce_loss = masked_ce_mean(per_step_ce, l_mask)
    weighted_label_ce_loss = mp_ce_weight * mp_ce_loss + l_ce_weight * l_ce_loss

    num_valid_labels = valid_mask.sum().clamp_min(1)
    preds = logits.argmax(dim=-1)
    label_accuracy = ((preds == labels) & valid_mask).sum().float() / num_valid_labels

    loss_dict = {
        "mp_ce_loss": mp_ce_loss.item(),
        "l_ce_loss": l_ce_loss.item(),
        "weighted_label_ce_loss": weighted_label_ce_loss.item(),
        "label_accuracy": label_accuracy.item(),
    }
    return weighted_label_ce_loss, loss_dict
