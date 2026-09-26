# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""BIO(+pad) label tokens for the ACT transformer encoder (action body only).

Padded steps use an extra embedding index ``num_label_classes`` so they do not
share a real BIO class. These tokens are appended after latent + robot/env
state tokens and are never fed to the VAE encoder.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def encoder_label_ids(
    batch: dict[str, Tensor],
    *,
    label_feature_key: str,
    num_label_classes: int,
    chunk_size: int | None = None,
) -> Tensor:
    """Return ``[B, T]`` label ids with padded steps replaced by the pad index.

    The pad index is ``num_label_classes`` (one past the last BIO class), matching
    ``nn.Embedding(num_label_classes + 1, dim_model)``.
    """
    if label_feature_key not in batch:
        raise KeyError(
            f"Label encoder tokens require {label_feature_key!r} in the batch "
            f"(got keys {sorted(batch)})"
        )
    ids = batch[label_feature_key]
    if ids.ndim == 3:
        ids = ids.squeeze(-1)
    ids = ids.long()

    pad_key = f"{label_feature_key}_is_pad"
    if pad_key in batch:
        valid = ~batch[pad_key]
    elif "action_is_pad" in batch:
        valid = ~batch["action_is_pad"]
    else:
        valid = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)

    if valid.shape != ids.shape:
        raise ValueError(
            f"Label pad mask shape {tuple(valid.shape)} does not match "
            f"label ids shape {tuple(ids.shape)}"
        )
    if chunk_size is not None and ids.shape[-1] != int(chunk_size):
        raise ValueError(
            f"Label encoder tokens expect T={chunk_size}, got {ids.shape[-1]}"
        )

    pad_id = int(num_label_classes)
    return torch.where(valid, ids, torch.full_like(ids, pad_id))


def append_label_encoder_tokens(
    encoder_in_tokens: list[Tensor],
    encoder_in_pos_embed: list[Tensor],
    label_ids: Tensor,
    label_embed: nn.Embedding,
    label_pos_embed: nn.Embedding,
) -> None:
    """Append ``chunk_size`` label tokens in-place after the current 1-D tokens.

    ``label_ids`` is ``[B, T]``. Each step becomes a ``[B, dim]`` token;
    positional ids are ``[1, dim]`` like other ACT 1-D encoder tokens.
    """
    tokens = label_embed(label_ids)
    chunk_size = int(label_pos_embed.num_embeddings)
    if tokens.shape[1] != chunk_size:
        raise ValueError(
            f"Label token count {tokens.shape[1]} != label_pos_embed size {chunk_size}"
        )
    for step in range(chunk_size):
        encoder_in_tokens.append(tokens[:, step])
    encoder_in_pos_embed.extend(list(label_pos_embed.weight.unsqueeze(1)))
