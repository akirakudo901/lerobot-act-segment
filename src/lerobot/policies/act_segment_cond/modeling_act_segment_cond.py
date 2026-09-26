#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""ACT action body conditioned on BIO label encoder tokens (no label CE head)."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor

from ..act.configuration_act import ACTConfig
from ..act.hybrid_act import HybridACTPolicy
from ..act.modeling_act import ACT
from hybrid_eval.segment.losses import label_targets
from .configuration_act_segment_cond import ACTSegmentCondConfig


class _LabelChunkPredictor(Protocol):
    def predict_label_chunk(self, batch: dict[str, Tensor]) -> Tensor: ...


class ACTSegmentCond(ACT):
    """Vanilla ACT plus BIO(+pad) tokens on the transformer encoder (not the VAE)."""

    def __init__(self, config: ACTSegmentCondConfig):
        super().__init__(config)
        self._init_label_encoder_tokens()


class ACTSegmentCondPolicy(HybridACTPolicy):
    """Action ACT trained with GT label tokens; eval labels from GT or frozen ``act_label``."""

    config_class = ACTSegmentCondConfig
    name = "act_segment_cond"

    def __init__(self, config: ACTSegmentCondConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._frozen_label_policy: _LabelChunkPredictor | None = None

    def _build_model(self, config: ACTConfig) -> ACTSegmentCond:
        return ACTSegmentCond(config)  # type: ignore[arg-type]

    def to(self, *args, **kwargs):
        out = super().to(*args, **kwargs)
        frozen = self._frozen_label_policy
        if isinstance(frozen, torch.nn.Module):
            frozen.to(*args, **kwargs)
        return out

    def _set_frozen_label_policy(self, policy: _LabelChunkPredictor) -> None:
        """Attach a frozen label model without registering it as an nn.Module child."""
        if isinstance(policy, torch.nn.Module):
            policy.eval()
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
        object.__setattr__(self, "_frozen_label_policy", policy)

    def _get_frozen_label_policy(self) -> _LabelChunkPredictor:
        if self._frozen_label_policy is not None:
            return self._frozen_label_policy
        path = self.config.label_policy_path
        if not path:
            raise ValueError(
                "act_segment_cond label_source='predicted' requires label_policy_path "
                "to a frozen act_label checkpoint"
            )
        from ..act_label.modeling_act_label import ACTLabelPolicy

        loaded = ACTLabelPolicy.from_pretrained(path)
        device = next(self.parameters()).device
        loaded.to(device)
        self._set_frozen_label_policy(loaded)
        return loaded

    def _inference_labels(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.label_source == "gt":
            key = self.config.label_feature_key
            if key not in batch:
                raise KeyError(
                    "label_source='gt' requires "
                    f"{key!r} in the batch (got keys {sorted(batch)})"
                )
            return label_targets(batch, key)
        return self._get_frozen_label_policy().predict_label_chunk(batch)

    def _batch_with_inference_labels(
        self, batch: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], Tensor]:
        batch = dict(batch)
        labels = self._inference_labels(batch)
        batch[self.config.label_feature_key] = labels
        if self.config.label_source == "predicted":
            batch.pop(f"{self.config.label_feature_key}_is_pad", None)
        return batch, labels

    @torch.no_grad()
    def predict_label_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the BIO ids used to condition the action body (GT or frozen act_label)."""
        self.eval()
        batch = self._prepare_batch(batch)
        return self._inference_labels(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        actions, _labels = self.predict_action_label_chunk(batch)
        return actions

    @torch.no_grad()
    def predict_action_label_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        sample_latent_prior: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(actions, labels)`` for hybrid rollout.

        Labels come from frozen ``act_label`` when ``label_source='predicted'``, else
        from the batch. The action trunk is always conditioned on those ids.
        """
        del kwargs
        self.eval()
        batch = self._prepare_batch(batch)
        batch, labels = self._batch_with_inference_labels(batch)
        actions, _vae_params = self.model(
            batch, sample_latent_prior=sample_latent_prior
        )
        return actions, labels

    @torch.no_grad()
    def per_step_val_losses(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """GT-conditioned per-step action L1. Label CE is zeros (no label head)."""
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, _vae_params = self.model(batch, sample_encoded_dist)
        action_l1, valid_mask = self._per_step_action_l1(batch, actions_hat)
        return action_l1, torch.zeros_like(action_l1), valid_mask

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, dict]:
        """Teacher-force GT labels into the encoder; MP/L-weighted action L1 (+ optional VAE KLD)."""
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, vae_params = self.model(batch, sample_encoded_dist)
        weighted_l1_loss, loss_dict = self._weighted_action_l1(batch, actions_hat)
        return self._maybe_add_kld(weighted_l1_loss, loss_dict, vae_params)
