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

"""ACT with a per-chunk-step segment label classification head."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from ..act.configuration_act import ACTConfig
from ..act.hybrid_act import HybridACTPolicy
from ..act.modeling_act import ACT
from hybrid_eval.segment.losses import (
    label_targets,
    label_valid_mask,
    segment_label_ce,
)
from .configuration_act_segment import ACTSegmentConfig


class ACTSegment(ACT):
    """ACT decoder extended with a linear label head on decoder tokens."""

    def __init__(self, config: ACTSegmentConfig):
        super().__init__(config)
        self.label_head = nn.Linear(config.dim_model, config.num_label_classes)

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool = False,
        sample_latent_prior: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        actions, vae_params, decoder_out = self._forward_from_batch(
            batch, sample_encoded_dist, sample_latent_prior=sample_latent_prior
        )
        labels_logits = self.label_head(decoder_out)
        return actions, labels_logits, vae_params


class ACTSegmentPolicy(HybridACTPolicy):
    """ACT policy with auxiliary BIO segment-label cross-entropy loss."""

    config_class = ACTSegmentConfig
    name = "act_segment"

    def _build_model(self, config: ACTConfig) -> ACTSegment:
        return ACTSegment(config)  # type: ignore[arg-type]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        actions, _labels_logits, _vae_params = self.model(batch)
        return actions

    @torch.no_grad()
    def predict_label_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return argmax segment labels for each step in the predicted chunk."""
        self.eval()
        batch = self._prepare_batch(batch)
        _actions, labels_logits, _vae_params = self.model(batch)
        return labels_logits.argmax(dim=-1)

    @torch.no_grad()
    def predict_action_label_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        sample_latent_prior: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Return both the actions and argmax segment labels for each step in the predicted chunk."""
        del kwargs
        self.eval()
        batch = self._prepare_batch(batch)
        actions, labels_logits, _vae_params = self.model(
            batch, sample_latent_prior=sample_latent_prior
        )
        return actions, labels_logits.argmax(dim=-1)

    @torch.no_grad()
    def per_step_val_losses(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-chunk-step action L1 and label CE for offline segment validation."""
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, labels_logits, _vae_params = self.model(batch, sample_encoded_dist)

        action_l1, valid_mask = self._per_step_action_l1(batch, actions_hat)
        targets = label_targets(batch, self.config.label_feature_key)
        per_step_ce = F.cross_entropy(
            labels_logits.reshape(-1, self.config.num_label_classes),
            targets.reshape(-1),
            reduction="none",
        ).view(labels_logits.shape[0], labels_logits.shape[1])
        return action_l1, per_step_ce, valid_mask

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, labels_logits, vae_params = self.model(batch, sample_encoded_dist)

        weighted_l1_loss, loss_dict = self._weighted_action_l1(batch, actions_hat)
        targets = label_targets(batch, self.config.label_feature_key)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        weighted_label_ce_loss, label_loss_dict = segment_label_ce(
            labels_logits,
            targets,
            valid_labels,
            mp_ce_weight=self.config.mp_ce_weight,
            l_ce_weight=self.config.l_ce_weight,
            num_label_classes=self.config.num_label_classes,
        )
        loss_dict.update(label_loss_dict)
        return self._maybe_add_kld(
            weighted_l1_loss + weighted_label_ce_loss, loss_dict, vae_params
        )
