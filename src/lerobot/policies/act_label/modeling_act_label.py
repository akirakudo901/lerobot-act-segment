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

"""Small ACT with a per-chunk-step BIO label head and no action / VAE loss."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import OBS_IMAGES

from ..act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from hybrid_eval.segment.losses import (
    label_targets,
    label_valid_mask,
    segment_label_ce,
)
from .configuration_act_label import ACTLabelConfig


class ACTLabel(ACT):
    """ACT decoder with a linear label head; action outputs are unused in the train loss."""

    def __init__(self, config: ACTLabelConfig):
        super().__init__(config)
        self.label_head = nn.Linear(config.dim_model, config.num_label_classes)

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool = False,
        sample_latent_prior: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        _actions, vae_params, decoder_out = self._forward_from_batch(
            batch, sample_encoded_dist, sample_latent_prior=sample_latent_prior
        )
        labels_logits = self.label_head(decoder_out)
        return _actions, labels_logits, vae_params


class ACTLabelPolicy(ACTPolicy):
    """ACT policy trained with BIO label cross-entropy only."""

    config_class = ACTLabelConfig
    name = "act_label"

    def __init__(self, config: ACTLabelConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTLabel(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    @torch.no_grad()
    def predict_label_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return argmax segment labels for each step in the predicted chunk."""
        self.eval()
        batch = self._prepare_batch(batch)
        _actions, labels_logits, _vae_params = self.model(batch)
        return labels_logits.argmax(dim=-1)

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Label-only: the action trunk is not a supported inference output."""
        raise NotImplementedError(
            "act_label does not support action outputs; use predict_label_chunk."
        )

    @torch.no_grad()
    def per_step_val_losses(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-chunk-step label CE for offline validation.

        ``action_l1`` is zeros: this policy has no action loss. ``valid_mask`` is
        the label pad mask so span aggregation can still group CE by MP/L type.
        """
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        _actions_hat, labels_logits, _vae_params = self.model(batch, sample_encoded_dist)

        targets = label_targets(batch, self.config.label_feature_key)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        per_step_ce = F.cross_entropy(
            labels_logits.reshape(-1, self.config.num_label_classes),
            targets.reshape(-1),
            reduction="none",
        ).view(labels_logits.shape[0], labels_logits.shape[1])

        action_l1 = torch.zeros_like(per_step_ce)
        return action_l1, per_step_ce, valid_labels

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        _actions_hat, labels_logits, _vae_params = self.model(batch, sample_encoded_dist)

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
        return weighted_label_ce_loss, label_loss_dict
