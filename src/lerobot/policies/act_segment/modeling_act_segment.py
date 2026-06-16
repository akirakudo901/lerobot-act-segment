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

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES

from ..act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from .configuration_act_segment import ACTSegmentConfig


class ACTSegment(ACT):
    """ACT decoder extended with a linear label head on decoder tokens."""

    def __init__(self, config: ACTSegmentConfig):
        super().__init__(config)
        self.label_head = nn.Linear(config.dim_model, config.num_label_classes)

    def forward(
        self, batch: dict[str, Tensor], sample_encoded_dist: bool = False
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        actions, vae_params, decoder_out = self._forward_from_batch(batch, sample_encoded_dist)
        labels_logits = self.label_head(decoder_out)
        return actions, labels_logits, vae_params


class ACTSegmentPolicy(ACTPolicy):
    """ACT policy with auxiliary BIO segment-label cross-entropy loss."""

    config_class = ACTSegmentConfig
    name = "act_segment"

    def __init__(self, config: ACTSegmentConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTSegment(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    def _label_targets(self, batch: dict[str, Tensor]) -> Tensor:
        labels = batch[self.config.label_feature_key]
        if labels.ndim == 3:
            labels = labels.squeeze(-1)
        return labels.long()

    def _label_valid_mask(self, batch: dict[str, Tensor]) -> Tensor:
        pad_key = f"{self.config.label_feature_key}_is_pad"
        if pad_key in batch:
            return ~batch[pad_key]
        return ~batch["action_is_pad"]

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

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, labels_logits, (mu_hat, log_sigma_x2_hat) = self.model(batch, sample_encoded_dist)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        action_valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid_actions = action_valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * action_valid_mask).sum() / num_valid_actions.clamp_min(1)

        label_targets = self._label_targets(batch)
        label_valid_mask = self._label_valid_mask(batch)
        per_step_ce = F.cross_entropy(
            labels_logits.reshape(-1, self.config.num_label_classes),
            label_targets.reshape(-1),
            reduction="none",
        ).view(labels_logits.shape[0], labels_logits.shape[1])
        num_valid_labels = label_valid_mask.sum().clamp_min(1)
        label_ce_loss = (per_step_ce * label_valid_mask).sum() / num_valid_labels

        preds = labels_logits.argmax(dim=-1)
        label_accuracy = ((preds == label_targets) & label_valid_mask).sum().float() / num_valid_labels

        loss_dict = {
            "l1_loss": l1_loss.item(),
            "label_ce_loss": label_ce_loss.item(),
            "label_accuracy": label_accuracy.item(),
        }

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight + label_ce_loss * self.config.label_weight
        else:
            loss = l1_loss + label_ce_loss * self.config.label_weight

        return loss, loss_dict
