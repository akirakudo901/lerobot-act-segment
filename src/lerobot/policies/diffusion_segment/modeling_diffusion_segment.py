#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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

"""Diffusion Policy with a BIO label head on the flattened observation conditioning vector."""

from __future__ import annotations

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from hybrid_eval.segment.losses import (
    label_targets,
    label_valid_mask,
    masked_action_loss_mean,
    mp_l_action_masks,
    segment_label_ce,
)
from lerobot.utils.constants import ACTION

from ..diffusion.modeling_diffusion import DiffusionModel
from .configuration_diffusion_segment import DiffusionSegmentConfig


class DiffusionSegmentModel(DiffusionModel):
    """UNet denoiser plus an MLP label head on ``global_cond`` (independent of the denoise loop)."""

    def __init__(self, config: DiffusionSegmentConfig):
        super().__init__(config)
        cond_dim = self.global_cond_dim
        self.label_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, config.horizon * config.num_label_classes),
        )

    def _label_logits_from_global_cond(self, global_cond: Tensor) -> Tensor:
        """Map flattened observation conditioning to per-horizon label logits ``(B, horizon, C)``."""
        batch_size = global_cond.shape[0]
        return self.label_head(global_cond).reshape(
            batch_size, self.config.horizon, self.config.num_label_classes
        )

    def generate_actions_and_labels(
        self, batch: dict[str, Tensor], noise: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Denoise a full-horizon action trajectory and classify each step from the same ``global_cond``.

        Returns unsliced ``(actions, label_logits)`` with time axis ``horizon``. The policy slices both
        to ``n_action_steps`` at rollout (``start = n_obs_steps - 1``).
        """
        actions, global_cond = self._generate_horizon_from_batch(batch, noise=noise)
        label_logits = self._label_logits_from_global_cond(global_cond)
        return actions, label_logits

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Diffusion MSE (optionally MP/L-split) plus pad-masked segment-label CE over the full horizon."""
        pred, target, global_cond = self._denoise_from_batch(batch)
        
        per_elem_mse = F.mse_loss(pred, target, reduction="none")
        action_valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        mp_action_mask, l_action_mask = mp_l_action_masks(
            batch,
            action_valid_mask,
            label_feature_key=self.config.label_feature_key,
        )
        mp_mse_loss = masked_action_loss_mean(per_elem_mse, mp_action_mask)
        l_mse_loss = masked_action_loss_mean(per_elem_mse, l_action_mask)
        weighted_mse_loss = l_mse_loss + self.config.mp_mse_weight * mp_mse_loss

        label_logits = self._label_logits_from_global_cond(global_cond)
        targets = label_targets(batch, self.config.label_feature_key)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        weighted_label_ce_loss, label_loss_dict = segment_label_ce(
            label_logits,
            targets,
            valid_labels,
            mp_ce_weight=self.config.mp_ce_weight,
            l_ce_weight=self.config.l_ce_weight,
            num_label_classes=self.config.num_label_classes,
        )

        loss = weighted_mse_loss + weighted_label_ce_loss
        loss_dict = {
            "mp_mse_loss": mp_mse_loss.item(),
            "l_mse_loss": l_mse_loss.item(),
            "weighted_mse_loss": weighted_mse_loss.item(),
            **label_loss_dict,
        }
        return loss, loss_dict
