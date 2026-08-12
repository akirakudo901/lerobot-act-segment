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

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig
from ..segment.configuration_segment import SegmentPolicyConfigMixin


@PreTrainedConfig.register_subclass("act_segment")
@dataclass
class ACTSegmentConfig(ACTConfig, SegmentPolicyConfigMixin):
    """ACT with an auxiliary per-chunk-step MP/L BIO label head.

    Hybrid rollout + segment CE knobs come from :class:`SegmentPolicyConfigMixin`.
    Fields below are ACT-specific (action L1 reweighting, VAE retry sampling, processors).
    """

    # Scales the MP execution-frame L1 term: weighted_l1 = l_l1_loss + mp_l1_weight * mp_l1_loss.
    mp_l1_weight: float = 1.0
    # On requery refill, sample ACT latent from N(0, I) instead of the deterministic zero vector.
    ompl_retry_sample_latent: bool = True

    # Reorder ``observation.state`` in the policy preprocessor to match the training dataset layout.
    # Default ``None``: no reordering. Set explicitly when eval env layout differs from training:
    # ``lerobot`` for datasets with ee_pos + ee_ori + gripper (no reorder step),
    # ``efficient_libero`` for legacy efficient exports (gripper + ee_pos + ee_ori).
    observation_state_layout: str | None = None

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))
