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
from typing import Literal

from hybrid_eval.segment.configuration_segment import SegmentPolicyConfigMixin

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_segment_cond")
@dataclass
class ACTSegmentCondConfig(ACTConfig, SegmentPolicyConfigMixin):
    """Full-size ACT action body conditioned on chunked BIO labels.

    Train teacher-forces GT ``frame_label_int`` into extra encoder tokens (no label CE).
    Eval ``label_source='predicted'`` runs a frozen ``act_label`` checkpoint first;
    ``label_source='gt'`` reads labels from the batch (offline val / GT replay).
    Hybrid rollout fields come from :class:`SegmentPolicyConfigMixin`.
    """

    # Scales the MP execution-frame L1 term: weighted_l1 = l_l1_loss + mp_l1_weight * mp_l1_loss.
    mp_l1_weight: float = 1.0
    # On requery refill, sample ACT latent from N(0, I) instead of the deterministic zero vector.
    ompl_retry_sample_latent: bool = True

    label_source: Literal["predicted", "gt"] = "predicted"
    label_policy_path: str | None = None

    # Reorder ``observation.state`` in the policy preprocessor to match the training dataset layout.
    # Default ``None``: no reordering. Same values as ``act_segment``.
    observation_state_layout: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.label_source not in ("predicted", "gt"):
            raise ValueError(
                f"label_source must be 'predicted' or 'gt', got {self.label_source!r}"
            )

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))
