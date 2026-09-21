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

from hybrid_eval.segment.configuration_segment import SegmentPolicyConfigMixin

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_label")
@dataclass
class ACTLabelConfig(ACTConfig, SegmentPolicyConfigMixin):
    """Small ACT that classifies per-chunk BIO labels (CE only, no VAE).

    Hybrid rollout knobs come along with :class:`SegmentPolicyConfigMixin` so CE
    weights and ``frame_label_int`` chunking match ``act_segment``. This policy
    does not run the hybrid orchestrator.
    """

    dim_model: int = 256
    n_heads: int = 4
    dim_feedforward: int = 1024
    n_encoder_layers: int = 2
    n_decoder_layers: int = 1
    use_vae: bool = False

    # Reorder ``observation.state`` in the policy preprocessor to match the training dataset layout.
    # Default ``None``: no reordering. Same values as ``act_segment``.
    observation_state_layout: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.use_vae:
            raise ValueError("act_label trains label CE only and does not support use_vae=True")

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))
