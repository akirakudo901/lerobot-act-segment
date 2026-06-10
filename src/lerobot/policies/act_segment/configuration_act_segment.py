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


@PreTrainedConfig.register_subclass("act_segment")
@dataclass
class ACTSegmentConfig(ACTConfig):
    """ACT with an auxiliary per-chunk-step MP/L BIO label head.

    Expects a dataset feature ``frame_label_int`` (0=B-MP, 1=I-MP, 2=B-L, 3=I-L)
    chunked over the same horizon as ``action``.
    """

    label_feature_key: str = "frame_label_int"
    label_weight: float = 1.0
    num_label_classes: int = 4

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))
