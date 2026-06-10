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

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, batch_to_transition
from lerobot.types import EnvTransition, TransitionKey

from ..act.processor_act import make_act_pre_post_processors
from .configuration_act_segment import ACTSegmentConfig


def _batch_to_transition_with_label(batch: dict[str, Any], label_feature_key: str) -> EnvTransition:
    """Preserve chunked label targets through the ACT preprocessor pipeline."""
    transition = batch_to_transition(batch)
    if label_feature_key in batch:
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary_data[label_feature_key] = batch[label_feature_key]
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
    return transition


def make_act_segment_pre_post_processors(
    config: ACTSegmentConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse ACT pre/post-processors while keeping label supervision in the batch."""
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=dataset_stats)
    label_feature_key = config.label_feature_key

    def to_transition(batch: dict[str, Any]) -> EnvTransition:
        return _batch_to_transition_with_label(batch, label_feature_key)

    preprocessor.to_transition = to_transition
    return preprocessor, postprocessor
