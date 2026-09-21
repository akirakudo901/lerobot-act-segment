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

"""Reuse act_segment processors so ``frame_label_int`` survives preprocessing."""

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from ..act_segment.processor_act_segment import (
    make_act_segment_pre_post_processors,
    prepend_act_segment_state_layout_step,
)
from .configuration_act_label import ACTLabelConfig


def prepend_act_label_state_layout_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    config: ACTLabelConfig,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """Prepend a state-layout reorder step when configured."""
    return prepend_act_segment_state_layout_step(preprocessor, config)


def make_act_label_pre_post_processors(
    config: ACTLabelConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Same ACT pre/post-processors as ``act_segment``, including label targets."""
    return make_act_segment_pre_post_processors(config, dataset_stats=dataset_stats)
