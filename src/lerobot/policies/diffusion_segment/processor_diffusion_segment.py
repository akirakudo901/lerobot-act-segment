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

from typing import Any

import torch

from lerobot.processor import (
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.types import EnvTransition

from ..act_segment.processor_act_segment import (
    EfficientLiberoStateReorderStep,
    _batch_to_transition_with_label,
)
from ..diffusion.processor_diffusion import make_diffusion_pre_post_processors
from .configuration_diffusion_segment import DiffusionSegmentConfig

_LEROBOT_STATE_LAYOUT = "lerobot"
_EFFICIENT_LIBERO_STATE_LAYOUT = "efficient_libero"
_SUPPORTED_STATE_LAYOUTS = (None, _LEROBOT_STATE_LAYOUT, _EFFICIENT_LIBERO_STATE_LAYOUT)


def _state_layout_step(layout: str | None) -> ObservationProcessorStep | None:
    if layout in (None, _LEROBOT_STATE_LAYOUT):
        return None
    if layout == _EFFICIENT_LIBERO_STATE_LAYOUT:
        return EfficientLiberoStateReorderStep()
    supported = ", ".join(repr(value) for value in _SUPPORTED_STATE_LAYOUTS)
    raise ValueError(
        f"Unsupported observation_state_layout={layout!r} for diffusion_segment. "
        f"Supported: {supported}."
    )


def prepend_diffusion_segment_state_layout_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    config: DiffusionSegmentConfig,
) -> PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]:
    """Prepend a state-layout reorder step when configured."""
    step = _state_layout_step(config.observation_state_layout)
    if step is None:
        return preprocessor
    if any(isinstance(existing, EfficientLiberoStateReorderStep) for existing in preprocessor.steps):
        return preprocessor
    return PolicyProcessorPipeline(
        steps=[step, *preprocessor.steps],
        name=preprocessor.name,
        to_transition=preprocessor.to_transition,
        to_output=preprocessor.to_output,
    )


def make_diffusion_segment_pre_post_processors(
    config: DiffusionSegmentConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Reuse Diffusion pre/post-processors while keeping label supervision in the batch."""
    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config, dataset_stats=dataset_stats
    )
    label_feature_key = config.label_feature_key

    def to_transition(batch: dict[str, Any]) -> EnvTransition:
        return _batch_to_transition_with_label(batch, label_feature_key)

    preprocessor.to_transition = to_transition
    preprocessor = prepend_diffusion_segment_state_layout_step(preprocessor, config)
    return preprocessor, postprocessor
