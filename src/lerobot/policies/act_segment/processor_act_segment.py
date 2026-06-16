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
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    batch_to_transition,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from ..act.processor_act import make_act_pre_post_processors
from .configuration_act_segment import ACTSegmentConfig

_EFFICIENT_LIBERO_STATE_LAYOUT = "efficient_libero"
# Default LIBERO flat state: ee_pos(3), ee_axis_angle(3), gripper(2).
# Efficient export / training order: gripper(2), ee_pos(3), ee_ori(3).
_LIBERO_TO_EFFICIENT_STATE_INDICES = (6, 7, 0, 1, 2, 3, 4, 5)
_EFFICIENT_LIBERO_STATE_DIM = 8


@dataclass
@ProcessorStepRegistry.register(name="efficient_libero_state_reorder")
class EfficientLiberoStateReorderStep(ObservationProcessorStep):
    """Reorder ``observation.state`` from default LIBERO layout to efficient-export layout."""

    def observation(self, observation):
        processed_obs = observation.copy()
        if OBS_STATE not in processed_obs:
            return processed_obs

        state = processed_obs[OBS_STATE]
        if not isinstance(state, torch.Tensor):
            raise TypeError(
                f"{OBS_STATE!r} must be a torch.Tensor for efficient_libero reorder, got {type(state)!r}"
            )
        if state.shape[-1] != _EFFICIENT_LIBERO_STATE_DIM:
            raise ValueError(
                f"efficient_libero state reorder expects {OBS_STATE!r} with last dim "
                f"{_EFFICIENT_LIBERO_STATE_DIM} (LIBERO flat state: ee_pos + axis-angle + gripper), "
                f"got shape {tuple(state.shape)}"
            )

        indices = torch.tensor(_LIBERO_TO_EFFICIENT_STATE_INDICES, device=state.device)
        processed_obs[OBS_STATE] = state.index_select(-1, indices)
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Permutation only; ``observation.state`` shape and keys are unchanged."""
        return features


def _state_layout_step(layout: str | None) -> ObservationProcessorStep | None:
    if layout is None:
        return None
    if layout == _EFFICIENT_LIBERO_STATE_LAYOUT:
        return EfficientLiberoStateReorderStep()
    raise ValueError(
        f"Unsupported observation_state_layout={layout!r} for act_segment. "
        f"Supported: {None!r}, {_EFFICIENT_LIBERO_STATE_LAYOUT!r}."
    )


def prepend_act_segment_state_layout_step(
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    config: ACTSegmentConfig,
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
    preprocessor = prepend_act_segment_state_layout_step(preprocessor, config)
    return preprocessor, postprocessor
