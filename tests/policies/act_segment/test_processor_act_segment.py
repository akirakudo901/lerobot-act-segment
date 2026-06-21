#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import tempfile

import numpy as np
import pytest
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act_segment.configuration_act_segment import ACTSegmentConfig
from lerobot.policies.act_segment.processor_act_segment import (
    EfficientLiberoStateReorderStep,
    make_act_segment_pre_post_processors,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor.env_processor import LiberoProcessorStep


def test_efficient_libero_state_reorder():
    """Reorder LIBERO flat state to efficient-export gripper-first layout."""
    libero_state = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
    )
    observation = {"observation.state": libero_state}

    reordered = EfficientLiberoStateReorderStep().observation(observation)
    expected = torch.tensor(
        [
            [7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(reordered["observation.state"], expected)


def test_efficient_libero_state_reorder_rejects_wrong_shape():
    observation = {"observation.state": torch.randn(2, 7)}
    with pytest.raises(ValueError, match="last dim 8"):
        EfficientLiberoStateReorderStep().observation(observation)


def test_act_segment_preprocessor_includes_reorder_when_configured():
    """Policy preprocessor prepends reorder when observation_state_layout is set."""
    config = ACTSegmentConfig(observation_state_layout="efficient_libero")
    preprocessor, _ = make_act_segment_pre_post_processors(config)
    assert isinstance(preprocessor.steps[0], EfficientLiberoStateReorderStep)


def test_act_segment_preprocessor_skips_reorder_by_default():
    config = ACTSegmentConfig()
    preprocessor, _ = make_act_segment_pre_post_processors(config)
    assert not any(isinstance(step, EfficientLiberoStateReorderStep) for step in preprocessor.steps)


def test_act_segment_preprocessor_loads_efficient_libero_step_from_checkpoint():
    """Saved efficient_libero reorder step resolves when loading from pretrained."""
    config = ACTSegmentConfig(observation_state_layout="efficient_libero")
    preprocessor, postprocessor = make_act_segment_pre_post_processors(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save_pretrained(tmpdir)
        postprocessor.save_pretrained(tmpdir)
        loaded_preprocessor, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=tmpdir,
        )

    assert isinstance(loaded_preprocessor.steps[0], EfficientLiberoStateReorderStep)


def test_libero_env_output_reordered_to_efficient_layout():
    """LiberoProcessorStep output matches efficient layout after policy reorder step."""
    B = 1
    obs = {
        "pixels": {
            "image": (np.random.rand(B, 128, 128, 3) * 255).astype(np.uint8),
            "image2": (np.random.rand(B, 128, 128, 3) * 255).astype(np.uint8),
        },
        "robot_state": {
            "eef": {
                "pos": np.array([[1.0, 2.0, 3.0]]),
                "quat": np.array([[0.0, 0.0, 0.0, 1.0]]),
                "mat": np.random.randn(B, 3, 3),
            },
            "gripper": {
                "qpos": np.array([[0.04, -0.04]]),
                "qvel": np.random.randn(B, 2),
            },
            "joints": {
                "pos": np.random.randn(B, 7),
                "vel": np.random.randn(B, 7),
            },
        },
    }
    env_obs = preprocess_observation(obs)
    env_processed = LiberoProcessorStep().observation(env_obs)
    processed_obs = EfficientLiberoStateReorderStep().observation(
        {"observation.state": env_processed["observation.state"]}
    )
    state = processed_obs["observation.state"]
    assert state.shape == (B, 8)
    assert state[0, 0].item() == pytest.approx(0.04)
    assert state[0, 1].item() == pytest.approx(-0.04)
    assert state[0, 2:5].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert state[0, 5:8].tolist() == pytest.approx([0.0, 0.0, 0.0])
