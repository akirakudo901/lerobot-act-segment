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

from lerobot.policies.act_label.configuration_act_label import ACTLabelConfig
from lerobot.policies.act_label.processor_act_label import make_act_label_pre_post_processors
from lerobot.policies.act_segment.processor_act_segment import EfficientLiberoStateReorderStep
from lerobot.policies.factory import make_pre_post_processors


def test_act_label_preprocessor_includes_reorder_when_configured():
    config = ACTLabelConfig(observation_state_layout="efficient_libero")
    preprocessor, _ = make_act_label_pre_post_processors(config)
    assert isinstance(preprocessor.steps[0], EfficientLiberoStateReorderStep)


def test_act_label_preprocessor_skips_reorder_by_default():
    config = ACTLabelConfig()
    preprocessor, _ = make_act_label_pre_post_processors(config)
    assert not any(isinstance(step, EfficientLiberoStateReorderStep) for step in preprocessor.steps)


def test_act_label_preprocessor_loads_efficient_libero_step_from_checkpoint():
    config = ACTLabelConfig(observation_state_layout="efficient_libero")
    preprocessor, postprocessor = make_act_label_pre_post_processors(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        preprocessor.save_pretrained(tmpdir)
        postprocessor.save_pretrained(tmpdir)
        loaded_preprocessor, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=tmpdir,
        )

    assert isinstance(loaded_preprocessor.steps[0], EfficientLiberoStateReorderStep)
