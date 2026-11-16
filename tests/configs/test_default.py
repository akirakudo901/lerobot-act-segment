# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

# MODIFIED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

import sys
import types
from types import SimpleNamespace

import pytest

from lerobot.configs.default import DatasetConfig


def test_dataset_config_valid():
    DatasetConfig(repo_id="user/repo", episodes=[0, 1, 2])


def test_dataset_config_negative_episodes():
    with pytest.raises(ValueError, match="non-negative"):
        DatasetConfig(repo_id="user/repo", episodes=[0, -1, 2])


def test_dataset_config_duplicate_episodes():
    with pytest.raises(ValueError, match="duplicates"):
        DatasetConfig(repo_id="user/repo", episodes=[0, 1, 1, 2])


def test_dataset_config_none_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=None)


def test_dataset_config_empty_episodes_ok():
    DatasetConfig(repo_id="user/repo", episodes=[])


def test_dataset_config_ablation_action_mode_defaults_to_none():
    assert DatasetConfig(repo_id="user/repo").ablation_action_mode == "none"


def test_dataset_config_accepts_ablation_action_mode():
    cfg = DatasetConfig(repo_id="user/repo", ablation_action_mode="original_dense")
    assert cfg.ablation_action_mode == "original_dense"


@pytest.mark.parametrize("mode", ["none", "original_dense"])
def test_factory_passes_ablation_action_mode_to_the_wrapper(monkeypatch, mode):
    from lerobot.datasets import factory

    captured = {}

    def fake_wrap(dataset, **kwargs):
        captured.update(kwargs)
        return dataset

    fake_module = types.ModuleType("dataset.loaders.mp_aug_ready_train_dataset")
    fake_module.wrap_mp_aug_ready_dataset = fake_wrap
    monkeypatch.setitem(sys.modules, "dataset.loaders.mp_aug_ready_train_dataset", fake_module)

    cfg = DatasetConfig(
        repo_id="user/repo",
        enable_mp_aug_ready_transforms=True,
        ablation_action_mode=mode,
    )
    dataset = SimpleNamespace(root=None)
    factory._maybe_wrap_mp_aug_ready_dataset(
        dataset,
        cfg,
        SimpleNamespace(chunk_size=5),
        enable_augmentation=True,
    )
    assert captured["ablation_action_mode"] == mode
