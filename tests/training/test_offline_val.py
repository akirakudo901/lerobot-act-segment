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

"""Tests for offline validation loss during training."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("accelerate", reason="accelerate is required (install lerobot[training])")

from accelerate import Accelerator

from lerobot.common.offline_eval_utils import (
    assert_no_episode_overlap,
    compute_offline_val_loss,
    split_episode_indices,
)
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from tests.utils import DEVICE

DUMMY_REPO_ID = "dummy/offline_val"
DUMMY_STATE_DIM = 6
DUMMY_ACTION_DIM = 6
IMAGE_SIZE = 8


def make_dummy_dataset(camera_keys: list[str], tmp_path: Path, n_episodes: int = 4):
    """Create a minimal local dataset for offline val smoke tests."""
    features = {
        "action": {"dtype": "float32", "shape": (DUMMY_ACTION_DIM,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (DUMMY_STATE_DIM,), "names": None},
    }
    for cam in camera_keys:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=DUMMY_REPO_ID,
        fps=30,
        features=features,
        root=tmp_path / "_dataset",
    )
    root = tmp_path / "_dataset"
    for ep_idx in range(n_episodes):
        for _ in range(3):
            frame = {
                "action": np.random.randn(DUMMY_ACTION_DIM).astype(np.float32),
                "observation.state": np.random.randn(DUMMY_STATE_DIM).astype(np.float32),
            }
            for cam in camera_keys:
                frame[f"observation.images.{cam}"] = np.random.randint(
                    0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8
                )
            frame["task"] = f"task_{ep_idx}"
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset, root


def test_split_episode_indices():
    episodes = list(range(10))
    train_eps, val_eps = split_episode_indices(episodes, fraction=0.2, seed=42)

    assert len(val_eps) >= 1
    assert len(train_eps) + len(val_eps) == len(episodes)
    assert set(train_eps).isdisjoint(val_eps)

    train_eps2, val_eps2 = split_episode_indices(episodes, fraction=0.2, seed=42)
    assert train_eps == train_eps2
    assert val_eps == val_eps2

    train_one, val_one = split_episode_indices([0], fraction=0.5, seed=0)
    assert train_one == [0]
    assert val_one == []


def test_assert_no_episode_overlap():
    assert_no_episode_overlap([0, 1, 2], [3, 4])
    with pytest.raises(ValueError, match="overlap"):
        assert_no_episode_overlap([0, 1, 2], [2, 3])


def test_validate_val_config_mutual_exclusion(tmp_path):
    _dataset, root = make_dummy_dataset(["camera1"], tmp_path)
    policy_config = make_policy_config("act", push_to_hub=False, device=DEVICE)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root),
        policy=policy_config,
        output_dir=tmp_path / "_output",
        val_freq=100,
    )
    with pytest.raises(ValueError, match="exactly one of val_split_fraction or val_dataset"):
        cfg.validate()

    cfg.val_split_fraction = 0.2
    cfg.validate()

    cfg.val_dataset = DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[0])
    with pytest.raises(ValueError, match="exactly one of val_split_fraction or val_dataset"):
        cfg.validate()


def test_validate_val_config_fraction_bounds(tmp_path):
    _dataset, root = make_dummy_dataset(["camera1"], tmp_path)
    policy_config = make_policy_config("act", push_to_hub=False, device=DEVICE)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root),
        policy=policy_config,
        output_dir=tmp_path / "_output",
        val_freq=100,
        val_split_fraction=1.5,
    )
    with pytest.raises(ValueError, match="val_split_fraction must be in"):
        cfg.validate()


def test_validate_val_config_requires_val_episodes_on_same_dataset(tmp_path):
    _dataset, root = make_dummy_dataset(["camera1"], tmp_path, n_episodes=4)
    policy_config = make_policy_config("act", push_to_hub=False, device=DEVICE)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[0, 1, 2]),
        val_dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root),
        policy=policy_config,
        output_dir=tmp_path / "_output",
        val_freq=100,
    )
    with pytest.raises(ValueError, match="val_dataset.episodes must be set explicitly"):
        cfg.validate()


def test_validate_val_config_overlap_detection(tmp_path):
    _dataset, root = make_dummy_dataset(["camera1"], tmp_path, n_episodes=4)
    policy_config = make_policy_config("act", push_to_hub=False, device=DEVICE)

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[0, 1, 2]),
        val_dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[2, 3]),
        policy=policy_config,
        output_dir=tmp_path / "_output",
        val_freq=100,
    )
    with pytest.raises(ValueError, match="overlap"):
        cfg.validate()


def test_compute_offline_val_loss(tmp_path):
    _dataset, root = make_dummy_dataset(["camera1"], tmp_path, n_episodes=4)
    policy_config = make_policy_config("act", push_to_hub=False, device=DEVICE)
    policy_config.chunk_size = 3
    policy_config.n_action_steps = 3
    policy_config.use_vae = False

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[0, 1, 2]),
        val_dataset=DatasetConfig(repo_id=DUMMY_REPO_ID, root=root, episodes=[3]),
        policy=policy_config,
        output_dir=tmp_path / "_output",
        val_freq=1,
        batch_size=2,
        num_workers=0,
    )

    from lerobot.datasets import make_dataset, make_val_dataset

    train_dataset = make_dataset(cfg)
    val_dataset = make_val_dataset(cfg, val_episodes=[3])
    policy = make_policy(cfg.policy, ds_meta=train_dataset.meta)
    preprocessor, _ = make_pre_post_processors(cfg.policy, dataset_stats=train_dataset.meta.stats)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    accelerator = Accelerator(cpu=True)
    policy = accelerator.prepare(policy)

    metrics = compute_offline_val_loss(
        policy=policy,
        val_dataloader=val_dataloader,
        preprocessor=preprocessor,
        accelerator=accelerator,
        camera_keys=train_dataset.meta.camera_keys,
    )

    assert metrics["val_batches"] > 0
    assert np.isfinite(metrics["loss"])
    assert metrics["val_s"] >= 0
