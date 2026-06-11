#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
from pprint import pformat

import torch

from lerobot.configs import PreTrainedConfig
from lerobot.configs.default import DatasetConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_PREFIX, REWARD

from .dataset_metadata import LeRobotDatasetMetadata
from .lerobot_dataset import LeRobotDataset
from .multi_dataset import MultiLeRobotDataset
from .streaming_dataset import StreamingLeRobotDataset


def resolve_delta_timestamps(
    cfg: PreTrainedConfig | RewardModelConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the config.

    Args:
        cfg (PreTrainedConfig | RewardModelConfig): The config to read delta_indices from. Both
            ``PreTrainedConfig`` and concrete ``RewardModelConfig`` subclasses expose the
            ``{observation,action,reward}_delta_indices`` properties used below.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Policies that supervise per-frame labels over an action chunk may expose optional
    ``label_feature_key`` and ``label_delta_indices`` attributes (e.g. ``act_segment``).
    When present and the label key exists in the dataset, it is chunked like ``action``.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    label_feature_key = getattr(cfg, "label_feature_key", None)
    label_delta_indices = getattr(cfg, "label_delta_indices", None)

    delta_timestamps = {}
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
        if (
            label_feature_key
            and label_delta_indices is not None
            and key == label_feature_key
        ):
            delta_timestamps[key] = [i / ds_meta.fps for i in label_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset_from_config(
    dataset_cfg: DatasetConfig,
    trainable_config: PreTrainedConfig | RewardModelConfig,
    *,
    tolerance_s: float,
    num_workers: int,
    episodes: list[int] | None = None,
    enable_image_transforms: bool | None = None,
) -> LeRobotDataset | MultiLeRobotDataset:
    """Build a dataset from a DatasetConfig with optional episode/transform overrides."""
    use_transforms = (
        dataset_cfg.image_transforms.enable
        if enable_image_transforms is None
        else enable_image_transforms
    )
    image_transforms = (
        ImageTransforms(dataset_cfg.image_transforms) if use_transforms else None
    )
    episode_list = episodes if episodes is not None else dataset_cfg.episodes

    if isinstance(dataset_cfg.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            dataset_cfg.repo_id, root=dataset_cfg.root, revision=dataset_cfg.revision
        )
        delta_timestamps = resolve_delta_timestamps(trainable_config, ds_meta)
        if not dataset_cfg.streaming:
            dataset = LeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=episode_list,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                video_backend=dataset_cfg.video_backend,
                return_uint8=True,
                tolerance_s=tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                dataset_cfg.repo_id,
                root=dataset_cfg.root,
                episodes=episode_list,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=dataset_cfg.revision,
                max_num_shards=num_workers,
                tolerance_s=tolerance_s,
                return_uint8=True,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")
        dataset = MultiLeRobotDataset(
            dataset_cfg.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=dataset_cfg.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if dataset_cfg.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


def make_dataset(cfg: TrainPipelineConfig) -> LeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
    """
    return make_dataset_from_config(
        cfg.dataset,
        cfg.trainable_config,
        tolerance_s=cfg.tolerance_s,
        num_workers=cfg.num_workers,
    )


def make_val_dataset(
    cfg: TrainPipelineConfig,
    val_episodes: list[int] | None = None,
) -> LeRobotDataset | MultiLeRobotDataset:
    """Build the offline validation dataset without image augmentations."""
    if cfg.val_dataset is not None:
        return make_dataset_from_config(
            cfg.val_dataset,
            cfg.trainable_config,
            tolerance_s=cfg.tolerance_s,
            num_workers=cfg.num_workers,
            enable_image_transforms=False,
        )

    if val_episodes is None:
        raise ValueError("val_episodes is required when val_dataset is not set.")

    return make_dataset_from_config(
        cfg.dataset,
        cfg.trainable_config,
        tolerance_s=cfg.tolerance_s,
        num_workers=cfg.num_workers,
        episodes=val_episodes,
        enable_image_transforms=False,
    )
