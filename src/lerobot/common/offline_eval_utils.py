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

# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

import random
import time
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.logging_utils import AverageMeter

if TYPE_CHECKING:
    from accelerate import Accelerator


def resolve_episode_pool(episodes: list[int] | None, total_episodes: int) -> list[int]:
    """Return the episode indices used for train or val when episodes is unset."""
    if episodes is not None:
        return list(episodes)
    return list(range(total_episodes))


def split_episode_indices(
    episodes: list[int], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Shuffle episodes and split into train/val holdout sets.

    Uses floor(n * fraction) val episodes, guaranteeing at least one val episode when n >= 2.
    """
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}.")

    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    if n == 1:
        return shuffled, []

    n_val = max(1, int(n * fraction))
    val_episodes = shuffled[:n_val]
    train_episodes = shuffled[n_val:]
    return train_episodes, val_episodes


def assert_no_episode_overlap(train_episodes: list[int], val_episodes: list[int]) -> None:
    overlap = sorted(set(train_episodes) & set(val_episodes))
    if overlap:
        raise ValueError(
            "Train and val datasets share repo_id+root but overlap on episodes: "
            f"{overlap}"
        )


def resolve_train_val_episodes(
    train_episodes: list[int] | None,
    total_episodes: int,
    *,
    val_split_fraction: float | None,
    val_episodes: list[int] | None,
    seed: int,
) -> tuple[list[int], list[int] | None]:
    """Resolve final train/val episode lists from config and dataset metadata."""
    episode_pool = resolve_episode_pool(train_episodes, total_episodes)
    if val_split_fraction is not None:
        train_eps, val_eps = split_episode_indices(episode_pool, val_split_fraction, seed)
        return train_eps, val_eps
    if val_episodes is not None:
        return episode_pool, list(val_episodes)
    return episode_pool, None


def compute_offline_val_loss(
    policy: PreTrainedPolicy,
    val_dataloader: DataLoader,
    preprocessor: PolicyProcessorPipeline,
    accelerator: "Accelerator",
    camera_keys: list[str],
) -> dict[str, float]:
    """Run a supervised validation pass without backward() or optimizer steps."""
    unwrapped = accelerator.unwrap_model(policy)
    was_training = unwrapped.training
    unwrapped.eval()

    meters: dict[str, AverageMeter] = {}
    start_time = time.perf_counter()
    n_batches = 0

    with torch.no_grad(), accelerator.autocast():
        for batch in val_dataloader:
            for cam_key in camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            loss, output_dict = unwrapped.forward(batch)

            if "loss" not in meters:
                meters["loss"] = AverageMeter("loss", ":.3f")
            meters["loss"].update(loss.item())

            for key, value in (output_dict or {}).items():
                if isinstance(value, (int, float)):
                    if key not in meters:
                        meters[key] = AverageMeter(key, ":.3f")
                    meters[key].update(float(value))

            n_batches += 1

    if was_training:
        unwrapped.train()

    metrics = {key: meter.avg for key, meter in meters.items()}
    metrics["val_s"] = time.perf_counter() - start_time
    metrics["val_batches"] = float(n_batches)
    return metrics
