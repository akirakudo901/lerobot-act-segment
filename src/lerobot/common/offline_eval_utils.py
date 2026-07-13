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

import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.policies import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.logging_utils import AverageMeter

if TYPE_CHECKING:
    from accelerate import Accelerator
    from dataset.core.types import SegmentLabel
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

EpisodeSpan = tuple[int, int, "SegmentLabel"]
EpisodeSpanTable = dict[int, list[EpisodeSpan]]
FrameLossKey = tuple[int, int]


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


def episode_pattern(spans: list[EpisodeSpan]) -> str:
    """Ordered MP/L pattern string, e.g. ``MP-L-MP``."""
    return "-".join(label for _, _, label in spans)


def local_frame_to_span_idx(local_frame: int, spans: list[EpisodeSpan]) -> int | None:
    """Return the span index containing ``local_frame``, or ``None`` if unlabeled."""
    for span_idx, (start, end, _label) in enumerate(spans):
        if start <= local_frame < end:
            return span_idx
    return None


def _episode_indices_for_span_tables(val_dataset: "LeRobotDataset") -> list[int]:
    if val_dataset.episodes is not None:
        return list(val_dataset.episodes)
    return list(range(val_dataset.num_episodes))


def _build_virtual_episode_span_tables(val_dataset: object) -> EpisodeSpanTable:
    """Build span tables from MpAugReadyTrainDataset virtual segments."""
    from dataset.core.segments import iter_labeled_spans

    episode_indices = set(_episode_indices_for_span_tables(val_dataset))
    episode_spans: EpisodeSpanTable = {}
    for segment in val_dataset._virtual_segments:
        episode_index = int(segment.virtual_episode_index)
        if episode_index not in episode_indices:
            continue
        episode_spans[episode_index] = list(iter_labeled_spans(segment.frame_labels_int))
    return episode_spans


def build_episode_span_tables(
    val_dataset: "LeRobotDataset",
    *,
    label_feature_key: str = "frame_label_int",
) -> EpisodeSpanTable | None:
    """Precompute GT MP/L spans for each validation episode."""
    if label_feature_key not in val_dataset.features:
        logger.warning(
            "Skipping segment validation metrics: dataset missing feature %r.",
            label_feature_key,
        )
        return None

    if hasattr(val_dataset, "_virtual_segments"):
        return _build_virtual_episode_span_tables(val_dataset)

    from dataset.core.segments import iter_labeled_spans

    episodes_meta = val_dataset.meta.episodes
    if episodes_meta is None:
        from lerobot.datasets.io_utils import load_episodes

        episodes_meta = load_episodes(val_dataset.root)
    episode_indices = _episode_indices_for_span_tables(val_dataset)
    hf_dataset = val_dataset.hf_dataset.with_format("numpy")
    reader = val_dataset._ensure_reader()
    abs_to_rel = reader._absolute_to_relative_idx

    episode_spans: EpisodeSpanTable = {}
    for episode_index in episode_indices:
        from_idx = int(episodes_meta["dataset_from_index"][episode_index])
        to_idx = int(episodes_meta["dataset_to_index"][episode_index])
        if abs_to_rel is not None:
            rel_indices = [
                abs_to_rel[abs_idx]
                for abs_idx in range(from_idx, to_idx)
                if abs_idx in abs_to_rel
            ]
            if not rel_indices:
                episode_spans[episode_index] = []
                continue
            labels = np.asarray(
                [
                    int(np.asarray(hf_dataset[row_idx][label_feature_key]).reshape(-1)[0])
                    for row_idx in rel_indices
                ],
                dtype=np.uint8,
            )
        else:
            labels = np.asarray(hf_dataset[label_feature_key][from_idx:to_idx], dtype=np.uint8).ravel()
        episode_spans[episode_index] = list(iter_labeled_spans(labels))

    return episode_spans


def _accumulate_deduped_frame_losses(
    frame_action_sums: dict[FrameLossKey, float],
    frame_action_counts: dict[FrameLossKey, int],
    frame_ce_sums: dict[FrameLossKey, float],
    frame_ce_counts: dict[FrameLossKey, int],
    *,
    episode_index: int,
    local_frame: int,
    action_l1: float,
    label_ce: float,
) -> None:
    key = (episode_index, local_frame)
    frame_action_sums[key] = frame_action_sums.get(key, 0.0) + action_l1
    frame_action_counts[key] = frame_action_counts.get(key, 0) + 1
    frame_ce_sums[key] = frame_ce_sums.get(key, 0.0) + label_ce
    frame_ce_counts[key] = frame_ce_counts.get(key, 0) + 1


def aggregate_segment_val_metrics(
    episode_spans: EpisodeSpanTable,
    frame_action_sums: dict[FrameLossKey, float],
    frame_action_counts: dict[FrameLossKey, int],
    frame_ce_sums: dict[FrameLossKey, float],
    frame_ce_counts: dict[FrameLossKey, int],
) -> dict[str, float]:
    """Aggregate deduped per-frame losses into span, pattern, and segment-type metrics."""
    type_action_values: dict[str, list[float]] = defaultdict(list)
    type_ce_values: dict[str, list[float]] = defaultdict(list)
    pattern_action_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    pattern_ce_values: dict[tuple[str, int, str], list[float]] = defaultdict(list)

    contributing_episodes = 0
    for episode_index, spans in episode_spans.items():
        if not spans:
            continue

        episode_had_span_loss = False
        pattern_key = episode_pattern(spans)
        for span_idx, (start, end, label) in enumerate(spans):
            action_values: list[float] = []
            ce_values: list[float] = []
            for local_frame in range(start, end):
                key = (episode_index, local_frame)
                if key not in frame_action_counts:
                    continue
                action_values.append(frame_action_sums[key] / frame_action_counts[key])
                ce_values.append(frame_ce_sums[key] / frame_ce_counts[key])

            if not action_values:
                continue

            episode_had_span_loss = True
            span_action_l1 = float(np.mean(action_values))
            span_label_ce = float(np.mean(ce_values))
            type_action_values[label].append(span_action_l1)
            type_ce_values[label].append(span_label_ce)
            pattern_action_values[(pattern_key, span_idx, label)].append(span_action_l1)
            pattern_ce_values[(pattern_key, span_idx, label)].append(span_label_ce)

        if episode_had_span_loss:
            contributing_episodes += 1

    metrics: dict[str, float] = {
        "segment_val_episodes": float(contributing_episodes),
        "segment_val_frames": float(len(frame_action_counts)),
    }

    for label, values in type_action_values.items():
        metrics[f"segment_type/{label}/action_l1"] = float(np.mean(values))
    for label, values in type_ce_values.items():
        metrics[f"segment_type/{label}/label_ce"] = float(np.mean(values))

    for (pattern_key, span_idx, label), values in pattern_action_values.items():
        metrics[f"pattern/{pattern_key}/span{span_idx}_{label}/action_l1"] = float(np.mean(values))
    for (pattern_key, span_idx, label), values in pattern_ce_values.items():
        metrics[f"pattern/{pattern_key}/span{span_idx}_{label}/label_ce"] = float(np.mean(values))

    return metrics


def compute_offline_val_segment_loss(
    policy: PreTrainedPolicy,
    val_dataloader: DataLoader,
    val_dataset: "LeRobotDataset",
    preprocessor: PolicyProcessorPipeline,
    accelerator: "Accelerator",
    camera_keys: list[str],
    episode_spans: EpisodeSpanTable,
    label_delta_indices: list[int],
) -> dict[str, float]:
    """Run segment-aware validation and return MP/L span and pattern metrics."""
    from lerobot.policies.act_segment.modeling_act_segment import ACTSegmentPolicy

    unwrapped = accelerator.unwrap_model(policy)
    if not isinstance(unwrapped, ACTSegmentPolicy):
        raise TypeError(
            "compute_offline_val_segment_loss requires ACTSegmentPolicy, "
            f"got {type(unwrapped).__name__}."
        )

    was_training = unwrapped.training
    unwrapped.eval()

    frame_action_sums: dict[FrameLossKey, float] = {}
    frame_action_counts: dict[FrameLossKey, int] = {}
    frame_ce_sums: dict[FrameLossKey, float] = {}
    frame_ce_counts: dict[FrameLossKey, int] = {}
    start_time = time.perf_counter()

    with torch.no_grad(), accelerator.autocast():
        for batch in val_dataloader:
            if batch is None:
                continue
            episode_indices = batch["episode_index"].tolist()
            frame_indices = batch["frame_index"].tolist()

            for cam_key in camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)

            action_l1, label_ce, valid_mask = unwrapped.per_step_val_losses(batch)
            batch_size = len(episode_indices)

            for row in range(batch_size):
                episode_index = int(episode_indices[row])
                spans = episode_spans.get(episode_index)
                if spans is None:
                    continue

                anchor_frame = int(frame_indices[row])
                for step_idx, delta in enumerate(label_delta_indices):
                    if step_idx >= valid_mask.shape[1] or not bool(valid_mask[row, step_idx].item()):
                        continue

                    local_frame = anchor_frame + int(delta)
                    if local_frame_to_span_idx(local_frame, spans) is None:
                        continue

                    _accumulate_deduped_frame_losses(
                        frame_action_sums,
                        frame_action_counts,
                        frame_ce_sums,
                        frame_ce_counts,
                        episode_index=episode_index,
                        local_frame=local_frame,
                        action_l1=float(action_l1[row, step_idx].item()),
                        label_ce=float(label_ce[row, step_idx].item()),
                    )

    if was_training:
        unwrapped.train()

    metrics = aggregate_segment_val_metrics(
        episode_spans,
        frame_action_sums,
        frame_action_counts,
        frame_ce_sums,
        frame_ce_counts,
    )
    metrics["segment_val_s"] = time.perf_counter() - start_time
    return metrics
