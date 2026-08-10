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

from lerobot.configs import PreTrainedConfig

from ..act.configuration_act import ACTConfig


@PreTrainedConfig.register_subclass("act_segment")
@dataclass
class ACTSegmentConfig(ACTConfig):
    """ACT with an auxiliary per-chunk-step MP/L BIO label head.

    Expects a dataset feature ``frame_label_int`` (0=B-MP, 1=I-MP, 2=B-L, 3=I-L)
    chunked over the same horizon as ``action``.
    """

    label_feature_key: str = "frame_label_int"
    # Scales per-type label CE: weighted_ce = mp_ce_weight * mp_ce_loss + l_ce_weight * l_ce_loss.
    mp_ce_weight: float = 1.0
    l_ce_weight: float = 1.0
    # Scales the MP execution-frame L1 term: weighted_l1 = l_l1_loss + mp_l1_weight * mp_l1_loss.
    mp_l1_weight: float = 1.0
    num_label_classes: int = 4

    use_hybrid_orchestrator: bool = False
    hybrid_connector: str = "mp_labeled_frames"
    # ``ik_pose_setter``: dummy action + after-step IK teleport (lerobot_eval hook).
    # ``ompl_waypoints``: Layer-1 OMPL plan + closed-loop OSC inside select_action.
    mp_executor_type: str = "ik_pose_setter"
    # ``full_chunk``: execute all ``n_action_steps`` before refill.
    # ``until_first_mp``: truncate at first MP-labeled step (inclusive), then refill.
    # ``until_first_mp_segment``: execute the first contiguous MP run (all consecutive
    # MP frames, ignoring intermediate ``B-MP`` waypoint boundaries), then refill at the
    # following L (that L is not executed from the cached chunk). Leading L before the
    # run is kept. If the run reaches the chunk end with no following L, execute it fully.
    # ``trust_contiguous_mp``: like ``until_first_mp_segment``, but only trust/execute the
    # contiguous MP run when an L follows it in the chunk (proof the full run was
    # predicted). If the run is truncated at the chunk end, stop before it and refill.
    # ``trust_near_mp``: like ``until_first_mp`` when the first MP frame lies within the
    # first ``hybrid_refill_mp_trust_steps`` predicted actions; otherwise stop before that
    # MP frame (exclusive) and refill so MP is re-predicted from live observations.
    hybrid_refill_mode: str = "full_chunk"
    # Only used when ``hybrid_refill_mode`` is ``trust_near_mp``. MP frames at indices
    # ``0 .. hybrid_refill_mp_trust_steps - 1`` are executed from the cached chunk;
    # the first MP at index ``>= hybrid_refill_mp_trust_steps`` is deferred.
    hybrid_refill_mp_trust_steps: int = 5

    # OMPL Layer-1 / closed-loop OSC knobs (used when mp_executor_type=ompl_waypoints).
    ompl_algorithm: str = "RRTConnect"
    ompl_time_limit: float = 1.0
    ompl_include_grasped_object_in_validity: bool = False
    # False: MuJoCo collision checks in Layer-1 OMPL. True: pure geometric planning.
    ompl_always_valid: bool = False
    ompl_on_ik_failure: str = "best_guess"  # raise | skip | best_guess
    # None: skip EE densification after OMPL (use interpolate_count waypoints only).
    ompl_max_ee_step_m: float | None = None
    ompl_path_interpolate_count: int | None = None
    # Fraction of joint-space extent between discrete collision samples (OMPL).
    ompl_validity_checking_resolution: float = 0.03
    # Shortcut the OMPL solution path after solve (can hug obstacles more tightly).
    ompl_simplify: bool = True
    # MuJoCo contact.dist below this (m) counts as collision. None = planner default.
    ompl_contact_dist_eps: float | None = None
    ompl_pos_tol_m: float = 0.01
    ompl_ori_tol_rad: float | None = None
    ompl_max_steps_per_waypoint: int = 50
    ompl_max_pos_delta_m: float | None = 0.05
    ompl_max_ori_delta_rad: float | None = 0.5
    ompl_pos_scale: float = 0.05
    ompl_rot_scale: float = 0.5
    # Extra OSC ticks holding the final waypoint after it is first reached.
    ompl_goal_hold_frames: int = 0
    # After a Layer-1 plan failure, requery the policy this many times before giving up.
    # Total plan attempts = 1 + ompl_plan_max_retries (initial + requeries).
    ompl_plan_max_retries: int = 3
    # On requery refill, sample ACT latent from N(0, I) instead of the deterministic zero vector.
    ompl_retry_sample_latent: bool = True
    # Multiply ``ompl_time_limit`` on retries whose previous status looks like a planner timeout.
    ompl_retry_time_limit_scale: float = 2.0
    # ``terminate_hold``: emit dummy OSC and mark the row exhausted (no chunk advance).
    # ``raise``: abort select_action with OmplPlanRetriesExhausted.
    ompl_plan_on_exhausted: str = "terminate_hold"

    # Reorder ``observation.state`` in the policy preprocessor to match the training dataset layout.
    # Default ``None``: no reordering. Set explicitly when eval env layout differs from training:
    # ``lerobot`` for datasets with ee_pos + ee_ori + gripper (no reorder step),
    # ``efficient_libero`` for legacy efficient exports (gripper + ee_pos + ee_ori).
    observation_state_layout: str | None = None

    # MP-action rescaling registry for eval-time inverse transform on predicted MP rows.
    # When unset, resolved automatically from {dataset_root}/meta/mp_action_rescaling.json
    # using dataset metadata at policy construction or train_config.json beside pretrained_path.
    mp_action_rescaling_registry_path: str | None = None
    mp_action_rescaling_strategy: str | None = None

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))
