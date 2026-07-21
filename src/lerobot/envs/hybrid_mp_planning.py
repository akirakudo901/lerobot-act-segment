#!/usr/bin/env python

# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Env-agnostic Layer-1 OMPL helpers for hybrid MP VectorEnv RPC.

``OmplPathPlanner`` needs a live sim handle (``context['replay_env']``) for IK and
MuJoCo validity checks. These helpers bind that handle explicitly so any gym env
compatible with the planner can expose the same RPC surface without duplicating
planner construction in each env class.

Worker envs (AsyncVectorEnv) still need thin methods that call these helpers —
``VectorEnv.call`` can only invoke methods that exist on the worker env instance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def plan_ompl(
    replay_env: Any,
    start_ee_pose: np.ndarray,
    goal_ee_pose: np.ndarray,
    *,
    algorithm: str = "RRTConnect",
    time_limit: float = 1.0,
    include_grasped_object_in_validity: bool = False,
    always_valid: bool = True,
    on_ik_failure: str = "raise",
    max_ee_step_m: float | None = 0.02,
    path_interpolate_count: int | None = 50,
) -> dict[str, Any] | None:
    """
    Layer-1 OMPL plan against *replay_env* → picklable ``ExecutionPlan`` mapping.

    Returns ``None`` when planning is skipped (``on_ik_failure='skip'``).
    With ``always_valid=True`` (default), OMPL skips MuJoCo collision checks.
    ``max_ee_step_m=None`` skips EE densification (see ``execution_plan_from_path``).
    """
    from hybrid_eval.execution.waypoint_osc import execution_plan_to_mapping
    from hybrid_eval.planning.ompl_motion_planner import OmplPathPlanner, OmplPlanSkipped

    planner = OmplPathPlanner(
        algorithm=algorithm,
        time_limit=float(time_limit),
        include_grasped_object_in_validity=bool(include_grasped_object_in_validity),
        always_valid=bool(always_valid),
        on_ik_failure=on_ik_failure,  # type: ignore[arg-type]
        max_ee_step_m=None if max_ee_step_m is None else float(max_ee_step_m),
        path_interpolate_count=path_interpolate_count,
        ik_max_iters=1000,
        ik_pos_tol=1e-2,
        ik_ori_tol=3e-2,
        ik_damping=5e-2,
        ik_null_space_gain=0.0,
        ik_waypoint_max_step_m=0.1,
    )
    try:
        plan = planner.plan(
            np.asarray(start_ee_pose, dtype=np.float64),
            np.asarray(goal_ee_pose, dtype=np.float64),
            context={"replay_env": replay_env},
        )
    except OmplPlanSkipped:
        return None
    return execution_plan_to_mapping(plan)


def plan_ompl_indexed(
    replay_env: Any,
    episode_index: int,
    targets: Sequence[Any | None],
    poses: Sequence[np.ndarray],
    mask: Sequence[bool],
    *,
    algorithm: str = "RRTConnect",
    time_limit: float = 1.0,
    include_grasped_object_in_validity: bool = False,
    always_valid: bool = True,
    on_ik_failure: str = "raise",
    max_ee_step_m: float | None = 0.02,
    path_interpolate_count: int | None = 50,
    pos_scale: float = 0.05,
    rot_scale: float = 0.5,
) -> dict[str, Any] | None:
    """
    Batched VectorEnv worker entry: plan for ``episode_index`` using *replay_env*.

    ``poses[i]`` is the live 6-D EE pose; ``targets[i]`` supplies the MP action
    used to resolve the absolute goal pose (and gripper hold).
    """
    from hybrid_eval.connectors.action_format import goal_pose_from_planning_target

    idx = int(episode_index)
    if idx < 0 or idx >= len(mask):
        return None
    if not mask[idx] or targets[idx] is None:
        return None
    start = np.asarray(poses[idx], dtype=np.float64).reshape(6)
    goal = goal_pose_from_planning_target(
        start,
        targets[idx],
        pos_scale=float(pos_scale),
        rot_scale=float(rot_scale),
    )
    return plan_ompl(
        replay_env,
        start,
        goal,
        algorithm=algorithm,
        time_limit=time_limit,
        include_grasped_object_in_validity=include_grasped_object_in_validity,
        always_valid=always_valid,
        on_ik_failure=on_ik_failure,
        max_ee_step_m=max_ee_step_m,
        path_interpolate_count=path_interpolate_count,
    )
