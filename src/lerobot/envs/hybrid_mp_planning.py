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

Plan results are picklable dicts:
- success: ``ExecutionPlan`` mapping (has ``joint_waypoints`` / ``ee_waypoints``)
- skip: ``None`` (``on_ik_failure='skip'``)
- failure: ``{"ok": False, "status": ..., "message": ..., ...}`` so AsyncVectorEnv
  workers do not die on ``OmplPlanFailed``. When failure-viz is enabled on the
  worker env, soft failures also include picklable PNG ``failure_stills``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def is_ompl_plan_failure(result: Any) -> bool:
    """True when *result* is a soft OMPL failure payload from :func:`plan_ompl`."""
    return isinstance(result, Mapping) and result.get("ok") is False


def is_ompl_plan_success(result: Any) -> bool:
    """True when *result* is a serialised :class:`ExecutionPlan` mapping."""
    if not isinstance(result, Mapping):
        return False
    if result.get("ok") is False:
        return False
    return "joint_waypoints" in result and "ee_waypoints" in result


def ompl_plan_failure_status(result: Any) -> str:
    """Best-effort status string for a soft failure / skip result."""
    if result is None:
        return "skipped"
    if is_ompl_plan_failure(result):
        return str(result.get("status") or "failed")
    return "unknown"


def _qpos_diagnosis_to_mapping(diagnosis: Any | None) -> dict[str, Any] | None:
    if diagnosis is None:
        return None
    return {
        "reason": diagnosis.reason_label(),
        "in_bounds": bool(diagnosis.in_bounds),
        "collision_valid": bool(diagnosis.collision_valid),
        "joint_limit_violations": [
            {
                "joint_index": int(v.joint_index),
                "value": float(v.value),
                "lower": float(v.lower),
                "upper": float(v.upper),
            }
            for v in getattr(diagnosis, "joint_limit_violations", ())
        ],
        "offending_contacts": [
            {
                "geom1_name": str(c.geom1_name),
                "geom2_name": str(c.geom2_name),
                "dist": float(c.dist),
            }
            for c in getattr(diagnosis, "offending_contacts", ())
        ],
    }


def _ompl_plan_failed_to_mapping(err: Any) -> dict[str, Any]:
    """Picklable soft-failure payload from :class:`OmplPlanFailed`."""
    goal = getattr(err, "goal_ee_pose", None)
    gripper = getattr(err, "gripper_qpos", None)
    q_start = getattr(err, "q_start", None)
    q_goal = getattr(err, "q_goal", None)
    return {
        "ok": False,
        "status": str(getattr(err, "status", "failed")),
        "message": str(err),
        "ik_success": bool(getattr(err, "ik_success", False)),
        "ik_pos_err": float(getattr(err, "ik_pos_err", float("nan"))),
        "ik_ori_err": float(getattr(err, "ik_ori_err", float("nan"))),
        "algorithm": getattr(err, "algorithm", None),
        "time_limit": getattr(err, "time_limit", None),
        "goal_ee_pose": (
            None if goal is None else np.asarray(goal, dtype=np.float64).reshape(6)
        ),
        "gripper_qpos": (
            None if gripper is None else np.asarray(gripper, dtype=np.float64).reshape(-1)
        ),
        "q_start": (
            None if q_start is None else np.asarray(q_start, dtype=np.float64).reshape(-1)
        ),
        "q_goal": (
            None if q_goal is None else np.asarray(q_goal, dtype=np.float64).reshape(-1)
        ),
        "diagnosis": _qpos_diagnosis_to_mapping(getattr(err, "diagnosis", None)),
        "start_diagnosis": _qpos_diagnosis_to_mapping(
            getattr(err, "start_diagnosis", None)
        ),
    }


def enable_ompl_failure_viz(replay_env: Any) -> None:
    """Inject mocap ghost / goal-marker bodies and mark *replay_env* for still capture."""
    from hybrid_eval.visualize.ompl_failure_render import ensure_ompl_failure_viz

    ensure_ompl_failure_viz(replay_env)
    replay_env._ompl_failure_viz_enabled = True


def is_ompl_failure_viz_enabled(replay_env: Any) -> bool:
    return bool(getattr(replay_env, "_ompl_failure_viz_enabled", False))


def plan_ompl(
    replay_env: Any,
    start_ee_pose: np.ndarray,
    goal_ee_pose: np.ndarray,
    *,
    algorithm: str = "RRTConnect",
    time_limit: float = 1.0,
    include_grasped_object_in_validity: bool = False,
    always_valid: bool = False,
    on_ik_failure: str = "raise",
    max_ee_step_m: float | None = 0.02,
    path_interpolate_count: int | None = 50,
    validity_checking_resolution: float = 0.03,
    simplify: bool = True,
    contact_dist_eps: float | None = None,
) -> dict[str, Any] | None:
    """
    Layer-1 OMPL plan against *replay_env* → picklable ``ExecutionPlan`` mapping.

    Returns ``None`` when planning is skipped (``on_ik_failure='skip'``).
    Returns a soft-failure dict (``ok=False``) when OMPL finds no path.
    With ``always_valid=False`` (default), OMPL uses MuJoCo collision checks;
    set ``always_valid=True`` for pure geometric planning.
    ``max_ee_step_m=None`` skips EE densification (see ``execution_plan_from_path``).

    When :func:`enable_ompl_failure_viz` has been called on *replay_env*, soft
    failures include ``failure_stills`` PNG bytes captured at the failure frame.
    """
    from hybrid_eval.execution.waypoint_osc import execution_plan_to_mapping
    from hybrid_eval.planning.ompl_motion_planner import (
        OmplPlanFailed,
        OmplPathPlanner,
        OmplPlanSkipped,
    )
    from hybrid_eval.visualize.ompl_failure_render import attach_ompl_failure_stills

    planner = OmplPathPlanner(
        algorithm=algorithm,
        time_limit=float(time_limit),
        include_grasped_object_in_validity=bool(include_grasped_object_in_validity),
        always_valid=bool(always_valid),
        on_ik_failure=on_ik_failure,  # type: ignore[arg-type]
        max_ee_step_m=None if max_ee_step_m is None else float(max_ee_step_m),
        path_interpolate_count=path_interpolate_count,
        validity_checking_resolution=float(validity_checking_resolution),
        simplify=bool(simplify),
        contact_dist_eps=(
            None if contact_dist_eps is None else float(contact_dist_eps)
        ),
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
    except OmplPlanFailed as err:
        mapping = _ompl_plan_failed_to_mapping(err)
        if is_ompl_failure_viz_enabled(replay_env):
            attach_ompl_failure_stills(replay_env, mapping)
        return mapping
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
    always_valid: bool = False,
    on_ik_failure: str = "raise",
    max_ee_step_m: float | None = 0.02,
    path_interpolate_count: int | None = 50,
    validity_checking_resolution: float = 0.03,
    simplify: bool = True,
    contact_dist_eps: float | None = None,
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
        validity_checking_resolution=validity_checking_resolution,
        simplify=simplify,
        contact_dist_eps=contact_dist_eps,
    )
