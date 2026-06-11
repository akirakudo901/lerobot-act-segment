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

# MODIFIED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.types import RobotObservation
from lerobot.utils.constants import OBS_STATE, OBS_STR

from .utils import _LazyAsyncVectorEnv, parse_camera_names


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def resolve_task_ids(
    suite: Any,
    *,
    task_ids: Iterable[int] | None = None,
    task_names: Sequence[str] | None = None,
) -> list[int]:
    """Map ``task_ids`` and/or ``task_names`` to validated suite task indices."""
    total_tasks = len(suite.tasks)
    if task_ids is None and task_names is None:
        return list(range(total_tasks))

    name_to_id = {task.name: idx for idx, task in enumerate(suite.tasks)}
    selected: set[int] = set()
    if task_ids is not None:
        selected.update(_select_task_ids(total_tasks, task_ids))
    if task_names is not None:
        for name in task_names:
            if name not in name_to_id:
                available = ", ".join(sorted(name_to_id))
                raise ValueError(
                    f"Unknown task name {name!r} in suite. Available: {available}"
                )
            selected.add(name_to_id[name])

    if not selected:
        raise ValueError("No tasks selected after resolving task_ids and task_names.")
    return sorted(selected)


# LIBERO-plus perturbation variants encode the perturbation in the filename
# but on disk only the base `.pruned_init` exists — strip the suffix to match
# LIBERO-plus's own suite.get_task_init_states() (we reimplement it here so we
# can pass weights_only=False for PyTorch 2.6+ numpy pickles).
_LIBERO_PERTURBATION_SUFFIX_RE = re.compile(r"_(?:language|view|light)_[^.]*|_(?:table|tb)_\d+")


def get_task_init_states(task_suite: Any, i: int, is_libero_plus: bool = False) -> np.ndarray:
    task = task_suite.tasks[i]
    filename = Path(task.init_states_file)
    root = Path(get_libero_path("init_states"))

    if not is_libero_plus:
        init_states_path = root / task.problem_folder / filename.name
        return torch.load(init_states_path, weights_only=False)  # nosec B614

    # LIBERO-plus: `_add_` / `_level` variants store extra-object layouts under
    # libero_newobj/ as a flat array that must be reshaped to (1, -1).
    if "_add_" in filename.name or "_level" in filename.name:
        init_states_path = root / "libero_newobj" / task.problem_folder / filename.name
        init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
        return init_states.reshape(1, -1)

    # LIBERO-plus perturbation variants encode the perturbation in the filename
    # but on disk only the base `.pruned_init` exists — strip the suffix to match.
    stripped = _LIBERO_PERTURBATION_SUFFIX_RE.sub("", filename.stem) + filename.suffix
    init_states_path = root / task.problem_folder / stripped
    return torch.load(init_states_path, weights_only=False)  # nosec B614


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        n_envs: int = 1,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        is_libero_plus: bool = False,
    ):
        super().__init__()
        self.task_id = task_id
        self.is_libero_plus = is_libero_plus
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)

        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = (
            get_task_init_states(task_suite, self.task_id, is_libero_plus=self.is_libero_plus)
            if self.init_states
            else None
        )
        self._reset_stride = n_envs  # when performing a reset, append `_reset_stride` to `init_state_id`.

        self.init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        # Extract task metadata without allocating GPU resources (safe before fork).
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        self._task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env: OffScreenRenderEnv | None = (
            None  # deferred — created on first reset() inside the worker subprocess
        )

        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _ensure_env(self) -> None:
        """Create the underlying OffScreenRenderEnv on first use.

        Called inside the worker subprocess after fork(), so each worker gets
        its own clean EGL context rather than inheriting a stale one from the
        parent process (which causes EGL_BAD_CONTEXT crashes with AsyncVectorEnv).
        """
        if self._env is not None:
            return
        env = OffScreenRenderEnv(
            bddl_file_name=self._task_bddl_file,
            camera_heights=self.observation_height,
            camera_widths=self.observation_width,
        )
        env.reset()
        self._env = env

    def render(self):
        self._ensure_env()
        raw_obs = self._env.env._get_observations()
        pixels = self._format_raw_obs(raw_obs)["pixels"]
        image = next(iter(pixels.values()))
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        assert self._env is not None, "_format_raw_obs called before _ensure_env()"
        images = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def reset(self, seed=None, **kwargs):
        self._ensure_env()
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self.init_state_id % len(self._init_states)])
            self.init_state_id += self._reset_stride  # Change init_state_id when reset

        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        assert self._env is not None
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def ik_obs_hook_class(self) -> type[LiberoEnv]:
        """Return the env class that implements batched IK observation helpers for ``rollout()``."""
        return type(self)

    @staticmethod
    def ee_poses_from_observation(
        observation: dict[str, Any],
        mask: Sequence[bool],
    ) -> list[np.ndarray]:
        """Extract pre-step 6-D EE poses (pos + axis-angle) for IK rows from the env obs batch."""
        from hybrid_eval.connectors.action_format import ee_pose_6d

        robot_state_key = f"{OBS_STR}.robot_state"
        if robot_state_key in observation:
            import robosuite.utils.transform_utils as T

            robot_state = observation[robot_state_key]
            eef_pos = robot_state["eef"]["pos"]
            eef_quat = robot_state["eef"]["quat"]
            batch_size = int(eef_pos.shape[0]) if hasattr(eef_pos, "shape") else len(eef_pos)
            poses: list[np.ndarray] = []
            for i in range(batch_size):
                if not mask[i]:
                    poses.append(np.zeros(6, dtype=np.float64))
                    continue
                pos = (
                    eef_pos[i].detach().cpu().numpy()
                    if isinstance(eef_pos, torch.Tensor)
                    else np.asarray(eef_pos[i])
                )
                quat = (
                    eef_quat[i].detach().cpu().numpy()
                    if isinstance(eef_quat, torch.Tensor)
                    else np.asarray(eef_quat[i])
                )
                ee_ori = np.asarray(T.quat2axisangle(quat), dtype=np.float64).ravel()
                poses.append(ee_pose_6d(pos, ee_ori))
            return poses

        if OBS_STATE in observation:
            state = observation[OBS_STATE]
            batch_size = int(state.shape[0])
            poses = []
            for i in range(batch_size):
                if not mask[i]:
                    poses.append(np.zeros(6, dtype=np.float64))
                    continue
                row = (
                    state[i].detach().cpu().numpy()
                    if isinstance(state, torch.Tensor)
                    else np.asarray(state[i])
                )
                poses.append(np.asarray(row[:6], dtype=np.float64))
            return poses

        raise KeyError(
            f"observation must contain {robot_state_key!r} or {OBS_STATE!r} for IK pre-step EE poses"
        )

    @staticmethod
    def _patch_observation_row(
        observation: dict[str, np.ndarray],
        index: int,
        fresh: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Replace one batch row in a vector-env observation dict with a single-env observation."""
        if "pixels" in fresh and "pixels" in observation:
            for cam, img in fresh["pixels"].items():
                if cam in observation["pixels"]:
                    observation["pixels"][cam][index] = np.asarray(img)

        if "robot_state" in fresh and "robot_state" in observation:
            for group in ("eef", "gripper", "joints"):
                if group not in fresh["robot_state"] or group not in observation["robot_state"]:
                    continue
                for key, value in fresh["robot_state"][group].items():
                    if key in observation["robot_state"][group]:
                        observation["robot_state"][group][key][index] = np.asarray(value)
        return observation

    @staticmethod
    def patch_observation_after_ik(
        env: gym.vector.VectorEnv,
        observation: dict[str, np.ndarray],
        mask: Sequence[bool],
    ) -> dict[str, np.ndarray]:
        """Refresh batch rows that received post-step IK via worker formatted observations."""
        if not any(mask):
            return observation

        try:
            formatted_list = list(env.call("get_current_observation"))
        except (AttributeError, NotImplementedError):
            return observation

        for i, active in enumerate(mask):
            if active:
                observation = LiberoEnv._patch_observation_row(observation, i, formatted_list[i])
        return observation

    def get_current_observation(self) -> dict[str, Any]:
        """
        Return formatted observation (same structure as ``step``/``reset``) without stepping
        but following forced update of environment observables.
        """
        self._ensure_env()
        assert self._env is not None
        self._env._update_observables(force=True)
        raw_obs = dict(self._env.env._get_observations())
        return self._format_raw_obs(raw_obs)

    def execute_ik(self, target: Any, current_ee_pose: np.ndarray) -> None:
        """Run IK teleport MP execution inside the worker (AsyncVectorEnv-safe)."""
        from hybrid_eval.execution.ik_pose_setter import IkPoseSetterMpExecutor

        executor = IkPoseSetterMpExecutor()
        executor.execute(
            target,
            current_ee_pose,
            context={"replay_env": self},
        )

    def execute_ik_indexed(
        self,
        targets: Sequence[Any | None],
        poses: Sequence[np.ndarray],
        mask: Sequence[bool],
    ) -> None:
        """Worker-local IK dispatch for batched ``VectorEnv.call`` (uses ``episode_index``)."""
        idx = int(self.episode_index)
        if idx < 0 or idx >= len(mask):
            return
        if not mask[idx] or targets[idx] is None:
            return
        self.execute_ik(targets[idx], np.asarray(poses[idx], dtype=np.float64))

    def close(self):
        if self._env is not None:
            self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks
    task_names_filter = gym_kwargs.pop("task_names", None)

    camera_names = parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None or task_names_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter} task_names={task_names_filter}")

    is_async = env_cls is gym.vector.AsyncVectorEnv

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        selected = resolve_task_ids(
            suite,
            task_ids=task_ids_filter,
            task_names=task_names_filter,
        )

        # All tasks in a suite share identical observation/action spaces.
        # Probe once and reuse to avoid creating a temp env per task.
        cached_obs_space: spaces.Space | None = None
        cached_act_space: spaces.Space | None = None
        cached_metadata: dict[str, Any] | None = None

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[suite_name][tid] = lazy
            else:
                out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    return {suite: dict(task_map) for suite, task_map in out.items()}
