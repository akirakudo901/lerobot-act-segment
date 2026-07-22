#!/usr/bin/env python

# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Evaluate every checkpoint in a training run and log metrics to Weights & Biases.

Discovers checkpoints under::

    <run-dir>/checkpoints/<step>/pretrained_model

For each checkpoint, runs :mod:`lerobot.scripts.lerobot_eval_hybrid_viz` and logs the
aggregated metrics from ``eval_info.json`` to a WandB run (typically the original
training run, resumed by ``--wandb-run-id``).

Usage
-----
::

    python -m lerobot.scripts.lerobot_eval_checkpoints \\
        --checkpoints-dir /path/to/.../checkpoints \\
        --output-dir ./eval_logs/batch \\
        --wandb-project my-project \\
        --wandb-run-id <training-run-id> \\
        -- \\
        --env.type=libero \\
        --env.task=libero_goal \\
        --env.task_names='["put_the_wine_bottle_on_top_of_the_cabinet"]' \\
        --eval.batch_size=5 \\
        --eval.n_episodes=10 \\
        --policy.n_action_steps=20 \\
        --policy.device=cuda \\
        --policy.observation_state_layout=efficient_libero \\
        --policy.use_hybrid_orchestrator=true
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lerobot.utils.constants import PRETRAINED_MODEL_DIR
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass(frozen=True)
class Checkpoint:
    step: int
    pretrained_path: Path


def resolve_checkpoints_dir(path: Path) -> Path:
    """Return the ``checkpoints`` directory from a run root or checkpoints path."""
    path = path.expanduser().resolve()
    if path.name == "checkpoints":
        return path
    nested = path / "checkpoints"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not find a checkpoints directory at {path} or {nested}. "
        "Pass the run directory or the checkpoints directory itself."
    )


def parse_checkpoint_step(name: str) -> int | None:
    if not name.isdigit():
        return None
    return int(name)


def discover_checkpoints(
    checkpoints_dir: Path,
    *,
    min_step: int | None = None,
    max_step: int | None = None,
    steps: set[int] | None = None,
) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        step = parse_checkpoint_step(child.name)
        if step is None:
            continue
        pretrained_path = child / PRETRAINED_MODEL_DIR
        if not pretrained_path.is_dir():
            logging.warning("Skipping %s: missing %s/", child, PRETRAINED_MODEL_DIR)
            continue
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if steps is not None and step not in steps:
            continue
        checkpoints.append(Checkpoint(step=step, pretrained_path=pretrained_path))
    checkpoints.sort(key=lambda ckpt: ckpt.step)
    return checkpoints


def flatten_eval_metrics(eval_info: dict) -> dict[str, float | int]:
    """Flatten eval_info.json into scalar metrics suitable for WandB logging."""
    metrics: dict[str, float | int] = {}

    def _add(prefix: str, values: dict) -> None:
        for key in ("pc_success", "avg_sum_reward", "avg_max_reward", "n_episodes", "eval_s", "eval_ep_s"):
            value = values.get(key)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
                metrics[f"{prefix}{key}"] = value

    # overall = eval_info.get("overall")
    # if isinstance(overall, dict):
    #     _add("", overall)

    # per_group = eval_info.get("per_group")
    # if isinstance(per_group, dict):
    #     for group_name, group_values in per_group.items():
    #         if isinstance(group_values, dict):
    #             _add(f"{group_name}/", group_values)

    per_task = eval_info.get("per_task")
    if isinstance(per_task, dict):
        for task_name, task_values in per_task.items():
            if isinstance(task_values, dict):
                safe_task = task_name.replace("/", "_")
                _add(f"task/{safe_task}/", task_values)

    return metrics


def run_hybrid_eval(
    *,
    pretrained_path: Path,
    output_dir: Path,
    eval_extra_args: list[str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval_hybrid_viz",
        f"--policy.path={pretrained_path}",
        f"--output_dir={output_dir}",
        *eval_extra_args,
    ]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    eval_info_path = output_dir / "eval_info.json"
    if not eval_info_path.exists():
        raise FileNotFoundError(f"Expected eval output at {eval_info_path}")
    return eval_info_path


def log_eval_to_wandb(
    wandb_module,
    *,
    step: int,
    eval_info: dict,
    checkpoint_path: Path,
    video_fps: int | None,
    log_video: bool,
) -> None:
    metrics = flatten_eval_metrics(eval_info)
    wandb_metrics = {f"eval/{key}": value for key, value in metrics.items()}
    wandb_metrics["eval/checkpoint_step"] = step
    wandb_metrics["eval/checkpoint_path"] = str(checkpoint_path)
    wandb_module.log(wandb_metrics, step=step)

    if not log_video:
        return

    overall = eval_info.get("overall", {})
    video_paths = overall.get("video_paths") if isinstance(overall, dict) else None
    if not video_paths:
        logging.warning("No eval videos found for checkpoint step %s", step)
        return

    video_path = video_paths[0]
    if not Path(video_path).exists():
        logging.warning("Eval video does not exist: %s", video_path)
        return

    wandb_module.log(
        {"eval/video": wandb_module.Video(video_path, fps=video_fps, format="mp4")},
        step=step,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Training run directory or its checkpoints/ subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base directory for per-checkpoint eval outputs and eval_info.json files.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lerobot",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB entity (team/user).",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Resume an existing WandB run (e.g. the original training run).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional WandB run name when creating a new run.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="WandB mode.",
    )
    parser.add_argument(
        "--wandb-notes",
        type=str,
        default=None,
        help="Optional WandB run notes.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step >= this value.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Only evaluate checkpoints with step <= this value.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate these checkpoint steps.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints whose output dir already contains eval_info.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue with remaining checkpoints if one eval fails (default: true).",
    )
    parser.add_argument(
        "--no-continue-on-error",
        action="store_false",
        dest="continue_on_error",
        help="Stop immediately when an eval fails.",
    )
    parser.add_argument(
        "--log-video",
        action="store_true",
        default=True,
        help="Upload the first eval rollout video to WandB (default: true).",
    )
    parser.add_argument(
        "--no-log-video",
        action="store_false",
        dest="log_video",
        help="Do not upload eval videos to WandB.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="FPS for uploaded WandB videos. Defaults to env fps if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered checkpoints without running eval or logging.",
    )
    parser.add_argument(
        "eval_extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to lerobot_eval_hybrid_viz after '--'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    register_third_party_plugins()
    init_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    eval_extra_args = list(args.eval_extra_args)
    if eval_extra_args and eval_extra_args[0] == "--":
        eval_extra_args = eval_extra_args[1:]

    checkpoints_dir = resolve_checkpoints_dir(args.checkpoints_dir)
    steps_filter = set(args.steps) if args.steps is not None else None
    checkpoints = discover_checkpoints(
        checkpoints_dir,
        min_step=args.min_step,
        max_step=args.max_step,
        steps=steps_filter,
    )
    if not checkpoints:
        logging.error("No checkpoints found under %s", checkpoints_dir)
        return 1

    logging.info("Discovered %d checkpoint(s) under %s", len(checkpoints), checkpoints_dir)
    for checkpoint in checkpoints:
        logging.info("  step=%06d  path=%s", checkpoint.step, checkpoint.pretrained_path)

    if args.dry_run:
        return 0

    os.environ["WANDB_SILENT"] = "True"
    import wandb

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        name=args.wandb_run_name,
        notes=args.wandb_notes,
        job_type="eval",
        resume="must" if args.wandb_run_id else None,
        mode=args.wandb_mode,
        dir=str(args.output_dir),
        config={
            "checkpoints_dir": str(checkpoints_dir),
            "eval_extra_args": eval_extra_args,
        },
    )

    summary: list[dict] = []
    failures: list[dict] = []

    try:
        for checkpoint in checkpoints:
            ckpt_output_dir = args.output_dir / f"checkpoint_{checkpoint.step:06d}"
            eval_info_path = ckpt_output_dir / "eval_info.json"

            logging.info("=== Evaluating checkpoint step %06d ===", checkpoint.step)
            try:
                if args.skip_existing and eval_info_path.exists():
                    logging.info("Skipping existing eval output: %s", eval_info_path)
                else:
                    eval_info_path = run_hybrid_eval(
                        pretrained_path=checkpoint.pretrained_path,
                        output_dir=ckpt_output_dir,
                        eval_extra_args=eval_extra_args,
                    )

                eval_info = json.loads(eval_info_path.read_text())
                log_eval_to_wandb(
                    wandb,
                    step=checkpoint.step,
                    eval_info=eval_info,
                    checkpoint_path=checkpoint.pretrained_path,
                    video_fps=args.video_fps,
                    log_video=args.log_video,
                )
                overall = eval_info.get("overall", {})
                summary.append(
                    {
                        "step": checkpoint.step,
                        "pretrained_path": str(checkpoint.pretrained_path),
                        "pc_success": overall.get("pc_success"),
                        "avg_sum_reward": overall.get("avg_sum_reward"),
                        "n_episodes": overall.get("n_episodes"),
                        "eval_info_path": str(eval_info_path),
                    }
                )
            except Exception as exc:
                logging.exception("Eval failed for checkpoint step %06d", checkpoint.step)
                failures.append(
                    {
                        "step": checkpoint.step,
                        "pretrained_path": str(checkpoint.pretrained_path),
                        "error": str(exc),
                    }
                )
                if not args.continue_on_error:
                    break
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "checkpoint_eval_summary.json").write_text(
            json.dumps({"completed": summary, "failures": failures}, indent=2)
        )
        wandb.finish()

    if failures and not summary:
        return 1
    if failures:
        logging.warning("%d checkpoint eval(s) failed", len(failures))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
