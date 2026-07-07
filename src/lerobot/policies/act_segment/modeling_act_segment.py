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

"""ACT with a per-chunk-step segment label classification head."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.envs.libero import get_libero_dummy_action
from lerobot.utils.constants import ACTION, OBS_IMAGES

from ..act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from .configuration_act_segment import ACTSegmentConfig

if TYPE_CHECKING:
    from dataset.core.mp_action_rescaling import MpActionRescalingRolloutContext
    from hybrid_eval.connectors.planning_target import PlanningTarget
    from hybrid_eval.protocols import MpSegmentConnector

_UNSET_MP_RESCALING_CTX = object()


@dataclass
class IkPending:
    """Batched IK targets stashed during ``select_action`` for the rollout hook.

    Each ``PlanningTarget.action`` is in environment-ready space when
    :meth:`ACTSegmentPolicy.set_rollout_action_processors` has been configured
    (unnormalized and MP-inverse-rescaled when applicable).
    """

    targets: list[PlanningTarget | None]
    mask: list[bool]


@dataclass(frozen=True)
class HybridStepTelemetry:
    """Per-env hybrid routing metadata from the last ``select_action`` call."""

    frame_label: int
    frame_label_str: str
    output_frame_index: int
    action_source: str
    is_new_chunk: bool
    chunk_anchor_step: int


@dataclass(frozen=True)
class HybridChunkTelemetry:
    """Completed policy chunk metadata for live-eval visualization."""

    row: int
    anchor_step: int
    orchestrator_output: Any


class ACTSegment(ACT):
    """ACT decoder extended with a linear label head on decoder tokens."""

    def __init__(self, config: ACTSegmentConfig):
        super().__init__(config)
        self.label_head = nn.Linear(config.dim_model, config.num_label_classes)

    def forward(
        self, batch: dict[str, Tensor], sample_encoded_dist: bool = False
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        actions, vae_params, decoder_out = self._forward_from_batch(batch, sample_encoded_dist)
        labels_logits = self.label_head(decoder_out)
        return actions, labels_logits, vae_params


class ACTSegmentPolicy(ACTPolicy):
    """ACT policy with auxiliary BIO segment-label cross-entropy loss."""

    config_class = ACTSegmentConfig
    name = "act_segment"

    def __init__(self, config: ACTSegmentConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTSegment(config)

        if config.use_hybrid_orchestrator and config.temporal_ensemble_coeff is not None:
            raise ValueError(
                "use_hybrid_orchestrator is incompatible with temporal_ensemble_coeff"
            )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self._connector: MpSegmentConnector | None = None
        if config.use_hybrid_orchestrator:
            self._connector = self._make_hybrid_connector()

        dataset_meta = kwargs.get("dataset_meta")
        dataset_root = getattr(dataset_meta, "root", None) if dataset_meta is not None else None
        self._rollout_postprocessor: Any | None = None
        self._mp_rescaling_ctx: MpActionRescalingRolloutContext | None = self._init_mp_rescaling_context(
            dataset_root=dataset_root,
        )
        self.reset()

    def _make_hybrid_connector(self) -> MpSegmentConnector:
        from hybrid_eval.connectors import MpLabeledFramesConnector, SegmentEndpointsConnector

        name = self.config.hybrid_connector
        if name == "mp_labeled_frames":
            return MpLabeledFramesConnector()
        if name == "segment_endpoints":
            return SegmentEndpointsConnector()
        raise ValueError(
            f"Unknown hybrid_connector {name!r}; expected 'mp_labeled_frames' or 'segment_endpoints'"
        )

    def _init_mp_rescaling_context(
        self,
        *,
        dataset_root: Path | str | None = None,
    ) -> MpActionRescalingRolloutContext | None:
        """Resolve the MP inverse-rescaling registry tied to this policy's training dataset."""
        from dataset.core.mp_action_rescaling import resolve_mp_action_rescaling_context

        pretrained_path = Path(self.config.pretrained_path) if self.config.pretrained_path else None
        return resolve_mp_action_rescaling_context(
            registry_path=self.config.mp_action_rescaling_registry_path,
            strategy_name=self.config.mp_action_rescaling_strategy,
            dataset_root=dataset_root,
            pretrained_path=pretrained_path,
        )

    def reset(self):
        """Clear ACT queues and hybrid orchestrator chunk state."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

        self._select_action_batchsize: int = None
        self._chunk_actions: Tensor | None = None
        self._chunk_labels: Tensor | None = None
        self._chunk_t: list[int] = []
        self._chunk_horizons: list[int] = []
        self._targets_by_frame: list[dict[int, PlanningTarget]] = []
        self._executed_target_ids: list[set[int]] = []
        self._ik_pending: IkPending | None = None
        self._chunk_anchor_steps: list[int] = []
        self._rollout_step: int = 0
        self._last_step_telemetry: list[HybridStepTelemetry | None] = []
        self._completed_chunks: list[HybridChunkTelemetry] = []
        self._last_step_ik_mask: list[bool] = []

    def set_rollout_action_processors(
        self,
        postprocessor: Any | None,
        *,
        mp_rescaling_ctx: Any | None = _UNSET_MP_RESCALING_CTX,
    ) -> None:
        """Attach eval-time action postprocessing used inside :meth:`select_action`.

        MP rescaling context is resolved once at policy construction from the training
        dataset registry. Pass ``mp_rescaling_ctx`` only when overriding that default.
        """
        self._rollout_postprocessor = postprocessor
        if mp_rescaling_ctx is not _UNSET_MP_RESCALING_CTX:
            self._mp_rescaling_ctx = mp_rescaling_ctx

    def _mp_rescaling_keys_for_batch(
        self,
        batch: dict[str, Tensor],
        batch_size: int,
    ) -> list[str]:
        """Return ``task`` keys for MP rescaling registry lookup."""
        if "task" not in batch:
            raise KeyError("Key 'task' missing from batch in _mp_rescaling_keys_for_batch")
        task = batch["task"]
       
        if isinstance(task, (list, tuple)):
            tasks = [str(t) for t in task]
            if len(tasks) < batch_size:
                tasks.extend([""] * (batch_size - len(tasks)))
            return tasks[:batch_size]
        return [str(task)] * batch_size

    def _frame_labels_for_batch(self, batch_size: int) -> list[int | None]:
        labels: list[int | None] = [
            int(t.frame_label) if t is not None else None for t in self._last_step_telemetry
        ]
        if len(labels) < batch_size:
            labels.extend([None] * (batch_size - len(labels)))
        return labels[:batch_size]

    def _finalize_rollout_action_tensor(
        self,
        action: Tensor,
        *,
        mp_rescaling_keys: Sequence[str],
        frame_labels: Sequence[int | None],
    ) -> Tensor:
        if self._rollout_postprocessor is None and self._mp_rescaling_ctx is None:
            return action

        finalized = action
        device = finalized.device
        if self._rollout_postprocessor is not None:
            finalized = self._rollout_postprocessor(finalized)

        if self._mp_rescaling_ctx is not None:
            from hybrid_eval.eval.mp_action_rescaling_rollout import (
                apply_mp_action_rescaling_to_actions,
            )

            action_np = finalized.detach().cpu().numpy().astype("float64", copy=False)
            action_np = apply_mp_action_rescaling_to_actions(
                action_np,
                mp_rescaling_keys,
                frame_labels,
                self._mp_rescaling_ctx,
            )
            finalized = torch.as_tensor(action_np, dtype=torch.float32, device=device)

        return finalized

    def _finalize_ik_pending(
        self,
        *,
        mp_rescaling_keys: Sequence[str],
        frame_labels: Sequence[int | None],
    ) -> None:
        if self._ik_pending is None:
            return
        if self._rollout_postprocessor is None and self._mp_rescaling_ctx is None:
            return

        from dataclasses import replace

        from hybrid_eval.connectors.planning_target import PlanningTarget
        from hybrid_eval.eval.mp_action_rescaling_rollout import (
            apply_mp_action_rescaling_to_actions,
        )

        finalized_targets: list[PlanningTarget | None] = []
        for row, target in enumerate(self._ik_pending.targets):
            if target is None or not self._ik_pending.mask[row]:
                finalized_targets.append(target)
                continue

            action_t = torch.as_tensor(target.action, dtype=torch.float32).unsqueeze(0)
            if self._rollout_postprocessor is not None:
                action_t = self._rollout_postprocessor(action_t)
            action_np = action_t.squeeze(0).detach().cpu().numpy().astype("float64", copy=False)

            if self._mp_rescaling_ctx is not None:
                label = frame_labels[row] if row < len(frame_labels) else None
                task = mp_rescaling_keys[row] if row < len(mp_rescaling_keys) else ""
                action_np = apply_mp_action_rescaling_to_actions(
                    action_np.reshape(1, -1),
                    [task],
                    [label],
                    self._mp_rescaling_ctx,
                )[0]

            finalized_targets.append(replace(target, action=action_np))

        self._ik_pending.targets = finalized_targets

    def _finalize_select_action_output(
        self,
        batch: dict[str, Tensor],
        action: Tensor,
    ) -> Tensor:
        batch_size = int(action.shape[0])
        mp_rescaling_keys = self._mp_rescaling_keys_for_batch(batch, batch_size)
        frame_labels = self._frame_labels_for_batch(batch_size)
        finalized = action.clone()

        policy_rows = [
            row
            for row in range(batch_size)
            if row >= len(self._last_step_ik_mask) or not self._last_step_ik_mask[row]
        ]
        if policy_rows:
            row_index = torch.tensor(policy_rows, dtype=torch.long, device=action.device)
            subset = self._finalize_rollout_action_tensor(
                finalized[row_index],
                mp_rescaling_keys=[mp_rescaling_keys[row] for row in policy_rows],
                frame_labels=[frame_labels[row] for row in policy_rows],
            )
            finalized[row_index] = subset

        self._finalize_ik_pending(
            mp_rescaling_keys=mp_rescaling_keys,
            frame_labels=frame_labels,
        )
        return finalized

    def set_rollout_step(self, step: int) -> None:
        """Set the current episode step index (used for chunk anchor bookkeeping)."""
        self._rollout_step = int(step)

    def consume_ik_pending(self) -> IkPending | None:
        """Return and clear IK targets from the last ``select_action`` call."""
        pending = self._ik_pending
        self._ik_pending = None
        if pending is None or not any(pending.mask):
            return None
        return pending

    def consume_hybrid_step_telemetry(self) -> list[HybridStepTelemetry | None]:
        """Return and clear per-row telemetry from the last ``select_action`` call."""
        telemetry = self._last_step_telemetry
        self._last_step_telemetry = []
        return telemetry

    def pop_completed_chunks(self) -> list[HybridChunkTelemetry]:
        """Return and clear policy chunks completed since the last pop."""
        completed = self._completed_chunks
        self._completed_chunks = []
        return completed

    def finalize_rollout_chunks(self) -> list[HybridChunkTelemetry]:
        """Emit any in-progress chunks at episode end (call before ``reset``)."""
        if self._chunk_actions is None:
            return []
        from hybrid_eval.orchestrator import HybridPolicyOutput

        batch_size = self._select_action_batchsize
        finalized: list[HybridChunkTelemetry] = []
        for row in range(batch_size):
            if self._chunk_t[row] <= 0:
                continue
            finalized.append(
                HybridChunkTelemetry(
                    row=row,
                    anchor_step=int(self._chunk_anchor_steps[row]),
                    orchestrator_output=self._snapshot_chunk_output(row),
                )
            )
        self._completed_chunks.extend(finalized)
        return self.pop_completed_chunks()

    def _snapshot_chunk_output(self, row: int) -> Any:
        from dataset.core.frame_labels import FrameLabelEnum
        from hybrid_eval.orchestrator import HybridPolicyOutput

        assert self._chunk_actions is not None
        assert self._chunk_labels is not None
        horizon = int(self._chunk_horizons[row])
        actions_np = self._chunk_actions[row, :horizon].detach().cpu().numpy()
        labels_np = self._chunk_labels[row, :horizon].detach().cpu().numpy()
        labels_str = tuple(FrameLabelEnum.to_str(int(v)) for v in labels_np)
        targets = tuple(self._targets_by_frame[row].values())
        return HybridPolicyOutput(
            actions=actions_np,
            frame_labels=labels_np,
            frame_labels_str=labels_str,
            planning_targets=targets,
        )

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    def _label_targets(self, batch: dict[str, Tensor]) -> Tensor:
        labels = batch[self.config.label_feature_key]
        if labels.ndim == 3:
            labels = labels.squeeze(-1)
        return labels.long()

    def _label_valid_mask(self, batch: dict[str, Tensor]) -> Tensor:
        pad_key = f"{self.config.label_feature_key}_is_pad"
        if pad_key in batch:
            return ~batch[pad_key]
        return ~batch["action_is_pad"]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        actions, _labels_logits, _vae_params = self.model(batch)
        return actions

    @torch.no_grad()
    def predict_label_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Return argmax segment labels for each step in the predicted chunk."""
        self.eval()
        batch = self._prepare_batch(batch)
        _actions, labels_logits, _vae_params = self.model(batch)
        return labels_logits.argmax(dim=-1)
    
    @torch.no_grad()
    def predict_action_label_chunk(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Return both the actions and argmax segment labels for each step in the predicted chunk."""
        self.eval()
        batch = self._prepare_batch(batch)
        actions, labels_logits, _vae_params = self.model(batch)
        return actions, labels_logits.argmax(dim=-1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select one action per env; hybrid orchestrator routes MP triggers to dummy IK steps."""
        if not self.config.use_hybrid_orchestrator:
            action = super().select_action(batch)
        else:
            # Extract the batch size from the 'batch' dict, assuming any tensor value
            first_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
            actual_batch_size = first_tensor.shape[0]

            # we fix a batch size for ```select_action``` that is fixed until policy is reset
            if self._select_action_batchsize is None:
                self._select_action_batchsize = actual_batch_size
            elif self._select_action_batchsize != actual_batch_size:
                raise ValueError(
                    f"Batch size mismatch in select_action: expected {self._select_action_batchsize}, "
                    f"but got {actual_batch_size}. Batch size must remain fixed across calls until reset."
                )

            action = self._select_action_hybrid(batch)

        if self._rollout_postprocessor is None and self._mp_rescaling_ctx is None:
            return action
        return self._finalize_select_action_output(batch, action)

    def _effective_chunk_horizon(self, labels_row: Tensor) -> int:
        from dataset.core.frame_labels import FrameLabelEnum

        for t in range(labels_row.shape[0]):
            if FrameLabelEnum.is_mp_frame_label(int(labels_row[t].item())):
                return t + 1
        return self.config.n_action_steps

    def _refill_hybrid_rows(self, batch: dict[str, Tensor], rows: Sequence[int]) -> None:
        """
        Recompute (refill) action and label chunk rows for the hybrid orchestrator.
        
        If called while `self._chunk_actions` is None, `rows` must cover the entire batch (0..batch_size-1), 
        otherwise throws a ValueError. This ensures we fill the full set of per-row memory.
        """
        assert self._connector is not None
        from hybrid_eval.eval.hybrid_rollout import group_targets_by_execution_frame

        # Remove duplicates and sort rows ascending
        rows = sorted(set(rows))
        batch_size = self._select_action_batchsize

        # Check for full batch coverage on first call if chunk actions are not yet initialized
        if self._chunk_actions is None and rows != list(range(batch_size)):
            raise ValueError(
                "On first call to _refill_hybrid_rows, rows must be the full batch range "
                f"(0..{batch_size - 1}) but got {rows}."
            )

        # Only run prediction for the requested rows, efficiently batching if possible
        if len(rows) == 0:
            return

        # Build mini-batch for inference
        # Check that values are tensors before indexing
        per_row_batch = {
            k: (v[rows] if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        actions, labels = self.predict_action_label_chunk(per_row_batch)
        actions = actions[:, : self.config.n_action_steps]
        labels = labels[:, : self.config.n_action_steps]

        # Initialize storage structures if needed
        if self._chunk_actions is None:
            self._chunk_actions = actions.clone()
            self._chunk_labels = labels.clone()
            self._targets_by_frame = [{} for _ in range(batch_size)]
            self._executed_target_ids = [set() for _ in range(batch_size)]
            self._chunk_t = [0] * batch_size
            self._chunk_horizons = [self.config.n_action_steps] * batch_size
            self._chunk_anchor_steps = [self._rollout_step] * batch_size

        # Now update only requested rows with fresh chunk
        for i, row in enumerate(rows):
            if self._chunk_t[row] > 0:
                self._completed_chunks.append(
                    HybridChunkTelemetry(
                        row=row,
                        anchor_step=int(self._chunk_anchor_steps[row]),
                        orchestrator_output=self._snapshot_chunk_output(row),
                    )
                )
            horizon = (
                self._effective_chunk_horizon(labels[i])
                if self.config.hybrid_refill_mode == "until_first_mp"
                else self.config.n_action_steps
            )
            self._chunk_actions[row] = actions[i]
            self._chunk_labels[row] = labels[i]
            self._chunk_horizons[row] = horizon
            self._chunk_t[row] = 0
            self._chunk_anchor_steps[row] = self._rollout_step

            # Connector copies policy-output actions into PlanningTarget; unnormalization
            # and MP inverse rescaling happen in select_action when rollout processors are set.
            actions_np = actions[i].detach().cpu().numpy()
            labels_np = labels[i].detach().cpu().numpy()
            targets = self._connector.planning_targets(actions_np, labels_np)
            grouped = group_targets_by_execution_frame(targets)
            self._targets_by_frame[row] = {frame: target for frame, target in grouped.items() if frame < horizon}
            self._executed_target_ids[row] = set()

    @torch.no_grad()
    def _select_action_hybrid(self, batch: dict[str, Tensor]) -> Tensor:
        from dataset.core.frame_labels import FrameLabelEnum

        batch_size = self._select_action_batchsize

        if self._chunk_actions is None:
            rows_to_refill = list(range(batch_size))
        else:
            rows_to_refill = [
                row for row in range(batch_size) if self._chunk_t[row] >= self._chunk_horizons[row]
            ]
        if rows_to_refill:
            self._refill_hybrid_rows(batch, rows_to_refill)

        assert self._chunk_actions is not None
        assert self._chunk_labels is not None

        batch_size = self._select_action_batchsize
        action_dim = self._chunk_actions.shape[-1]
        device = self._chunk_actions.device
        dtype = self._chunk_actions.dtype
        dummy = torch.tensor(get_libero_dummy_action(), device=device, dtype=dtype)

        actions_out = torch.empty(batch_size, action_dim, device=device, dtype=dtype)
        ik_targets: list[Any | None] = [None] * batch_size
        ik_mask = [False] * batch_size
        step_telemetry: list[HybridStepTelemetry | None] = [None] * batch_size

        for row in range(batch_size):
            chunk_t = self._chunk_t[row]
            target = self._targets_by_frame[row].get(chunk_t)
            label = int(self._chunk_labels[row, chunk_t].item())
            action_source = "none"

            if target is not None and id(target) not in self._executed_target_ids[row]:
                if self.config.mp_executor_type == "ik_pose_setter":
                    # Dummy OSC action for this step; the real MP motion comes from
                    # ``IkPending`` targets finalized inside :meth:`select_action`.
                    actions_out[row] = dummy
                    ik_targets[row] = target
                    ik_mask[row] = True
                    action_source = "mp"
                    self._executed_target_ids[row].add(id(target))
                else:
                    raise NotImplementedError(
                        f"mp_executor_type {self.config.mp_executor_type!r} is not supported yet"
                    )
            elif FrameLabelEnum.is_l_frame_label(label) or FrameLabelEnum.is_mp_frame_label(label):
                actions_out[row] = self._chunk_actions[row, chunk_t]
                action_source = "policy"
            else:
                raise ValueError(f"unknown frame label {label!r} at chunk index {chunk_t}")

            step_telemetry[row] = HybridStepTelemetry(
                frame_label=label,
                frame_label_str=FrameLabelEnum.to_str(label),
                output_frame_index=chunk_t,
                action_source=action_source,
                is_new_chunk=chunk_t == 0,
                chunk_anchor_step=int(self._chunk_anchor_steps[row]),
            )
            self._chunk_t[row] += 1

        self._ik_pending = IkPending(targets=ik_targets, mask=ik_mask)
        self._last_step_telemetry = step_telemetry
        self._last_step_ik_mask = list(ik_mask)
        return actions_out

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, dict]:
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, labels_logits, (mu_hat, log_sigma_x2_hat) = self.model(batch, sample_encoded_dist)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        action_valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid_actions = action_valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * action_valid_mask).sum() / num_valid_actions.clamp_min(1)

        label_targets = self._label_targets(batch)
        label_valid_mask = self._label_valid_mask(batch)
        per_step_ce = F.cross_entropy(
            labels_logits.reshape(-1, self.config.num_label_classes),
            label_targets.reshape(-1),
            reduction="none",
        ).view(labels_logits.shape[0], labels_logits.shape[1])
        num_valid_labels = label_valid_mask.sum().clamp_min(1)
        label_ce_loss = (per_step_ce * label_valid_mask).sum() / num_valid_labels

        preds = labels_logits.argmax(dim=-1)
        label_accuracy = ((preds == label_targets) & label_valid_mask).sum().float() / num_valid_labels

        loss_dict = {
            "l1_loss": l1_loss.item(),
            "label_ce_loss": label_ce_loss.item(),
            "label_accuracy": label_accuracy.item(),
        }

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight + label_ce_loss * self.config.label_weight
        else:
            loss = l1_loss + label_ce_loss * self.config.label_weight

        return loss, loss_dict
