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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.envs.libero import get_libero_dummy_action
from lerobot.utils.constants import ACTION, OBS_IMAGES

from ..act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from .configuration_act_segment import ACTSegmentConfig

if TYPE_CHECKING:
    from hybrid_eval.connectors.planning_target import PlanningTarget
    from hybrid_eval.protocols import MpSegmentConnector


@dataclass
class IkPending:
    """Batched IK targets stashed during ``select_action`` for the rollout hook."""

    targets: list[PlanningTarget | None]
    mask: list[bool]


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

        self.reset()

    def _make_hybrid_connector(self) -> MpSegmentConnector:
        from hybrid_eval.connectors import ConsecutiveMpConnector, SegmentEndpointsConnector

        name = self.config.hybrid_connector
        if name == "consecutive_mp":
            return ConsecutiveMpConnector()
        if name == "segment_endpoints":
            return SegmentEndpointsConnector()
        raise ValueError(
            f"Unknown hybrid_connector {name!r}; expected 'consecutive_mp' or 'segment_endpoints'"
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
        self._chunk_t = 0
        self._targets_by_frame: list[dict[int, PlanningTarget]] = []
        self._executed_target_ids: list[set[int]] = []
        self._ik_pending: IkPending | None = None

    def consume_ik_pending(self) -> IkPending | None:
        """Return and clear IK targets from the last ``select_action`` call."""
        pending = self._ik_pending
        self._ik_pending = None
        if pending is None or not any(pending.mask):
            return None
        return pending

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
            return super().select_action(batch)
        
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
     
        return self._select_action_hybrid(batch)

    def _refill_hybrid_chunk(self, batch: dict[str, Tensor]) -> None:
        assert self._connector is not None
        from hybrid_eval.eval.hybrid_rollout import group_targets_by_execution_frame

        actions, labels = self.predict_action_label_chunk(batch)
        actions = actions[:, : self.config.n_action_steps]
        labels = labels[:, : self.config.n_action_steps]
        self._chunk_actions = actions
        self._chunk_labels = labels
        self._chunk_t = 0
        self._targets_by_frame = []
        self._executed_target_ids = []

        batch_size = self._select_action_batchsize
        for row in range(batch_size):
            actions_np = actions[row].detach().cpu().numpy()
            labels_np = labels[row].detach().cpu().numpy()
            targets = self._connector.planning_targets(actions_np, labels_np)
            self._targets_by_frame.append(group_targets_by_execution_frame(targets))
            self._executed_target_ids.append(set())

    @torch.no_grad()
    def _select_action_hybrid(self, batch: dict[str, Tensor]) -> Tensor:
        from dataset.core.frame_labels import FrameLabelEnum

        if self._chunk_actions is None or self._chunk_t >= self.config.n_action_steps:
            self._refill_hybrid_chunk(batch)

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

        chunk_t = self._chunk_t
        for row in range(batch_size):
            target = self._targets_by_frame[row].get(chunk_t)
            label = int(self._chunk_labels[row, chunk_t].item())

            if target is not None and id(target) not in self._executed_target_ids[row]:
                if self.config.mp_executor_type == "ik_pose_setter":
                    actions_out[row] = dummy
                    ik_targets[row] = target
                    ik_mask[row] = True
                    self._executed_target_ids[row].add(id(target))
                else:
                    raise NotImplementedError(
                        f"mp_executor_type {self.config.mp_executor_type!r} is not supported yet"
                    )
            elif FrameLabelEnum.is_l_frame_label(label) or FrameLabelEnum.is_mp_frame_label(label):
                actions_out[row] = self._chunk_actions[row, chunk_t]
            else:
                raise ValueError(f"unknown frame label {label!r} at chunk index {chunk_t}")

        self._ik_pending = IkPending(targets=ik_targets, mask=ik_mask)
        self._chunk_t += 1
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
