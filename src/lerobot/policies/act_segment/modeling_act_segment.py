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
from typing import Any, Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES

from ..act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from ..pretrained import PreTrainedPolicy
from ..segment.losses import (
    label_targets,
    label_valid_mask,
    masked_action_loss_mean,
    mp_l_action_masks,
    segment_label_ce,
)
from ..segment import rollout_wrapper as _segment_rollout_mod
from ..segment.rollout_wrapper import (
    HybridChunkTelemetry,
    HybridStepTelemetry,
    IkPending,
    OmplPlanRetriesExhausted,
    SegmentRolloutWrapper,
)
from .configuration_act_segment import ACTSegmentConfig

# Re-export hybrid telemetry types for callers that import from this module.
__all__ = [
    "ACTSegment",
    "ACTSegmentPolicy",
    "HybridChunkTelemetry",
    "HybridStepTelemetry",
    "IkPending",
    "OmplPlanRetriesExhausted",
]


class ACTSegment(ACT):
    """ACT decoder extended with a linear label head on decoder tokens."""

    def __init__(self, config: ACTSegmentConfig):
        super().__init__(config)
        self.label_head = nn.Linear(config.dim_model, config.num_label_classes)

    def forward(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool = False,
        sample_latent_prior: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        actions, vae_params, decoder_out = self._forward_from_batch(
            batch, sample_encoded_dist, sample_latent_prior=sample_latent_prior
        )
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

        dataset_meta = kwargs.get("dataset_meta")
        dataset_root = getattr(dataset_meta, "root", None) if dataset_meta is not None else None
        self._segment_rollout = SegmentRolloutWrapper(
            self,
            config,
            dataset_root=dataset_root,
            pretrained_path=config.pretrained_path,
        )
        self.reset()

    def __getattr__(self, name: str) -> Any:
        """Forward hybrid orchestrator state to the composed rollout wrapper.

        Lets eval hooks and existing tests keep reading ``policy._connector``,
        ``policy._chunk_t``, ``policy._ompl_trackers``, etc. Falls through to
        ``nn.Module.__getattr__`` for registered parameters / submodules.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name == "_segment_rollout":
            raise AttributeError(name)
        try:
            rollout = object.__getattribute__(self, "_segment_rollout")
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from exc
        if hasattr(rollout, name):
            return getattr(rollout, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def reset(self):
        """Clear ACT queues and hybrid orchestrator chunk state."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._segment_rollout.reset()

    def set_rollout_action_processors(
        self,
        postprocessor: Any | None,
        *,
        mp_rescaling_ctx: Any | None = _segment_rollout_mod._UNSET_MP_RESCALING_CTX,
    ) -> None:
        """Attach eval-time action postprocessing used inside :meth:`select_action`."""
        self._segment_rollout.set_rollout_action_processors(
            postprocessor, mp_rescaling_ctx=mp_rescaling_ctx
        )

    def set_dummy_action(self, action: Sequence[float] | None) -> None:
        """Override the no-op action used for ``ik_pose_setter`` MP trigger frames."""
        self._segment_rollout.set_dummy_action(action)

    def bind_eval_env(self, env: Any | None) -> None:
        """Associate this policy with a VectorEnv for hybrid MP (Layer-1 OMPL RPC)."""
        self._segment_rollout.bind_eval_env(env)

    def set_rollout_step(self, step: int) -> None:
        """Set the current episode step index (used for chunk anchor bookkeeping)."""
        self._segment_rollout.set_rollout_step(step)

    def consume_ik_pending(self) -> IkPending | None:
        """Return and clear IK targets from the last ``select_action`` call."""
        return self._segment_rollout.consume_ik_pending()

    def consume_hybrid_step_telemetry(self) -> list[HybridStepTelemetry | None]:
        """Return and clear per-row telemetry from the last ``select_action`` call."""
        return self._segment_rollout.consume_hybrid_step_telemetry()

    def pop_completed_chunks(self) -> list[HybridChunkTelemetry]:
        """Return and clear policy chunks completed since the last pop."""
        return self._segment_rollout.pop_completed_chunks()

    def finalize_rollout_chunks(self) -> list[HybridChunkTelemetry]:
        """Emit any in-progress chunks at episode end (call before ``reset``)."""
        return self._segment_rollout.finalize_rollout_chunks()

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

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
    def predict_action_label_chunk(
        self,
        batch: dict[str, Tensor],
        *,
        sample_latent_prior: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return both the actions and argmax segment labels for each step in the predicted chunk."""
        self.eval()
        batch = self._prepare_batch(batch)
        actions, labels_logits, _vae_params = self.model(
            batch, sample_latent_prior=sample_latent_prior
        )
        return actions, labels_logits.argmax(dim=-1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select one action per env; hybrid orchestrator routes MP triggers to dummy IK steps.

        Vanilla (non-hybrid) ACT leaves postprocessing to the outer eval loop.
        Hybrid mode finalizes inside :class:`SegmentRolloutWrapper`.
        """
        if not self.config.use_hybrid_orchestrator:
            return super().select_action(batch)
        
        return self._segment_rollout.select_action(batch)

    @torch.no_grad()
    def per_step_val_losses(
        self,
        batch: dict[str, Tensor],
        sample_encoded_dist: bool | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-chunk-step action L1 and label CE for offline segment validation.

        Returns:
            action_l1: ``[B, T]`` mean L1 over action dims per valid step.
            label_ce: ``[B, T]`` cross-entropy per valid step.
            valid_mask: ``[B, T]`` bool mask (action and label both valid).
        """
        batch = self._prepare_batch(batch)
        sample_encoded_dist = self._resolve_sample_encoded_dist(batch, sample_encoded_dist)
        actions_hat, labels_logits, _vae_params = self.model(batch, sample_encoded_dist)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        action_valid_mask = ~batch["action_is_pad"]
        action_l1 = abs_err.mean(dim=-1)

        targets = label_targets(batch, self.config.label_feature_key)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        per_step_ce = F.cross_entropy(
            labels_logits.reshape(-1, self.config.num_label_classes),
            targets.reshape(-1),
            reduction="none",
        ).view(labels_logits.shape[0], labels_logits.shape[1])

        valid_mask = action_valid_mask & valid_labels
        return action_l1, per_step_ce, valid_mask

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
        mp_action_mask, l_action_mask = mp_l_action_masks(
            batch,
            action_valid_mask,
            label_feature_key=self.config.label_feature_key,
        )
        mp_l1_loss = masked_action_loss_mean(abs_err, mp_action_mask)
        l_l1_loss = masked_action_loss_mean(abs_err, l_action_mask)
        weighted_l1_loss = l_l1_loss + self.config.mp_l1_weight * mp_l1_loss

        targets = label_targets(batch, self.config.label_feature_key)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        weighted_label_ce_loss, label_loss_dict = segment_label_ce(
            labels_logits,
            targets,
            valid_labels,
            mp_ce_weight=self.config.mp_ce_weight,
            l_ce_weight=self.config.l_ce_weight,
            num_label_classes=self.config.num_label_classes,
        )

        loss_dict = {
            "mp_l1_loss": mp_l1_loss.item(),
            "l_l1_loss": l_l1_loss.item(),
            "weighted_l1_loss": weighted_l1_loss.item(),
            **label_loss_dict,
        }

        if self.config.use_vae:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = weighted_l1_loss + mean_kld * self.config.kl_weight + weighted_label_ce_loss
        else:
            loss = weighted_l1_loss + weighted_label_ce_loss

        return loss, loss_dict
