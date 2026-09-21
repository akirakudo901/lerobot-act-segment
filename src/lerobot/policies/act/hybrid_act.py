# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Shared hybrid action-ACT scaffolding for ``act_segment`` and ``act_segment_cond``.

Does not own a label head or label-encoder tokens. Subclasses supply ``_build_model``
and keep labels-as-output vs labels-as-input in their own ``forward`` / ``predict_*``.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_IMAGES

from ..pretrained import PreTrainedPolicy
from hybrid_eval.segment.losses import (
    label_valid_mask,
    masked_action_loss_mean,
    mp_l_action_masks,
)
from hybrid_eval.segment.policy_mixin import SegmentRolloutPolicyMixin
from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy, ACTTemporalEnsembler


class HybridACTPolicy(SegmentRolloutPolicyMixin, ACTPolicy):
    """Queues, hybrid ``select_action``, MP/L-weighted action L1, optional VAE KLD."""

    def __init__(self, config: ACTConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = self._build_model(config)

        if config.use_hybrid_orchestrator and config.temporal_ensemble_coeff is not None:
            raise ValueError(
                "use_hybrid_orchestrator is incompatible with temporal_ensemble_coeff"
            )

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self._init_segment_rollout(config, **kwargs)
        self.reset()

    def _build_model(self, config: ACTConfig) -> nn.Module:
        raise NotImplementedError

    def reset(self):
        """Clear ACT queues and hybrid orchestrator chunk state."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._segment_rollout.reset()

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return batch

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select one action per env; hybrid orchestrator routes MP/L when enabled."""
        if not self.config.use_hybrid_orchestrator:
            return super().select_action(batch)
        return self._segment_rollout.select_action(batch)

    def _weighted_action_l1(
        self, batch: dict[str, Tensor], actions_hat: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """MP/L-separated action L1: ``weighted = l_l1 + mp_l1_weight * mp_l1``."""
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
        return weighted_l1_loss, {
            "mp_l1_loss": mp_l1_loss.item(),
            "l_l1_loss": l_l1_loss.item(),
            "weighted_l1_loss": weighted_l1_loss.item(),
        }

    def _maybe_add_kld(
        self,
        loss: Tensor,
        loss_dict: dict[str, float],
        vae_params: tuple[Tensor, Tensor] | tuple[None, None],
    ) -> tuple[Tensor, dict[str, float]]:
        if not self.config.use_vae:
            return loss, loss_dict
        mu_hat, log_sigma_x2_hat = vae_params
        mean_kld = (
            (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
            .sum(-1)
            .mean()
        )
        loss_dict = {**loss_dict, "kld_loss": mean_kld.item()}
        return loss + mean_kld * self.config.kl_weight, loss_dict

    def _per_step_action_l1(
        self, batch: dict[str, Tensor], actions_hat: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return ``(action_l1 [B, T], valid_mask [B, T])`` for offline val."""
        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        action_valid_mask = ~batch["action_is_pad"]
        action_l1 = abs_err.mean(dim=-1)
        valid_labels = label_valid_mask(batch, self.config.label_feature_key)
        return action_l1, action_valid_mask & valid_labels
