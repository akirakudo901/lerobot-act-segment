# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Protocol for policies that emit action + BIO label chunks for hybrid rollout."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class SegmentChunkPolicy(Protocol):
    """Minimal surface required by :class:`~lerobot.policies.segment.rollout_wrapper.SegmentRolloutWrapper`.

    Implementations must predict a joint action/label chunk.
    """

    def predict_action_label_chunk(
        self,
        batch: dict[str, Tensor],
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(actions, labels)`` with shapes ``(B, k, act_dim)`` and ``(B, k)``.

        Optional keyword arguments may include ``sample_latent_prior``. 
        Implementations should document additional accepted arguments.
        """
        ...
   