# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

"""Shared ACT knobs for hybrid action policies (``act_segment`` / ``act_segment_cond``)."""

from dataclasses import dataclass


@dataclass
class ACTHybridConfigMixin:
    """Action-body fields shared by joint-head and label-conditioned ACT segment policies.

    Host config must also provide ``chunk_size`` (from :class:`~lerobot.policies.act.configuration_act.ACTConfig`).
    Hybrid / OMPL rollout fields stay on :class:`~hybrid_eval.segment.configuration_segment.SegmentPolicyConfigMixin`.
    Cond-only eval knobs (``label_source``, ``label_policy_path``) stay on ``ACTSegmentCondConfig``.
    """

    # Scales the MP execution-frame L1 term: weighted_l1 = l_l1_loss + mp_l1_weight * mp_l1_loss.
    mp_l1_weight: float = 1.0
    # On requery refill, sample ACT latent from N(0, I) instead of the deterministic zero vector.
    ompl_retry_sample_latent: bool = True

    # Reorder ``observation.state`` in the policy preprocessor to match the training dataset layout.
    # Default ``None``: no reordering. Set explicitly when eval env layout differs from training:
    # ``lerobot`` for datasets with ee_pos + ee_ori + gripper (no reorder step),
    # ``efficient_libero`` for legacy efficient exports (gripper + ee_pos + ee_ori).
    observation_state_layout: str | None = None

    @property
    def label_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))  # type: ignore[attr-defined]
