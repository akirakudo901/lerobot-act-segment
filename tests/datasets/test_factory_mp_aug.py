# IMPLEMENTED BY akirakudo901 for the hybrid-motion-planner project
# see: https://github.com/akirakudo901/lerobot-act-segment

from types import SimpleNamespace

import pytest

from lerobot.datasets.factory import _mp_aug_ready_action_window_size


def test_mp_aug_ready_action_window_prefers_chunk_size():
    cfg = SimpleNamespace(chunk_size=100, horizon=40)
    assert _mp_aug_ready_action_window_size(cfg) == 100


def test_mp_aug_ready_action_window_falls_back_to_horizon():
    cfg = SimpleNamespace(horizon=40)
    assert _mp_aug_ready_action_window_size(cfg) == 40


def test_mp_aug_ready_action_window_requires_chunk_or_horizon():
    with pytest.raises(ValueError, match="chunk_size"):
        _mp_aug_ready_action_window_size(SimpleNamespace())
