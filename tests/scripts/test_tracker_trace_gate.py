from lerobot.configs.default import EvalConfig
from lerobot.scripts.lerobot_eval import _tracker_trace_row_count


def _n(**kw):
    base = dict(
        write_tracker_trace_plots=True,
        max_episodes_rendered=10,
        n_episodes_rendered=0,
        num_envs=5,
        uses_ompl=True,
    )
    base.update(kw)
    return _tracker_trace_row_count(**base)


def test_default_config_writes_plots():
    assert EvalConfig().write_tracker_trace_plots is True


def test_enabled_collects_as_before():
    assert _n() == 5
    assert _n(n_episodes_rendered=8) == 2
    assert _n(n_episodes_rendered=10) == 0


def test_disabled_collects_nothing():
    assert _n(write_tracker_trace_plots=False) == 0


def test_no_rendering_or_non_ompl_collects_nothing():
    assert _n(max_episodes_rendered=0) == 0
    assert _n(write_tracker_trace_plots=False, max_episodes_rendered=0) == 0
    assert _n(uses_ompl=False) == 0
