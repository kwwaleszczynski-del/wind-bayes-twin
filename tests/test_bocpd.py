import numpy as np

from wind_bayes_twin.bocpd import BOCPD


def test_finds_known_step():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 150), rng.normal(2.5, 1, 60)]
    cp = BOCPD(hazard=1 / 200)
    recent = []
    for v in x:
        cp.update(v)
        recent.append(cp.p_recent(7))
    recent = np.array(recent)
    assert recent[10:140].max() < 0.5  # quiet before the step (after the first days, when every run is short)
    assert recent[150:160].max() > 0.5  # change found within ten days
    assert abs(cp.map_run_length() - 60) <= 3
    assert abs(cp.recent_mean(80) - 2.5) < 0.5


def test_missing_values_are_skipped():
    cp = BOCPD()
    for v in (0.1, np.nan, -0.2):
        cp.update(v)
    assert np.isclose(cp.p.sum(), 1.0) and len(cp.p) == 3
