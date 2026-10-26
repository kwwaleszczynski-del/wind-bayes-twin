import numpy as np
from conftest import RATED

from wind_bayes_twin.attribution import Economics, decision_values, shapley_losses
from wind_bayes_twin.baselines import Charts, cusum, trailing_median
from wind_bayes_twin.model import turbine_power


def test_cusum_detects_low_shift_and_resets():
    z = np.r_[np.zeros(50), np.full(30, -2.0)]
    days, dirs, _ = cusum(z, 0.5, 5.0, "low")
    assert days[0] == 53 and dirs[0] == -1 and len(days) > 1


def test_trailing_median_is_causal():
    x = np.r_[np.zeros(200), np.ones(100)]
    tm = trailing_median(x)
    assert np.all(tm[120:231] == 0.0)  # the step only enters 31 days later
    assert tm[-1] == 1.0


def test_chart_calibration_has_no_alarm_on_calibration_days():
    rng = np.random.default_rng(0)
    y = {s: rng.normal(0, 0.02, (400, 3)) for s in ("pow", "rated", "anemo", "tgear", "tgen")}
    y["yaw"] = rng.normal(0, 1.5, (400, 3))
    cal = np.zeros(400, dtype=bool)
    cal[150:300] = True
    charts = Charts.calibrate(y, cal)
    for i in range(3):
        assert not [a for a in charts.run(y, i, 150, 300) if a[1] in ("pow", "rated", "anemo", "yaw")]


def test_shapley_split_adds_up(table):
    e = np.linspace(5, 15, 120)
    theta = np.zeros((3, 7))
    theta[0, 0] = np.log(0.97)
    theta[1, 4] = 0.05
    theta[2, [0, 3, 4]] = np.log(0.97), 8.0, 0.05
    single = shapley_losses(theta[:2], e, table, RATED)
    assert np.allclose(single[0, 1:], 0) and single[0, 0] > 0
    assert np.allclose(single[1, [0, 1]], 0) and single[1, 2] > 0
    loss = shapley_losses(theta[2:], e, table, RATED)
    healthy = np.minimum(table(e), RATED).sum() / 6
    total = healthy - turbine_power(theta[2:], e, table, RATED).sum(axis=1) / 6
    assert np.isclose(loss[0].sum(), total[0])


def test_value_of_information():
    eco = Economics(horizon_days=100, inspection_cost=100.0, fix_cost={"eff": 1000.0})
    certain = decision_values(np.full(1000, 20.0), "eff", eco)  # 2000 at stake, for sure
    assert np.isclose(certain["evpi"], 0.0) and np.isclose(certain["value_of_inspection"], 900.0)
    unsure = decision_values(np.r_[np.zeros(600), np.full(400, 50.0)], "eff", eco)
    assert unsure["evpi"] > 0  # information pays when the decision is uncertain
