import numpy as np
import pandas as pd
import pytest

from wind_bayes_twin.farm import event_mask
from wind_bayes_twin.faults import SCENARIOS, Fault, yaw_power_factor
from wind_bayes_twin.stats import DayIndex


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_day_index_does_not_depend_on_datetime_resolution(unit):
    t = pd.date_range("2019-12-30 22:00", periods=600, freq="10min", tz="UTC").as_unit(unit)
    days = DayIndex(t)
    assert days.n_days == 6
    assert days.code[0] == 0 and days.code[-1] == 5
    assert days.of("2020-01-01") == 2 and days.of(pd.Timestamp("2020-01-01 13:00", tz="UTC")) == 2
    assert days.mask("2019-12-31", "2020-01-02").sum() == 2


@pytest.mark.parametrize("unit", ["s", "us", "ns"])
def test_event_mask_mixed_units(unit):
    t = pd.date_range("2020-01-01", periods=12, freq="10min", tz="UTC").as_unit(unit)
    starts = pd.DatetimeIndex(["2020-01-01 00:25"], tz="UTC").as_unit("ms")
    ends = pd.DatetimeIndex(["2020-01-01 00:45"], tz="UTC").as_unit("us")
    m = event_mask(t, starts, ends)
    assert m.tolist() == [False, False, True, True, True] + [False] * 7


def test_fault_truths():
    assert Fault("y", "yaw", 4.0, signed=True).truth(-1) == 4.0
    assert np.isclose(Fault("a", "anemometer", 0.03, signed=True).truth(1), np.log(1.03))
    assert np.isclose(Fault("e", "eff", 0.05).truth(), np.log(0.95))
    assert {f.fault_type for f in SCENARIOS} == {"yaw", "anemo", "eff", "derate", "gear", "gen"}


def test_yaw_power_factor():
    curve = pd.DataFrame({"ws": np.arange(3.0, 25.01, 0.5)})
    curve["power_kw"] = np.clip(2050 * (curve["ws"] / 12.0) ** 3, 0, 2050)  # pure cubic: factor = cos^2
    f = yaw_power_factor(np.zeros(3), np.array([7.0, 9.0, 18.0]), 8.0, curve)
    assert np.allclose(f[:2], np.cos(np.deg2rad(8.0)) ** 2, atol=0.01)
    assert np.isclose(f[2], 1.0)  # still at rated power in strong wind
