import numpy as np

from wind_bayes_twin.reference import (farm_curve, farm_deficit, leave_one_out_direction, leave_one_out_median,
                                       nacelle_offsets, power_from_wind, wind_from_power, wrap180)


def test_curve_and_inverse():
    rng = np.random.default_rng(0)
    ws = rng.uniform(2, 20, 20000)
    p = np.clip(2000 * ((ws - 3) / 9.5) ** 3, 0, 2000) + rng.normal(0, 10, ws.size)
    curve = farm_curve(ws[:, None], p[:, None], np.ones((ws.size, 1), dtype=bool))
    assert np.all(np.diff(curve["power_kw"]) >= 0)
    e = wind_from_power(np.array([500.0, 1000.0, 3000.0]), curve)
    assert np.isnan(e[2])
    assert np.allclose(power_from_wind(e[:2], curve), [500, 1000], rtol=0.02)


def test_directions_and_offsets():
    rng = np.random.default_rng(1)
    wd = rng.uniform(0, 360, 5000)
    offsets_true = np.array([2.0, -1.0, 0.5, -1.5])
    nac = (wd[:, None] + offsets_true[None, :] + rng.normal(0, 3, (5000, 4))) % 360
    valid = np.ones_like(nac, dtype=bool)
    off = nacelle_offsets(nac, valid)
    assert np.allclose(off, offsets_true - offsets_true.mean(), atol=0.3)
    theta = leave_one_out_direction(nac, valid, off)
    assert np.abs(wrap180(theta[:, 0] - wd)).mean() < 3


def test_leave_one_out_median_excludes_self():
    v = np.array([[1.0, 2.0, 100.0]])
    out = leave_one_out_median(v, np.ones_like(v, dtype=bool))
    assert np.allclose(out, [[51.0, 50.5, 1.5]])


def test_wake_superposition_and_stopped_turbines():
    tables = np.zeros((2, 2, 360, 45))
    tables[0, 1, 270, :] = 0.3  # turbine 0 wakes turbine 1 for wind from 270 deg
    theta = np.array([270.0, 90.0])
    ws = np.array([8.0, 8.0])
    d = farm_deficit(tables, theta, ws, np.ones((2, 2)))
    assert np.allclose(d, [[0, 0.3], [0, 0]])
    d_off = farm_deficit(tables, theta, ws, np.array([[0.0, 1.0], [1.0, 1.0]]))
    assert np.allclose(d_off[0], 0.0)
