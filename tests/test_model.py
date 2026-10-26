import numpy as np
from conftest import RATED

from wind_bayes_twin.model import (ModelSpec, anemo_stat, material, power_stats, slab_logpdf, slab_sample,
                                   yaw_factor)


def test_power_statistics(table):
    e = np.linspace(5, 15, 80)
    p_hat = table(e)
    full = e >= 13.0
    theta = np.zeros((4, 7))
    theta[1, 0] = np.log(0.95)  # efficiency
    theta[2, 4] = 0.10  # derating
    theta[3, 3] = 8.0  # yaw
    pw, rt = power_stats(theta, e, p_hat, full, table, RATED)
    assert np.allclose([pw[0], rt[0]], 0.0)
    assert np.allclose([pw[1], rt[1]], np.log(0.95))
    assert np.isclose(rt[2], np.log(0.9), atol=0.02) and pw[2] > rt[2]
    assert pw[3] < -0.015 and rt[3] > pw[3]  # yaw costs in partial load, little at rated
    assert np.isclose(yaw_factor(np.array([8.0]))[0] ** 3, np.cos(np.deg2rad(8.0)) ** 2)


def test_materiality():
    spec = ModelSpec()
    th = np.zeros((5, 7))
    th[1, 0] = -0.02
    th[2, 3] = -3.0
    th[3, 1] = 0.02
    th[4, 5] = 0.05
    m = material(th, spec)
    assert not m[0].any()
    assert m[1, 0] and m[2, 2] and m[3, 1] and not m[4, 4]


def test_slab_density_integrates_to_one():
    for slab in (("sym", 2.0), ("neg", 0.05), ("pos", 0.3)):
        lo, hi = (-12.0, 12.0) if slab[0] == "sym" else ((-0.4, 0.0) if slab[0] == "neg" else (0.0, 2.5))
        x = np.linspace(lo, hi, 200001)
        dens = np.exp(slab_logpdf(x, slab))
        assert np.isclose(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(x)), 1.0, atol=2e-3)
    rng = np.random.default_rng(0)
    assert (slab_sample(rng, ("pos", 0.1), 1000, floor=0.01) >= 0.01).all()


def test_anemometer_statistic():
    th = np.zeros((2, 7))
    th[1, 1], th[1, 2] = np.log(1.03), 0.0
    assert np.allclose(anemo_stat(th, 7.0), [0.0, np.log(1.03)])
