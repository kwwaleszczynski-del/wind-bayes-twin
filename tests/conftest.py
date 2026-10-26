"""Synthetic data generated from the twin's own model (no network, no Kelmarsh data)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_bayes_twin.model import CurveTable, power_stats
from wind_bayes_twin.stats import STATS
from wind_bayes_twin.twin_filter import DayInput, Noise

RATED = 2050.0


def synthetic_curve() -> pd.DataFrame:
    ws = np.arange(3.0, 25.01, 0.5)
    p = np.clip(RATED * ((ws - 3.0) / (12.5 - 3.0)) ** 3, 0.0, RATED)
    return pd.DataFrame({"ws": ws, "power_kw": p})


@pytest.fixture
def table() -> CurveTable:
    return CurveTable(synthetic_curve())


def make_noise(sigma_e=None, sigma_w=None, phi=None, mu=None) -> Noise:
    n = len(STATS)
    sigma_e = np.array([0.05, 0.012, 0.03, 0.025, 1.6, 0.15, 0.35]) if sigma_e is None else np.asarray(sigma_e)
    sigma_w = np.array([0.012, 0.003, 0.003, 0.003, 0.5, 0.03, 0.4]) if sigma_w is None else np.asarray(sigma_w)
    phi = np.full(n, 0.8) if phi is None else np.asarray(phi)
    mu = np.zeros(n) if mu is None else np.asarray(mu)
    return Noise(mu_mean=mu, mu_var=np.full(n, 1e-8), sigma_e=sigma_e, sigma_w=sigma_w, phi=phi,
                 c_ref=np.full(n, 100.0), level_walk=np.zeros(n))


@pytest.fixture
def noise() -> Noise:
    return make_noise()


def simulate(noise: Noise, theta_fn, n_days: int, table: CurveTable, seed: int = 0, rated_share: float = 0.4,
             r_spread: float = 0.0):
    """Daily inputs drawn from the twin's observation model with health parameters ``theta_fn(day)`` (7,).

    Records: 100 per day of expected equivalent wind speed 5-11 m/s, plus full-load records (13-15 m/s) on a
    ``rated_share`` of the days. Statistics = model prediction + healthy level + AR(1) wander + noise.
    """
    rng = np.random.default_rng(seed)
    n_s = len(STATS)
    n = np.zeros(n_s)
    days = []
    for d in range(n_days):
        e = rng.uniform(5.0, 11.0, 100)
        if rng.random() < rated_share:
            e = np.r_[e, rng.uniform(13.0, 15.0, 30)]
        p_hat = table(e)
        full = e >= 13.0
        th = np.asarray(theta_fn(d), dtype=float)[None, :]
        p_mod = np.minimum(np.exp(th[0, 0]) * table(e), np.inf if th[0, 4] <= 0 else (1 - th[0, 4]) * RATED)
        if r_spread > 0:  # record-level dispersion of the actual power
            p_mod = np.minimum(np.exp(th[0, 0]) * p_hat * rng.normal(1.0, r_spread, len(e)),
                               np.inf if th[0, 4] <= 0 else (1 - th[0, 4]) * RATED)
            pw = np.log(p_mod[~full].sum() / p_hat[~full].sum())
            rt = np.log(p_mod[full].sum() / p_hat[full].sum()) if full.any() else np.nan
        else:
            pw, rt = (v[0] for v in power_stats(th, e, p_hat, full, table, RATED))
        u_lo, u_hi = 5.5, 9.0
        h = np.array([pw, rt, np.log(np.exp(th[0, 1]) + th[0, 2] / u_lo), np.log(np.exp(th[0, 1]) + th[0, 2] / u_hi),
                      th[0, 3], th[0, 5] * 13.0, th[0, 6] * 19.0])
        n = noise.phi * n + rng.normal(0.0, noise.sigma_w)
        y = h + noise.mu_mean + n + rng.normal(0.0, noise.sigma_e)
        days.append(DayInput(y=y, count=np.full(n_s, 100.0), e_hat=e, p_hat=p_hat, full=full, u_lo=u_lo,
                             u_hi=u_hi, rise_gear=13.0, rise_gen=19.0))
    return days
