"""First-order (RC) thermal model of a drivetrain temperature and its fleet-referenced daily residual.

Re-implemented compactly from the earlier project wind-thermal-nbm. The component is one thermal capacity behind
one thermal resistance R to what cools it; over a 10-minute step the temperature is an exponential filter of the
steady state, which is linear in the loss terms (P, P^2, rotor speed) and the cooling-side temperatures:

    T^[t] = a T^[t-1] + (1 - a) T_ss[t],   a = exp(-dt / tau)
    T_ss  = theta_0 + sum_k theta_k T_ref,k + theta_P P + theta_P2 P^2 + theta_w w

The model is simulated from its inputs only (it never sees the measured temperature). For a fixed tau it is linear
in theta, so the fit is a search over tau with least squares inside. Two channels are used by the twin:

    gear   gearbox rear bearing, cooled by the gear oil (plus nacelle and outside air)
    gen    generator rear bearing, cooled by the nacelle air

A higher thermal resistance (worse cooling, a damaged bearing) scales the temperature difference to the cooling
side, rise = T^ - T_cool (gear oil for the gearbox bearing, nacelle air for the generator bearing), by rho = R'/R.
The twin observes the RC residual referenced to the fleet (minus the median residual of the other producing
turbines), which cancels the model error the six identical turbines share.

Usage:
    rc = fit_rc(farm, "gear", farm.period(2017))
    sim = simulate(farm, rc)                     # {"t_hat": ..., "rise": ...} (time, turbine)
    cool = cooling_side(farm, "gear")            # measured temperature of what cools the component
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

STEP_MIN = 10.0
RATED_RPM = 15.0
CHANNELS = {
    "gear": ("gear_bearing", ("gear_oil", "nacelle_temp", "ambient_temp")),
    "gen": ("gen_bearing", ("nacelle_temp", "ambient_temp")),
}
PRODUCING_KW = 100.0
WARMUP = 36  # 6 hours after a gap
GAP_FILL = 3
log = logging.getLogger(__name__)


def segments(valid: np.ndarray) -> list[tuple[int, int]]:
    v = np.concatenate([[False], valid, [False]]).astype(np.int8)
    d = np.diff(v)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def exp_filter(x: np.ndarray, valid: np.ndarray, a: float) -> np.ndarray:
    """y[t] = a y[t-1] + (1-a) x[t] inside each valid segment, started in steady state."""
    y = np.full(x.shape, np.nan)
    for s, e in segments(valid):
        y[s:e] = lfilter([1.0 - a], [1.0, -a], x[s:e], axis=0, zi=a * x[s:s + 1])[0]
    return y


def warm(valid: np.ndarray, n: int = WARMUP) -> np.ndarray:
    out = np.zeros(len(valid), dtype=bool)
    for s, e in segments(valid):
        out[s + n:e] = True
    return out


def _fill(x: np.ndarray, limit: int = GAP_FILL) -> np.ndarray:
    """Linear interpolation over interior gaps of at most ``limit`` records."""
    x = x.copy()
    bad = np.isnan(x)
    if not bad.any() or bad.all():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~bad], x[~bad])
    run = np.cumsum(~bad)
    gap_len = np.bincount(run, weights=bad)[run]
    inside = (idx > np.flatnonzero(~bad)[0]) & (idx < np.flatnonzero(~bad)[-1])
    use = bad & (gap_len <= limit) & inside
    x[use] = filled[use]
    return x


def inputs(farm, channel: str, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(design matrix [1, refs..., P, P^2, w], valid rows) for turbine column ``i``."""
    _, refs = CHANNELS[channel]
    p = np.clip(_fill(farm.power[:, i]), 0.0, None) / 1000.0
    w = np.clip(_fill(farm.rotor_rpm[:, i]), 0.0, None) / RATED_RPM
    cols = [np.ones(len(p))] + [_fill(getattr(farm, r)[:, i]) for r in refs] + [p, p * p, w]
    x = np.column_stack(cols)
    return x, np.isfinite(x).all(axis=1)


@dataclass
class RC:
    channel: str
    tau_h: np.ndarray  # per turbine
    theta: np.ndarray  # (turbine, feature)
    rmse: np.ndarray


def _fit_one(x, valid, y, use) -> tuple[float, np.ndarray, float]:
    xv = np.where(valid[:, None], x, 0.0)

    def solve(log_tau):
        basis = exp_filter(xv, valid, float(np.exp(-STEP_MIN / 60.0 / np.exp(log_tau))))[use]
        theta = np.linalg.lstsq(basis, y[use], rcond=None)[0]
        return float(np.mean((basis @ theta - y[use]) ** 2)), theta

    grid = np.linspace(np.log(0.05), np.log(12.0), 15)
    k = int(np.argmin([solve(g)[0] for g in grid]))
    res = minimize_scalar(lambda g: solve(g)[0], bounds=(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]),
                          method="bounded", options={"xatol": 1e-3})
    mse, theta = solve(res.x)
    return float(np.exp(res.x)), theta, float(np.sqrt(mse))


def fit_rc(farm, channel: str, rows: np.ndarray) -> RC:
    target = CHANNELS[channel][0]
    taus, thetas, rmses = [], [], []
    for i in range(farm.n):
        x, valid = inputs(farm, channel, i)
        y = getattr(farm, target)[:, i]
        use = warm(valid) & np.isfinite(y) & rows
        tau, theta, rmse = _fit_one(x, valid, y, use)
        taus.append(tau), thetas.append(theta), rmses.append(rmse)
        log.info("RC %s T%d: tau %.2f h, rmse %.2f K", channel, farm.turbines[i], tau, rmse)
    return RC(channel, np.array(taus), np.array(thetas), np.array(rmses))


def cooling_side(farm, channel: str) -> np.ndarray:
    """Measured temperature of the medium that cools the channel's component (its first reference)."""
    return getattr(farm, CHANNELS[channel][1][0])


def simulate(farm, rc: RC) -> dict[str, np.ndarray]:
    """Simulated temperature t_hat and its rise above the cooling side (both (time, turbine), NaN in warm-up)."""
    shape = farm.power.shape
    t_hat = np.full(shape, np.nan)
    for i in range(farm.n):
        x, valid = inputs(farm, rc.channel, i)
        a = float(np.exp(-STEP_MIN / 60.0 / rc.tau_h[i]))
        ok = warm(valid)
        t_hat[ok, i] = exp_filter(np.where(valid[:, None], x, 0.0) @ rc.theta[i], valid, a)[ok]
    return {"t_hat": t_hat, "rise": t_hat - cooling_side(farm, rc.channel)}


def fleet_residual(measured: np.ndarray, t_hat: np.ndarray, producing: np.ndarray, min_others: int = 3) -> np.ndarray:
    """RC residual minus the median RC residual of the other producing turbines at the same moment."""
    r = np.where(producing, measured - t_hat, np.nan)
    out = np.full(r.shape, np.nan)
    for i in range(r.shape[1]):
        others = np.delete(r, i, axis=1)
        enough = np.isfinite(others).sum(axis=1) >= min_others
        med = np.full(len(r), np.nan)
        med[enough] = np.nanmedian(others[enough], axis=1)
        out[:, i] = r[:, i] - med
    return out
