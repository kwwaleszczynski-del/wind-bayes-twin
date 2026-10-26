"""Run the twin over days for one turbine: inputs from the observations, filter steps, daily posterior summaries.

Usage:
    noise = turbine_noise(posteriors, i)
    daily, snaps = run_turbine(obs, i, noise, ModelSpec(), table, cap_kw, days=range(d0, d1), seed=1,
                               snapshot_days={d0 + 30})
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .attribution import ENERGY_CAUSES, shapley_losses
from .changepoint import BlockTwin
from .model import ModelSpec
from .stats import STATS
from .twin_filter import DayInput, Noise, TwinFilter

ATTRIBUTION_SAMPLES = 200


def turbine_noise(posteriors: dict, i: int) -> Noise:
    """Noise model of turbine column ``i`` from the hierarchical posteriors of the seven statistics."""
    get = lambda attr: np.array([getattr(posteriors[s], attr)[i] for s in STATS])  # noqa: E731
    fleet = lambda attr: np.array([getattr(posteriors[s], attr) for s in STATS], dtype=float)  # noqa: E731
    return Noise(mu_mean=get("mu_mean"), mu_var=get("mu_sd") ** 2, sigma_e=get("sigma_e"), sigma_w=get("sigma_w"),
                 phi=get("phi"), c_ref=fleet("c_ref"), level_walk=fleet("level_walk"))


def day_input(obs, i: int, d: int, y: dict | None = None) -> DayInput:
    """Inputs of turbine column ``i`` on day ``d``; ``y`` overrides the statistics (fault injection)."""
    rec = obs.records[i]
    sl = rec.of_day(d)
    src = obs.y if y is None else y
    return DayInput(y=np.array([src[s][d, i] for s in STATS]), count=np.array([obs.count[s][d, i] for s in STATS]),
                    e_hat=rec.e_hat[sl], p_hat=rec.p_hat[sl],
                    full=rec.full[sl], u_lo=obs.u_lo[d, i], u_hi=obs.u_hi[d, i],
                    rise_gear=obs.rise["tgear"][d, i], rise_gen=obs.rise["tgen"][d, i])


def make_engine(engine: str, noise: Noise, spec: ModelSpec, table, cap_kw: float, r_quantiles=None,
                n_particles: int = 2000, seed: int = 0, guided: bool = True):
    """``block``: run-length Kalman engine (primary); ``particle``: joint particle filter."""
    if engine == "block":
        return BlockTwin(noise, spec, table, cap_kw, r_quantiles)
    return TwinFilter(noise, spec, table, cap_kw, n_particles=n_particles, seed=seed, guided=guided,
                      r_quantiles=r_quantiles)


def run_turbine(obs, i: int, noise: Noise, spec: ModelSpec, table, cap_kw: float, days, engine: str = "block",
                r_quantiles=None, n_particles: int = 2000, seed: int = 0, guided: bool = True, snapshot_days=(),
                attribute: bool = False, quantiles: bool = False, final_quantiles: bool = False,
                y: dict | None = None, start_snapshot: dict | None = None, final_snapshot: bool = False):
    """Filter turbine column ``i`` over ``days``; returns (daily summary frame, {day: engine snapshot}).

    Snapshots are taken before the day is processed; ``final_snapshot`` adds the state after the last day as
    ``snaps["final"]``.
    """
    flt = make_engine(engine, noise, spec, table, cap_kw, r_quantiles, n_particles, seed, guided)
    if start_snapshot is not None:
        flt.restore(start_snapshot)
    rows, snaps = [], {}
    t0 = time.perf_counter()
    days = list(days)
    for d in days:
        if d in snapshot_days:
            snaps[d] = flt.snapshot()
        inp = day_input(obs, i, d, y)
        q = quantiles or (final_quantiles and d == days[-1])
        row = {"day": d, **(flt.step(inp, q) if engine == "block" else flt.step(inp))}
        if attribute:
            rng = np.random.default_rng(seed * 100_003 + d)
            loss = shapley_losses(flt.sample(ATTRIBUTION_SAMPLES, rng), inp.e_hat, table, cap_kw)
            for k, c in enumerate(ENERGY_CAUSES):
                row[f"loss_{c}_kwh"] = float(loss[:, k].mean())
                row[f"loss_{c}_q05"], row[f"loss_{c}_q95"] = map(float, np.quantile(loss[:, k], [0.05, 0.95]))
        rows.append(row)
    if final_snapshot:
        snaps["final"] = flt.snapshot()
    out = pd.DataFrame(rows)
    out.attrs["seconds"] = time.perf_counter() - t0
    return out, snaps
