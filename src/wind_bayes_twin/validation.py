"""Validation: unmodified test years (false alarms) and known faults injected into real data (detection, delay,
cause, parameter recovery), the twin side by side with the WARP and NBM charts on the very same statistics.

Protocol. Every turbine is filtered from 1 January 2019 to the end of 2021 on unmodified data ("clean run"). For
each fault scenario, turbine and one of eight start dates spread over 2019-2020, the fault is injected into that
turbine's records for 90 days, and the twin is restarted from its clean state on the start day (the filter state
before the fault does not depend on it). Signs alternate between start dates for yaw and anemometer faults.

Alarms. Fault types that explain the same statistics compete (efficiency loss and derating both lower the power at
full load, and for weeks the data may not tell which), so the twin raises an alarm for a group of types
(``model.ALARM_GROUPS``) when the probability that the group has a material fault crosses ``p_alarm`` (0.9), and
re-arms when it falls below ``p_rearm`` (0.5); its diagnosis is the group's most probable type at that moment. A chart system (WARP, NBM, or both)
alarms when any of its charts does, and its diagnosis is the type that chart targets. A fault counts as detected
when the system alarms within the 90 days; the chance level is the same count on the unmodified data in the same
windows. Parameter recovery compares the twin's posterior mean and 90 % credible interval on the last day of the
window with the injected value.

Usage:
    runs, clean, p_runs = campaign(cache_dir, engine="block", workers=6)
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .faults import SCENARIOS, inject
from .model import ALARM_GROUPS, FAULTS, ModelSpec
from .pipeline import TEST_YEARS, load_cached
from .twin import run_turbine, turbine_noise

HORIZON = 90
N_STARTS = 8
LEAD_DAYS = 14
P_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)  # alarm thresholds of the twin for the trade-off curves
H_MULTIPLIERS = (0.6, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5)  # chart thresholds h x m for the trade-off curves
SYSTEMS = {"WARP": ("pow", "rated", "anemo", "yaw"), "NBM": ("tgear", "tgen"),
           "WARP + NBM": ("pow", "rated", "anemo", "yaw", "tgear", "tgen")}
log = logging.getLogger(__name__)


GROUP_IDX = [[FAULTS.index(f) for f in g] for g in ALARM_GROUPS]


def twin_alarms(p: np.ndarray, armed: np.ndarray, spec: ModelSpec) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Alarms (day index, fault index) from daily fault probabilities (days, 6); ``armed`` is per alarm group.

    Returns the alarms and the final armed state."""
    armed = armed.copy()
    pg = np.column_stack([np.clip(p[:, g].sum(axis=1), 0.0, 1.0) for g in GROUP_IDX])
    out = []
    for d in range(len(p)):
        fire = armed & (pg[d] > spec.p_alarm)
        for k in np.flatnonzero(fire)[np.argsort(-pg[d, fire])]:
            g = GROUP_IDX[k]
            out.append((d, g[int(np.argmax(p[d, g]))]))
        armed &= ~fire
        armed |= pg[d] < spec.p_rearm
    return out, armed


def spec_at(threshold: float, spec: ModelSpec = ModelSpec()) -> ModelSpec:
    """Model spec with another alarm threshold (re-armed 0.4 below it)."""
    return replace(spec, p_alarm=threshold, p_rearm=max(0.05, threshold - 0.4))


def scaled_charts(charts, m: float):
    return replace(charts, h={c: v * m for c, v in charts.h.items()})


def probs(df: pd.DataFrame) -> np.ndarray:
    return df[[f"p_{f}" for f in FAULTS]].to_numpy()


def start_days(prep) -> list[int]:
    first = prep.days.of(f"{TEST_YEARS[0]}-01-01") + LEAD_DAYS
    last = prep.days.of(f"{TEST_YEARS[-1]}-12-31") - HORIZON
    return [int(round(v)) for v in np.linspace(first, last, N_STARTS)]


def _first(alarms, start, stop):
    """First alarm (day, cause) with start <= day < stop, else None."""
    inside = [a for a in alarms if start <= a[0] < stop]
    return inside[0] if inside else None


def chart_alarms(charts, y, i, first, last, system: str):
    """Alarms (day, chart) of the charts of ``system`` for turbine column ``i``, charts started on ``first``."""
    return [(d, c) for d, c, _ in charts.run(y, i, first, last) if c in SYSTEMS[system]]


def turbine_campaign(cache_dir: Path, i: int, engine: str, scenarios=SCENARIOS, n_particles: int = 2000,
                     guided: bool = True, seed: int = 1, attribute: bool = True):
    """Clean run 2019-2021 and every injection run of turbine column ``i``.

    Returns the run rows, the clean daily posteriors and the daily fault probabilities of every run (runs, 90, 6).
    """
    from .baselines import CHART_CAUSE

    prep, cal = load_cached(cache_dir)
    spec, table, cap = ModelSpec(), prep.table(), prep.cap_kw
    noise = turbine_noise(cal.noise, i)
    rq = cal.r_quantiles[i]
    starts = start_days(prep)
    days = prep.day_range(TEST_YEARS[0], 2021)
    t0 = time.perf_counter()
    clean, snaps = run_turbine(prep.obs, i, noise, spec, table, cap, days, engine=engine, r_quantiles=rq,
                               n_particles=n_particles, seed=seed, guided=guided, snapshot_days=set(starts),
                               attribute=attribute, quantiles=True)
    clean_seconds = time.perf_counter() - t0
    clean["turbine"] = int(prep.farm.turbines[i])
    clean["date"] = prep.obs.dates[clean["day"]]
    p_clean = probs(clean)
    first = days[0]
    ctx = prep.context()
    charts_m = {m: scaled_charts(cal.charts, m) for m in H_MULTIPLIERS}
    rows, trajectories = [], []
    for n_s, start in enumerate(starts):
        # armed state of the twin's alarms on the start day, from the clean history
        _, armed0 = twin_alarms(p_clean[:start - first], np.ones(len(ALARM_GROUPS), dtype=bool), spec)
        chance_twin, _ = twin_alarms(p_clean[start - first:start - first + HORIZON], armed0, spec)
        chance_charts = {s: _first(chart_alarms(cal.charts, prep.obs.y, i, first, start + HORIZON, s), start,
                                   start + HORIZON) for s in SYSTEMS}
        sign = 1 if n_s % 2 == 0 else -1
        for fault in scenarios:
            t1 = time.perf_counter()
            y = inject(ctx, i, fault, start, HORIZON, sign)
            df, _ = run_turbine(prep.obs, i, noise, spec, table, cap, range(start, start + HORIZON), engine=engine,
                                r_quantiles=rq, n_particles=n_particles, seed=seed + 1000 + start, guided=guided,
                                final_quantiles=True, y=y, start_snapshot=snaps[start])
            alarms, _ = twin_alarms(probs(df), armed0, spec)
            trajectories.append(probs(df).astype(np.float32))
            last = df.iloc[-1]
            row = {"engine": engine, "turbine": int(prep.farm.turbines[i]), "scenario": fault.name,
                   "fault": fault.fault_type, "sign": sign if fault.signed else 0, "start": prep.obs.dates[start],
                   "truth": fault.truth(sign), "twin_detected": bool(alarms),
                   "twin_days": alarms[0][0] if alarms else np.nan,
                   "twin_cause": FAULTS[alarms[0][1]] if alarms else "none",
                   "twin_chance": bool(chance_twin),
                   "p_end": float(last[f"p_{fault.fault_type}"]),
                   "cause_end": FAULTS[int(np.argmax(probs(df.iloc[[-1]])[0]))]
                   if probs(df.iloc[[-1]])[0].max() > 0.5 else "none",
                   "est_mean": float(last[f"mean_{fault.param}"]), "est_q05": float(last[f"q05_{fault.param}"]),
                   "est_q95": float(last[f"q95_{fault.param}"]), "seconds": time.perf_counter() - t1}
            if engine == "block":
                age = last.get(f"onset_age_{fault.fault_type}", np.nan)
                row["onset_error_days"] = float(age - (HORIZON - 1)) if np.isfinite(age) else np.nan
            for s in SYSTEMS:
                a = _first(chart_alarms(cal.charts, y, i, first, start + HORIZON, s), start, start + HORIZON)
                row[f"{s}_detected"] = a is not None
                row[f"{s}_days"] = a[0] - start if a else np.nan
                row[f"{s}_cause"] = CHART_CAUSE[a[1]] if a else "none"
                row[f"{s}_chance"] = chance_charts[s] is not None
            for m, ch in charts_m.items():  # trade-off: detection with scaled chart thresholds
                a = _first(chart_alarms(ch, y, i, first, start + HORIZON, "WARP + NBM"), start, start + HORIZON)
                row[f"charts_det_m{m:g}"] = a is not None
                row[f"charts_cause_m{m:g}"] = CHART_CAUSE[a[1]] if a else "none"
            rows.append(row)
        log.info("T%d start %s done", prep.farm.turbines[i], prep.obs.dates[start].date())
    clean.attrs["seconds"] = clean_seconds
    clean["engine"] = engine
    return rows, clean, np.array(trajectories)


def _worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return turbine_campaign(*args)


def campaign(cache_dir: Path, engine: str = "block", workers: int = 6, turbines=None, **kw):
    """Injection runs (DataFrame) and clean daily posteriors (DataFrame) of all turbines."""
    prep, _ = load_cached(cache_dir)
    turbines = range(prep.farm.n) if turbines is None else turbines
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    args = [(cache_dir, i, engine, kw.get("scenarios", SCENARIOS), kw.get("n_particles", 2000),
             kw.get("guided", True), kw.get("seed", 1), kw.get("attribute", True)) for i in turbines]
    rows, cleans, secs, trajs = [], [], [], []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_worker, args))
    else:
        results = [_worker(a) for a in args]
    for r, c, tr in results:
        rows += r
        secs.append(c.attrs.get("seconds", np.nan))
        cleans.append(c)
        trajs.append(tr)
    clean = pd.concat(cleans, ignore_index=True)
    clean.attrs["seconds"] = secs
    return pd.DataFrame(rows), clean, np.concatenate(trajs)


def false_alarms(clean: pd.DataFrame, prep, cal, spec: ModelSpec = ModelSpec(), charts=None):
    """Alarms per turbine-year on the unmodified test years: the twin by fault type, the charts by system."""
    test = clean["date"].dt.year.isin(TEST_YEARS)
    first = prep.days.of(f"{TEST_YEARS[0]}-01-01")
    last = prep.days.of(f"{TEST_YEARS[-1]}-12-31") + 1
    turb_years = 0.0
    counts = {f"twin {f}": 0 for f in FAULTS}
    counts.update({s: 0 for s in SYSTEMS})
    episodes = []
    for t, g in clean[test].groupby("turbine"):
        i = int(np.flatnonzero(prep.farm.turbines == t)[0])
        turb_years += len(g) / 365.25
        alarms, _ = twin_alarms(probs(g), np.ones(len(ALARM_GROUPS), dtype=bool), spec)
        for d, f in alarms:
            counts[f"twin {FAULTS[f]}"] += 1
            episodes.append({"turbine": t, "date": g["date"].iloc[d], "system": "twin", "cause": FAULTS[f]})
        for s in SYSTEMS:
            a = chart_alarms(cal.charts if charts is None else charts, prep.obs.y, i, first, last, s)
            counts[s] += len(a)
            if s != "WARP + NBM":
                episodes += [{"turbine": t, "date": prep.obs.dates[d], "system": s, "cause": c} for d, c in a]
    rows = [{"system": k, "alarms": v, "per_turbine_year": v / turb_years} for k, v in counts.items()]
    twin_total = sum(v for k, v in counts.items() if k.startswith("twin"))
    rows.append({"system": "twin (all types)", "alarms": twin_total, "per_turbine_year": twin_total / turb_years})
    return pd.DataFrame(rows), pd.DataFrame(episodes)
