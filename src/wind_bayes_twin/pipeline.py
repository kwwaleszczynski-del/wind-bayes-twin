"""From the raw archives to everything the twin needs, with the protocol of the earlier projects.

    2017 (baseline)      farm power curve, nacelle offsets, site factors of the free-stream reference, RC models
    2018 (calibration)   hierarchical noise model of every statistic, record dispersion at full load, chart
                         thresholds of the baselines; assumed fault-free
    2019-2020 (test)     false alarms on unmodified data, fault injection
    2021 (application)   unmodified data compared with the turbines' status logs

Prepared data (farm arrays, reference terms, thermal simulations, daily observations) and the calibration are cached
as pickles in ``data/processed``; the calibration is also written to ``results/noise_model.json``. The pickles are a
local cache written by this pipeline itself (never downloaded or shared), so loading them is safe.

Usage:
    prep = prepare(raw_dir, cache_dir)
    cal = calibrate(prep, cache_dir, results_dir)
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import reference as ref_mod
from . import thermal
from .baselines import Charts
from .data import layout, load
from .farm import build_farm
from .faults import Context
from .hierarchy import NoisePosterior, fit_statistic
from .model import CurveTable
from .stats import STATS, DayIndex, daily, dispersion_quantiles, rise_by_stat, thermal_residuals

YEARS = (2017, 2018, 2019, 2020, 2021)
BASE_YEAR, CAL_YEAR, TEST_YEARS, APPLY_YEAR = 2017, 2018, (2019, 2020), 2021
log = logging.getLogger(__name__)


@dataclass
class Prepared:
    farm: object
    ref: object
    terms: object
    rcs: dict
    sims: dict
    residuals: dict
    obs: object
    days: DayIndex
    status: object

    @property
    def cap_kw(self) -> float:
        return max(self.farm.rated_kw, self.ref.p_max)

    def table(self) -> CurveTable:
        return CurveTable(self.ref.curve)

    def context(self) -> Context:
        return Context(self.farm, self.terms, rise_by_stat(self.sims), self.residuals, self.obs, self.days.code,
                       self.ref.curve, self.ref.p_max)

    def day_range(self, first_year: int, last_year: int) -> range:
        return range(self.days.of(f"{first_year}-01-01"), self.days.of(f"{last_year}-12-31") + 1)


@dataclass
class Calibration:
    noise: dict  # stat -> NoisePosterior
    r_quantiles: np.ndarray  # (turbine, 20)
    charts: Charts
    seconds: float

    def to_json(self) -> dict:
        return {"noise": {s: p.to_dict() for s, p in self.noise.items()},
                "r_quantiles": np.round(self.r_quantiles, 5).tolist(), "fit_seconds": round(self.seconds, 1)}


def prepare(raw_dir: Path, cache_dir: Path, refresh: bool = False) -> Prepared:
    cache = cache_dir / "prepared.pkl"
    if cache.exists() and not refresh:
        with cache.open("rb") as fh:
            return pickle.load(fh)
    t0 = time.perf_counter()
    scada, status = load(list(YEARS), raw_dir, cache_dir, refresh=refresh)
    farm = build_farm(scada, status, layout(raw_dir / "Kelmarsh_WT_static.csv"))
    tables = ref_mod.wake_tables(farm, cache=cache_dir / "wake_tables.npy")
    ref = ref_mod.fit_reference(farm, farm.period(BASE_YEAR), tables)
    terms = ref_mod.predict(farm, ref)
    rcs = {ch: thermal.fit_rc(farm, ch, farm.period(BASE_YEAR)) for ch in thermal.CHANNELS}
    sims = {ch: thermal.simulate(farm, rc) for ch, rc in rcs.items()}
    days = DayIndex(farm.time)
    obs = daily(farm, terms, sims, days, ref.p_max)
    prep = Prepared(farm, ref, terms, rcs, sims, thermal_residuals(farm, sims), obs, days, status)
    with cache.open("wb") as fh:
        pickle.dump(prep, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("prepared %d days x %d turbines in %.0f s", days.n_days, farm.n, time.perf_counter() - t0)
    return prep


def calibrate(prep: Prepared, cache_dir: Path, results_dir: Path | None = None, refresh: bool = False,
              seed: int = 0) -> Calibration:
    cache = cache_dir / "calibration.pkl"
    if cache.exists() and not refresh:
        with cache.open("rb") as fh:
            return pickle.load(fh)
    t0 = time.perf_counter()
    years = prep.obs.dates.year
    cal, base = years == CAL_YEAR, years == BASE_YEAR
    noise = {}
    for s in STATS:
        post = fit_statistic(prep.obs.y[s][cal], prep.obs.count[s][cal], seed=seed)
        # slow drift of the healthy level: the year-to-year change of every turbine's median, baseline -> calibration
        change = np.nanmedian(prep.obs.y[s][cal], axis=0) - np.nanmedian(prep.obs.y[s][base], axis=0)
        post.level_walk = float(np.sqrt(np.mean(change ** 2)) / np.sqrt(365.0))
        noise[s] = post
        log.info("noise model %-6s R-hat %.3f, tau %.4f", s, post.rhat_max, post.tau)
    rq = dispersion_quantiles(prep.farm, prep.terms, prep.farm.period(CAL_YEAR), prep.ref.p_max)
    charts = Charts.calibrate(prep.obs.y, cal)
    out = Calibration(noise, rq, charts, time.perf_counter() - t0)
    with cache.open("wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "noise_model.json").write_text(json.dumps(out.to_json(), indent=1))
    return out


def load_cached(cache_dir: Path) -> tuple[Prepared, Calibration]:
    """Prepared data and calibration from the local cache (written by ``prepare`` and ``calibrate``)."""
    with (cache_dir / "prepared.pkl").open("rb") as fh:
        prep = pickle.load(fh)
    with (cache_dir / "calibration.pkl").open("rb") as fh:
        cal = pickle.load(fh)
    return prep, cal


def load_noise(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {s: NoisePosterior.from_dict(v) for s, v in d["noise"].items()}
