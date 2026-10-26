"""Daily observations of the twin: seven statistics per turbine and day, plus the records behind the power ones.

Each statistic reduces one residual of the free-stream reference (``reference.Terms``) or of the fleet-referenced
thermal model (``thermal``) to one value per day:

    pow     ln(sum P / sum P^)       normal operation, 100 kW <= P^ < 97 % of the curve maximum  (>= 36 records)
    full    ln(sum P / sum P^)       normal operation, e^ >= 13 m/s, well above rated wind        (>= 6 records)
    alo     median ln(u / u^)        normal operation, u^ in [4, 7) m/s                           (>= 18 records)
    ahi     median ln(u / u^)        normal operation, u^ in [7, 11] m/s                          (>= 18 records)
    yaw     median Delta [deg]       normal operation, free stream >= 4 m/s                       (>= 36 records)
    tgear   mean fleet-referenced RC residual of the gearbox bearing [K], producing          (>= 36 records)
    tgen    mean fleet-referenced RC residual of the generator bearing [K], producing        (>= 36 records)

``full`` uses only records whose expected wind is well above rated (the power curve is flat there): a healthy turbine
then produces rated power whatever the exact wind, so the statistic is about twice as precise and much less skewed
than the full-load statistic of WARP (P^ >= 97 %, which includes records near rated wind where the wind estimate
matters). Records in between (P^ >= 97 %, e^ < 13 m/s) are not used by the twin. Two further statistics are computed
only for the WARP charts used as a baseline: ``rated`` (WARP's full-load statistic, P^ >= 97 %) and ``anemo``
(median ln(u / u^) over u^ in [4, 11] m/s).

The power statistics are kept with their records (expected equivalent wind speed e^ of every record), because the
twin's observation model re-evaluates them under each hypothesis about the turbine's health (a power cap or a yaw
offset acts record by record). The anemometer statistics keep the band median of u^ and the thermal ones the
daily mean temperature rise of the RC model above the cooling side, which is all their observation models need.

Usage:
    days = DayIndex(farm.time)
    obs = daily(farm, terms, thermal_sims, days)
    obs.y["pow"].shape          # (n_days, n_turbines)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .thermal import PRODUCING_KW, fleet_residual

STATS = ("pow", "full", "alo", "ahi", "yaw", "tgear", "tgen")
EXTRA = ("rated", "anemo")
FULL_LOAD = 0.97  # WARP's full-load records: P^ >= 97 % of the curve maximum
FULL_WS = 13.0  # the twin's full-load records: expected wind at least this [m/s]
POWER_MIN_KW = 100.0
ALO, AHI = (4.0, 7.0), (7.0, 11.0)
YAW_MIN_WS = 4.0
MIN_RECORDS = {"pow": 36, "full": 6, "rated": 6, "alo": 18, "ahi": 18, "anemo": 36, "yaw": 36, "tgear": 36, "tgen": 36}
THERMAL = {"tgear": ("gear", "gear_bearing"), "tgen": ("gen", "gen_bearing")}


class DayIndex:
    """Consecutive day numbers 0..n_days-1 of 10-minute timestamps (computed without the datetime unit)."""

    def __init__(self, time: pd.DatetimeIndex):
        days = time.normalize()
        self.dates = pd.date_range(days[0], days[-1], freq="D")
        self.code = np.asarray((days - days[0]) // pd.Timedelta("1D"), dtype=np.int64)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def _ts(self, t) -> pd.Timestamp:
        t = pd.Timestamp(t)
        return t.tz_localize(self.dates.tz) if t.tzinfo is None and self.dates.tz is not None else t

    def of(self, timestamp) -> int:
        return int((self._ts(timestamp).normalize() - self.dates[0]) // pd.Timedelta("1D"))

    def mask(self, start, end) -> np.ndarray:
        """Days in [start, end)."""
        return np.asarray((self.dates >= self._ts(start)) & (self.dates < self._ts(end)))


@dataclass
class Records:
    """Records behind the power statistics of one turbine, sorted by day."""

    day: np.ndarray
    e_hat: np.ndarray
    p_hat: np.ndarray
    full: np.ndarray  # bool: full-load record (e^ >= 13 m/s), else partial load
    start: np.ndarray = field(default=None)  # first record of each day (n_days + 1 pointer array)

    def of_day(self, d: int) -> slice:
        return slice(int(self.start[d]), int(self.start[d + 1]))


@dataclass
class Observations:
    dates: pd.DatetimeIndex
    y: dict  # stat -> (n_days, n)
    count: dict  # stat -> (n_days, n) records behind each value
    u_lo: np.ndarray  # band medians of u^ (n_days, n)
    u_hi: np.ndarray
    rise: dict  # "tgear"/"tgen" -> daily mean rise above the cooling side [K] (n_days, n)
    records: list  # Records per turbine
    p_max: float


def _bincount_mean(code, values, valid, n_days):
    w = valid & np.isfinite(values)
    s = np.bincount(code[w], weights=values[w], minlength=n_days)
    c = np.bincount(code[w], minlength=n_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / c, c


def _daily_median(code, values, valid, n_days):
    w = valid & np.isfinite(values)
    s = pd.Series(values[w]).groupby(code[w])
    out = np.full(n_days, np.nan)
    med = s.median()
    out[med.index.to_numpy()] = med.to_numpy()
    cnt = np.bincount(code[w], minlength=n_days)
    return out, cnt


def power_records(code, normal, p_obs, terms, i: int, p_max: float, n_days: int):
    """Records behind ``pow`` and ``full`` for turbine ``i`` and the observed power statistics (with ``rated``)."""
    p_hat, e_hat = terms.p_hat[:, i], terms.e_hat[:, i]
    use = normal & np.isfinite(p_obs) & np.isfinite(p_hat) & (p_hat >= POWER_MIN_KW)
    rated = p_hat >= FULL_LOAD * p_max
    full = e_hat >= FULL_WS
    twin = use & (~rated | full)
    rec = Records(code[twin], e_hat[twin], p_hat[twin], full[twin])
    rec.start = np.searchsorted(rec.day, np.arange(n_days + 1))
    y, cnt = {}, {}
    for name, sel in (("pow", use & ~rated), ("full", use & full), ("rated", use & rated)):
        sp = np.bincount(code[sel], weights=p_obs[sel], minlength=n_days)
        sh = np.bincount(code[sel], weights=p_hat[sel], minlength=n_days)
        cnt[name] = np.bincount(code[sel], minlength=n_days)
        with np.errstate(invalid="ignore", divide="ignore"):
            y[name] = np.log(np.maximum(sp, 1e-3 * sh) / sh)
    return rec, y, cnt


def turbine_observations(i: int, sig: dict, terms, rise: dict, residuals: dict, code, n_days: int, normal, p_max,
                         rows=None):
    """Statistics of turbine column ``i`` on the timestamps ``rows`` (a slice, default all; day codes ``code``).

    ``sig`` (power, ws, delta) and ``residuals`` (fleet-referenced thermal residuals) are the turbine's own 1-D
    signals on those rows; ``terms`` and ``rise`` (temperature rise above the cooling side) are full (time,
    turbine) arrays."""
    rows = slice(None) if rows is None else rows
    sub = lambda a: a[rows, i]  # noqa: E731
    sub_terms = type(terms)(**{k: (v[rows] if isinstance(v, np.ndarray) else v) for k, v in vars(terms).items()})
    rec, y, cnt = power_records(code, normal, sig["power"], sub_terms, i, p_max, n_days)
    u_hat = sub(terms.u_hat)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.log(sig["ws"] / u_hat)
    bands = {}
    for name, (lo, hi) in (("alo", ALO), ("ahi", AHI), ("anemo", (ALO[0], AHI[1]))):
        sel = normal & (u_hat >= lo) & ((u_hat < hi) if name == "alo" else (u_hat <= hi))
        y[name], cnt[name] = _daily_median(code, lr, sel, n_days)
        bands[name] = _daily_median(code, u_hat, sel & np.isfinite(lr), n_days)[0]
    y["yaw"], cnt["yaw"] = _daily_median(code, sig["delta"], normal & (sub(terms.e_free) >= YAW_MIN_WS), n_days)
    rise_d = {}
    for name in THERMAL:
        prod = np.isfinite(residuals[name]) & (sig["power"] > PRODUCING_KW)
        y[name], cnt[name] = _bincount_mean(code, residuals[name], prod, n_days)
        rise_d[name] = _bincount_mean(code, sub(rise[name]), prod, n_days)[0]
    for name in STATS + EXTRA:
        y[name][cnt[name] < MIN_RECORDS[name]] = np.nan
    return y, cnt, bands, rise_d, rec


def rise_by_stat(sims: dict) -> dict:
    return {s: sims[ch]["rise"] for s, (ch, _) in THERMAL.items()}


def thermal_residuals(farm, sims: dict, temps: dict | None = None) -> dict:
    """Fleet-referenced residual of each thermal channel, (time, turbine); ``temps`` overrides measured temps."""
    producing = np.nan_to_num(farm.power) > PRODUCING_KW
    out = {}
    for name, (channel, target) in THERMAL.items():
        measured = getattr(farm, target) if temps is None else temps[target]
        out[name] = fleet_residual(measured, sims[channel]["t_hat"], producing)
    return out


def daily(farm, terms, sims: dict, days: DayIndex, p_max: float) -> Observations:
    """Observations of every turbine and day."""
    res = thermal_residuals(farm, sims)
    n_days, n = days.n_days, farm.n
    obs = Observations(days.dates, {s: np.full((n_days, n), np.nan) for s in STATS + EXTRA},
                       {s: np.zeros((n_days, n), dtype=int) for s in STATS + EXTRA}, np.full((n_days, n), np.nan),
                       np.full((n_days, n), np.nan), {s: np.full((n_days, n), np.nan) for s in THERMAL}, [], p_max)
    for i in range(n):
        sig = {"power": farm.power[:, i], "ws": farm.ws[:, i], "delta": terms.delta[:, i]}
        y, cnt, bands, rise, rec = turbine_observations(i, sig, terms, rise_by_stat(sims),
                                                         {s: res[s][:, i] for s in THERMAL}, days.code, n_days,
                                                         farm.normal[:, i], p_max)
        for s in STATS + EXTRA:
            obs.y[s][:, i], obs.count[s][:, i] = y[s], cnt[s]
        obs.u_lo[:, i], obs.u_hi[:, i] = bands["alo"], bands["ahi"]
        for s in THERMAL:
            obs.rise[s][:, i] = rise[s]
        obs.records.append(rec)
    return obs


def dispersion_quantiles(farm, terms, rows: np.ndarray, p_max: float, n_q: int = 20) -> np.ndarray:
    """(turbine, n_q) quantiles of P / P^ over normal full-load records in ``rows``, scaled to mean 1.

    The record-to-record spread of a healthy turbine around its expected power at full load: a power cap removes
    only the part of it above the cap, which the derating model needs.
    """
    out = []
    for i in range(farm.n):
        p_hat = terms.p_hat[:, i]
        sel = rows & farm.normal[:, i] & np.isfinite(farm.power[:, i]) & (p_hat >= FULL_LOAD * p_max)
        q = np.quantile(farm.power[sel, i] / p_hat[sel], (np.arange(n_q) + 0.5) / n_q)
        out.append(q / q.mean())
    return np.array(out)
