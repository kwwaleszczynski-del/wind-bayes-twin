"""The two earlier detectors as baselines, run on the same daily statistics and the same injected faults.

WARP charts (wind-wake-aware-verification): a CUSUM per turbine on four daily statistics, each standardised with
the median and robust spread (1.4826 MAD) of the calibration year, tuned to the smallest fault it should catch
(k = target / 2 sigma, target 3 % or 3 deg), threshold h = the highest value any turbine's chart reached in the
calibration year (at least 5); one-sided low for ``pow`` and ``rated``, two-sided for ``anemo`` and ``yaw``.

NBM charts (wind-thermal-nbm, "RC + fleet" with the trailing baseline): the daily fleet-referenced RC residual of
the two bearings minus the median of days t-120..t-31, divided by the robust spread of the calibration year,
clipped to +-5; two-sided CUSUM with k = 1 and h = the smallest threshold without alarm on the calibration year.

Both reset after an alarm. Each chart is mapped to the fault type it targets, which is its diagnosis.

Usage:
    charts = Charts.calibrate(obs.y, cal_days)
    alarms = charts.run(y, first_day, last_day)          # list of (day, turbine, chart, cause)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WARP = {"pow": ("low", 0.03, "eff"), "rated": ("low", 0.03, "derate"), "anemo": ("both", 0.03, "anemo"),
        "yaw": ("both", 3.0, "yaw")}
NBM = {"tgear": "gear", "tgen": "gen"}
H_MIN = 5.0
NBM_K = 1.0
TRAIL = (120, 31)
Z_CLIP = 5.0
CHART_CAUSE = {**{c: v[2] for c, v in WARP.items()}, **NBM}


def cusum(z: np.ndarray, k: float, h: float, side: str):
    """Tabular CUSUM; returns (alarm days, alarm directions, running max(S-, S+)). NaN days leave S unchanged."""
    lo = hi = 0.0
    days, dirs = [], []
    stat = np.zeros(len(z))
    for d, v in enumerate(z):
        if np.isfinite(v):
            if side in ("low", "both"):
                lo = max(0.0, lo - v - k)
            if side in ("high", "both"):
                hi = max(0.0, hi + v - k)
        stat[d] = max(lo, hi)
        if lo > h or hi > h:
            days.append(d)
            dirs.append(-1 if lo > h else 1)
            lo = hi = 0.0
    return days, dirs, stat


def trailing_median(x: np.ndarray, lo: int = TRAIL[0], hi: int = TRAIL[1]) -> np.ndarray:
    """Causal median of days t-lo .. t-hi of a daily series (NaN where fewer than 30 values)."""
    out = np.full(len(x), np.nan)
    for t in range(lo, len(x)):
        w = x[t - lo:t - hi + 1]
        if np.isfinite(w).sum() >= 30:
            out[t] = np.nanmedian(w)
    return out


@dataclass
class Charts:
    mu: dict  # chart -> (n,) median of the calibration year (WARP) / robust spread basis (NBM)
    sigma: dict
    k: dict
    h: dict

    @classmethod
    def calibrate(cls, y: dict, cal: np.ndarray) -> "Charts":
        """Standardisation and thresholds from the fault-free calibration days ``cal`` (bool over days)."""
        mu, sigma, k, h = {}, {}, {}, {}
        for c, (side, target, _) in WARP.items():
            v = y[c][cal]
            mu[c] = np.nanmedian(v, axis=0)
            sigma[c] = 1.4826 * np.nanmedian(np.abs(v - mu[c]), axis=0)
            k[c] = target / 2.0 / sigma[c]
            z = (v - mu[c]) / sigma[c]
            peak = max(cusum(z[:, i], k[c][i], np.inf, side)[2].max() for i in range(v.shape[1]))
            h[c] = max(H_MIN, peak * 1.0001)
        for c in NBM:
            x = y[c] - np.column_stack([trailing_median(y[c][:, i]) for i in range(y[c].shape[1])])
            v = x[cal]
            mu[c] = np.zeros(v.shape[1])
            sigma[c] = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, axis=0)), axis=0)
            k[c] = np.full(v.shape[1], NBM_K)
            z = np.clip(v / sigma[c], -Z_CLIP, Z_CLIP)
            h[c] = max(cusum(z[:, i], NBM_K, np.inf, "both")[2].max() for i in range(v.shape[1])) * 1.0001
        return cls(mu, sigma, k, h)

    def z(self, chart: str, x: np.ndarray, i: int) -> np.ndarray:
        """Standardised daily values of turbine column ``i`` (x: the full daily series of that turbine)."""
        if chart in NBM:
            return np.clip((x - trailing_median(x)) / self.sigma[chart][i], -Z_CLIP, Z_CLIP)
        return (x - self.mu[chart][i]) / self.sigma[chart][i]

    def run(self, y: dict, i: int, first: int, last: int) -> list[tuple[int, str, int]]:
        """Alarms (day, chart, direction) of turbine column ``i`` with the charts started on day ``first``."""
        out = []
        for c in list(WARP) + list(NBM):
            side = WARP[c][0] if c in WARP else "both"
            z = self.z(c, y[c][:last, i], i)[first:last]
            days, dirs, _ = cusum(z, self.k[c][i], self.h[c], side)
            out += [(first + d, c, s) for d, s in zip(days, dirs)]
        return sorted(out)
