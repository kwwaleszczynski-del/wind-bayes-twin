"""Free-stream reference of every turbine from its neighbours: what a healthy turbine would see and produce.

This is the WARP method (Wake-Adjusted Reference Power) of the earlier project wind-wake-aware-verification,
re-implemented compactly here because the twin needs its record-level terms, not its daily charts. For turbine i
and 10-minute record t, everything comes from the *other* turbines, so a fault on i cannot leak into its own
reference:

1. Direction: theta_-i = circular mean of the other turbines' nacelle positions minus their static offsets c_j.
2. Wakes: single-wake deficit tables from PyWake (Bastankhah & Porte-Agel Gaussian wake, generic 92 m / 2.05 MW
   thrust curve), smoothed over direction, combined by root-sum-square over the turbines that are running.
3. Free stream: E_-i = median over the free turbines j (deficit < 2 %, normal operation) of e_j / (1 - d_j), where
   e_j = F^-1(P_j) is the power-equivalent wind speed (inverse farm power curve, partial load) or kappa_j u_j.
   E^P_-i uses power only and is the anemometer-independent reference for the anemometer check.
4. Expected equivalent wind speed and power at i: e^_i = E_-i (1 - d_i) s_i(theta, E, TI), P^_i = F(e^_i), with a
   smooth site-and-transfer factor ln s_i = b0 + 6 direction harmonics + c ln(E/8) + g ln(TI/0.12) fitted on the
   baseline year (robust least squares in log wind speed, b0 then set so baseline energy is matched).
5. Expected anemometer reading u^_i = E^P_-i (1 - d_i) s^u_i, and the nacelle direction offset
   Delta_i = N_i - c_i - theta_-i.

The wake parameters are the values calibrated in wind-wake-aware-verification on the same baseline year
(wake expansion k = 0.035, direction smoothing 3 deg, north correction +2 deg); they are not re-tuned here.

Usage:
    tables = wake_tables(farm, cache=Path("data/processed/wake_tables.npy"))
    ref = fit_reference(farm, farm.period(2017), tables)
    terms = predict(farm, ref)          # Terms: e_hat, p_hat, u_hat, delta ... as (time, turbine) arrays
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

WD_GRID = np.arange(0.0, 360.0, 1.0)
WS_GRID = np.arange(3.0, 25.01, 0.5)
WAKE_K = 0.035
SIGMA_DEG = 3.0
THETA0 = 2.0
FREE_DEFICIT = 0.02
P_RANGE = (50.0, 1850.0)  # kW, partial load: power inverted to a wind speed only here
CUT_OUT = 25.0
TI_REF = 0.12
E_REF = 8.0
N_HARMONICS = 6
FIT_RANGE = (4.5, 10.0)
ENERGY_MIN_KW = 100.0
log = logging.getLogger(__name__)


def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- power curve
def farm_curve(ws, power, mask, bin_width: float = 0.5, min_count: int = 30) -> pd.DataFrame:
    """Mean power per wind speed bin over the masked records, made non-decreasing (columns ws, power_kw)."""
    ws, power = np.asarray(ws, dtype=float)[mask], np.asarray(power, dtype=float)[mask]
    ok = ~(np.isnan(ws) | np.isnan(power))
    df = pd.DataFrame({"ws": ws[ok], "power_kw": power[ok]})
    df["bin"] = np.floor(df["ws"] / bin_width + 0.5) * bin_width
    g = df.groupby("bin")
    curve = pd.DataFrame({"ws": g["ws"].mean(), "power_kw": g["power_kw"].mean(), "n": g.size()})
    curve = curve[curve["n"] >= min_count].reset_index(drop=True)
    curve["power_kw"] = np.maximum.accumulate(curve["power_kw"].clip(lower=0.0).to_numpy())
    return curve


def power_from_wind(ws, curve: pd.DataFrame, cut_out: float = CUT_OUT) -> np.ndarray:
    """Farm power curve F: linear between bins, 0 below the first bin and above cut-out."""
    ws = np.asarray(ws, dtype=float)
    xs, ys = curve["ws"].to_numpy(), curve["power_kw"].to_numpy()
    p = np.interp(ws, xs, ys, left=0.0, right=ys[-1])
    p[(ws > cut_out) | (ws < xs[0])] = 0.0
    p[np.isnan(ws)] = np.nan
    return p


def wind_from_power(power, curve: pd.DataFrame, p_range=P_RANGE) -> np.ndarray:
    """Power-equivalent wind speed F^-1(P); NaN outside the partial-load range."""
    p = np.asarray(power, dtype=float)
    xs = curve["power_kw"].to_numpy() + np.arange(len(curve)) * 1e-6  # strictly increasing
    out = np.interp(p, xs, curve["ws"].to_numpy())
    out[~((p >= p_range[0]) & (p <= p_range[1]))] = np.nan
    return out


# ---------------------------------------------------------------- directions and medians
def leave_one_out_direction(nacelle, valid, offsets=None) -> np.ndarray:
    """(time, turbine) wind direction at each turbine: circular mean of the *other* turbines' nacelle positions."""
    corrected = np.asarray(nacelle, dtype=float) - (0.0 if offsets is None else np.asarray(offsets)[None, :])
    rad = np.deg2rad(np.where(valid, corrected, np.nan))
    s, c = np.nan_to_num(np.sin(rad)), np.nan_to_num(np.cos(rad))
    cnt = (valid & ~np.isnan(rad)).astype(float)
    s_o, c_o, n_o = s.sum(1, keepdims=True) - s, c.sum(1, keepdims=True) - c, cnt.sum(1, keepdims=True) - cnt
    out = np.rad2deg(np.arctan2(s_o, c_o)) % 360.0
    out[n_o == 0] = np.nan
    return out


def nacelle_offsets(nacelle, valid, max_iter: int = 20, tol: float = 0.01) -> np.ndarray:
    """Static nacelle position offset of each turbine relative to the others (fixed point, mean zero)."""
    offsets = np.zeros(nacelle.shape[1])
    for _ in range(max_iter):
        delta = wrap180(nacelle - leave_one_out_direction(nacelle, valid, offsets))
        new = np.array([np.nanmedian(np.where(valid[:, i], delta[:, i], np.nan)) for i in range(nacelle.shape[1])])
        new -= new.mean()
        done = np.max(np.abs(new - offsets)) < tol
        offsets = new
        if done:
            break
    return offsets


def leave_one_out_median(values, mask) -> np.ndarray:
    """(time, turbine) median over the *other* turbines of ``values`` where ``mask``; NaN if none."""
    v = np.where(mask, values, np.nan)
    out = np.full(v.shape, np.nan)
    for i in range(v.shape[1]):
        others = np.delete(v, i, axis=1)
        ok = ~np.isnan(others).all(axis=1)
        out[ok, i] = np.nanmedian(others[ok], axis=1)
    return out


def _row_median(x, mask) -> np.ndarray:
    v = np.where(mask, x, np.nan)
    out = np.full(len(v), np.nan)
    ok = mask.any(axis=1)
    out[ok] = np.nanmedian(v[ok], axis=1)
    return out


# ---------------------------------------------------------------- wakes
def wake_tables(farm, k: float = WAKE_K, sigma_deg: float = SIGMA_DEG, cache: Path | None = None) -> np.ndarray:
    """Direction-smoothed single-wake deficits ``[j, i, wd, ws]`` that turbine j alone causes at turbine i."""
    if cache is not None and cache.exists():
        return np.load(cache)
    from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
    from py_wake.rotor_avg_models import CGIRotorAvg
    from py_wake.site import UniformSite
    from py_wake.superposition_models import SquaredSum
    from py_wake.wind_farm_models import PropagateDownwind
    from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

    x, y, h = farm.x, farm.y, farm.hub_height
    turbine = GenericWindTurbine("generic", farm.diameter, float(np.median(h)), farm.rated_kw)
    wfm = PropagateDownwind(UniformSite(p_wd=[1.0], ti=0.1), turbine,
                            wake_deficitModel=BastankhahGaussianDeficit(k=k, rotorAvgModel=CGIRotorAvg(4)),
                            superpositionModel=SquaredSum())
    n = len(x)
    out = np.zeros((n, n, len(WD_GRID), len(WS_GRID)))
    for j in range(n):
        for i in range(j + 1, n):
            ws_eff = np.asarray(wfm(x[[j, i]], y[[j, i]], h=h[[j, i]], wd=WD_GRID, ws=WS_GRID).WS_eff.values)
            out[j, i] = 1.0 - ws_eff[1] / WS_GRID[None, :]
            out[i, j] = 1.0 - ws_eff[0] / WS_GRID[None, :]
    out = gaussian_filter1d(np.clip(out, 0.0, 1.0), sigma_deg, axis=-2, mode="wrap")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


def farm_deficit(tables: np.ndarray, theta, ws, operating) -> np.ndarray:
    """Root-sum-square deficit at every turbine, (time, turbine); only running turbines cast a wake."""
    theta, ws = np.asarray(theta, dtype=float), np.asarray(ws, dtype=float)
    f = np.clip((np.nan_to_num(ws) - WS_GRID[0]) / (WS_GRID[1] - WS_GRID[0]), 0.0, len(WS_GRID) - 1 - 1e-9)
    lo = np.floor(f).astype(int)
    w = f - lo
    hi = np.minimum(lo + 1, len(WS_GRID) - 1)
    wd = np.round(np.nan_to_num(theta)).astype(int) % len(WD_GRID)
    n = tables.shape[0]
    total = np.zeros((len(theta), n))
    for j in range(n):
        for i in range(n):
            if i != j:
                t = tables[j, i]
                total[:, i] += operating[:, j] * ((1 - w) * t[wd, lo] + w * t[wd, hi]) ** 2
    out = np.sqrt(total)
    out[np.isnan(theta) | np.isnan(ws)] = np.nan
    return out


# ---------------------------------------------------------------- site factor
def features(theta, e, ti) -> np.ndarray:
    """Design matrix of the site factor: 1, cos/sin(m theta) m = 1..6, ln(E/8), ln(TI/0.12)."""
    t = np.deg2rad(np.asarray(theta, dtype=float))
    le = np.log(np.clip(np.asarray(e, dtype=float), 3.0, 15.0) / E_REF)
    lt = np.log(np.clip(np.nan_to_num(np.asarray(ti, dtype=float), nan=TI_REF), 0.04, 0.35) / TI_REF)
    cols = [np.ones_like(t)]
    for m in range(1, N_HARMONICS + 1):
        cols += [np.cos(m * t), np.sin(m * t)]
    return np.stack(cols + [le, lt], axis=-1)


def robust_lstsq(design, y, k: float = 4.0, iterations: int = 3) -> np.ndarray:
    """Least squares refitted after dropping points further than k robust standard deviations."""
    keep = np.ones(len(y), dtype=bool)
    for _ in range(iterations):
        coef = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        r = y - design @ coef
        med = np.median(r[keep])
        keep = np.abs(r - med) <= k * max(1.4826 * np.median(np.abs(r[keep] - med)), 1e-6)
    return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]


@dataclass
class Reference:
    tables: np.ndarray
    curve: pd.DataFrame
    offsets: np.ndarray
    kappa: np.ndarray
    coef_p: np.ndarray
    coef_u: np.ndarray

    @property
    def p_max(self) -> float:
        return float(self.curve["power_kw"].max())


@dataclass
class Terms:
    """Record-level reference terms, all (time, turbine)."""

    theta: np.ndarray
    deficit: np.ndarray
    e_free: np.ndarray
    e_free_p: np.ndarray
    ti: np.ndarray
    e_hat: np.ndarray = None
    p_hat: np.ndarray = None
    u_hat: np.ndarray = None
    delta: np.ndarray = None


def _free_stream(farm, ref: Reference, rows) -> Terms:
    normal, power, ws = farm.normal[rows], farm.power[rows], farm.ws[rows]
    e = wind_from_power(np.where(normal, power, np.nan), ref.curve)
    e_mix = np.where(np.isnan(e), ref.kappa[None, :] * ws, e)
    theta = leave_one_out_direction(farm.nacelle[rows], normal & ~np.isnan(farm.nacelle[rows]), ref.offsets)
    u_look = leave_one_out_median(ws, normal)
    ti = leave_one_out_median(farm.turbulence[rows], normal)
    shape = power.shape
    out = Terms(theta, np.full(shape, np.nan), np.full(shape, np.nan), np.full(shape, np.nan), ti)
    operating = farm.operating[rows].astype(float)
    for i in range(farm.n):
        d = farm_deficit(ref.tables, theta[:, i] + THETA0, u_look[:, i], operating)
        free = normal & (d < FREE_DEFICIT)
        free[:, i] = False
        out.deficit[:, i] = d[:, i]
        out.e_free[:, i] = _row_median(e_mix / (1 - d), free)
        free_p = free & ~np.isnan(e)
        all_p = (free_p.sum(axis=1) == free.sum(axis=1)) & free.any(axis=1)
        out.e_free_p[:, i] = np.where(all_p, _row_median(e / (1 - d), free_p), np.nan)
    return out


def fit_reference(farm, base: np.ndarray, tables: np.ndarray) -> Reference:
    """Farm curve, nacelle offsets, anemometer scales and site factors fitted on the baseline rows ``base``."""
    norm_b = farm.normal & base[:, None]
    curve = farm_curve(farm.ws, farm.power, norm_b)
    offsets = nacelle_offsets(farm.nacelle[base], norm_b[base] & (farm.ws[base] >= 4.0))
    e = wind_from_power(np.where(norm_b, farm.power, np.nan), curve)
    kappa = np.nanmedian(np.where(np.isnan(e), np.nan, e / farm.ws), axis=0)
    n_feat = features(0.0, E_REF, TI_REF).shape[-1]
    ref = Reference(tables, curve, offsets, kappa, np.zeros((farm.n, n_feat)), np.zeros((farm.n, n_feat)))
    terms = _free_stream(farm, ref, base)
    e_b, ws_b, normal_b = e[base], farm.ws[base], farm.normal[base]
    expected = terms.e_free_p * (1 - terms.deficit)
    for i in range(farm.n):
        rows = normal_b[:, i] & (expected[:, i] >= FIT_RANGE[0]) & (expected[:, i] <= FIT_RANGE[1])
        x = features(terms.theta[:, i], terms.e_free_p[:, i], terms.ti[:, i])
        for target, coef in ((e_b[:, i], ref.coef_p), (ws_b[:, i], ref.coef_u)):
            ok = rows & ~np.isnan(target)
            coef[i] = robust_lstsq(x[ok], np.log(target[ok]) - np.log(expected[ok, i]))
    # energy calibration of b0: the log fit is median-unbiased in wind speed, not in energy (convex curve)
    s_p = np.exp(np.einsum("tif,if->ti", features(terms.theta, terms.e_free, terms.ti), ref.coef_p))
    e_hat = terms.e_free * (1 - terms.deficit) * s_p
    power_b = farm.power[base]
    for i in range(farm.n):
        ok = normal_b[:, i] & np.isfinite(power_b[:, i]) & np.isfinite(e_hat[:, i])
        scale = 1.0
        for _ in range(30):
            p_hat = power_from_wind(e_hat[ok, i] * scale, curve)
            sel = p_hat >= ENERGY_MIN_KW
            ratio = power_b[ok, i][sel].sum() / p_hat[sel].sum()
            scale *= ratio ** (1.0 / 3.0)
            if abs(ratio - 1.0) < 1e-5:
                break
        ref.coef_p[i, 0] += np.log(scale)
    return ref


def predict(farm, ref: Reference, rows=None) -> Terms:
    """Expected equivalent wind speed and power, expected anemometer reading and nacelle offset Delta."""
    rows = np.ones(len(farm.time), dtype=bool) if rows is None else rows
    terms = _free_stream(farm, ref, rows)
    s_p = np.exp(np.einsum("tif,if->ti", features(terms.theta, terms.e_free, terms.ti), ref.coef_p))
    s_u = np.exp(np.einsum("tif,if->ti", features(terms.theta, terms.e_free_p, terms.ti), ref.coef_u))
    terms.e_hat = terms.e_free * (1 - terms.deficit) * s_p
    terms.p_hat = power_from_wind(terms.e_hat, ref.curve)
    terms.u_hat = terms.e_free_p * (1 - terms.deficit) * s_u
    terms.delta = wrap180(farm.nacelle[rows] - ref.offsets[None, :] - terms.theta)
    return terms
