"""The twin's state-space model: hidden health parameters, their dynamics and the observation model.

Hidden state of turbine i on day d (deviation from its own healthy, calibration-year behaviour):

    eff      ln eta            multiplicative power loss (blade damage, pitch offset, soiling)
    ascale   ln a              nacelle anemometer scale error against the free stream
    abias    b [m/s]           nacelle anemometer offset
    yaw      gamma [deg]       static yaw misalignment against the neighbours' wind direction
    derate   kappa             power limited to (1 - kappa) of rated
    gear     rho_g - 1         thermal resistance of the gearbox bearing to its oil, relative
    gen      rho_n - 1         thermal resistance of the generator bearing to the nacelle air, relative

Six fault types switch these on: eff, anemo (ascale + abias), yaw, derate, gear, gen. A fault type is either off
(its parameters are exactly 0, the spike) or on (parameters drawn from a slab at onset, then a random walk), a
spike-and-slab / regime-switching prior:

    off -> on   with hazard h per day, parameters ~ slab
    on  -> on   random walk with daily step q; a further jump with hazard h
    on  -> off  with hazard h_repair (repair, a transient fault ending)

Observation model of the seven daily statistics (``stats``), given the health parameters theta:

    pow, full    ln( sum_t min(eta F(e^_t cos(gamma)^(n/3)), (1 - kappa) P_r) / sum_t F(e^_t) )
                 over the day's records, e^_t the expected equivalent wind speed from the neighbours (wake model
                 included); n = 2: the cos^n yaw loss applied as an effective wind speed (WARP's yaw fault model)
    alo, ahi     ln(a + b / u~)       u~ = the day's median expected anemometer reading in the band
    yaw          gamma
    tgear, tgen  (rho - 1) R~         R~ = the day's mean RC temperature rise above the cooling side [K]

and every statistic y_s = h_s(theta) + mu_s + n_s + eps_s with the healthy level mu_s, AR(1) wander n_s and noise
eps_s of the hierarchical model (``hierarchy``). A fault is *material* when it is worth acting on (``MATERIAL``);
P(fault k) is the posterior probability that type k is on and material.

Usage:
    spec = ModelSpec()
    h = power_stats(theta, e_hat, p_hat, full_flag, curve_table, cap_kw)      # (N,), (N,)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PARAMS = ("eff", "ascale", "abias", "yaw", "derate", "gear", "gen")
FAULTS = ("eff", "anemo", "yaw", "derate", "gear", "gen")
# fault types that explain the same statistics compete; an alarm is raised on the probability that the group has a
# material fault, and the diagnosis is its most probable type
ALARM_GROUPS = (("eff", "derate"), ("yaw",), ("anemo",), ("gear",), ("gen",))
FAULT_PARAMS = {"eff": (0,), "anemo": (1, 2), "yaw": (3,), "derate": (4,), "gear": (5,), "gen": (6,)}
FAULT_LABEL = {"eff": "efficiency loss", "anemo": "anemometer", "yaw": "yaw misalignment", "derate": "derating",
               "gear": "gearbox bearing cooling", "gen": "generator bearing cooling"}
# statistics that carry information on each fault type (used to guide the jump proposals)
FAULT_STATS = {"eff": ("full", "pow"), "anemo": ("alo", "ahi"), "yaw": ("yaw",), "derate": ("full",),
               "gear": ("tgear",), "gen": ("tgen",)}
YAW_EXPONENT = 2.0
ANEMO_CHECK = (5.0, 9.0)  # wind speeds at which an anemometer error is judged [m/s]


@dataclass
class ModelSpec:
    """Prior of the dynamics and the materiality thresholds (all per day; units as in the module docstring)."""

    hazard: dict = field(default_factory=lambda: {k: 1 / 1000 for k in FAULTS})
    hazard_repair: float = 1 / 200
    # slab of the onset size: (kind, scale) per parameter; kind "neg"/"pos" = half-normal, "sym" = normal
    slab: dict = field(default_factory=lambda: {
        "eff": ("neg", 0.05), "ascale": ("sym", 0.04), "abias": ("sym", 0.3), "yaw": ("sym", 6.0),
        "derate": ("pos", 0.10), "gear": ("pos", 0.30), "gen": ("pos", 0.30)})
    slab_floor: dict = field(default_factory=lambda: {"derate": 0.01})  # derating starts at >= 1 % of rated
    walk: dict = field(default_factory=lambda: {
        "eff": 0.002, "ascale": 0.001, "abias": 0.01, "yaw": 0.2, "derate": 0.003, "gear": 0.01, "gen": 0.01})
    material: dict = field(default_factory=lambda: {
        "eff": 0.01, "anemo": 0.015, "yaw": 2.5, "derate": 0.02, "gear": 0.10, "gen": 0.10})
    p_alarm: float = 0.9
    p_rearm: float = 0.5


def slab_sample(rng, kind_scale, size, floor=0.0):
    kind, scale = kind_scale
    z = rng.normal(0.0, scale, size)
    if kind == "neg":
        return -np.abs(z) - floor
    if kind == "pos":
        return np.abs(z) + floor
    return z


def slab_logpdf(x, kind_scale, floor=0.0):
    kind, scale = kind_scale
    x = np.asarray(x, dtype=float)
    if kind == "sym":
        return -0.5 * (x / scale) ** 2 - np.log(scale * np.sqrt(2 * np.pi))
    v = -x - floor if kind == "neg" else x - floor
    out = -0.5 * (v / scale) ** 2 - np.log(scale * np.sqrt(np.pi / 2))
    return np.where(v >= 0, out, -np.inf)


def material(theta: np.ndarray, spec: ModelSpec) -> np.ndarray:
    """(N, 6) bool: is each fault type's parameter beyond its materiality threshold (ignores the regime)."""
    m = spec.material
    anemo = np.maximum(*(np.abs(np.log(np.clip(np.exp(theta[:, 1]) + theta[:, 2] / u, 1e-3, None)))
                         for u in ANEMO_CHECK))
    return np.column_stack([theta[:, 0] < -m["eff"], anemo > m["anemo"], np.abs(theta[:, 3]) > m["yaw"],
                            theta[:, 4] > m["derate"], theta[:, 5] > m["gear"], theta[:, 6] > m["gen"]])


class CurveTable:
    """The farm power curve F tabulated on a fine wind speed grid for fast vectorised look-up."""

    def __init__(self, curve, step: float = 0.01, ws_max: float = 30.0):
        from .reference import power_from_wind

        self.step = step
        self.grid = np.arange(0.0, ws_max + step, step)
        self.values = power_from_wind(self.grid, curve)
        self.values[np.isnan(self.values)] = 0.0

    def __call__(self, ws: np.ndarray) -> np.ndarray:
        idx = np.clip((ws / self.step + 0.5).astype(np.int64), 0, len(self.grid) - 1)
        return self.values[idx]


def yaw_factor(gamma_deg: np.ndarray) -> np.ndarray:
    """Effective wind speed factor of a static misalignment: cos(gamma)^(n/3) (cos^n in partial load)."""
    return np.clip(np.cos(np.deg2rad(gamma_deg)), 1e-3, None) ** (YAW_EXPONENT / 3.0)


def turbine_power(theta: np.ndarray, e_hat: np.ndarray, table: CurveTable, cap_kw: float,
                  r_q: np.ndarray | None = None) -> np.ndarray:
    """(N, R) modelled power of each particle on each record (no cap when kappa = 0).

    With ``r_q`` (quantiles of the healthy record-to-record ratio P / P^, mean 1) the cap is applied to the
    dispersed power and averaged: E_r[min(r eta F(e^ g), c)].
    """
    ws = e_hat[None, :] * yaw_factor(theta[:, 3])[:, None]
    p = np.exp(theta[:, 0])[:, None] * table(ws)
    cap = np.where(theta[:, 4] > 0, (1.0 - theta[:, 4]) * cap_kw, np.inf)[:, None]
    if r_q is None:
        return np.minimum(p, cap)
    return np.minimum(p[:, :, None] * r_q[None, None, :], cap[:, :, None]).mean(axis=2)


def power_stats(theta, e_hat, p_hat, full, table: CurveTable, cap_kw: float, r_q: np.ndarray | None = None):
    """Modelled ``pow`` and ``full`` statistics of every particle for one day's records (NaN if no records).

    The dispersion ``r_q`` is applied only to records above 75 % of the cap, the only ones a cap of up to 20 %
    can reach.
    """
    n = len(theta)
    if len(e_hat) == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    if r_q is None:
        p = turbine_power(theta, e_hat, table, cap_kw)
    else:
        high = p_hat >= 0.75 * cap_kw
        p = np.empty((n, len(e_hat)))
        p[:, ~high] = turbine_power(theta, e_hat[~high], table, cap_kw)
        p[:, high] = turbine_power(theta, e_hat[high], table, cap_kw, r_q)
    out = []
    for sel in (~full, full):
        den = p_hat[sel].sum()
        out.append(np.log(np.maximum(p[:, sel].sum(axis=1), 1e-3 * den) / den) if sel.any() else np.full(n, np.nan))
    return out[0], out[1]


def anemo_stat(theta: np.ndarray, u_band: float) -> np.ndarray:
    return np.log(np.clip(np.exp(theta[:, 1]) + theta[:, 2] / u_band, 1e-3, None))
