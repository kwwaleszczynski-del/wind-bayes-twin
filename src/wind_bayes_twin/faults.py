"""Known faults injected into the real SCADA records of one turbine, for validation against ground truth.

The faults follow the injection studies of wind-wake-aware-verification (yaw, anemometer, derating, power loss) and
wind-thermal-nbm (cooling degradation), extended by gradual versions (drifts). Only the faulty turbine's own
signals change, inside the fault window; its references come from the other turbines and are not affected, and
the record states (normal operation, stops ...) are kept as classified on the clean data.

    yaw          static misalignment delta: the vane is offset, so the rotor runs delta off the wind. The nacelle
                 position moves by delta and power follows a cos^n law of the true inflow angle (vane + delta),
                 applied as an effective wind speed so a misaligned turbine still reaches rated power:
                 r = [cos(v + delta) / cos v]^(n/3),  P' = P F(r u) / F(u)   (n = 2; WARP's yaw fault)
    anemometer   both nacelle anemometers read (1 + b) times the truth; power unchanged
    eff          power x (1 - s) at all wind speeds (blade damage, pitch offset)
    derate       power capped at (1 - s) x rated
    gear / gen   cooling degradation: the temperature difference of the bearing to its cooling side (gear oil /
                 nacelle air) grows by the factor rho, T' = T + (rho - 1) max(T - T_cool, 0)

Steps act from the onset; ``ramp_days`` > 0 turns a fault into a linear drift reaching its full size after that
many days (the cooling faults ramp over 30 days as in wind-thermal-nbm). Each scenario also states the value of
the twin's parameter that corresponds to the full fault ("truth"), used to score parameter recovery.

Usage:
    y = inject(ctx, i=2, fault=SCENARIOS[0], start_day=800, n_days=90, sign=-1)    # statistics with the fault
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FAULTS
from .reference import power_from_wind
from .stats import STATS, EXTRA, THERMAL, turbine_observations
from .thermal import cooling_side

YAW_EXPONENT = 2.0
VANE_CLIP_DEG = 30.0
STEPS_PER_DAY = 144


@dataclass(frozen=True)
class Fault:
    name: str
    kind: str  # yaw, anemometer, eff, derate, gear, gen
    size: float
    ramp_days: float = 0.0
    signed: bool = False  # alternate the sign between start dates

    @property
    def fault_type(self) -> str:
        return "anemo" if self.kind == "anemometer" else self.kind

    @property
    def param(self) -> str:
        return {"anemometer": "anemo7", "yaw": "yaw", "eff": "eff", "derate": "derate", "gear": "gear",
                "gen": "gen"}[self.kind]

    def truth(self, sign: int = 1) -> float:
        """Twin parameter of the full fault (yaw: Delta moves by -delta; anemometer: error at 7 m/s, ln(1 + b))."""
        s = self.size * (sign if self.signed else 1)
        if self.kind == "yaw":
            return -s
        if self.kind == "anemometer":
            return float(np.log1p(s))
        if self.kind == "eff":
            return float(np.log1p(-s))
        return s


SCENARIOS = [
    Fault("yaw 4°", "yaw", 4.0, signed=True),
    Fault("yaw 8°", "yaw", 8.0, signed=True),
    Fault("yaw drift to 8° (60 d)", "yaw", 8.0, 60.0, signed=True),
    Fault("anemometer ±3 %", "anemometer", 0.03, signed=True),
    Fault("anemometer drift to ±5 % (60 d)", "anemometer", 0.05, 60.0, signed=True),
    Fault("efficiency -2 %", "eff", 0.02),
    Fault("efficiency -5 %", "eff", 0.05),
    Fault("efficiency drift to -5 % (60 d)", "eff", 0.05, 60.0),
    Fault("derating to 95 %", "derate", 0.05),
    Fault("gear bearing cooling +15 %", "gear", 0.15, 30.0),
    Fault("gear bearing cooling +25 %", "gear", 0.25, 30.0),
    Fault("generator bearing cooling +15 %", "gen", 0.15, 30.0),
    Fault("generator bearing cooling +25 %", "gen", 0.25, 30.0),
]
assert all(f.fault_type in FAULTS for f in SCENARIOS)


def yaw_power_factor(vane, ws, delta_deg: float | np.ndarray, curve) -> np.ndarray:
    """Factor P'/P for a static misalignment given the measured vane angle and wind speed."""
    v = np.deg2rad(np.clip(np.nan_to_num(np.asarray(vane, dtype=float)), -VANE_CLIP_DEG, VANE_CLIP_DEG))
    r = (np.clip(np.cos(v + np.deg2rad(delta_deg)), 0.0, None) / np.cos(v)) ** (YAW_EXPONENT / 3.0)
    p0 = power_from_wind(ws, curve)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = power_from_wind(ws * r, curve) / p0
    return np.where(np.isfinite(factor) & (p0 > 10.0), factor, r ** 3)


@dataclass
class Context:
    """Everything needed to recompute one turbine's statistics after modifying its signals."""

    farm: object
    terms: object
    rise: dict  # stat -> (time, turbine) temperature rise above the cooling side
    residuals: dict  # stat -> (time, turbine) fleet-referenced thermal residual
    obs: object
    code: np.ndarray  # day code of every timestamp
    curve: object
    p_max: float


def inject(ctx: Context, i: int, fault: Fault, start_day: int, n_days: int, sign: int = 1) -> dict:
    """Statistics (all days, all turbines) with ``fault`` injected into turbine column ``i`` on [start, start+n)."""
    farm, terms = ctx.farm, ctx.terms
    rows = slice(int(np.searchsorted(ctx.code, start_day)), int(np.searchsorted(ctx.code, start_day + n_days)))
    code = ctx.code[rows] - start_day
    elapsed = np.arange(rows.stop - rows.start) / STEPS_PER_DAY
    ramp = np.clip(elapsed / fault.ramp_days, 0.0, 1.0) if fault.ramp_days > 0 else np.ones(len(elapsed))
    size = fault.size * (sign if fault.signed else 1) * ramp
    power, ws = farm.power[rows, i].copy(), farm.ws[rows, i].copy()
    delta = terms.delta[rows, i].copy()
    residuals = {s: ctx.residuals[s][rows, i].copy() for s in THERMAL}
    if fault.kind == "yaw":
        power = power * yaw_power_factor(farm.vane[rows, i], ws, size, ctx.curve)
        delta = delta - size
    elif fault.kind == "anemometer":
        ws = ws * (1.0 + size)
    elif fault.kind == "eff":
        power = power * (1.0 - size)
    elif fault.kind == "derate":
        power = np.minimum(power, (1.0 - size) * farm.rated_kw)
    elif fault.kind in ("gear", "gen"):
        stat = "tgear" if fault.kind == "gear" else "tgen"
        temp = getattr(farm, THERMAL[stat][1])[rows, i]
        residuals[stat] = residuals[stat] + size * np.clip(temp - cooling_side(farm, fault.kind)[rows, i], 0.0, None)
    else:
        raise ValueError(f"unknown fault kind {fault.kind!r}")
    sig = {"power": power, "ws": ws, "delta": delta}
    y_new = turbine_observations(i, sig, terms, {s: ctx.rise[s] for s in THERMAL}, residuals, code, n_days,
                                 farm.normal[rows, i], ctx.p_max, rows=rows)[0]
    out = {s: ctx.obs.y[s].copy() for s in STATS + EXTRA}
    for s in STATS + EXTRA:
        out[s][start_day:start_day + n_days, i] = y_new[s]
    return out
