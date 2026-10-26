"""The farm as (time x turbine) arrays on a complete 10-minute grid, with every record classified.

Classification rules (first match wins) follow the cleaning of the earlier project wind-power-curve-kelmarsh:

    missing     no wind speed or no power
    sensor      anemometer problem: out of range, frozen for 30 minutes, or the two anemometers disagree
    stop_grid   stop requested by the grid or the park controller (IEC "Out of Electrical Specification",
                "Requested Shutdown")
    stop        any other logged stop, or no output although the wind is well above cut-in
    curtailed   logged power reduction, or blades pitched out while power is low for the wind speed
    icing       cold (<= 2 degC) and power far below the turbine's own warm-weather curve
    normal      everything else

The wind speed is rebuilt as the mean of the two nacelle anemometers (the SCADA "Wind speed" signal changes
definition in April 2018) and normalised to 1.225 kg/m3 with the density from the nacelle ambient temperature and
the standard-atmosphere pressure at hub height.

Usage:
    farm = build_farm(scada, status, layout(static_csv))
    farm.power.shape            # (n_timestamps, 6)
    farm.normal                 # bool, normal operation
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

STEP = pd.Timedelta("10min")
STATES = ("normal", "missing", "sensor", "stop_grid", "stop", "curtailed", "icing")
GRID_CATEGORIES = {"Out of Electrical Specification", "Requested Shutdown"}
DERATE_PATTERN = r"P red|externally reduced|[Rr]educed power|[Pp]ower reduction|[Pp]ower limit"
RHO_0 = 1.225
ELEVATION_M = 130.0  # site elevation (static file: 123-137 m); only enters the density through the pressure
SIGNALS = ("power", "ws", "ws_scada", "ws_std", "nacelle", "vane", "pitch", "rotor_rpm", "ambient_temp",
           "nacelle_temp", "gear_oil", "gear_bearing", "gen_bearing", "stator")


@dataclass
class Farm:
    """SCADA signals as (n_timestamps, n_turbines) float arrays (NaN where missing) plus the record states."""

    time: pd.DatetimeIndex
    turbines: np.ndarray
    x: np.ndarray
    y: np.ndarray
    hub_height: np.ndarray
    diameter: float
    rated_kw: float
    state: np.ndarray  # int8 index into STATES
    signals: dict = field(default_factory=dict)

    def __getattr__(self, name):
        signals = self.__dict__.get("signals", {})
        if name in signals:
            return signals[name]
        raise AttributeError(name)

    @property
    def n(self) -> int:
        return len(self.turbines)

    @property
    def normal(self) -> np.ndarray:
        return self.state == 0

    @property
    def operating(self) -> np.ndarray:
        """Rotor running (casting a wake): not stopped and producing."""
        stopped = np.isin(self.state, (STATES.index("stop"), STATES.index("stop_grid")))
        return ~stopped & (np.nan_to_num(self.power) > 0)

    @property
    def turbulence(self) -> np.ndarray:
        """Nacelle turbulence intensity ws_std / ws (SCADA signal), NaN below 3 m/s."""
        with np.errstate(divide="ignore", invalid="ignore"):
            ti = self.ws_std / self.ws_scada
        return np.where(self.ws_scada >= 3.0, ti, np.nan)

    def period(self, start, end=None) -> np.ndarray:
        """Mask for a year (``period(2019)``), a range of years (``period(2019, 2020)``) or two timestamps."""
        if isinstance(start, (int, np.integer)):
            end = start if end is None else end
            start, end = pd.Timestamp(f"{start}-01-01", tz="UTC"), pd.Timestamp(f"{end + 1}-01-01", tz="UTC")
        return np.asarray((self.time >= start) & (self.time < end))

    def day_code(self) -> np.ndarray:
        """Day number of every timestamp counted from the first day of the grid (unit-safe)."""
        days = self.time.normalize()
        return np.asarray((days - days[0]) // pd.Timedelta("1D"), dtype=np.int64)

    def copy(self) -> "Farm":
        return replace(self, signals={k: v.copy() for k, v in self.signals.items()}, state=self.state.copy())

    def slice(self, rows: np.ndarray) -> "Farm":
        return replace(self, time=self.time[rows], state=self.state[rows],
                       signals={k: v[rows] for k, v in self.signals.items()})


def event_mask(timestamps: pd.DatetimeIndex, starts, ends) -> np.ndarray:
    """True for 10-minute intervals [t, t + 10 min) overlapping any event [start, end); open events last 10 min.

    All times are converted to nanoseconds first: pandas may store them in different units.
    """
    ts = pd.DatetimeIndex(timestamps).as_unit("ns").asi8
    starts = pd.DatetimeIndex(starts).as_unit("ns")
    ends = pd.DatetimeIndex(ends).as_unit("ns")
    ends = ends.where(~ends.isna(), starts + STEP)
    first = np.searchsorted(ts, (starts - STEP).asi8, side="right")
    last = np.searchsorted(ts, ends.asi8, side="left")
    diff = np.zeros(len(ts) + 1, dtype=np.int64)
    np.add.at(diff, first, 1)
    np.add.at(diff, last, -1)
    return np.cumsum(diff[:-1]) > 0


def air_density(temp_c, height_m) -> np.ndarray:
    """Dry-air density from temperature and the standard-atmosphere pressure at ``height_m``."""
    p = 101325.0 * (1.0 - 2.25577e-5 * np.asarray(height_m, dtype=float)) ** 5.25588
    return p / (287.05 * (np.asarray(temp_c, dtype=float) + 273.15))


def frozen(x: np.ndarray, n: int = 3) -> np.ndarray:
    """True where a value repeats exactly in at least ``n`` consecutive records (per column)."""
    out = np.zeros(x.shape, dtype=bool)
    for j in range(x.shape[1]):
        s = x[:, j]
        same = np.r_[False, s[1:] == s[:-1]]
        run = np.cumsum(~same)
        length = np.bincount(run)[run]
        out[:, j] = (length >= n) & ~np.isnan(s)
    return out


def _bin_median(values, bins, mask):
    """Median of ``values`` per wind speed bin over ``mask``, mapped back to every record (NaN if no bin)."""
    s = pd.Series(values[mask]).groupby(bins[mask]).median()
    return pd.Series(bins).map(s).to_numpy(dtype=float)


def classify(sig: dict, status: pd.DataFrame, time: pd.DatetimeIndex, turbines: np.ndarray) -> np.ndarray:
    """State code (index into STATES) of every record, shape (time, turbine)."""
    ws, p = sig["ws"], sig["power"]
    n_t, n = ws.shape
    stop_grid = np.zeros((n_t, n), dtype=bool)
    stop = np.zeros((n_t, n), dtype=bool)
    derate = np.zeros((n_t, n), dtype=bool)
    is_stop = status["status"].eq("Stop")
    is_grid = status["iec_category"].isin(GRID_CATEGORIES)
    is_derate = (status["message"].fillna("").str.contains(DERATE_PATTERN)
                 | status["iec_category"].eq("Partial Performance")) & ~is_stop
    for j, t in enumerate(turbines):
        own = status["turbine"].eq(t)
        for arr, sel in ((stop_grid, is_stop & is_grid), (stop, is_stop & ~is_grid), (derate, is_derate)):
            ev = status[own & sel]
            if len(ev):
                arr[:, j] = event_mask(time, ev["start"], ev["end"])
    missing = np.isnan(ws) | np.isnan(p)
    with np.errstate(invalid="ignore"):
        pair = np.abs(sig["ws1"] - sig["ws2"]) > np.maximum(1.0, 0.15 * 0.5 * (sig["ws1"] + sig["ws2"]))
        sensor = (ws < 0) | (ws > 40) | (frozen(ws) & (ws > 1.0)) | np.nan_to_num(pair).astype(bool)
        stop |= (p <= 0) & (ws >= 4.5)
    bins = np.floor(np.nan_to_num(ws, nan=-1.0) / 0.5 + 0.5) * 0.5
    pitched = np.zeros((n_t, n), dtype=bool)
    for j in range(n):
        ok = ~missing[:, j]
        med_pitch = _bin_median(sig["pitch"][:, j], bins[:, j], ok & np.isfinite(sig["pitch"][:, j]))
        med_power = _bin_median(p[:, j], bins[:, j], ok)
        with np.errstate(invalid="ignore"):
            pitched[:, j] = (sig["pitch"][:, j] > med_pitch + 4.0) & (p[:, j] < 0.85 * med_power) & (ws[:, j] >= 5.0)
    state = np.zeros((n_t, n), dtype=np.int8)
    for name, flag in (("curtailed", derate | pitched), ("stop", stop), ("stop_grid", stop_grid),
                       ("sensor", sensor), ("missing", missing)):  # reverse priority: later rules overwrite
        state[flag] = STATES.index(name)
    temp = sig["ambient_temp"]
    for j in range(n):
        warm = (state[:, j] == 0) & (temp[:, j] > 5)
        curve = pd.Series(p[warm, j]).groupby(bins[warm, j]).median()
        expected = np.interp(np.nan_to_num(ws[:, j]), curve.index.to_numpy(), curve.to_numpy(), left=0.0,
                             right=float(curve.iloc[-1]))
        with np.errstate(invalid="ignore"):
            icing = (temp[:, j] <= 2) & (expected > 100) & (p[:, j] < 0.7 * expected)
        state[(state[:, j] == 0) & icing, j] = STATES.index("icing")
    return state


def build_farm(scada: pd.DataFrame, status: pd.DataFrame, lay: pd.DataFrame) -> Farm:
    """(turbine, timestamp) SCADA frame -> Farm on a complete 10-minute grid with classified records."""
    ts = scada.index.get_level_values("timestamp")
    grid = pd.date_range(ts.min(), ts.max(), freq="10min", name="timestamp")
    turbines = np.array(sorted(scada.index.get_level_values("turbine").unique()))

    def wide(col):
        return scada[col].unstack("turbine").reindex(index=grid, columns=turbines).to_numpy(dtype=float)

    raw = {c: wide(c) for c in scada.columns}
    sig = {c: raw[c] for c in SIGNALS if c in raw and c != "ws"}
    ws = np.where(np.isnan(raw["ws1"]) | np.isnan(raw["ws2"]), raw["ws_scada"], 0.5 * (raw["ws1"] + raw["ws2"]))
    lay = lay.loc[turbines]
    temp = np.where(np.isnan(raw["ambient_temp"]), np.nanmedian(raw["ambient_temp"]), raw["ambient_temp"])
    rho = air_density(temp, ELEVATION_M + lay["hub_height_m"].to_numpy()[None, :])
    sig["ws"] = ws * (rho / RHO_0) ** (1.0 / 3.0)
    state = classify({**sig, "ws1": raw["ws1"], "ws2": raw["ws2"]}, status, grid, turbines)
    return Farm(time=grid, turbines=turbines, x=lay["x_m"].to_numpy(), y=lay["y_m"].to_numpy(),
                hub_height=lay["hub_height_m"].to_numpy(), diameter=float(lay["rotor_diameter_m"].iloc[0]),
                rated_kw=float(lay["rated_kw"].iloc[0]), state=state, signals=sig)
