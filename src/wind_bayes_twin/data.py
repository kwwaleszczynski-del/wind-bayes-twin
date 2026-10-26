"""Read the open Kelmarsh SCADA archives: 10-minute signals and status logs of the six turbines.

Each yearly ``Kelmarsh_SCADA_<year>_<id>.zip`` (Zenodo record 16807551, CC BY 4.0) holds, per turbine, one
10-minute Greenbyte export (~300 signals) and one status (event) log. Only the signals used here are kept and
every year is cached as Parquet. The reader follows the ones of the earlier projects wind-power-curve-kelmarsh
and wind-thermal-nbm (same archives), but keeps the power, wind, direction *and* temperature channels together.

The archives are looked up in ``$KELMARSH_DATA_DIR/raw`` (several projects can share one download), otherwise
in ``data/raw`` of this repository.

Usage:
    scada, status = load([2017, 2018], raw_dir=Path("data/raw"), cache_dir=Path("data/processed"))
    lay = layout(raw_dir / "Kelmarsh_WT_static.csv")
"""
from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COL = "# Date and time"
DEG = "°"
SIGNALS = {
    "Power (kW)": "power",
    "Wind speed (m/s)": "ws_scada",
    "Wind speed, Standard deviation (m/s)": "ws_std",
    "Wind speed Sensor 1 (m/s)": "ws1",
    "Wind speed Sensor 2 (m/s)": "ws2",
    f"Nacelle position ({DEG})": "nacelle",
    f"Vane position 1+2 ({DEG})": "vane",
    f"Blade angle (pitch position) A ({DEG})": "pitch",
    "Rotor speed (RPM)": "rotor_rpm",
    f"Nacelle ambient temperature ({DEG}C)": "ambient_temp",
    f"Nacelle temperature ({DEG}C)": "nacelle_temp",
    f"Gear oil temperature ({DEG}C)": "gear_oil",
    f"Rear bearing temperature ({DEG}C)": "gear_bearing",  # gearbox (high-speed) rear bearing
    f"Generator bearing rear temperature ({DEG}C)": "gen_bearing",
    f"Stator temperature 1 ({DEG}C)": "stator",
}
TEMPERATURES = ("ambient_temp", "nacelle_temp", "gear_oil", "gear_bearing", "gen_bearing", "stator")
TEMP_RANGE = (-40.0, 150.0)  # the export holds sensor error values such as -273.2 degC
STATUS_COLUMNS = {
    "Timestamp start": "start",
    "Timestamp end": "end",
    "Status": "status",
    "Code": "code",
    "Message": "message",
    "IEC category": "iec_category",
}
TURBINE_RE = re.compile(r"_Kelmarsh_(\d+)_")
EARTH_RADIUS_M = 6_371_000.0
log = logging.getLogger(__name__)


def raw_dir_default(repo_data: Path) -> Path:
    env = os.environ.get("KELMARSH_DATA_DIR")
    return Path(env) / "raw" if env else repo_data / "raw"


def _header_row(text: str, marker: str) -> int:
    """Index of the column header line (exports start with a block of '#' comment lines)."""
    for i, line in enumerate(io.StringIO(text)):
        if line.startswith(marker):
            return i
        if i > 60:
            break
    raise ValueError(f"header line starting with {marker!r} not found")


def read_turbine_csv(text: str, turbine: int) -> pd.DataFrame:
    """One 10-minute export -> frame indexed by (turbine, timestamp UTC), float32 columns."""
    skip = _header_row(text, TIME_COL)
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=lambda c: c in SIGNALS or c == TIME_COL,
                     na_values=["NaN"])
    ts = pd.to_datetime(df[TIME_COL], format="%Y-%m-%d %H:%M:%S").dt.tz_localize("UTC")
    out = pd.DataFrame(index=pd.DatetimeIndex(ts, name="timestamp"))
    for src, dst in SIGNALS.items():
        out[dst] = df[src].to_numpy(dtype="float32") if src in df else np.float32(np.nan)
    for c in TEMPERATURES:
        out[c] = out[c].where((out[c] > TEMP_RANGE[0]) & (out[c] < TEMP_RANGE[1]))
    out.insert(0, "turbine", np.int8(turbine))
    return out.set_index("turbine", append=True).swaplevel().sort_index()


def read_status_csv(text: str, turbine: int) -> pd.DataFrame:
    """One status log -> rows of start/end/status/code/message/iec_category."""
    skip = _header_row(text, "Timestamp start")
    df = pd.read_csv(io.StringIO(text), skiprows=skip, usecols=list(STATUS_COLUMNS), dtype=str)
    df = df.rename(columns=STATUS_COLUMNS)
    for c in ("start", "end"):  # events still open at export time have end "-"
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d %H:%M:%S", errors="coerce").dt.tz_localize("UTC")
    df["code"] = pd.to_numeric(df["code"], errors="coerce")
    df.insert(0, "turbine", np.int8(turbine))
    return df


def read_zip(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scada, status = [], []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            m = TURBINE_RE.search(base)
            if not m or not base.endswith(".csv"):
                continue
            text = z.read(name).decode("utf-8-sig")
            if base.startswith("Turbine_Data"):
                scada.append(read_turbine_csv(text, int(m.group(1))))
            elif base.startswith("Status"):
                status.append(read_status_csv(text, int(m.group(1))))
    if not scada:
        raise ValueError(f"{path.name}: no turbine data files found")
    return pd.concat(scada).sort_index(), pd.concat(status, ignore_index=True)


def load(years: list[int], raw_dir: Path, cache_dir: Path, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SCADA and status data for ``years`` from the Parquet cache or the raw archives."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    scada, status = [], []
    for year in years:
        s_path, e_path = cache_dir / f"scada_{year}.parquet", cache_dir / f"status_{year}.parquet"
        if s_path.exists() and e_path.exists() and not refresh:
            s, e = pd.read_parquet(s_path), pd.read_parquet(e_path)
        else:
            archives = sorted(raw_dir.glob(f"Kelmarsh_SCADA_{year}_*.zip"))
            if not archives:
                raise FileNotFoundError(f"no archive for {year} in {raw_dir} (set KELMARSH_DATA_DIR)")
            log.info("reading %s", archives[0].name)
            s, e = read_zip(archives[0])
            s.to_parquet(s_path)
            e.to_parquet(e_path)
            log.info("%d: %d rows, %d status events", year, len(s), len(e))
        scada.append(s)
        status.append(e)
    scada_all = pd.concat(scada)
    scada_all = scada_all[~scada_all.index.duplicated(keep="first")].sort_index()
    return scada_all, pd.concat(status, ignore_index=True).drop_duplicates()


def layout(static_csv: Path) -> pd.DataFrame:
    """Turbine positions in metres (x east, y north) around the farm centre, hub height and rotor diameter."""
    df = pd.read_csv(static_csv, encoding="utf-8-sig")
    lat0, lon0 = np.deg2rad(df["Latitude"].mean()), np.deg2rad(df["Longitude"].mean())
    lat, lon = np.deg2rad(df["Latitude"]), np.deg2rad(df["Longitude"])
    return pd.DataFrame({
        "turbine": df["Title"].str.extract(r"(\d+)$", expand=False).astype(int),
        "x_m": EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0),
        "y_m": EARTH_RADIUS_M * (lat - lat0),
        "hub_height_m": df["Hub Height (m)"],
        "rotor_diameter_m": df["Rotor Diameter (m)"],
        "rated_kw": df["Rated power (kW)"],
    }).set_index("turbine").sort_index()
