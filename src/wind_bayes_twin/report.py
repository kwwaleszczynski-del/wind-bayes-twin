"""Figures for the README (docs/img) and the interactive dashboard (docs/index.html).

Usage:
    report(cache_dir, results_dir, docs_dir)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .faults import SCENARIOS, inject  # noqa: E402
from .model import ALARM_GROUPS, FAULT_LABEL, FAULTS, ModelSpec  # noqa: E402
from .pipeline import TEST_YEARS, load_cached  # noqa: E402
from .twin import run_turbine, turbine_noise  # noqa: E402
from .validation import HORIZON, chart_alarms, start_days, twin_alarms  # noqa: E402
from .experiments import decisions  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TWIN, CHARTS, WARP_C, NBM_C, BAD = "#2a78d6", "#898781", "#e3913b", "#8c5bd6", "#d03b3b"
FAULT_COLOR = {"eff": "#2a78d6", "anemo": "#1b9e77", "yaw": "#e3913b", "derate": "#d03b3b", "gear": "#8c5bd6",
               "gen": "#a6761d"}
PARAM_LABEL = {"eff": "efficiency ln η", "anemo7": "anemometer error at 7 m/s", "yaw": "yaw offset γ [°]",
               "derate": "derating κ", "gear": "gear bearing ρ − 1", "gen": "generator bearing ρ − 1"}
REPLAYS = [("yaw 4°", 2, 3), ("efficiency drift to -5 % (60 d)", 0, 6), ("generator bearing cooling +25 %", 4, 1),
           ("derating to 95 %", 3, 0)]
log = logging.getLogger(__name__)


def _style():
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- replays of injected faults
def replay(prep, cal, scenario: str, i: int, n_start: int, lead: int = 45) -> dict:
    """Daily posterior of one injected fault, from ``lead`` days before onset to the end of the window."""
    fault = next(f for f in SCENARIOS if f.name == scenario)
    start = start_days(prep)[n_start]
    sign = 1 if n_start % 2 == 0 else -1
    spec, table, cap = ModelSpec(), prep.table(), prep.cap_kw
    noise = turbine_noise(cal.noise, i)
    first = prep.days.of(f"{TEST_YEARS[0]}-01-01")
    lead = min(lead, start - first)
    _, snaps = run_turbine(prep.obs, i, noise, spec, table, cap, range(first, start - lead + 1), r_quantiles=
                           cal.r_quantiles[i], snapshot_days={start - lead})
    y = inject(prep.context(), i, fault, start, HORIZON, sign)
    df, end = run_turbine(prep.obs, i, noise, spec, table, cap, range(start - lead, start + HORIZON),
                          r_quantiles=cal.r_quantiles[i], quantiles=True, y=y, start_snapshot=snaps[start - lead],
                          final_snapshot=True)
    dec = decisions(prep, cal, {start + HORIZON - 1: end["final"]}, i)
    charts = [(d - start, c) for d, c in chart_alarms(cal.charts, y, i, first, start + HORIZON, "WARP + NBM")
              if d >= start - lead]
    alarms, _ = twin_alarms(df[[f"p_{f}" for f in FAULTS]].to_numpy(), np.ones(len(ALARM_GROUPS), dtype=bool), spec)
    return {"scenario": scenario, "fault": fault.fault_type, "param": fault.param, "turbine": int(prep.farm.turbines[i]),
            "start": str(prep.obs.dates[start].date()), "truth": fault.truth(sign), "ramp": fault.ramp_days,
            "df": df, "rel": (df["day"] - start).to_numpy(), "charts": charts,
            "twin_alarms": [(int(df["day"].iloc[d] - start), FAULTS[f]) for d, f in alarms],
            "decisions": sorted(({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                  if k in ("fault", "p_fault", "cost_rate_per_day", "expected_loss",
                                           "value_of_inspection", "evpi")} for r in dec),
                                key=lambda r: -r["value_of_inspection"])}


def fig_replay(rp: dict, path: Path):
    df, rel = rp["df"], rp["rel"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True, height_ratios=[1.1, 1])
    for f in FAULTS:
        ax1.plot(rel, df[f"p_{f}"], color=FAULT_COLOR[f], lw=1.6 if f == rp["fault"] else 1.0,
                 alpha=1 if f == rp["fault"] else 0.6, label=FAULT_LABEL[f])
    ax1.axhline(0.9, color=MUTED, lw=0.8, ls="--")
    ax1.set_ylabel("P(material fault)")
    ax1.set_ylim(-0.03, 1.38)  # headroom: the legend sits above the probabilities, not on them
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.legend(ncol=3, fontsize=8, loc="upper left")
    p = rp["param"]
    truth = np.where(rel < 0, 0.0, rp["truth"] * (np.clip(rel / rp["ramp"], 0, 1) if rp["ramp"] else 1.0))
    ax2.fill_between(rel, df[f"q05_{p}"], df[f"q95_{p}"], color=TWIN, alpha=0.2, lw=0, label="90 % credible interval")
    ax2.plot(rel, df[f"mean_{p}"], color=TWIN, lw=1.6, label="posterior mean")
    ax2.plot(rel, truth, color=INK, lw=1.2, ls="--", label="injected")
    ax2.set_ylabel(PARAM_LABEL[p])
    ax2.set_xlabel("days from fault onset")
    for ax in (ax1, ax2):
        ax.axvline(0, color=INK2, lw=0.8)
        for d, c in rp["twin_alarms"][:1]:
            ax.axvline(d, color=TWIN, lw=1.2, ls=":")
        for d, c in [a for a in rp["charts"] if a[0] >= 0][:1]:
            ax.axvline(d, color=WARP_C, lw=1.2, ls=":")
    ax2.legend(fontsize=8, loc="best")
    first_twin = next((d for d, _ in rp["twin_alarms"] if d >= 0), None)
    first_chart = next((d for d, _ in rp["charts"] if d >= 0), None)
    ax1.set_title(f"T{rp['turbine']}, {rp['scenario']} from {rp['start']}: twin alarm day {first_twin}, "
                  f"WARP/NBM chart alarm day {first_chart}", fontsize=10, loc="left", color=INK)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- validation figures
def fig_detection(det: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(det))[::-1]
    ax.barh(y + 0.2, det["twin detected"] * 100, height=0.38, color=TWIN, label="Bayesian twin")
    ax.barh(y - 0.2, det["WARP + NBM detected"] * 100, height=0.38, color=CHARTS, label="WARP + NBM charts")
    for k, (_, r) in enumerate(det.iterrows()):
        for off, col in ((0.2, "twin"), (-0.2, "WARP + NBM")):
            d = r[f"{col} median days"]
            if np.isfinite(d):
                ax.text(r[f"{col} detected"] * 100 + 1, y[k] + off, f"{d:.0f} d", va="center", fontsize=7.5,
                        color=INK2)
    ax.set_yticks(y, det["scenario"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("detected within 90 days [%] (label: median days to detection)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(conf: dict, path: Path):
    fig, axes = plt.subplots(1, len(conf), figsize=(4.2 * len(conf), 3.8))
    for ax, (title, tab) in zip(np.atleast_1d(axes), conf.items()):
        ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=40, ha="right")
        ax.set_yticks(range(tab.shape[0]), tab.index)
        ax.grid(False)
        for r in range(tab.shape[0]):
            for c in range(tab.shape[1]):
                v = tab.iat[r, c]
                if v >= 0.005:
                    ax.text(c, r, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.55 else INK)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("diagnosis of the first alarm")
    np.atleast_1d(axes)[0].set_ylabel("injected fault")
    fig.savefig(path)
    plt.close(fig)


def fig_recovery(runs: pd.DataFrame, path: Path):
    kinds = [("yaw", "yaw γ [°]", 1.0), ("anemo", "anemometer error at 7 m/s [%]", 100.0),
             ("eff", "efficiency ln η [%]", 100.0), ("derate", "derating κ [%]", 100.0),
             ("gear", "gear bearing ρ − 1 [%]", 100.0), ("gen", "generator bearing ρ − 1 [%]", 100.0)]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
    fig.subplots_adjust(hspace=0.55, wspace=0.3)
    for ax, (f, lab, sc) in zip(axes.ravel(), kinds):
        g = runs[runs["fault"] == f]
        x = g["truth"] * sc + np.random.default_rng(0).normal(0, 0.012 * np.ptp(g["truth"] * sc) + 1e-9, len(g))
        inside = (g["est_q05"] <= g["truth"]) & (g["truth"] <= g["est_q95"])
        for ok, col in ((True, TWIN), (False, BAD)):
            s = inside == ok
            lo = np.clip(g.loc[s, "est_mean"] - g.loc[s, "est_q05"], 0, None) * sc  # a mixture's mean can lie
            hi = np.clip(g.loc[s, "est_q95"] - g.loc[s, "est_mean"], 0, None) * sc  # outside its 90 % interval
            ax.errorbar(x[s], g.loc[s, "est_mean"] * sc, yerr=[lo, hi],
                        fmt="o", ms=2.5, color=col, ecolor=col, elinewidth=0.6, alpha=0.6, lw=0)
        lim = np.array([min(g["truth"].min(), g["est_q05"].min()), max(g["truth"].max(), g["est_q95"].max())]) * sc
        ax.plot(lim, lim, color=MUTED, lw=0.8)
        ax.set_title(f"{lab}\n90 % interval covers the truth in {inside.mean() * 100:.0f} %", fontsize=8.5,
                     loc="left")
        ax.set_xlabel("injected")
    axes[0, 0].set_ylabel("estimated (day 90, 90 % CI)")
    axes[1, 0].set_ylabel("estimated (day 90, 90 % CI)")
    fig.savefig(path)
    plt.close(fig)


def fig_fleet(clean: pd.DataFrame, events: pd.DataFrame, path: Path):
    """Weekly maximum P(material fault) per turbine and type, 2019-2021, with logged events."""
    weeks = clean.assign(week=clean["date"].dt.to_period("W").dt.start_time)
    rows, labels = [], []
    for t in sorted(clean["turbine"].unique()):
        w = weeks[weeks["turbine"] == t].groupby("week")[[f"p_{f}" for f in FAULTS]].max()
        for f in FAULTS:
            rows.append(w[f"p_{f}"].to_numpy())
            labels.append(f"T{t} {f}")
    z = np.array(rows)
    xs = w.index
    fig, ax = plt.subplots(figsize=(10, 7.4))
    im = ax.imshow(z, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest",
                   extent=(0, len(xs), len(rows), 0))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="weekly maximum P(material fault)")
    ax.set_yticks(np.arange(len(rows)) + 0.5, labels, fontsize=7)
    ticks = [k for k, d in enumerate(xs) if d.month in (1, 7) and d.day <= 7]
    ax.set_xticks(ticks, [xs[k].strftime("%b %Y") for k in ticks])
    ax.grid(False)
    for k in range(1, 6):
        ax.axhline(k * len(FAULTS), color="white", lw=2)
    for _, e in events.iterrows():
        r = sorted(clean["turbine"].unique()).index(e["turbine"]) * len(FAULTS) + FAULTS.index(e["fault"])
        x = (pd.Timestamp(e["start"]).tz_localize(None) - xs[0]).days / 7.0
        ax.plot([x], [r + 0.5], marker="|", ms=8, mew=1.6, color=BAD)
    ax.set_title("Weekly maximum P(material fault), unmodified data; red ticks: related status-log messages",
                 fontsize=10, loc="left")
    fig.savefig(path)
    plt.close(fig)


def fig_waterfall(loss: pd.DataFrame, year: int, path: Path, price: float = 50.0):
    y = loss[loss["year"] == year].sum(numeric_only=True)
    steps = [("downtime", "downtime"), ("curtailment_icing", "curtailment, icing"), ("no_data_sensor", "no data"),
             ("health_eff", "efficiency loss"), ("health_yaw", "yaw misalignment"), ("health_derate", "derating"),
             ("performance_other", "other performance"), ("unassigned", "unassigned")]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    level = y["potential"]
    ax.bar(0, level, color=MUTED)
    ax.text(0, level, f"{level:,.0f}", ha="center", va="bottom", fontsize=8)
    x = 1
    for key, lab in steps:
        v = y[key]
        col = TWIN if key.startswith("health") else (CHARTS if v >= 0 else "#1b9e77")
        ax.bar(x, -v, bottom=level, color=col)
        ax.text(x, max(level, level - v), f"{-v:+,.0f}\n£{-v * price / 1000:+,.1f}k", ha="center",
                va="bottom", fontsize=7.5)
        level -= v
        x += 1
    ax.bar(x, y["actual"], color=INK2)
    ax.text(x, y["actual"], f"{y['actual']:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(x + 1), ["potential"] + [s[1] for s in steps] + ["actual"], rotation=30, ha="right")
    ax.set_ylim(y["actual"] * 0.85, y["potential"] * 1.04)
    ax.set_ylabel("MWh")
    ax.set_title(f"Kelmarsh {year}: energy by cause [MWh] (blue: health causes estimated by the twin, "
                 f"£{price:.0f}/MWh; axis cut)", fontsize=10, loc="left")
    fig.savefig(path)
    plt.close(fig)


def fig_tradeoff(tr: pd.DataFrame, path: Path):
    """Detection (and detection with the right cause) against false alarms, over the alarm thresholds."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, col, title in ((axes[0], "detected", "detected within 90 days"),
                           (axes[1], "right_cause", "detected, first alarm names the right cause")):
        # twin labels above-left of the point, chart labels below-right, so the two curves' labels do not collide
        for sys_name, color, lab, off, ha in (("twin", TWIN, "Bayesian twin (P threshold)", (-4, 2), "right"),
                                              ("WARP + NBM", CHARTS, "WARP + NBM charts (h x factor)", (3, -9), "left")):
            g = tr[tr["system"] == sys_name].sort_values("false_alarms_per_turbine_year")
            ax.plot(g["false_alarms_per_turbine_year"], g[col] * 100, "o-", color=color, ms=4, lw=1.4, label=lab)
            for _, r in g.iterrows():
                ax.annotate(f"{r['setting']:g}", (r["false_alarms_per_turbine_year"], r[col] * 100), fontsize=7,
                            color=color, xytext=off, textcoords="offset points", ha=ha)
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per turbine-year (unmodified 2019-2020)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("% of 624 injected faults")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- dashboard
def _r(a, nd=4):
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(f"{v:.{nd}g}") for v in a]


def dashboard_data(prep, clean, events, loss, dec, replays) -> dict:
    out = {"start": str(clean["date"].min().date()), "turbines": [int(t) for t in sorted(clean["turbine"].unique())],
           "faults": [{"key": f, "label": FAULT_LABEL[f], "color": FAULT_COLOR[f]} for f in FAULTS],
           "params": [{"key": p, "label": PARAM_LABEL[p]} for p in ("eff", "anemo7", "yaw", "derate", "gear", "gen")],
           "series": {}, "events": [], "waterfall": {}, "decisions": {}, "replays": []}
    for t, g in clean.groupby("turbine"):
        s = {f"p_{f}": _r(g[f"p_{f}"], 3) for f in FAULTS}
        for p in ("eff", "anemo7", "yaw", "derate", "gear", "gen"):
            for q in ("mean", "q05", "q95"):
                s[f"{q}_{p}"] = _r(g[f"{q}_{p}"])
        for c in ("eff", "yaw", "derate"):
            s[f"loss_{c}"] = _r(g[f"loss_{c}_kwh"], 3)
        out["series"][str(t)] = s
    d0 = clean["date"].min()
    for _, e in events.iterrows():
        out["events"].append({"turbine": int(e["turbine"]), "fault": e["fault"],
                              "day": int((e["start"].normalize() - d0).days), "messages": e["messages"]})
    for y, g in loss.groupby("year"):
        out["waterfall"][str(y)] = {k: float(round(v, 2)) for k, v in g.sum(numeric_only=True).items()
                                    if k not in ("turbine", "year")}
    for d, g in dec.groupby("date"):
        out["decisions"][str(pd.Timestamp(d).date())] = g.sort_values("value_of_inspection", ascending=False)[
            ["turbine", "fault", "p_fault", "cost_rate_per_day", "expected_loss", "value_of_inspection", "evpi"]
        ].round(3).to_dict("records")
    for rp in replays:
        df = rp["df"]
        out["replays"].append({"scenario": rp["scenario"], "turbine": rp["turbine"], "start": rp["start"],
                               "fault": rp["fault"], "param": rp["param"], "truth": rp["truth"], "ramp": rp["ramp"],
                               "rel": rp["rel"].tolist(), "charts": rp["charts"], "twin_alarms": rp["twin_alarms"],
                               "decisions": rp["decisions"],
                               **{f"p_{f}": _r(df[f"p_{f}"], 3) for f in FAULTS},
                               **{f"{q}_{p}": _r(df[f"{q}_{p}"]) for q in ("mean", "q05", "q95")
                                  for p in ("eff", "anemo7", "yaw", "derate", "gear", "gen")}})
    return out


def report(cache_dir: Path, results_dir: Path, docs_dir: Path) -> None:
    _style()
    img = docs_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    prep, cal = load_cached(cache_dir)
    runs = pd.read_csv(results_dir / "injection_runs.csv")
    det = pd.read_csv(results_dir / "detection.csv")
    fig_detection(det, img / "detection.png")
    conf = {"Bayesian twin": pd.read_csv(results_dir / "confusion_twin.csv", index_col=0),
            "WARP + NBM charts": pd.read_csv(results_dir / "confusion_warp_nbm.csv", index_col=0)}
    fig_confusion(conf, img / "confusion.png")
    fig_recovery(runs, img / "recovery.png")
    if (results_dir / "tradeoff.csv").exists():
        fig_tradeoff(pd.read_csv(results_dir / "tradeoff.csv"), img / "tradeoff.png")
    clean = pd.read_parquet(cache_dir / "clean_block.parquet")
    events = pd.read_csv(results_dir / "log_events.csv", parse_dates=["start", "end"])
    fig_fleet(clean, events, img / "fleet_2019_2021.png")
    loss = pd.read_csv(results_dir / "lost_energy_by_cause.csv")
    fig_waterfall(loss, 2021, img / "waterfall_2021.png")
    dec = pd.read_csv(results_dir / "decisions_quarterly.csv", parse_dates=["date"])
    replays = [replay(prep, cal, s, i, k) for s, i, k in REPLAYS]
    for rp, name in zip(replays, ("replay_yaw", "replay_eff_drift", "replay_gen", "replay_derate")):
        fig_replay(rp, img / f"{name}.png")
    data = dashboard_data(prep, clean, events, loss, dec, replays)
    template = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "index.html").write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))),
                                         encoding="utf-8")
    log.info("report written to %s", docs_dir)
