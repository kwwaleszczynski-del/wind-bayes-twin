"""Experiments behind the README: validation campaign, logged events, losses by cause and inspection ranking.

Usage:
    validate(cache_dir, results_dir, engine="block", workers=6)
    cases(cache_dir, results_dir)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import ENERGY_CAUSES, Economics, cost_rates, decision_values
from .changepoint import BlockTwin
from .farm import STATES
from .model import ALARM_GROUPS, FAULTS, ModelSpec
from .pipeline import APPLY_YEAR, TEST_YEARS, load_cached
from .validation import (H_MULTIPLIERS, P_THRESHOLDS, SYSTEMS, campaign, chart_alarms, false_alarms,
                         scaled_charts, spec_at, twin_alarms)
from .twin import turbine_noise

DECISION_SAMPLES = 2000
RECENT_DAYS = 30
# status-log messages related to each fault type (Kelmarsh, Senvion MM92 export)
LOG_GROUPS = {
    "anemo": r"[Aa]nemometer",
    "yaw": r"[Vv]ane|[Yy]aw error|Uncontrolled yaw",
    "gen": r"[Gg]enerator fan|generator heating|generator brushes",
    "gear": r"gear bearing|Oil filter gear|gear oil pump|fan gear",
    "eff": r"[Pp]itch|Limit switch error|Wind < power",
    "derate": r"Reduced power|dev\. electr\. power",
}
log = logging.getLogger(__name__)


# ---------------------------------------------------------------- summaries of the injection campaign
def detection_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per scenario: detection rate within 90 days and median days to detection, twin and chart systems."""
    rows = []
    for name, g in runs.groupby("scenario", sort=False):
        row = {"scenario": name, "runs": len(g)}
        for s in ("twin",) + tuple(SYSTEMS):
            det = g[f"{s}_detected"]
            row[f"{s} detected"] = det.mean()
            row[f"{s} median days"] = g.loc[det, f"{s}_days"].median()
            row[f"{s} right cause"] = (g[f"{s}_cause"] == g["fault"])[det].mean() if det.any() else np.nan
            row[f"{s} chance"] = g[f"{s}_chance"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def confusion(runs: pd.DataFrame, system: str = "twin") -> pd.DataFrame:
    """Injected fault type (rows) x diagnosis of the first alarm (columns), share of runs."""
    cols = list(FAULTS) + ["none"]
    tab = pd.crosstab(runs["fault"], runs[f"{system}_cause"], normalize="index")
    return tab.reindex(index=list(FAULTS), columns=cols, fill_value=0.0)


def recovery_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Parameter recovery at the end of the window: bias and coverage of the 90 % credible interval."""
    rows = []
    for name, g in runs.groupby("scenario", sort=False):
        err = g["est_mean"] - g["truth"]
        rows.append({"scenario": name, "truth (abs)": g["truth"].abs().median(),
                     "median estimate (abs)": (g["est_mean"] * np.sign(g["truth"])).median(),
                     "median error": float(np.median(err * np.sign(g["truth"]))),
                     "90% CI coverage": float(((g["est_q05"] <= g["truth"]) & (g["truth"] <= g["est_q95"])).mean()),
                     "median CI width": float((g["est_q95"] - g["est_q05"]).median()),
                     "median onset error [d]": (g.loc[g["twin_detected"], "onset_error_days"].median()
                                                if "onset_error_days" in g and g["fault"].iloc[0] != "derate"
                                                else np.nan)})
    return pd.DataFrame(rows)


def overall(runs: pd.DataFrame) -> dict:
    out = {}
    for s in ("twin",) + tuple(SYSTEMS):
        det = runs[f"{s}_detected"]
        out[s] = {"detected": float(det.mean()), "median_days": float(runs.loc[det, f"{s}_days"].median()),
                  "right_cause_of_detected": float((runs[f"{s}_cause"] == runs["fault"])[det].mean()),
                  "chance": float(runs[f"{s}_chance"].mean())}
    return out


def tradeoff(runs: pd.DataFrame, p_runs: np.ndarray, clean: pd.DataFrame, prep, cal) -> pd.DataFrame:
    """Detection rate, right diagnosis and false alarms per turbine-year at several alarm thresholds."""
    rows = []
    first = prep.days.of(f"{TEST_YEARS[0]}-01-01")
    starts = {pd.Timestamp(s): prep.days.of(s) for s in runs["start"].unique()}
    p_clean = {t: g[[f"p_{f}" for f in FAULTS]].to_numpy() for t, g in clean.groupby("turbine")}
    for thr in P_THRESHOLDS:
        spec = spec_at(thr)
        det, right = [], []
        for k, r in enumerate(runs.itertuples()):
            d0 = starts[pd.Timestamp(r.start)] - first
            _, armed = twin_alarms(p_clean[r.turbine][:d0], np.ones(len(ALARM_GROUPS), dtype=bool), spec)
            alarms, _ = twin_alarms(p_runs[k], armed, spec)
            det.append(bool(alarms))
            right.append(bool(alarms) and FAULTS[alarms[0][1]] == r.fault)
        fa, _ = false_alarms(clean, prep, cal, spec)
        rows.append({"system": "twin", "setting": thr, "detected": np.mean(det),
                     "right_cause": np.mean(right), "false_alarms_per_turbine_year":
                     float(fa.loc[fa["system"] == "twin (all types)", "per_turbine_year"].iloc[0])})
    for m in H_MULTIPLIERS:
        fa, _ = false_alarms(clean, prep, cal, charts=scaled_charts(cal.charts, m))
        cause = runs.get(f"charts_cause_m{m:g}")
        rows.append({"system": "WARP + NBM", "setting": m, "detected": runs[f"charts_det_m{m:g}"].mean(),
                     "right_cause": (cause == runs["fault"]).mean() if cause is not None else np.nan,
                     "false_alarms_per_turbine_year":
                     float(fa.loc[fa["system"] == "WARP + NBM", "per_turbine_year"].iloc[0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decisions (value of information)
def decisions(prep, cal, clean_snaps: dict, i: int, eco: Economics = Economics(), seed: int = 0) -> list[dict]:
    """Inspection values of every fault type of turbine column ``i`` at the snapshot days."""
    spec, table, cap = ModelSpec(), prep.table(), prep.cap_kw
    rec = prep.obs.records[i]
    out = []
    for d, snap in sorted(clean_snaps.items()):
        twin = BlockTwin(turbine_noise(cal.noise, i), spec, table, cap, cal.r_quantiles[i])
        twin.restore(snap)
        rng = np.random.default_rng(seed + d)
        samples = twin.sample(DECISION_SAMPLES, rng)
        on = np.zeros((DECISION_SAMPLES, len(FAULTS)), dtype=bool)
        on[:, FAULTS.index("anemo")] = np.abs(samples[:, 1] + samples[:, 2] / 7.0) > spec.material["anemo"]
        e_recent = rec.e_hat[rec.start[max(d - RECENT_DAYS + 1, 0)]:rec.start[d + 1]]
        rates = cost_rates(samples, e_recent, RECENT_DAYS, table, cap, eco, on)
        probs = twin.fault_probability()
        for k, f in enumerate(FAULTS):
            out.append({"turbine": int(prep.farm.turbines[i]), "date": prep.obs.dates[d], "fault": f,
                        "p_fault": float(probs[k]), "cost_rate_per_day": float(rates[f].mean()),
                        **decision_values(rates[f], f, eco)})
    return out


def _decision_worker(args):
    cache_dir, i, days = args
    from .twin import run_turbine

    prep, cal = load_cached(cache_dir)
    noise = turbine_noise(cal.noise, i)
    rng_days = prep.day_range(TEST_YEARS[0], APPLY_YEAR)
    _, snaps = run_turbine(prep.obs, i, noise, ModelSpec(), prep.table(), prep.cap_kw, rng_days, engine="block",
                           r_quantiles=cal.r_quantiles[i], snapshot_days=set(days))
    return decisions(prep, cal, snaps, i)


# ---------------------------------------------------------------- validation
def validate(cache_dir: Path, results_dir: Path, engine: str = "block", workers: int = 6,
             n_particles: int = 2000, guided: bool = True) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    runs, clean, p_runs = campaign(cache_dir, engine=engine, workers=workers, n_particles=n_particles,
                                   guided=guided, attribute=engine == "block")
    seconds = time.perf_counter() - t0
    clean_seconds = clean.attrs.get("seconds", [])
    tag = "" if engine == "block" else f"_{engine}" + ("" if guided else "_unguided")
    clean.to_parquet(cache_dir / f"clean_{engine}{tag.replace('_' + engine, '')}.parquet")
    np.save(cache_dir / f"injection_probs{tag}.npy", p_runs)
    runs.to_csv(results_dir / f"injection_runs{tag}.csv", index=False, float_format="%.5g")
    detection_table(runs).to_csv(results_dir / f"detection{tag}.csv", index=False, float_format="%.4g")
    recovery_table(runs).to_csv(results_dir / f"recovery{tag}.csv", index=False, float_format="%.4g")
    for s in ("twin", "WARP + NBM"):
        confusion(runs, s).to_csv(results_dir / f"confusion_{s.replace(' + ', '_').lower()}{tag}.csv",
                                  float_format="%.3f")
    prep, cal = load_cached(cache_dir)
    fa, episodes = false_alarms(clean, prep, cal)
    fa.to_csv(results_dir / f"false_alarms{tag}.csv", index=False, float_format="%.3f")
    tradeoff(runs, p_runs, clean, prep, cal).to_csv(results_dir / f"tradeoff{tag}.csv", index=False,
                                                    float_format="%.4g")
    episodes.to_csv(results_dir / f"alarms_test_clean{tag}.csv", index=False)
    summary = {"engine": engine, "runs": len(runs), "overall": overall(runs),
               "false_alarms_per_turbine_year": dict(zip(fa["system"], fa["per_turbine_year"].round(3))),
               "campaign_seconds": round(seconds, 1), "workers": workers,
               "clean_run_seconds_per_turbine": [round(s, 1) for s in clean_seconds],
               "clean_run_days": int(clean.groupby("turbine").size().iloc[0]),
               "injection_run_seconds_median": round(float(runs["seconds"].median()), 2)}
    if engine == "particle":
        summary["particles"] = n_particles
        summary["guided"] = guided
    (results_dir / f"validation{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("validation (%s): %d runs in %.0f s", engine, len(runs), seconds)


# ---------------------------------------------------------------- logged events
def log_episodes(status: pd.DataFrame, first_year: int, last_year: int, merge_days: int = 14) -> pd.DataFrame:
    """Status-log events of each fault-related group, merged per turbine within ``merge_days``."""
    st = status[(status["start"].dt.year >= first_year) & (status["start"].dt.year <= last_year)]
    rows = []
    for fault, pattern in LOG_GROUPS.items():
        sel = st[st["message"].fillna("").str.contains(pattern, regex=True)]
        for t, g in sel.groupby("turbine"):
            g = g.sort_values("start")
            start, last, msgs = None, None, []
            for _, ev in g.iterrows():
                if start is not None and (ev["start"] - last).days > merge_days:
                    rows.append({"turbine": int(t), "fault": fault, "start": start, "end": last,
                                 "messages": "; ".join(sorted(set(msgs)))})
                    start, msgs = None, []
                start = ev["start"] if start is None else start
                last = ev["start"]
                msgs.append(ev["message"])
            if start is not None:
                rows.append({"turbine": int(t), "fault": fault, "start": start, "end": last,
                             "messages": "; ".join(sorted(set(msgs)))})
    return pd.DataFrame(rows)


def cases(cache_dir: Path, results_dir: Path, lead_days: int = 60) -> None:
    """Logged events 2019-2021 against the twin and the charts on unmodified data; 2021 state and decisions."""
    from concurrent.futures import ProcessPoolExecutor

    prep, cal = load_cached(cache_dir)
    clean = pd.read_parquet(cache_dir / "clean_block.parquet")
    spec = ModelSpec()
    first_day = prep.days.of(f"{TEST_YEARS[0]}-01-01")
    last_day = prep.days.of(f"{APPLY_YEAR}-12-31") + 1
    # alarms of the twin and the charts over 2019-2021 (unmodified data)
    alarm_rows = []
    for t, g in clean.groupby("turbine"):
        i = int(np.flatnonzero(prep.farm.turbines == t)[0])
        alarms, _ = twin_alarms(g[[f"p_{f}" for f in FAULTS]].to_numpy(), np.ones(len(ALARM_GROUPS), dtype=bool),
                                spec)
        alarm_rows += [{"turbine": t, "date": g["date"].iloc[d], "system": "twin", "cause": FAULTS[f]}
                       for d, f in alarms]
        for s in ("WARP", "NBM"):
            from .baselines import CHART_CAUSE
            alarm_rows += [{"turbine": t, "date": prep.obs.dates[d], "system": s, "cause": CHART_CAUSE[c]}
                           for d, c in chart_alarms(cal.charts, prep.obs.y, i, first_day, last_day, s)]
    alarms = pd.DataFrame(alarm_rows)
    alarms.to_csv(results_dir / "alarms_2019_2021.csv", index=False)
    eps = log_episodes(prep.status, TEST_YEARS[0], APPLY_YEAR)
    # score: an alarm of the matching type in the 60 days before the logged episode (early warning), or around it
    rows = []
    span_days = (prep.obs.dates[last_day - 1] - prep.obs.dates[first_day]).days + 1
    for _, ep in eps.iterrows():
        row = ep.to_dict()
        g = clean[clean["turbine"] == ep["turbine"]].set_index("date")
        win = g.loc[ep["start"].normalize() - pd.Timedelta(days=lead_days):ep["start"].normalize()]
        row["twin_p_max_before"] = float(win[f"p_{ep['fault']}"].max()) if len(win) else np.nan
        around = g.loc[ep["start"].normalize() - pd.Timedelta(days=7):ep["end"].normalize() + pd.Timedelta(days=30)]
        row["twin_p_max_around"] = float(around[f"p_{ep['fault']}"].max()) if len(around) else np.nan
        for s in ("twin", "WARP", "NBM"):
            a = alarms[(alarms["system"] == s) & (alarms["turbine"] == ep["turbine"]) & (alarms["cause"] == ep["fault"])]
            before = a[(a["date"] < ep["start"]) & (a["date"] >= ep["start"] - pd.Timedelta(days=lead_days))]
            near = a[(a["date"] >= ep["start"] - pd.Timedelta(days=7)) & (a["date"] <= ep["end"] + pd.Timedelta(days=30))]
            row[f"{s}_before"] = len(before) > 0
            row[f"{s}_around"] = len(near) > 0
            # chance: share of days of the period with such an alarm in the preceding lead_days
            n_alarm = len(alarms[(alarms["system"] == s) & (alarms["cause"] == ep["fault"])
                                 & (alarms["turbine"] == ep["turbine"])])
            row[f"{s}_chance_before"] = min(1.0, n_alarm * lead_days / span_days)
        rows.append(row)
    scored = pd.DataFrame(rows)
    scored.to_csv(results_dir / "log_events.csv", index=False, float_format="%.3f")
    # decisions at quarter ends and the 2021 state per turbine
    q_days = [prep.days.of(d) for d in pd.date_range(f"{TEST_YEARS[0]}-03-31", f"{APPLY_YEAR}-12-31", freq="QE")]
    with ProcessPoolExecutor(max_workers=6) as ex:
        dec = sum(ex.map(_decision_worker, [(cache_dir, i, q_days) for i in range(prep.farm.n)]), [])
    dec = pd.DataFrame(dec)
    dec.to_csv(results_dir / "decisions_quarterly.csv", index=False, float_format="%.4g")
    end = dec[dec["date"] == dec["date"].max()].sort_values("value_of_inspection", ascending=False)
    end.to_csv(results_dir / "inspection_ranking_2021.csv", index=False, float_format="%.4g")
    losses(prep, clean).to_csv(results_dir / "lost_energy_by_cause.csv", index=False, float_format="%.4g")
    log.info("cases: %d logged episodes, %d alarms", len(scored), len(alarms))


# ---------------------------------------------------------------- energy waterfall
def losses(prep, clean: pd.DataFrame, eco: Economics = Economics()) -> pd.DataFrame:
    """Energy per turbine and year [MWh]: potential, operational losses, health losses by cause, actual."""
    farm, terms = prep.farm, prep.terms
    years = farm.time.year
    rows = []
    state = farm.state
    for i, t in enumerate(farm.turbines):
        g = clean[clean["turbine"] == t]
        gy = g["date"].dt.year
        for y in sorted(set(gy)):
            sel = years == y
            p_hat = terms.p_hat[sel, i]
            p = np.nan_to_num(farm.power[sel, i])
            st = state[sel, i]
            ok = np.isfinite(p_hat)
            row = {"turbine": int(t), "year": int(y),
                   "potential": np.nansum(p_hat) / 6000.0, "actual": p.sum() / 6000.0}
            for name, states in (("downtime", ("stop", "stop_grid")), ("curtailment_icing", ("curtailed", "icing")),
                                 ("no_data_sensor", ("missing", "sensor"))):
                m = ok & np.isin(st, [STATES.index(s) for s in states])
                row[name] = np.sum(p_hat[m] - p[m]) / 6000.0
            normal = ok & (st == 0)
            perf = np.sum(p_hat[normal] - p[normal]) / 6000.0
            for c in ENERGY_CAUSES:
                row[f"health_{c}"] = g.loc[gy == y, f"loss_{c}_kwh"].sum() / 1000.0
            row["performance_other"] = perf - sum(row[f"health_{c}"] for c in ENERGY_CAUSES)
            # records outside every category (reference missing) close the balance
            row["unassigned"] = (row["potential"] - row["actual"] - row["downtime"] - row["curtailment_icing"]
                                 - row["no_data_sensor"] - perf)
            row["health_value_gbp"] = sum(row[f"health_{c}"] for c in ENERGY_CAUSES) * eco.price_per_mwh
            rows.append(row)
    return pd.DataFrame(rows)
