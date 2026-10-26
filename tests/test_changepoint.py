import numpy as np
import pytest
from conftest import RATED, make_noise, simulate

from wind_bayes_twin.changepoint import BLOCKS, BlockFilter, BlockTwin, power_offsets
from wind_bayes_twin.model import ALARM_GROUPS, FAULTS, ModelSpec

R_ONES = np.ones(20)


def theta_step(day0, **params):
    names = ("eff", "ascale", "abias", "yaw", "derate", "gear", "gen")

    def fn(d):
        th = np.zeros(7)
        if d >= day0:
            for k, v in params.items():
                th[names.index(k)] = v
        return th
    return fn


def run(days, noise, table, quantiles_last=True):
    twin = BlockTwin(noise, ModelSpec(), table, RATED, R_ONES)
    out = [twin.step(d, quantiles=quantiles_last and k == len(days) - 1) for k, d in enumerate(days)]
    return out, twin


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.isclose(a[k], b[k], rtol=0, atol=0, equal_nan=True) for k in a)


def first_alarm(out, fault, p=0.9):
    """First day on which the alarm group of ``fault`` has a material fault with probability > p."""
    group = next(g for g in ALARM_GROUPS if fault in g)
    hit = [k for k, o in enumerate(out) if sum(o[f"p_{f}"] for f in group) > p]
    return hit[0] if hit else None


def test_healthy_stays_quiet(noise, table):
    days = simulate(noise, theta_step(10 ** 6), 400, table, seed=3)
    out, _ = run(days, noise, table)
    worst = max(max(o[f"p_{f}"] for f in FAULTS) for o in out)
    assert worst < 0.9


@pytest.mark.parametrize("fault,param,value,stat_param", [
    ("yaw", "yaw", 6.0, "yaw"),
    ("gear", "gear", 0.3, "gear"),
    ("anemo", "ascale", 0.05, "anemo7"),
    ("eff", "eff", -0.05, "eff"),
])
def test_step_fault_is_found_and_sized(noise, table, fault, param, value, stat_param):
    days = simulate(noise, theta_step(60, **{param: value}), 160, table, seed=11)
    out, _ = run(days, noise, table)
    hit = first_alarm(out, fault)
    assert hit is not None and 60 <= hit < 100
    last = out[-1]
    assert last[f"q05_{stat_param}"] <= value <= last[f"q95_{stat_param}"]
    # the fault is attributed to the right type, not another one
    assert max(last[f"p_{f}"] for f in FAULTS if f != fault) < 0.5


def test_derating_on_grid(noise, table):
    days = simulate(noise, theta_step(40, derate=0.05), 140, table, seed=5, rated_share=0.6)
    out, _ = run(days, noise, table)
    assert first_alarm(out, "derate") is not None
    assert out[-1]["q05_derate"] <= 0.05 <= out[-1]["q95_derate"]
    assert out[-1]["p_eff"] < 0.5


def test_onset_is_dated(noise, table):
    days = simulate(noise, theta_step(80, yaw=8.0), 140, table, seed=2)
    out, _ = run(days, noise, table)
    assert abs(out[-1]["onset_age_yaw"] - (139 - 80)) <= 3


def test_deterministic(noise, table):
    days = simulate(noise, theta_step(30, gen=0.3), 80, table, seed=4)
    a, _ = run(days, noise, table)
    b, _ = run(days, noise, table)
    assert same(a[-1], b[-1])


def test_snapshot_restart_equals_continuous(noise, table):
    days = simulate(noise, theta_step(40, yaw=5.0), 90, table, seed=8)
    twin = BlockTwin(noise, ModelSpec(), table, RATED, R_ONES)
    for d in days[:50]:
        twin.step(d)
    snap = twin.snapshot()
    cont = [twin.step(d) for d in days[50:]]
    twin2 = BlockTwin(noise, ModelSpec(), table, RATED, R_ONES)
    twin2.restore(snap)
    again = [twin2.step(d) for d in days[50:]]
    assert same(cont[-1], again[-1])


def test_interval_coverage_is_nominal(table):
    """90 % intervals of the gear cooling parameter cover the truth in about 90 % of replications."""
    noise = make_noise()
    block = next(b for b in BLOCKS if b.name == "gear")
    hits, n_rep = 0, 120
    rng = np.random.default_rng(0)
    for rep in range(n_rep):
        truth = rng.uniform(0.15, 0.4)
        days = simulate(noise, theta_step(20, gear=truth), 90, table, seed=100 + rep)
        flt = BlockFilter(block, noise, ModelSpec())
        for d in days:
            flt.predict()
            flt.update(np.array([d.y[5]]), np.array([d.count[5]]), np.array([[d.rise_gear]]), np.zeros(1))
        s = flt.summary(quantiles=True)
        hits += s["q05_gear"] <= truth <= s["q95_gear"]
    assert 0.82 <= hits / n_rep <= 0.97


def test_power_offsets_no_cap_is_yaw_only(table):
    e = np.linspace(5, 15, 50)
    p_hat = table(e)
    full = e >= 13.0
    off = power_offsets(e, p_hat, full, 0.0, R_ONES, table, RATED)
    assert np.allclose(off[0], 0.0, atol=1e-9)
    off8 = power_offsets(e, p_hat, full, 8.0, R_ONES, table, RATED)
    assert off8[0, 0] < -0.01  # yaw loss in partial load
    assert np.all(np.diff(off[1:, 1]) < 0)  # deeper cap, lower full-load statistic
