from conftest import RATED, simulate
from test_changepoint import theta_step

from wind_bayes_twin.model import FAULTS, ModelSpec
from wind_bayes_twin.twin_filter import TwinFilter


def run(days, noise, table, seed=1, guided=True, n=1500):
    flt = TwinFilter(noise, ModelSpec(), table, RATED, n_particles=n, seed=seed, guided=guided)
    return [flt.step(d) for d in days]


def test_particle_filter_recovers_yaw_and_efficiency(noise, table):
    days = simulate(noise, theta_step(40, yaw=6.0, eff=-0.05), 130, table, seed=21, rated_share=0.6)
    out = run(days, noise, table)
    last = out[-1]
    assert last["p_yaw"] > 0.9 and last["p_eff"] > 0.9
    assert last["q05_yaw"] <= 6.0 <= last["q95_yaw"]
    assert last["q05_eff"] <= -0.05 <= last["q95_eff"]


def test_particle_filter_quiet_when_healthy(noise, table):
    days = simulate(noise, theta_step(10 ** 6), 250, table, seed=5)
    out = run(days, noise, table)
    assert max(max(o[f"p_{f}"] for f in FAULTS) for o in out) < 0.9


def test_same_seed_same_result(noise, table):
    days = simulate(noise, theta_step(20, gear=0.3), 60, table, seed=3)
    a = run(days, noise, table, seed=7, n=500)
    b = run(days, noise, table, seed=7, n=500)
    c = run(days, noise, table, seed=8, n=500)
    keys = [k for k in a[-1] if k.startswith(("p_", "mean_"))]
    assert all(a[-1][k] == b[-1][k] for k in keys)
    assert any(a[-1][k] != c[-1][k] for k in keys)  # a different seed is a different Monte Carlo sample
    assert abs(a[-1]["mean_gear"] - c[-1]["mean_gear"]) < 0.05  # ... of the same posterior
