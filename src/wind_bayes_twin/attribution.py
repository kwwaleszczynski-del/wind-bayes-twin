"""Lost energy by cause (Shapley attribution) and the value-of-information ranking of inspections.

Lost energy. Three of the fault types cost energy directly: efficiency loss, yaw misalignment and derating. For a
parameter vector theta and a set of records, the energy lost against a healthy turbine is

    L(theta) = sum_t [ F(e^_t) - P_t(theta) ] dt,    P_t = min(eta F(e^_t cos(gamma)^(n/3)), (1 - kappa) P_r)

The causes interact (a power cap hides part of an efficiency loss), so L is split between them by the Shapley
value: the average extra loss of switching a cause on, over all orders in which the causes can be switched on.
The split adds up exactly to L. Applied to posterior samples it gives the expected lost energy per cause with a
credible interval.

Value of information (simplified, every assumption explicit in ``Economics``). For turbine i and fault type k the
decision is whether to inspect now. An inspection costs C_insp and reveals the true state; a fault found is fixed
at cost C_fix_k, which removes its cost for the rest of the horizon H (the time until the fault would be found
anyway, here half a year to the next scheduled service). With V_k = cost rate of k under the posterior x H:

    value of inspecting   = E[ max(V_k - C_fix_k, 0) ] - C_insp
    EVPI (information)    = E[ max(V_k - C_fix_k, 0) ] - max( E[V_k] - C_fix_k, 0 )

The ranking is by the value of inspecting (expected avoided loss net of fixing and inspection costs); the EVPI
shows how much of it comes from resolving uncertainty rather than from an already certain fault. Cost rates:
energy x price for efficiency, yaw and derating (on the last 30 days' wind; a posterior draw that is better than
healthy costs nothing, not a negative amount); for the bearing cooling faults an
illustrative failure-risk cost (annual failure probability up by ``fail_prob_per_25pct`` per +25 % thermal
resistance, times ``failure_cost``); for an anemometer a flat illustrative cost (degraded control and reporting).

Usage:
    losses = shapley_losses(samples, e_hat, table, cap_kw)            # (M, 3) kWh
    table = inspection_ranking(posteriors, Economics())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from .model import FAULTS, turbine_power

ENERGY_CAUSES = ("eff", "yaw", "derate")
CAUSE_PARAM = {"eff": 0, "yaw": 3, "derate": 4}
STEP_H = 10.0 / 60.0


@dataclass
class Economics:
    price_per_mwh: float = 50.0  # GBP, assumed flat price
    horizon_days: float = 180.0  # until the fault would be found anyway (next scheduled service)
    inspection_cost: float = 500.0
    fix_cost: dict = field(default_factory=lambda: {"eff": 4000.0, "anemo": 800.0, "yaw": 600.0, "derate": 200.0,
                                                    "gear": 3000.0, "gen": 2000.0})
    failure_cost: dict = field(default_factory=lambda: {"gear": 150_000.0, "gen": 60_000.0})
    fail_prob_per_25pct: float = 0.10  # added annual failure probability per +25 % thermal resistance
    anemo_cost_per_day: float = 5.0


def healthy_power(e_hat, table, cap_kw) -> np.ndarray:
    return np.minimum(table(e_hat), cap_kw)


def shapley_losses(theta: np.ndarray, e_hat: np.ndarray, table, cap_kw: float) -> np.ndarray:
    """(M, 3) energy lost [kWh] over the records, split between efficiency, yaw and derating (Shapley)."""
    m = len(theta)
    if len(e_hat) == 0:
        return np.zeros((m, len(ENERGY_CAUSES)))
    base = healthy_power(e_hat, table, cap_kw).sum() * STEP_H
    value = {}
    for r in range(len(ENERGY_CAUSES) + 1):
        for subset in combinations(range(len(ENERGY_CAUSES)), r):
            th = np.zeros_like(theta)
            for c in subset:
                th[:, CAUSE_PARAM[ENERGY_CAUSES[c]]] = theta[:, CAUSE_PARAM[ENERGY_CAUSES[c]]]
            value[subset] = base - turbine_power(th, e_hat, table, cap_kw).sum(axis=1) * STEP_H
    n = len(ENERGY_CAUSES)
    out = np.zeros((m, n))
    for c in range(n):
        others = [k for k in range(n) if k != c]
        for r in range(n):
            for subset in combinations(others, r):
                wgt = factorial(r) * factorial(n - r - 1) / factorial(n)
                with_c = tuple(sorted(subset + (c,)))
                out[:, c] += wgt * (value[with_c] - value[subset])
    return out


def cost_rates(samples: np.ndarray, e_hat_recent: np.ndarray, n_days_recent: int, table, cap_kw: float,
               eco: Economics, on: np.ndarray) -> dict:
    """Cost per day [GBP] of each fault type for each posterior sample (energy on the recent records)."""
    kwh = shapley_losses(samples, e_hat_recent, table, cap_kw) / max(n_days_recent, 1)
    rates = {c: np.clip(kwh[:, k], 0.0, None) / 1000.0 * eco.price_per_mwh for k, c in enumerate(ENERGY_CAUSES)}
    for c, j in (("gear", 5), ("gen", 6)):
        excess = np.clip(samples[:, j], 0.0, None)
        rates[c] = excess / 0.25 * eco.fail_prob_per_25pct * eco.failure_cost[c] / 365.0
    rates["anemo"] = on[:, FAULTS.index("anemo")] * eco.anemo_cost_per_day
    return rates


def decision_values(rate: np.ndarray, fault: str, eco: Economics) -> dict:
    v = rate * eco.horizon_days
    fix = eco.fix_cost[fault]
    with_info = float(np.mean(np.maximum(v - fix, 0.0)))
    return {"expected_loss": float(v.mean()), "value_of_inspection": with_info - eco.inspection_cost,
            "evpi": with_info - max(float(v.mean()) - fix, 0.0)}


def inspection_ranking(rows: list[dict]) -> pd.DataFrame:
    """Rows of (turbine, fault, p_fault, expected_loss, value_of_inspection, evpi) ranked by value of inspection."""
    df = pd.DataFrame(rows)
    return df.sort_values("value_of_inspection", ascending=False).reset_index(drop=True)
