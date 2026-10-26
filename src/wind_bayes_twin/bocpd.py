"""Bayesian online change-point detection (Adams & MacKay 2007) for a standardised daily statistic.

The series is cut into segments with a constant mean; inside a segment the values are Gaussian with unit
variance around an unknown mean m ~ N(0, s0^2). With a constant hazard H the run length r (days since the last
change) has the recursion

    p(r_t = r + 1, x_1:t) = p(r_{t-1} = r, x_1:t-1) pi_r(x_t) (1 - H)
    p(r_t = 0,     x_1:t) = sum_r p(r_{t-1} = r, x_1:t-1) pi_r(x_t) H

where pi_r is the posterior predictive of the segment of length r: N(m_r, 1 + 1/k_r), k_r = 1/s0^2 + r and
m_r = (sum of the segment's values) / k_r. Run lengths beyond ``r_max`` are merged into the last one. Days
without a value leave the posterior unchanged.

In the twin it runs on every statistic and is used only to *propose* jumps in the particle filter: when it says
that something changed recently, more particles try a fault onset of the size the new segment suggests. Because
the particle weights correct for the proposal, this changes the Monte Carlo efficiency, not the posterior.

Usage:
    cp = BOCPD(hazard=1 / 200)
    for x in z:
        cp.update(x)
        cp.p_recent(7), cp.recent_mean(30)
"""
from __future__ import annotations

import numpy as np


class BOCPD:
    def __init__(self, hazard: float = 1 / 200, s0: float = 2.0, r_max: int = 400):
        self.h, self.s0, self.r_max = hazard, s0, r_max
        self.p = np.array([1.0])  # run-length posterior p(r_t | x_1:t)
        self.sum = np.array([0.0])  # segment sums per run length
        self.cnt = np.array([0.0])  # segment counts per run length

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        k = 1.0 / self.s0 ** 2 + self.cnt
        m = self.sum / k
        var = 1.0 + 1.0 / k
        pred = np.exp(-0.5 * (x - m) ** 2 / var) / np.sqrt(2 * np.pi * var)
        growth = self.p * pred * (1 - self.h)
        cp = float(np.sum(self.p * pred) * self.h)
        p = np.concatenate([[cp], growth])
        s = np.concatenate([[0.0], self.sum]) + x
        c = np.concatenate([[0.0], self.cnt]) + 1.0
        if len(p) > self.r_max:  # merge the oldest run lengths
            p[self.r_max - 1] += p[self.r_max:].sum()
            p, s, c = p[:self.r_max], s[:self.r_max], c[:self.r_max]
        total = p.sum()
        self.p = p / total if total > 0 else np.r_[1.0, np.zeros(len(p) - 1)]
        self.sum, self.cnt = s, c

    def p_recent(self, days: int) -> float:
        """Posterior probability that the current segment started within the last ``days`` observations."""
        return float(self.p[:days].sum())

    def recent_mean(self, days: int) -> float:
        """Posterior mean level of the current segment, given that it started within the last ``days``."""
        w = self.p[:days]
        if w.sum() <= 0:
            return 0.0
        m = self.sum[:days] / (1.0 / self.s0 ** 2 + self.cnt[:days])
        return float(np.sum(w * m) / w.sum())

    def map_run_length(self) -> int:
        return int(np.argmax(self.p))
