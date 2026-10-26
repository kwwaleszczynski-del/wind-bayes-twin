"""Online inference: a Rao-Blackwellised particle filter with change-point-guided jump proposals.

Each particle carries the seven health parameters and the on/off state of the six fault types (``model``). The
healthy level mu_s and the AR(1) wander n_s of every statistic enter the observations linearly and with Gaussian
noise, so they are integrated out exactly with a two-state Kalman filter per particle and statistic (Rao-
Blackwellisation; Doucet et al. 2000). The Kalman covariance depends only on which days were observed, so it is
shared by all particles and only the means are per particle. The particle weight of a day is the product of the
Kalman predictive densities of the observed statistics:

    y_s ~ N( h_s(theta) + m_mu + m_n ,  S_s ),   S_s = [1 1] P_s [1 1]' + sigma_e,s^2

Proposal. Most particles move with the prior dynamics (spike-and-slab onsets, random walk, repair). A change-point
detector (``bocpd``, Adams & MacKay 2007) runs on every statistic; when it reports a recent change in a statistic
linked to fault type k, onsets of k are proposed more often (rate a_k instead of the prior hazard h_k) and their size
is drawn half from the prior slab and half from a normal centred on the level of the new segment. The weights are
corrected by prior / proposal for both the timing and the size, so the filter still targets the posterior of the
prior model; the detector only puts particles where the data point.

Healthy level. mu_s follows a slow random walk (step ``level_walk``, from the year-to-year change of the healthy
levels between the baseline and the calibration year): what is normal for a turbine is not fixed for ever.

Robustness. Observations are modelled as contaminated: with probability ``P_OUTLIER`` a day's value comes from a
component ``OUT_SCALE`` times wider (icing, maintenance, a logging glitch). A day whose posterior outlier probability
under the particle mixture exceeds 0.5 is skipped, unless the two previous values of that statistic were skipped
in the same direction (a real jump persists, an outlier does not); on the other days the particle weights use the
contaminated likelihood, which bounds the influence of a single day. Particles are resampled (systematic) when the
effective sample size falls below half the particle count.

Usage:
    flt = TwinFilter(noise, ModelSpec(), CurveTable(curve), cap_kw=2050, n_particles=2000, seed=1)
    for d in range(start, end):
        summary = flt.step(day_input(obs, i, d))
    snap = flt.snapshot(); flt.restore(snap)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .bocpd import BOCPD
from .hierarchy import OUT_SCALE, P_OUTLIER, count_scale
from .model import (FAULT_PARAMS, FAULT_STATS, FAULTS, PARAMS, ModelSpec, anemo_stat, material, power_stats,
                    slab_logpdf, slab_sample)
from .stats import STATS

PERSIST = 2
GUIDE_MAX = 0.2  # largest proposed onset rate per fault type and day
GUIDE_GAIN = 0.5
RECENT_DAYS = 7
SEGMENT_DAYS = 30
GUIDE_FLOOR = {"eff": 0.005, "ascale": 0.005, "yaw": 1.0, "derate": 0.01, "gear": 0.03, "gen": 0.03}
S = {s: k for k, s in enumerate(STATS)}
P = {p: k for k, p in enumerate(PARAMS)}


@dataclass
class Noise:
    """Healthy level and noise of the seven statistics for one turbine (from ``hierarchy``)."""

    mu_mean: np.ndarray
    mu_var: np.ndarray
    sigma_e: np.ndarray
    sigma_w: np.ndarray
    phi: np.ndarray
    c_ref: np.ndarray
    level_walk: np.ndarray

    @property
    def stationary_sd(self) -> np.ndarray:
        return np.sqrt(self.sigma_e ** 2 + self.sigma_w ** 2 / (1 - self.phi ** 2))


@dataclass
class DayInput:
    y: np.ndarray  # (7,) statistics, NaN if missing
    count: np.ndarray  # (7,) records behind each statistic
    e_hat: np.ndarray  # records behind pow / full
    p_hat: np.ndarray
    full: np.ndarray  # full-load record (e^ >= 13 m/s)
    u_lo: float
    u_hi: float
    rise_gear: float
    rise_gen: float


def _lognorm(v, var):
    return -0.5 * (np.log(2 * np.pi * var) + v ** 2 / var)


def weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    return x[order][np.minimum(np.searchsorted(cw, qs), len(x) - 1)]


class TwinFilter:
    def __init__(self, noise: Noise, spec: ModelSpec, table, cap_kw: float, n_particles: int = 2000, seed: int = 0,
                 guided: bool = True, r_quantiles: np.ndarray | None = None):
        self.noise, self.spec, self.table, self.cap, self.r_q = noise, spec, table, cap_kw, r_quantiles
        self.n, self.guided = n_particles, guided
        self.rng = np.random.default_rng(seed)
        n_s = len(STATS)
        self.theta = np.zeros((self.n, len(PARAMS)))
        self.on = np.zeros((self.n, len(FAULTS)), dtype=bool)
        self.m = np.zeros((self.n, n_s, 2))
        self.m[:, :, 0] = noise.mu_mean[None, :]
        self.P = np.zeros((n_s, 2, 2))
        self.P[:, 0, 0] = noise.mu_var
        self.P[:, 1, 1] = noise.sigma_w ** 2 / (1 - noise.phi ** 2)
        self.logw = np.zeros(self.n)
        self.cp = [BOCPD() for _ in STATS]
        self.gated = np.zeros(n_s, dtype=int)
        self.rise_mem = {"gear": 15.0, "gen": 15.0}

    # ------------------------------------------------------------------ state handling
    def snapshot(self) -> dict:
        return copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in ("noise", "spec", "table", "r_q")})

    def restore(self, snap: dict) -> None:
        self.__dict__.update(copy.deepcopy(snap))

    def weights(self) -> np.ndarray:
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    # ------------------------------------------------------------------ proposal
    def _targets(self) -> dict:
        """Onset rate and target level proposed for each fault type from the change-point detectors."""
        sd = self.noise.stationary_sd
        out = {}
        for k in FAULTS:
            best = max(FAULT_STATS[k], key=lambda s: self.cp[S[s]].p_recent(RECENT_DAYS))
            p_rec = self.cp[S[best]].p_recent(RECENT_DAYS)
            shift = self.cp[S[best]].recent_mean(SEGMENT_DAYS) * sd[S[best]]
            if k == "derate":
                level = -shift
            elif k in ("gear", "gen"):
                level = shift / max(self.rise_mem[k], 1.0)
            else:
                level = shift
            out[k] = (max(self.spec.hazard[k], min(GUIDE_MAX, GUIDE_GAIN * p_rec)), level)
        return out

    def _propagate(self) -> None:
        spec, rng, n = self.spec, self.rng, self.n
        targets = self._targets() if self.guided else {}
        for f, k in enumerate(FAULTS):
            idx = FAULT_PARAMS[k]
            # repair: a fault that is on switches off
            rep = self.on[:, f] & (rng.random(n) < spec.hazard_repair)
            self.on[rep, f] = False
            self.theta[np.ix_(rep, idx)] = 0.0
            # random walk of active faults
            for j in idx:
                step = rng.normal(0.0, spec.walk[PARAMS[j]], n)
                self.theta[:, j] += np.where(self.on[:, f], step, 0.0)
            if k == "derate":
                self.theta[:, P["derate"]] = np.clip(self.theta[:, P["derate"]], 0.0, 0.9)
            # onsets / further jumps
            h = spec.hazard[k]
            rate, level = targets.get(k, (h, None))
            jump = rng.random(n) < rate
            self.logw += np.where(jump, np.log(h / rate), np.log((1 - h) / (1 - rate)))
            if not jump.any():
                continue
            for pos, j in enumerate(idx):
                name = PARAMS[j]
                slab, floor = spec.slab[name], spec.slab_floor.get(name, 0.0)
                size = slab_sample(rng, slab, n, floor)
                if level is not None and pos == 0:  # guided size for the main parameter of the fault
                    want = level - self.theta[:, j]
                    spread = 0.3 * np.abs(want) + GUIDE_FLOOR.get(name, 0.01)
                    guided = rng.normal(want, spread)
                    use = rng.random(n) < 0.5
                    size = np.where(use, guided, size)
                    q_g = -0.5 * ((size - want) / spread) ** 2 - np.log(spread * np.sqrt(2 * np.pi))
                    lp = slab_logpdf(size, slab, floor)
                    mix = np.logaddexp(np.log(0.5) + lp, np.log(0.5) + q_g)
                    self.logw += np.where(jump, lp - mix, 0.0)
                self.theta[:, j] += np.where(jump, size, 0.0)
            self.on[jump, f] = True
            if k == "derate":
                self.theta[:, P["derate"]] = np.clip(self.theta[:, P["derate"]], 0.0, 0.9)
        self.logw[~np.isfinite(self.logw)] = -np.inf

    # ------------------------------------------------------------------ observation
    def _predictions(self, day: DayInput) -> dict:
        th = self.theta
        pow_, full = power_stats(th, day.e_hat, day.p_hat, day.full, self.table, self.cap, self.r_q)
        return {"pow": pow_, "full": full, "alo": anemo_stat(th, day.u_lo), "ahi": anemo_stat(th, day.u_hi),
                "yaw": th[:, P["yaw"]], "tgear": th[:, P["gear"]] * day.rise_gear,
                "tgen": th[:, P["gen"]] * day.rise_gen}

    def step(self, day: DayInput) -> dict:
        """Advance one day: change-point update, propagation, Kalman/particle update, summary, resampling."""
        nz = self.noise
        sd = nz.stationary_sd
        for k, ch in (("gear", day.rise_gear), ("gen", day.rise_gen)):
            if np.isfinite(ch):
                self.rise_mem[k] = 0.9 * self.rise_mem[k] + 0.1 * ch
        for s in range(len(STATS)):
            self.cp[s].update((day.y[s] - nz.mu_mean[s]) / sd[s])
        self._propagate()
        # Kalman prediction of [mu, n] for every statistic
        self.m[:, :, 1] *= nz.phi[None, :]
        self.P[:, 0, 0] += nz.level_walk ** 2
        self.P[:, 1, 1] = nz.phi ** 2 * self.P[:, 1, 1] + nz.sigma_w ** 2
        self.P[:, 0, 1] = self.P[:, 1, 0] = nz.phi * self.P[:, 0, 1]
        observed = np.isfinite(day.y)
        h = self._predictions(day) if observed.any() else {}
        w = self.weights()
        n_gated = 0
        var_e = nz.sigma_e ** 2 * count_scale(day.count, nz.c_ref)
        var_out = (OUT_SCALE * sd) ** 2
        for s, name in enumerate(STATS):
            if not observed[s] or not np.all(np.isfinite(h[name])):
                continue
            pred = h[name] + self.m[:, s, 0] + self.m[:, s, 1]
            Ps = self.P[s]
            var = Ps[0, 0] + 2 * Ps[0, 1] + Ps[1, 1] + var_e[s]
            mean_mix = np.sum(w * pred)
            r = day.y[s] - mean_mix
            v_mix = var + np.sum(w * (pred - mean_mix) ** 2)
            l_in = np.log(1 - P_OUTLIER) + _lognorm(r, v_mix)
            l_out = np.log(P_OUTLIER) + _lognorm(r, v_mix + var_out[s])
            sign = int(np.sign(r))
            if l_out > l_in and not (self.gated[s] * sign >= PERSIST):
                self.gated[s] = self.gated[s] + sign if self.gated[s] * sign > 0 else sign
                n_gated += 1
                continue
            self.gated[s] = 0
            v = day.y[s] - pred
            self.logw += np.logaddexp(np.log(1 - P_OUTLIER) + _lognorm(v, var),
                                      np.log(P_OUTLIER) + _lognorm(v, var + var_out[s]))
            gain = (Ps @ np.ones(2)) / var
            self.m[:, s, :] += v[:, None] * gain[None, :]
            self.P[s] = Ps - np.outer(gain, np.ones(2) @ Ps)
        w = self.weights()
        summary = self.summary(w)
        summary["gated"] = n_gated
        ess = 1.0 / np.sum(w ** 2)
        summary["ess"] = ess
        if ess < self.n / 2:
            self._resample(w)
        return summary

    def _resample(self, w: np.ndarray) -> None:
        u = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), self.n - 1)
        self.theta, self.on, self.m = self.theta[idx], self.on[idx], self.m[idx]
        self.logw = np.zeros(self.n)

    # ------------------------------------------------------------------ outputs
    def fault_probability(self, w=None) -> np.ndarray:
        w = self.weights() if w is None else w
        return w @ (self.on & material(self.theta, self.spec))

    def summary(self, w=None) -> dict:
        w = self.weights() if w is None else w
        out = {f"p_{k}": float(p) for k, p in zip(FAULTS, self.fault_probability(w))}
        out.update({f"on_{k}": float(p) for k, p in zip(FAULTS, w @ self.on)})
        for j, name in enumerate(PARAMS):
            q = weighted_quantiles(self.theta[:, j], w, [0.05, 0.5, 0.95])
            out[f"mean_{name}"] = float(w @ self.theta[:, j])
            out[f"q05_{name}"], out[f"q50_{name}"], out[f"q95_{name}"] = map(float, q)
        e7 = self.theta[:, P["ascale"]] + self.theta[:, P["abias"]] / 7.0
        q = weighted_quantiles(e7, w, [0.05, 0.5, 0.95])
        out["mean_anemo7"] = float(w @ e7)
        out["q05_anemo7"], out["q50_anemo7"], out["q95_anemo7"] = map(float, q)
        for s, name in enumerate(STATS):
            out[f"cp_{name}"] = self.cp[s].p_recent(RECENT_DAYS)
        return out

    def sample(self, m: int, rng=None) -> np.ndarray:
        """``m`` parameter vectors drawn from the current posterior (for loss attribution and decisions)."""
        rng = np.random.default_rng(0) if rng is None else rng
        idx = rng.choice(self.n, size=m, p=self.weights())
        return self.theta[idx] * (self.on[idx][:, [FAULTS.index(k) for k in
                                                   ("eff", "anemo", "anemo", "yaw", "derate", "gear", "gen")]])
