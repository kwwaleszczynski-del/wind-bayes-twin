"""Primary inference engine: run-length (change-point) recursion with a Kalman filter in every segment.

The model of ``model`` is split into five blocks that share no statistic: power (pow, full: efficiency loss and
derating), yaw (yaw), anemometer (alo, ahi), gearbox bearing (tgear) and generator bearing (tgen). Within a block
the health parameters enter the observations linearly once a few known terms are fixed:

    power   pow   = ln eta + o_pow(kappa, gamma^) + mu + n + eps      full = ln eta + o_full(kappa) + mu + n + eps
    yaw     yaw   = gamma + mu + n + eps
    anemo   alo   = ln a + b / u~_lo + mu + n + eps                   (first order in the anemometer error)
    gear    tgear = (rho_g - 1) R~ + mu + n + eps                     R~: RC temperature rise above the coolant
    gen     tgen  = (rho_n - 1) R~ + mu + n + eps

The derating cap kappa enters nonlinearly and is put on a grid; the offsets o(kappa, gamma^) integrate the cap over
the turbine's own record-to-record dispersion of P/P^ at full load (calibration year) and include the yaw power loss
at the yaw block's current estimate gamma^ (plug-in coupling between the two blocks).

Hypotheses. A block is either healthy, or in a fault segment of one of its fault types that started r days ago
(run length r, Adams & MacKay 2007) - with the derating cap on the grid as part of the hypothesis. Given a
hypothesis everything else is linear-Gaussian: the fault parameter (slab prior at onset, then a random walk), the
healthy level mu (slow random walk) and the AR(1) wander n of every statistic form one Kalman state, so each
hypothesis carries a mean and a covariance and its exact predictive likelihood. Transitions per day:

    healthy  -> fault k (new segment)      rate lambda_k            fault  -> healthy (repair)   rate h_repair
    fault    -> fault k (a further jump)   rate lambda_k            fault  -> same segment       otherwise

All healthy histories have the same future, so they are merged into one hypothesis every day; at a change point
the new segment starts from the moment-matched mixture of the nuisance states (mu, n) of all hypotheses (the only
approximation besides merging run lengths beyond ``R_MAX``, by moment matching). Hypotheses whose weight falls
below 1e-8 are dropped.
This is the recursion of Fearnhead & Liu (2007) with Kalman segment models; unlike a particle filter it has no
Monte Carlo error, so evidence that accumulates slowly over weeks is never lost by resampling.

Robust likelihood: each value is an outlier with probability ``P_OUTLIER`` (a component ``OUT_SCALE`` times wider);
the Kalman update is the moment-matched mixture of "update" and "ignore". A value that the mixture of hypotheses
calls an outlier is skipped altogether unless the two previous values of that statistic were skipped in the same
direction.

Usage:
    twin = BlockTwin(noise, spec, table, cap_kw, r_quantiles)
    out = twin.step(day_input)          # fault probabilities and parameter means of all blocks
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .hierarchy import OUT_SCALE, P_OUTLIER, count_scale
from .model import FAULTS, ModelSpec, yaw_factor
from .stats import STATS

R_MAX = 120  # run lengths beyond this are merged (per fault type) ...
R_MAX_GRID = 30  # ... and beyond this for the derating grid, whose segments carry no parameter of their own
PRUNE = 1e-8
PERSIST = 2
KAPPA_GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40])
ANEMO_JUDGE_WS = 7.0  # the anemometer error is judged at this wind speed (first order: ln a + b / 7)
PARAM_RANGE = {"eff": (-0.4, 0.2), "ascale": (-0.3, 0.3), "abias": (-2.0, 2.0), "yaw": (-30.0, 30.0),
               "derate": (0.0, 0.5), "gear": (-1.0, 2.0), "gen": (-1.0, 2.0), "anemo7": (-0.3, 0.3)}


@dataclass(frozen=True)
class Block:
    name: str
    stats: tuple
    faults: tuple
    params: tuple  # parameter names in the Kalman state (before the nuisance states)


BLOCKS = (
    Block("yaw", ("yaw",), ("yaw",), ("yaw",)),
    Block("power", ("pow", "full"), ("eff", "derate"), ("eff",)),
    Block("anemo", ("alo", "ahi"), ("anemo",), ("ascale", "abias")),
    Block("gear", ("tgear",), ("gear",), ("gear",)),
    Block("gen", ("tgen",), ("gen",), ("gen",)),
)
FAULT_PARAMS = {"eff": ("eff",), "derate": (), "yaw": ("yaw",), "anemo": ("ascale", "abias"), "gear": ("gear",),
                "gen": ("gen",)}
SLAB_SD = {"eff": 0.05, "ascale": 0.04, "abias": 0.3, "yaw": 6.0, "gear": 0.3, "gen": 0.3}


def kappa_prior(spec: ModelSpec) -> np.ndarray:
    """Prior weights of the derating grid: half-normal slab density times the grid spacing."""
    _, scale = spec.slab["derate"]
    edges = np.r_[KAPPA_GRID[0] - 0.005, 0.5 * (KAPPA_GRID[1:] + KAPPA_GRID[:-1]), KAPPA_GRID[-1] + 0.05]
    w = np.exp(-0.5 * (KAPPA_GRID / scale) ** 2) * np.diff(edges)
    return w / w.sum()


def power_offsets(e_hat, p_hat, full, gamma_hat: float, r_q: np.ndarray, table, cap_kw: float):
    """(G + 1, 2) modelled pow / full offsets for kappa = 0 and every grid value (yaw loss at gamma_hat included).

    Power of a record under a cap c is integrated over the turbine's healthy dispersion r of P/P^ (quantiles r_q,
    mean 1): E_r[min(r F(e^ g), c)].
    """
    out = np.full((len(KAPPA_GRID) + 1, 2), np.nan)
    if len(e_hat) == 0:
        return out
    f = table(e_hat * yaw_factor(np.array([gamma_hat]))[0])
    caps = np.r_[np.inf, (1.0 - KAPPA_GRID) * cap_kw]
    p = np.minimum(r_q[None, None, :] * f[None, :, None], caps[:, None, None]).mean(axis=2)  # (G+1, R)
    for k, sel in enumerate((~full, full)):
        if sel.any():
            out[:, k] = np.log(np.maximum(p[:, sel].sum(axis=1), 1e-6) / p_hat[sel].sum())
    return out


class BlockFilter:
    """Run-length recursion with Kalman segment models for one block of one turbine."""

    def __init__(self, block: Block, noise, spec: ModelSpec):
        self.block, self.spec = block, spec
        self.n_p, self.n_s = len(block.params), len(block.stats)
        self.d = self.n_p + 2 * self.n_s
        self.si = np.array([STATS.index(s) for s in block.stats])
        self.mu_mean, self.mu_var = noise.mu_mean[self.si], noise.mu_var[self.si]
        self.sigma_e, self.sigma_w = noise.sigma_e[self.si], noise.sigma_w[self.si]
        self.phi, self.walk_mu = noise.phi[self.si], noise.level_walk[self.si]
        self.c_ref, self.sd = noise.c_ref[self.si], noise.stationary_sd[self.si]
        self.has_grid = "derate" in block.faults
        self.kprior = kappa_prior(spec) if self.has_grid else None
        m0 = np.zeros(self.d)
        p0 = np.zeros((self.d, self.d))
        for k in range(self.n_s):
            m0[self.n_p + 2 * k] = self.mu_mean[k]
            p0[self.n_p + 2 * k, self.n_p + 2 * k] = self.mu_var[k]
            p0[self.n_p + 2 * k + 1, self.n_p + 2 * k + 1] = self.sigma_w[k] ** 2 / (1 - self.phi[k] ** 2)
        self.w = np.ones(1)
        self.label = np.array([-1])  # -1 healthy, else index into block.faults
        self.grid = np.array([-1])  # derating grid index, -1 none
        self.age = np.array([0])
        self.m = m0[None, :]
        self.P = p0[None, :, :]
        self.gated = np.zeros(self.n_s, dtype=int)
        # dynamics shared by all hypotheses
        self.A = np.eye(self.d)
        self.Q = np.zeros((self.d, self.d))
        for k in range(self.n_s):
            j = self.n_p + 2 * k
            self.A[j + 1, j + 1] = self.phi[k]
            self.Q[j, j] = self.walk_mu[k] ** 2
            self.Q[j + 1, j + 1] = self.sigma_w[k] ** 2
        self.q_param = np.array([spec.walk[p] for p in block.params]) ** 2
        self.param_of = {f: [block.params.index(p) for p in FAULT_PARAMS[f]] for f in block.faults}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _moments(w, m, P):
        tot = w.sum()
        mean = (w[:, None] * m).sum(0) / tot
        dm = m - mean
        cov = (np.einsum("h,hij->ij", w, P) + np.einsum("h,hi,hj->ij", w, dm, dm)) / tot
        return tot, mean, cov

    def _zero_params(self, m, P):
        m = m.copy()
        P = P.copy()
        m[..., :self.n_p] = 0.0
        P[..., :self.n_p, :] = 0.0
        P[..., :, :self.n_p] = 0.0
        return m, P

    # ------------------------------------------------------------------ prediction
    def predict(self) -> None:
        spec = self.spec
        lam = np.array([spec.hazard[f] for f in self.block.faults])
        lam_tot = lam.sum()
        healthy = self.label < 0
        w_stay = self.w * np.where(healthy, 1.0 - lam_tot, 1.0 - spec.hazard_repair - lam_tot)
        w_rep = np.where(healthy, 0.0, self.w * spec.hazard_repair)
        # merged healthy hypothesis: healthy that stays + faults that are repaired (parameters dropped)
        src = healthy | (w_rep > 0)
        mz, pz = self._zero_params(self.m[src], self.P[src])
        wh, mh, ph = self._moments(np.where(healthy, w_stay, w_rep)[src], mz, pz)
        # new segments start from the mixture of the nuisance states of all hypotheses
        _, mn, pn = self._moments(self.w, *self._zero_params(self.m, self.P))
        new_w, new_label, new_grid, new_m, new_p = [], [], [], [], []
        total = self.w.sum()
        for f_idx, f in enumerate(self.block.faults):
            if f == "derate":
                for g, pg in enumerate(self.kprior):
                    new_w.append(total * lam[f_idx] * pg)
                    new_label.append(f_idx), new_grid.append(g), new_m.append(mn), new_p.append(pn)
                continue
            p = pn.copy()
            for j in self.param_of[f]:
                p[j, j] = SLAB_SD[self.block.params[j]] ** 2
            m = mn.copy()
            if f == "eff":  # the slab of a loss is centred on a loss
                m[self.param_of[f][0]] = -0.5 * SLAB_SD["eff"]
            new_w.append(total * lam[f_idx])
            new_label.append(f_idx), new_grid.append(-1), new_m.append(m), new_p.append(p)
        keep = ~healthy
        self.w = np.r_[wh, w_stay[keep], new_w]
        self.label = np.r_[-1, self.label[keep], new_label]
        self.grid = np.r_[-1, self.grid[keep], new_grid]
        self.age = np.r_[0, self.age[keep] + 1, np.zeros(len(new_w), dtype=int)]
        self.m = np.concatenate([mh[None], self.m[keep], np.array(new_m).reshape(-1, self.d)])
        self.P = np.concatenate([ph[None], self.P[keep], np.array(new_p).reshape(-1, self.d, self.d)])
        # Kalman prediction: nuisance dynamics for all, random walk of the parameters of active faults
        self.m = self.m @ self.A.T
        self.P = np.einsum("ij,hjk,lk->hil", self.A, self.P, self.A) + self.Q[None]
        for f_idx, f in enumerate(self.block.faults):
            act = self.label == f_idx
            for j in self.param_of[f]:
                self.P[act, j, j] += self.q_param[j]
        self._merge_old()

    def _merge_old(self) -> None:
        r_max = np.where(self.grid >= 0, R_MAX_GRID, R_MAX)
        old = self.age > r_max
        if not old.any():
            return
        for lab, g in {(int(a), int(b)) for a, b in zip(self.label[old], self.grid[old])}:
            cap = R_MAX_GRID if g >= 0 else R_MAX
            sel = (self.label == lab) & (self.grid == g) & (self.age >= cap)
            if sel.sum() < 2:
                self.age[sel] = cap
                continue
            tot, mean, cov = self._moments(self.w[sel], self.m[sel], self.P[sel])
            first = np.flatnonzero(sel)[0]
            self.w[first], self.m[first], self.P[first], self.age[first] = tot, mean, cov, cap
            drop = sel.copy()
            drop[first] = False
            self._keep(~drop)

    def _keep(self, keep) -> None:
        self.w, self.label, self.grid, self.age = self.w[keep], self.label[keep], self.grid[keep], self.age[keep]
        self.m, self.P = self.m[keep], self.P[keep]

    # ------------------------------------------------------------------ update
    def update(self, y, count, coef, offsets) -> int:
        """Bayes update with the block's statistics.

        y, count: (n_s,) values and record counts; coef: (n_s, n_p) loadings of the parameters;
        offsets: (n_s,) known offset, or (G+1, n_s) per derating grid value (row 0: no cap). Returns the number of
        statistics skipped as outliers.
        """
        n_gated = 0
        var_e = self.sigma_e ** 2 * count_scale(count, self.c_ref)
        var_out = (OUT_SCALE * self.sd) ** 2
        offsets = np.atleast_2d(offsets)
        for k in range(self.n_s):
            if not np.isfinite(y[k]) or not np.all(np.isfinite(coef[k])):
                continue
            off = offsets[np.clip(self.grid + 1, 0, None), k] if offsets.shape[0] > 1 else np.full(len(self.w),
                                                                                                    offsets[0, k])
            if not np.all(np.isfinite(off)):
                continue
            h = np.zeros(self.d)
            h[:self.n_p] = coef[k]
            h[self.n_p + 2 * k] = h[self.n_p + 2 * k + 1] = 1.0
            pred = self.m @ h + off
            ph = self.P @ h
            s = ph @ h + var_e[k]
            wn = self.w / self.w.sum()
            mean = wn @ pred
            v_mix = wn @ (s + (pred - mean) ** 2)
            r = y[k] - mean
            l_in = np.log(1 - P_OUTLIER) + _lognorm(r, v_mix)
            l_out = np.log(P_OUTLIER) + _lognorm(r, v_mix + var_out[k])
            sign = int(np.sign(r))
            if l_out > l_in and not (self.gated[k] * sign >= PERSIST):
                self.gated[k] = self.gated[k] + sign if self.gated[k] * sign > 0 else sign
                n_gated += 1
                continue
            self.gated[k] = 0
            v = y[k] - pred
            a = np.log(1 - P_OUTLIER) + _lognorm(v, s)
            b = np.log(P_OUTLIER) + _lognorm(v, s + var_out[k])
            lik = np.logaddexp(a, b)
            resp = np.exp(a - lik)  # probability that the value is an inlier, per hypothesis
            gain = ph / s[:, None]
            kv = gain * v[:, None]
            self.m = self.m + resp[:, None] * kv
            self.P = (self.P - resp[:, None, None] * np.einsum("hi,hj->hij", gain, ph)
                      + (resp * (1 - resp))[:, None, None] * np.einsum("hi,hj->hij", kv, kv))
            logw = np.log(np.maximum(self.w, 1e-300)) + lik
            self.w = np.exp(logw - logw.max())
        self.w /= self.w.sum()
        keep = (self.w >= PRUNE) | (self.label < 0)
        if not keep.all():
            self._keep(keep)
            self.w /= self.w.sum()
        return n_gated

    # ------------------------------------------------------------------ outputs
    def material_prob(self, fault: str) -> np.ndarray:
        """Per hypothesis: probability that the fault's parameter is beyond its materiality threshold."""
        thr = self.spec.material[fault]
        if fault == "derate":
            kap = np.where(self.grid >= 0, KAPPA_GRID[np.clip(self.grid, 0, None)], 0.0)
            return (kap > thr).astype(float)
        idx = self.param_of[fault]
        if fault == "anemo":
            h = np.zeros(self.d)
            h[idx[0]], h[idx[1]] = 1.0, 1.0 / ANEMO_JUDGE_WS
            mean, sd = self.m @ h, np.sqrt(np.maximum(np.einsum("i,hij,j->h", h, self.P, h), 1e-12))
            return ndtr((-thr - mean) / sd) + ndtr((mean - thr) / sd)
        mean, sd = self.m[:, idx[0]], np.sqrt(np.maximum(self.P[:, idx[0], idx[0]], 1e-12))
        if fault == "yaw":
            return ndtr((-thr - mean) / sd) + ndtr((mean - thr) / sd)
        if fault == "eff":
            return ndtr((-thr - mean) / sd)
        return ndtr((mean - thr) / sd)  # gear, gen: worse cooling only

    def summary(self, quantiles: bool = False) -> dict:
        out = {}
        for f_idx, f in enumerate(self.block.faults):
            act = self.label == f_idx
            out[f"on_{f}"] = float(self.w[act].sum())
            out[f"p_{f}"] = float(self.w[act] @ self.material_prob(f)[act])
            if act.any():
                age = np.bincount(self.age[act], weights=self.w[act], minlength=1)
                out[f"onset_age_{f}"] = float(np.argmax(age)) if out[f"on_{f}"] > 0 else np.nan
            else:
                out[f"onset_age_{f}"] = np.nan
        for j, name in enumerate(self.block.params):
            out[f"mean_{name}"] = float(self.w @ self.m[:, j])
            if quantiles:
                out.update(self._quantiles(name, self.m[:, j], self.P[:, j, j]))
        if self.has_grid:
            kap = np.where(self.grid >= 0, KAPPA_GRID[np.clip(self.grid, 0, None)], 0.0)
            out["mean_derate"] = float(self.w @ kap)
            if quantiles:
                out.update(self._quantiles("derate", kap, np.zeros(len(kap))))
        if self.block.name == "anemo":  # effective anemometer error at 7 m/s, ln a + b / 7
            h = np.zeros(self.d)
            h[0], h[1] = 1.0, 1.0 / ANEMO_JUDGE_WS
            e7 = self.m @ h
            out["mean_anemo7"] = float(self.w @ e7)
            if quantiles:
                out.update(self._quantiles("anemo7", e7, np.einsum("i,hij,j->h", h, self.P, h)))
        return out

    def _quantiles(self, name, mean, var, qs=(0.05, 0.5, 0.95)) -> dict:
        """Quantiles of the mixture (point masses + Gaussians) of the hypotheses, by bisection on its CDF."""
        lo, hi = PARAM_RANGE[name]
        w = self.w / self.w.sum()
        sd = np.sqrt(np.maximum(var, 0.0))
        point = sd < 1e-9
        mp, wp = mean[point], w[point]
        mg, sg, wg = mean[~point], sd[~point], w[~point]
        q = np.asarray(qs, dtype=float)

        def cdf(x):
            c = (mp[None, :] <= x[:, None]).astype(float) @ wp
            if len(mg):
                c += ndtr((x[:, None] - mg[None, :]) / sg[None, :]) @ wg
            return c

        grid = np.linspace(lo, hi, 129)  # bracket on a coarse grid, then bisect
        k = np.clip(np.searchsorted(cdf(grid), q), 1, len(grid) - 1)
        a, b = grid[k - 1], grid[k]
        for _ in range(16):
            mid = 0.5 * (a + b)
            below = cdf(mid) < q
            a, b = np.where(below, mid, a), np.where(below, b, mid)
        tol = 1e-6 * (hi - lo)
        for j in range(len(b)):  # a quantile that falls on a point mass is that point exactly
            near = np.abs(mp - b[j]) <= tol
            if near.any():
                b[j] = mp[near][0]
        return {f"q{int(round(qq * 100)):02d}_{name}": float(v) for qq, v in zip(q, b)}

    def sample(self, n: int, rng) -> np.ndarray:
        """(n, n_params [+ kappa]) draws from the block's parameter posterior."""
        idx = rng.choice(len(self.w), size=n, p=self.w / self.w.sum())
        cols = []
        for j in range(self.n_p):
            sd = np.sqrt(np.maximum(self.P[idx, j, j], 0.0))
            cols.append(self.m[idx, j] + sd * rng.standard_normal(n))
        if self.has_grid:
            cols.append(np.where(self.grid[idx] >= 0, KAPPA_GRID[np.clip(self.grid[idx], 0, None)], 0.0))
        return np.column_stack(cols) if cols else np.zeros((n, 0))


def _lognorm(v, var):
    return -0.5 * (np.log(2 * np.pi * var) + v ** 2 / var)


class BlockTwin:
    """The five block filters of one turbine, run day by day (yaw first: its estimate feeds the power block)."""

    def __init__(self, noise, spec: ModelSpec, table, cap_kw: float, r_quantiles: np.ndarray):
        self.spec, self.table, self.cap, self.r_q = spec, table, cap_kw, r_quantiles
        self.blocks = {b.name: BlockFilter(b, noise, spec) for b in BLOCKS}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.blocks)

    def restore(self, snap: dict) -> None:
        self.blocks = copy.deepcopy(snap)

    def step(self, day, quantiles: bool = False) -> dict:
        y = dict(zip(STATS, day.y))
        cnt = dict(zip(STATS, day.count))
        out, gated = {}, 0
        for name, flt in self.blocks.items():
            flt.predict()
            stats = flt.block.stats
            yy = np.array([y[s] for s in stats])
            cc = np.array([cnt[s] for s in stats])
            if name == "yaw":
                coef, off = np.ones((1, 1)), np.zeros(1)
            elif name == "power":
                gamma = out.get("mean_yaw", 0.0)
                coef, off = np.ones((2, 1)), power_offsets(day.e_hat, day.p_hat, day.full, gamma, self.r_q,
                                                           self.table, self.cap)
            elif name == "anemo":
                coef = np.array([[1.0, 1.0 / day.u_lo], [1.0, 1.0 / day.u_hi]])
                off = np.zeros(2)
            else:
                rise = day.rise_gear if name == "gear" else day.rise_gen
                coef, off = np.array([[rise]]), np.zeros(1)
            gated += flt.update(yy, cc, coef, off)
            out.update(flt.summary(quantiles))
        out["gated"] = gated
        out["hypotheses"] = sum(len(f.w) for f in self.blocks.values())
        return out

    def fault_probability(self) -> np.ndarray:
        s = {}
        for flt in self.blocks.values():
            s.update(flt.summary())
        return np.array([s[f"p_{f}"] for f in FAULTS])

    def sample(self, n: int, rng) -> np.ndarray:
        """(n, 7) joint draws of the parameters in ``model.PARAMS`` order (blocks independent)."""
        yaw = self.blocks["yaw"].sample(n, rng)[:, 0]
        pw = self.blocks["power"].sample(n, rng)
        an = self.blocks["anemo"].sample(n, rng)
        ge = self.blocks["gear"].sample(n, rng)[:, 0]
        gn = self.blocks["gen"].sample(n, rng)[:, 0]
        return np.column_stack([pw[:, 0], an[:, 0], an[:, 1], yaw, pw[:, 1], ge, gn])
