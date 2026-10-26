"""Offline hierarchical model of healthy behaviour: fleet priors for the online filter (NumPyro, NUTS).

For every daily statistic s, turbine i and day d of the calibration year (assumed fault-free) the healthy turbine is

    y_{i,d} = mu_i + n_{i,d} + eps_{i,d},     n_{i,d} = phi_i n_{i,d-1} + w_{i,d},     w ~ N(0, sigma_w,i^2)
    eps ~ (1 - p) N(0, sigma_e,i^2 c_ref / c_{i,d}) + p N(0, (k sd_i)^2)

mu_i is the turbine's own healthy level relative to the reference (terrain, sensor position, a reference fitted on a
different year), n a slow AR(1) wander of the reference error with the weather, eps the day-to-day noise, whose
variance falls with the number of records c behind the day's value (c_ref: median count, ratio clipped to
[0.25, 9]). As in the online filter, a day is an outlier with probability p = 2 % and then comes from a component
k = 10 times wider than the stationary spread sd_i (icing, a service visit): the healthy level is then the level of
the ordinary days, not pulled by a skewed tail. Six turbines and one year are few, so the turbines share information
(partial pooling):

    mu_i = mu_0 + tau z_i,                 z_i ~ N(0, 1)
    log sigma_e,i = a_e + b_e z^e_i,       log sigma_w,i = a_w + b_w z^w_i,       logit phi_i = a_p + b_p z^p_i

with weakly informative hyperpriors on the standardised scale. The AR(1) states are integrated out with a Kalman
filter inside the likelihood (``jax.lax.scan``; exact for the Gaussian part, the outlier mixture is moment-matched
after each day exactly as online), so NUTS only samples the 4 x 6 turbine parameters and the 8 hyperparameters.
Days without a value are skipped by the filter (prediction only).

The posterior means of sigma_e, sigma_w and phi and the posterior mean and variance of mu are the priors and noise
model of the online filter.

Usage:
    post = fit_statistic(y_cal, count_cal)   # (n_days, n_turbines), NaN for missing days, record counts
    post.mu_mean, post.sigma_e, post.stationary_sd()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

P_OUTLIER = 0.02
OUT_SCALE = 10.0
log = logging.getLogger(__name__)


@dataclass
class NoisePosterior:
    c_ref: float
    mu_mean: np.ndarray
    mu_sd: np.ndarray
    sigma_e: np.ndarray
    sigma_w: np.ndarray
    phi: np.ndarray
    mu0: float
    tau: float
    tau_q: tuple
    rhat_max: float
    unpooled_mu: np.ndarray  # per-turbine median (no pooling), for comparison
    level_walk: float = 0.0  # daily random-walk step of the healthy level (set from the year-to-year change)

    def stationary_sd(self) -> np.ndarray:
        return np.sqrt(self.sigma_e ** 2 + self.sigma_w ** 2 / (1.0 - self.phi ** 2))

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "NoisePosterior":
        return cls(**{k: (np.asarray(v) if isinstance(v, list) and k != "tau_q" else v) for k, v in d.items()})


def count_scale(count, c_ref: float):
    """Noise variance multiplier c_ref / c of a day's value built from ``count`` records (clipped)."""
    with np.errstate(divide="ignore"):
        return np.clip(c_ref / np.maximum(np.asarray(count, dtype=float), 1.0), 0.25, 9.0)


def kalman_loglik(y, observed, mu, sigma_e, sigma_w, phi, var_scale, p_out=P_OUTLIER, out_scale=OUT_SCALE):
    """Log likelihood of y (days, turbines) under level + AR(1) + contaminated noise, AR state filtered (JAX).

    With ``p_out = 0`` it is the exact Gaussian likelihood. Otherwise each day's Kalman update is the moment-matched
    mixture of "update" (probability r that the day is ordinary) and "ignore".
    """
    import jax
    import jax.numpy as jnp

    var0 = sigma_w ** 2 / (1.0 - phi ** 2)
    var_out = out_scale ** 2 * (sigma_e ** 2 + var0)
    log_in, log_out = jnp.log1p(-p_out), jnp.log(jnp.maximum(p_out, 1e-300))

    def lognorm(v, s):
        return -0.5 * (jnp.log(2 * jnp.pi * s) + v ** 2 / s)

    def step(carry, inp):
        m, p, ll = carry
        yt, ot, ct = inp
        m, p = phi * m, phi ** 2 * p + sigma_w ** 2
        s = p + sigma_e ** 2 * ct
        v = yt - mu - m
        a = log_in + lognorm(v, s)
        b = log_out + lognorm(v, s + var_out)
        lik = jnp.logaddexp(a, b)
        r = jnp.exp(a - lik)
        ll = ll + jnp.where(ot, lik, 0.0)
        kv = p / s * v
        m = jnp.where(ot, m + r * kv, m)
        p = jnp.where(ot, p - r * p * p / s + r * (1 - r) * kv ** 2, p)
        return (m, p, ll), None

    init = (jnp.zeros_like(mu), var0, jnp.zeros_like(mu))
    (_, _, ll), _ = jax.lax.scan(step, init, (y, observed, var_scale))
    return ll.sum()


def _model(y, observed, var_scale, n):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.nn import sigmoid

    mu0 = numpyro.sample("mu0", dist.Normal(0.0, 1.0))
    tau = numpyro.sample("tau", dist.HalfNormal(0.5))
    a_e = numpyro.sample("a_e", dist.Normal(np.log(0.8), 0.5))
    b_e = numpyro.sample("b_e", dist.HalfNormal(0.3))
    a_w = numpyro.sample("a_w", dist.Normal(np.log(0.3), 1.0))
    b_w = numpyro.sample("b_w", dist.HalfNormal(0.3))
    a_p = numpyro.sample("a_p", dist.Normal(1.5, 1.0))
    b_p = numpyro.sample("b_p", dist.HalfNormal(0.5))
    with numpyro.plate("turbine", n):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        ze = numpyro.sample("ze", dist.Normal(0.0, 1.0))
        zw = numpyro.sample("zw", dist.Normal(0.0, 1.0))
        zp = numpyro.sample("zp", dist.Normal(0.0, 1.0))
    mu = numpyro.deterministic("mu", mu0 + tau * z)
    sigma_e = numpyro.deterministic("sigma_e", jnp.exp(a_e + b_e * ze))
    sigma_w = numpyro.deterministic("sigma_w", jnp.exp(a_w + b_w * zw))
    phi = numpyro.deterministic("phi", 0.995 * sigmoid(a_p + b_p * zp))
    numpyro.factor("ll", kalman_loglik(y, observed, mu, sigma_e, sigma_w, phi, var_scale))


def fit_statistic(y: np.ndarray, count: np.ndarray, seed: int = 0, warmup: int = 600, samples: int = 600,
                  chains: int = 2) -> NoisePosterior:
    """NUTS fit of the hierarchical noise model to one statistic (n_days, n_turbines), original units out.

    ``count`` holds the number of records behind each daily value.
    """
    import jax
    from numpyro.diagnostics import gelman_rubin
    from numpyro.infer import MCMC, NUTS

    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(y - med), axis=0)
    scale = float(np.nanmedian(mad))
    centre = float(np.nanmedian(med))
    z = (y - centre) / scale
    observed = np.isfinite(z)
    z = np.where(observed, z, 0.0)
    c_ref = float(np.median(count[observed]))
    var_scale = count_scale(count, c_ref)
    mcmc = MCMC(NUTS(_model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), z, observed, var_scale, y.shape[1])
    s = mcmc.get_samples(group_by_chain=True)
    rhat = (max(float(np.max(gelman_rubin(np.asarray(s[k])))) for k in ("mu", "sigma_e", "sigma_w", "phi"))
            if chains > 1 else float("nan"))
    flat = {k: np.asarray(v).reshape(-1, *np.asarray(v).shape[2:]) for k, v in s.items()}
    tau = flat["tau"] * scale
    return NoisePosterior(
        c_ref=c_ref, mu_mean=centre + scale * flat["mu"].mean(0), mu_sd=scale * flat["mu"].std(0),
        sigma_e=scale * flat["sigma_e"].mean(0), sigma_w=scale * flat["sigma_w"].mean(0),
        phi=flat["phi"].mean(0), mu0=float(centre + scale * flat["mu0"].mean()), tau=float(tau.mean()),
        tau_q=(float(np.quantile(tau, 0.05)), float(np.quantile(tau, 0.95))), rhat_max=rhat, unpooled_mu=med)
