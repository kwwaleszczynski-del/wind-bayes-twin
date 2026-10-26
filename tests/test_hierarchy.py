import numpy as np

from wind_bayes_twin.hierarchy import count_scale, fit_statistic, kalman_loglik


def numpy_loglik(y, obs, mu, se, sw, phi, scale, p_out=0.0, k_out=10.0):
    """Reference implementation in plain Python of the (contaminated) Kalman likelihood."""
    ll = 0.0
    norm = lambda v, s: np.exp(-0.5 * v * v / s) / np.sqrt(2 * np.pi * s)  # noqa: E731
    for i in range(y.shape[1]):
        m, p = 0.0, sw[i] ** 2 / (1 - phi[i] ** 2)
        var_out = k_out ** 2 * (se[i] ** 2 + sw[i] ** 2 / (1 - phi[i] ** 2))
        for t in range(y.shape[0]):
            m, p = phi[i] * m, phi[i] ** 2 * p + sw[i] ** 2
            if obs[t, i]:
                s = p + se[i] ** 2 * scale[t, i]
                v = y[t, i] - mu[i] - m
                a, b = (1 - p_out) * norm(v, s), p_out * norm(v, s + var_out)
                ll += np.log(a + b)
                r = a / (a + b)
                m_upd, p_upd = m + p / s * v, p - p * p / s
                m, p = r * m_upd + (1 - r) * m, r * p_upd + (1 - r) * p + r * (1 - r) * (p / s * v) ** 2
    return ll


def test_kalman_likelihood_matches_reference():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(40, 3))
    obs = rng.random((40, 3)) > 0.2
    scale = rng.uniform(0.5, 2, (40, 3))
    args = (np.array([0.1, -0.2, 0.0]), np.array([0.8, 1.0, 0.6]), np.array([0.3, 0.5, 0.2]),
            np.array([0.7, 0.9, 0.5]))
    jx = float(kalman_loglik(np.where(obs, y, 0.0), obs, *args, scale, p_out=0.0))
    assert np.isclose(jx, numpy_loglik(y, obs, *args, scale), rtol=1e-5)
    y[5, 1] = 25.0  # an outlier: the contaminated version keeps it from dominating
    jr = float(kalman_loglik(np.where(obs, y, 0.0), obs, *args, scale, p_out=0.02))
    assert np.isclose(jr, numpy_loglik(y, obs, *args, scale, p_out=0.02), rtol=1e-5)


def test_count_scale():
    assert np.allclose(count_scale([100, 50, 400, 0], 100.0), [1.0, 2.0, 0.25, 9.0])


def test_partial_pooling_recovers_levels():
    rng = np.random.default_rng(1)
    n_days, n = 150, 4
    mu = np.array([0.5, -0.5, 0.2, 0.0])
    noise = np.zeros(n)
    y = np.zeros((n_days, n))
    for t in range(n_days):
        noise = 0.7 * noise + rng.normal(0, 0.3, n)
        y[t] = mu + noise + rng.normal(0, 0.8, n)
    y[rng.random((n_days, n)) < 0.1] = np.nan
    post = fit_statistic(y, np.full(y.shape, 100), warmup=100, samples=100, chains=2)
    assert post.rhat_max < 1.2
    assert np.all(np.abs(post.mu_mean - mu) < 3 * post.mu_sd + 0.05)
    assert 0.5 < np.median(post.sigma_e) < 1.1
    # the turbines differ (true spread of the levels 0.39): the fleet-level spread is not shrunk to zero
    assert post.tau_q[0] < 0.39 < post.tau_q[1] + 0.2 and post.tau > 0.1
