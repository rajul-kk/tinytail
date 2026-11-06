"""Inequality-based pWCET baselines (MEMIK and ATAN with RESTK) on synthetic samples."""
import numpy as np
import pytest
from scipy.stats import norm
from pwcet import inequality as iq
from pwcet import synthetic as syn


def test_k1_is_plain_markov_and_atan_bound_inverts_exactly():
    x = np.array([1.0, 2.0, 3.0, 6.0])
    lm = iq._log_moments(x, "memik", (1.0,), k_max=3)
    b = iq._bounds(lm, "memik", (1.0,), 0.1)
    assert b[0, 0] == pytest.approx(x.mean() / 0.1)                  # Markov: E[X] / p
    assert b[0, 1] == pytest.approx(np.sqrt(np.mean(x ** 2) / 0.1))  # second moment
    d = 4.0
    lm = iq._log_moments(x, "atan", (d,), k_max=2)
    b = iq._bounds(lm, "atan", (d,), 0.5)
    # P(X >= b) <= E[atan(X/d)^2] / atan(b/d)^2 = p exactly at the returned b
    assert np.mean(np.arctan(x / d) ** 2) / np.arctan(b[0, 1] / d) ** 2 == pytest.approx(0.5)


def test_atan_has_no_finite_bound_beyond_saturation():
    x = np.full(100, 10.0)
    lm = iq._log_moments(x, "atan", (1.0,), k_max=1)  # atan(10) = 1.471, close to pi/2
    assert np.isinf(iq._bounds(lm, "atan", (1.0,), 1e-3)[0, 0])


def test_log_domain_handles_cycle_counts_at_k_150():
    x = 2e5 + np.arange(1000.0)
    lm = iq._log_moments(x, "memik", (1.0,))
    assert np.all(np.isfinite(lm)) and lm.shape == (1, iq.K_MAX)
    assert np.all(np.isfinite(iq._bounds(lm, "memik", (1.0,), 1e-9)))


@pytest.fixture(scope="module")
def normal_fits():
    """One normal sample, fitted with the protocol defaults (100 RESTK simulations: with fewer,
    the three-point linearity check of RESTK was observed to fail on this same sample)."""
    x = 1000.0 + 30.0 * np.random.default_rng(3).standard_normal(100_000)
    return {m: iq.fit_inequality(x, m) for m in iq.METHODS}


def test_bounds_are_safe_and_monotone_on_a_normal_sample(normal_fits):
    f = normal_fits["memik"]
    assert f.applicable and abs(f.restk_corr) >= iq.RESTK_CORR
    assert all(1 <= k <= iq.K_MAX for k in f.restk_maxk)
    ps = [1e-3, 1e-4, 1e-6, 1e-9]
    b = iq.inequality_pwcet(f, ps)
    assert np.all(np.diff(b) > 0)
    for p, bound in zip(ps, b):
        assert bound >= 1000.0 + 30.0 * norm.isf(p)  # no underestimation of the true quantile
    assert b[-1] < 1.06 * (1000.0 + 30.0 * norm.isf(1e-9))  # and not wildly loose here
    # RESTK grows the k cap towards smaller p, clipped to [1, K_MAX]
    assert f.maxk(1e-9) >= f.maxk(1e-3)


def test_atan_reduces_to_memik_when_the_large_d_end_of_the_grid_wins(normal_fits):
    """Observed on high-offset samples: the largest d (nearly linear arctan) gives the
    tightest ATAN bound, so ATAN matches MEMIK instead of improving on it."""
    m, a = normal_fits["memik"], normal_fits["atan"]
    assert m.applicable and a.applicable
    for p in (1e-3, 1e-6, 1e-9):
        assert iq.inequality_pwcet(a, p) == pytest.approx(iq.inequality_pwcet(m, p), rel=2e-3)


def test_declines_on_short_samples_and_rejects_non_positive_times():
    f = iq.fit_inequality(syn.iid_gpd(5_000, seed=5), "memik")
    assert not f.applicable and "10,000" in f.reason
    with pytest.raises(ValueError):
        iq.inequality_pwcet(f, 1e-6)
    with pytest.raises(ValueError):
        iq.fit_inequality(np.array([0.0, 1.0] * 10_000), "memik")
    with pytest.raises(ValueError):
        iq.fit_inequality(np.ones(20_000), "tanh")
