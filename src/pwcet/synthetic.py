"""Synthetic timing traces with known properties (units arbitrary, think cycles).

i.i.d. generators expose the exact tail so tests can compare estimates with the truth.
Non-i.i.d. generators are deliberately invalid inputs that the test battery must flag.
All generators take an explicit seed.
"""
import numpy as np
from scipy.stats import genpareto


def iid_gpd(n, loc=1000.0, scale=10.0, shape=0.0, seed=0):
    """i.i.d. GPD(loc, scale, shape). shape=0 is an exponential tail. Peaks over any
    threshold are again GPD with the same shape, so the right answer is known."""
    return genpareto.rvs(shape, loc=loc, scale=scale, size=n,
                         random_state=np.random.default_rng(seed))


def true_quantile(p, loc=1000.0, scale=10.0, shape=0.0):
    """Exact value exceeded with probability p by iid_gpd samples."""
    return float(genpareto.isf(p, shape, loc=loc, scale=scale))


def ar1(n, phi=0.8, base=1000.0, sigma=5.0, seed=0):
    """Autocorrelated Gaussian trace (violates independence; stationary)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return base + x


def regime_shift(n, at=0.5, delta=20.0, base=1000.0, sigma=5.0, seed=0):
    """Independent noise whose mean jumps by delta at fraction `at` (non-stationary)."""
    x = base + np.random.default_rng(seed).normal(0.0, sigma, n)
    x[int(n * at):] += delta
    return x


def periodic_spikes(n, period=50, height=200.0, base=1000.0, sigma=5.0, seed=0):
    """Noise plus a spike every `period` runs (periodic interference; serially dependent)."""
    x = base + np.random.default_rng(seed).normal(0.0, sigma, n)
    x[::period] += height
    return x


def mode_switching(n, p_switch=0.01, delta=40.0, base=1000.0, sigma=5.0, seed=0):
    """Two latency modes (e.g. cache-warm / cache-cold phases) that persist for about
    1/p_switch runs before flipping: a sticky Markov chain. Marginally bimodal and
    stationary in the long run, but strongly serially dependent, so EVT must be refused."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < p_switch
    mode = np.cumsum(flips) % 2
    return base + delta * mode + rng.normal(0.0, sigma, n)
