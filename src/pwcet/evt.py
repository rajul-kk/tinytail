"""Peaks-over-threshold EVT: GPD fit, threshold choice, goodness of fit, pWCET.

The block-maxima / GEV estimator lives in validate.py (fit_gev, gev_pwcet).
Not implemented: declustering, bootstrap CIs on pWCET.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import genpareto


@dataclass(frozen=True)
class PotFit:
    threshold: float
    shape: float
    scale: float
    n_exceed: int
    n: int
    gof_pvalue: float = float("nan")

    @property
    def zeta(self) -> float:
        """Probability that a run exceeds the threshold."""
        return self.n_exceed / self.n


def is_degenerate(x, min_distinct=20) -> bool:
    """Near-constant traces (e.g. a cacheless core with no interference): EVT not applicable."""
    return np.unique(np.asarray(x)).size < min_distinct


def ad_statistic(exc, shape, scale) -> float:
    """Anderson-Darling A^2 of excesses against GPD(shape, scale)."""
    return ad_from_cdf(genpareto.cdf(exc, shape, scale=scale))


def ad_from_cdf(u) -> float:
    """Anderson-Darling A^2 from fitted-CDF values u = F(x_i) (any fitted family)."""
    z = np.sort(np.asarray(u, float))
    z = np.clip(z, 1e-12, 1 - 1e-12)
    n = z.size
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(z) + np.log1p(-z[::-1]))))


def _fit_excesses(exc):
    shape, _, scale = genpareto.fit(exc, floc=0.0)
    return float(shape), float(scale)


def gof_pvalue(exc, shape, scale, reps=100, seed=0) -> float:
    """Parametric-bootstrap p-value of the AD statistic (parameters re-estimated per
    replicate, so the usual AD tables are not needed)."""
    rng = np.random.default_rng(seed)
    a0 = ad_statistic(exc, shape, scale)
    count = 0
    for _ in range(reps):
        sim = genpareto.rvs(shape, scale=scale, size=len(exc), random_state=rng)
        s, sc = _fit_excesses(sim)
        count += ad_statistic(sim, s, sc) >= a0
    return (count + 1) / (reps + 1)


def fit_pot(x, quantile=0.95, gof_reps=100, seed=0) -> PotFit:
    """Fit a GPD to exceedances over the empirical `quantile` of x."""
    x = np.asarray(x, float)
    if is_degenerate(x):
        raise ValueError("degenerate trace: too few distinct values for EVT")
    u = float(np.quantile(x, quantile))
    exc = x[x > u] - u
    shape, scale = _fit_excesses(exc)
    p = gof_pvalue(exc, shape, scale, gof_reps, seed) if gof_reps else float("nan")
    return PotFit(u, shape, scale, int(exc.size), int(x.size), p)


def choose_threshold(x, quantiles=(0.90, 0.95, 0.975, 0.99), alpha=0.05,
                     gof_reps=100, seed=0) -> PotFit:
    """Lowest candidate quantile whose GPD fit is not rejected (most data, least bias);
    if every fit is rejected, the highest quantile is returned with its small p-value
    so callers can see the poor fit."""
    fit = None
    for q in quantiles:
        fit = fit_pot(x, q, gof_reps, seed)
        if fit.gof_pvalue >= alpha:
            return fit
    return fit


def pwcet(fit: PotFit, p) -> np.ndarray:
    """Value exceeded with per-run probability p (scalar or array); needs p < zeta."""
    p = np.asarray(p, float)
    if np.any(p >= fit.zeta):
        raise ValueError("p must be below the threshold exceedance rate zeta")
    return fit.threshold + genpareto.isf(p / fit.zeta, fit.shape, scale=fit.scale)
