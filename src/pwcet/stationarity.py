"""Stationarity test on a timing trace."""
import warnings
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.tsa.stattools import kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from .iid import TestResult


def ks_halves(x, alpha=0.05) -> TestResult:
    """Two-sample Kolmogorov-Smirnov test, first half vs second half (null: same
    distribution), the usual MBPTA identical-distribution check. Integer cycle counts
    have ties, which makes the asymptotic p-value conservative (refuses less often)."""
    x = np.asarray(x, float)
    h = x.size // 2
    r = ks_2samp(x[:h], x[h:2 * h])
    return TestResult("ks_halves", float(r.statistic), float(r.pvalue), alpha)


def kpss_test(x, alpha=0.05) -> TestResult:
    """KPSS level-stationarity test (null: stationary). statsmodels clips the
    p-value to [0.01, 0.10]; the interpolation warning is silenced for that reason."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p, _, _ = kpss(np.asarray(x, float), regression="c", nlags="auto")
    return TestResult("kpss", float(stat), float(p), alpha)
