"""Stationarity test on a timing trace."""
import warnings
import numpy as np
from statsmodels.tsa.stattools import kpss, InterpolationWarning
from .iid import TestResult


def kpss_test(x, alpha=0.05) -> TestResult:
    """KPSS level-stationarity test (null: stationary). statsmodels clips the
    p-value to [0.01, 0.10]; the interpolation warning is silenced for that reason."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p, _, _ = kpss(np.asarray(x, float), regression="c", nlags="auto")
    return TestResult("kpss", float(stat), float(p), alpha)
