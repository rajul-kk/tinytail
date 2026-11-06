"""Independence tests on a timing trace. Each returns a TestResult."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest class
    name: str
    statistic: float
    pvalue: float
    alpha: float = 0.05

    @property
    def reject(self) -> bool:
        """True when the null (independence / stationarity) is rejected at alpha."""
        return self.pvalue < self.alpha


def runs_test(x, alpha=0.05) -> TestResult:
    """Wald-Wolfowitz runs test about the median (null: independent). Values equal to
    the median are dropped; heavily tied traces lose power, so check degeneracy first."""
    x = np.asarray(x, float)
    s = x[x != np.median(x)] > np.median(x)
    n1, n2 = int(s.sum()), int((~s).sum())
    if n1 == 0 or n2 == 0:
        return TestResult("runs", float("nan"), float("nan"), alpha)
    runs = 1 + int(np.count_nonzero(s[1:] != s[:-1]))
    n = n1 + n2
    mu = 2.0 * n1 * n2 / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / (n - 1.0)
    z = (runs - mu) / np.sqrt(var)
    return TestResult("runs", float(z), float(2 * norm.sf(abs(z))), alpha)


def ljung_box(x, lags=20, alpha=0.05) -> TestResult:
    """Ljung-Box portmanteau test up to `lags` (null: no autocorrelation)."""
    r = acorr_ljungbox(np.asarray(x, float), lags=[lags])
    return TestResult("ljung_box", float(r["lb_stat"].iloc[0]),
                      float(r["lb_pvalue"].iloc[0]), alpha)
