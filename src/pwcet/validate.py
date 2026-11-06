"""Second estimators, baselines and held-out validation of pWCET bounds.

- Block maxima with a GEV fit (the alternative to the GPD peaks-over-threshold fit in evt.py).
- MBPTA-CV exponential-tail baseline (Abella et al. 2017 rules, coefficient of variation of excesses).
- Maximum observed value plus a fixed margin (industrial-practice baseline).
- Held-out validation: count how often held-out runs exceed each bound and compare the rate
  with the target exceedance probability using an exact (Clopper-Pearson) binomial interval.

Decision rules and default parameters are fixed in protocol/PROTOCOL.md.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.stats import genextreme, binomtest
from .evt import ad_from_cdf

# ---------------------------------------------------------------- GEV / block maxima

GEV_N_BLOCKS = 100


@dataclass(frozen=True)
class GevFit:
    shape: float  # xi in the EVT convention (xi > 0 heavy tail); scipy's c is -xi
    loc: float
    scale: float
    block_size: int
    n_blocks: int
    gof_pvalue: float = float("nan")


def block_maxima(x, block_size) -> np.ndarray:
    """Maxima of consecutive non-overlapping blocks; a trailing partial block is dropped."""
    x = np.asarray(x, float)
    m = x.size // block_size
    if m < 1:
        raise ValueError("trace shorter than one block")
    return x[: m * block_size].reshape(m, block_size).max(axis=1)


def _fit_gev(m):
    """MLE on standardised maxima, started from the Gumbel moment solution. Without the
    standardisation scipy's optimiser diverged on cycle-count maxima (values near 2e5, spread
    near 2e2), returning shape -4 on a synthetic interference trace."""
    m = np.asarray(m, float)
    mu, sd = m.mean(), m.std(ddof=1)
    c, loc, scale = genextreme.fit((m - mu) / sd, 0.0, loc=-0.45, scale=0.78)
    return float(c), float(loc * sd + mu), float(scale * sd)


def gev_gof_pvalue(m, c, loc, scale, reps=100, seed=0) -> float:
    """Parametric-bootstrap Anderson-Darling p-value for a GEV fit (parameters re-fitted
    on every replicate)."""
    rng = np.random.default_rng(seed)
    a0 = ad_from_cdf(genextreme.cdf(m, c, loc=loc, scale=scale))
    count = 0
    for _ in range(reps):
        sim = genextreme.rvs(c, loc=loc, scale=scale, size=len(m), random_state=rng)
        cs, ls, ss = _fit_gev(sim)
        count += ad_from_cdf(genextreme.cdf(sim, cs, loc=ls, scale=ss)) >= a0
    return (count + 1) / (reps + 1)


def fit_gev(x, n_blocks=GEV_N_BLOCKS, gof_reps=100, seed=0) -> GevFit:
    """GEV by maximum likelihood on `n_blocks` block maxima (block size n // n_blocks)."""
    x = np.asarray(x, float)
    b = x.size // n_blocks
    if b < 2:
        raise ValueError("trace too short for the requested number of blocks")
    m = block_maxima(x, b)
    if np.unique(m).size < 5:
        raise ValueError("block maxima nearly constant: GEV not applicable")
    c, loc, scale = _fit_gev(m)
    p = gev_gof_pvalue(m, c, loc, scale, gof_reps, seed) if gof_reps else float("nan")
    return GevFit(-c, loc, scale, b, int(m.size), p)


def gev_pwcet(fit: GevFit, p) -> np.ndarray:
    """Per-run exceedance p converted to per-block 1 - (1 - p)^b, then the GEV quantile.
    This assumes runs inside a block are independent (the gates check that first)."""
    p = np.asarray(p, float)
    pb = -np.expm1(fit.block_size * np.log1p(-p))
    return genextreme.isf(pb, -fit.shape, loc=fit.loc, scale=fit.scale)


# ---------------------------------------------------------------- MBPTA-CV baseline

CV_MIN_TAIL = 10      # Rule 1: tails of fewer than 10 values are not tested
CV_MIN_EXCEED = 50    # Const1: the fitted tail has at least 50 exceedances
CV_Z = 1.96           # 95 % band: under exponentiality the residual CV is about N(1, 1/k)


@dataclass(frozen=True)
class CvFit:
    applicable: bool
    threshold: float
    cv: float
    n_exceed: int
    n: int
    mean_excess: float
    reason: str = ""

    @property
    def zeta(self) -> float:
        return self.n_exceed / self.n


def cv_plot(x):
    """Empirical residual CV for every tail size k = 1 .. n // 2 (the lower half of the sample
    is never treated as tail, Rule 4). With values sorted in decreasing order, tail k has
    threshold th = x_(k+1) and excesses x_(1..k) - th. A tail size is `valid` only if its k
    values lie strictly above th (no tie across the threshold). Returns (k, threshold, mean
    excess, cv, valid) as arrays indexed by k - 1."""
    xs = np.sort(np.asarray(x, float))[::-1]
    n = xs.size
    kmax = n // 2
    z = xs[0] - xs[: kmax + 1]                # >= 0, increasing; excess_i = z[k] - z[i]
    c1 = np.cumsum(z[:kmax])
    c2 = np.cumsum(z[:kmax] ** 2)
    k = np.arange(1, kmax + 1, dtype=float)
    th = xs[1: kmax + 1]
    mean = z[1: kmax + 1] - c1 / k
    with np.errstate(divide="ignore", invalid="ignore"):
        var = np.maximum(c2 - c1 ** 2 / k, 0.0) / (k - 1)
        cv = np.sqrt(var) / mean
    valid = xs[:kmax] > th                    # x_(k) > x_(k+1)
    valid &= mean > 0
    return k.astype(int), th, mean, cv, valid


def mbpta_cv(x, min_tail=CV_MIN_TAIL, min_exceed=CV_MIN_EXCEED, z=CV_Z) -> CvFit:
    """MBPTA-CV following Abella et al., ACM TODAES 22(4) 2017, section 5 (re-implemented from
    the paper, not from the authors' code):

    - the exponentiality test for a tail of k exceedances passes if cv <= 1 + z / sqrt(k):
      exponential (inside the 95 % band) or light (below it), the latter being upper-bounded
      by an exponential and therefore accepted (Rule 5);
    - Rule 1/2: the test must pass for every tail of 10 to 49 values, otherwise the method
      declines (the paper asks for more runs; with the protocol's fixed sample size it
      declines);
    - Const1/Const2/Rule 3: the chosen tail has k >= 50 and the test passes for it and for
      every smaller tail down to 10; among those the k whose cv is closest to 1 is chosen;
    - Rule 4: at most half of the sample is tail;
    - section 5.3: exponential fit with rate 1 / mean excess, shifted by the threshold.
    The band 1 +- z / sqrt(k) is the asymptotic normal band for the residual CV of an
    exponential sample (the paper cites del Castillo et al. 2014 without printing it).
    Tail sizes whose threshold value is tied with a tail value are skipped."""
    x = np.asarray(x, float)
    n = x.size
    if n // 2 < min_exceed:
        return CvFit(False, float("nan"), float("nan"), 0, n, float("nan"), "trace too short")
    k, th, mean, cv, valid = cv_plot(x)
    passed = cv <= 1 + z / np.sqrt(k)
    small = valid & (k >= min_tail) & (k < min_exceed)
    if np.any(small & ~passed):
        k_bad = int(k[small & ~passed][0])
        return CvFit(False, float(th[k_bad - 1]), float(cv[k_bad - 1]), k_bad, n,
                     float(mean[k_bad - 1]),
                     f"exponentiality rejected for the {k_bad} largest values "
                     "(heavier than exponential among the 10-49 largest)")
    big = valid & (k >= min_exceed)
    fails = np.flatnonzero(big & ~passed)
    stop = int(k[fails[0]]) if fails.size else int(k[-1]) + 1
    elig = np.flatnonzero(big & (k < stop))
    if elig.size == 0:
        if not np.any(big):
            return CvFit(False, float("nan"), float("nan"), 0, n, float("nan"),
                         "too many ties: no tail of >= 50 values strictly above its threshold")
        i = fails[0]
        return CvFit(False, float(th[i]), float(cv[i]), int(k[i]), n, float(mean[i]),
                     "exponentiality rejected at the smallest admissible tail (k >= 50)")
    i = elig[np.argmin(np.abs(cv[elig] - 1.0))]
    return CvFit(True, float(th[i]), float(cv[i]), int(k[i]), n, float(mean[i]))


def cv_pwcet(fit: CvFit, p) -> np.ndarray:
    """Exponential tail: u + mean_excess * ln(zeta / p)."""
    if not fit.applicable:
        raise ValueError(f"MBPTA-CV not applicable: {fit.reason}")
    p = np.asarray(p, float)
    if np.any(p >= fit.zeta):
        raise ValueError("p must be below the tail fraction zeta")
    return fit.threshold + fit.mean_excess * np.log(fit.zeta / p)


# ---------------------------------------------------------------- max observed + margin

MARGIN = 0.20


def max_plus_margin(x, margin=MARGIN) -> float:
    """High-water mark times (1 + margin). Not a probabilistic bound: it has no exceedance
    probability attached, which is why it is a baseline."""
    return float(np.max(x) * (1.0 + margin))


# ---------------------------------------------------------------- held-out validation

CONFIDENCE = 0.95
MIN_EXPECTED = 5.0
TOLERANCE = 2.0

PASS = "pass"
PASS_PESSIMISTIC = "pass-pessimistic"
FAIL = "fail"
UNTESTABLE = "untestable"


@dataclass(frozen=True)
class HoldoutCheck:
    p: float
    bound: float
    n: int
    exceed: int
    rate: float
    ci_low: float
    ci_high: float
    expected: float
    verdict: str

    def as_dict(self) -> dict:
        return asdict(self)


def split_trace(x, fit_frac=0.5):
    """Earlier part for fitting, later part held out (time order is kept)."""
    x = np.asarray(x, float)
    k = int(round(fit_frac * x.size))
    if k < 1 or k >= x.size:
        raise ValueError("fit_frac leaves an empty part")
    return x[:k], x[k:]


def exceedance_check(bound, holdout, p, confidence=CONFIDENCE,
                     min_expected=MIN_EXPECTED, tolerance=TOLERANCE) -> HoldoutCheck:
    """Compare the held-out exceedance rate of `bound` with the target probability p,
    using the exact (Clopper-Pearson) interval [lo, hi] and tolerance factor T
    (PROTOCOL.md section 9), applied in this order:

    fail             : lo > T * p (optimistic beyond sampling error and tolerance: unsafe).
    untestable       : n * p < min_expected, so a pass would carry no evidence. A fail can
                       still be detected at such p, and is reported as fail.
    pass-pessimistic : hi < p / T (safe, but looser than claimed).
    pass             : otherwise.
    """
    h = np.asarray(holdout, float)
    n = int(h.size)
    k = int(np.count_nonzero(h > bound))
    ci = binomtest(k, n).proportion_ci(confidence, method="exact")
    lo, hi = float(ci.low), float(ci.high)
    expected = n * float(p)
    if lo > tolerance * p:
        verdict = FAIL
    elif expected < min_expected:
        verdict = UNTESTABLE
    elif hi < p / tolerance:
        verdict = PASS_PESSIMISTIC
    else:
        verdict = PASS
    return HoldoutCheck(float(p), float(bound), n, k, k / n, lo, hi, expected, verdict)


def validate_bounds(bounds: dict, holdout, confidence=CONFIDENCE,
                    min_expected=MIN_EXPECTED, tolerance=TOLERANCE) -> list:
    """`bounds` maps target probability p to the pWCET value; returns one check per p."""
    return [exceedance_check(v, holdout, p, confidence, min_expected, tolerance)
            for p, v in sorted(bounds.items(), reverse=True)]
