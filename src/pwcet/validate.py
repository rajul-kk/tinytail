"""Second estimators, baselines and held-out validation of pWCET bounds.

- Block maxima with a GEV fit (the alternative to the GPD peaks-over-threshold fit in evt.py).
- MBPTA-CV-style exponential-tail baseline (coefficient of variation of excesses).
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

CV_MIN_EXCEED = 50
CV_MAX_FRAC = 0.10
CV_Z = 1.96


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


def mbpta_cv(x, min_exceed=CV_MIN_EXCEED, max_frac=CV_MAX_FRAC, z=CV_Z, n_grid=40) -> CvFit:
    """Simplified re-implementation of the MBPTA-CV idea (Abella et al., ACM TODAES 2017):
    excesses of an exponential tail have coefficient of variation 1, with approximate
    standard error 1/sqrt(k) for k excesses. Tail sizes k are scanned from the largest
    (max_frac * n) down to `min_exceed`; the first tail whose CV is not significantly above 1
    (CV <= 1 + z / sqrt(k)) is modelled as exponential. A CV below 1 means a lighter tail,
    for which the exponential bound is pessimistic and therefore accepted. If every tail
    has CV above the band the tail is heavier than exponential and the method declines.
    Not checked against the authors' reference implementation."""
    x = np.asarray(x, float)
    n = x.size
    xs = np.sort(x)[::-1]
    k_max = int(max_frac * n)
    if k_max < min_exceed:
        return CvFit(False, float("nan"), float("nan"), 0, n, float("nan"), "trace too short")
    ks = np.unique(np.geomspace(min_exceed, k_max, n_grid).astype(int))[::-1]
    last = None
    for k in ks:
        u = xs[k]
        exc = x[x > u] - u
        if exc.size < min_exceed:
            continue
        mean = float(exc.mean())
        cv = float(exc.std(ddof=1) / mean)
        last = (u, cv, exc.size, mean)
        if cv <= 1 + z / np.sqrt(exc.size):
            return CvFit(True, float(u), cv, int(exc.size), n, mean)
    if last is None:
        return CvFit(False, float("nan"), float("nan"), 0, n, float("nan"),
                     "too many ties: no tail with enough strict exceedances")
    u, cv, k, mean = last
    return CvFit(False, float(u), cv, int(k), n, mean,
                 "CV above the exponential band at every tail size (heavier than exponential)")


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
