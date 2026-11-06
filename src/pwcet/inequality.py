"""Inequality-based pWCET bounds: the third baseline (PROTOCOL.md section 14, deviation D2).

Both methods bound the exceedance probability with a generalised Markov inequality: for a
non-negative, increasing f, P(X >= b) <= E[f(X)] / f(b). They need no tail model, so they
have no model uncertainty, but E[f(X)] is estimated from the sample.

- MEMIK (Vilardell, Serra, Mezzetti, Abella, Cazorla, del Castillo, ECRTS 2022):
  f(x) = x^k, minimised over k = 1 .. maxk.
- ATAN (Toba, Yano, Azumi, arXiv 2511.11682, 2025): f(x) = arctan(x / d)^k, minimised over
  a (d, k) grid. The paper gives no grid; ours is d = median(x) * D_FACTORS (large d tends
  to MEMIK, since arctan(x/d) is about x/d), k = 1 .. maxk. TANH is not implemented.
- RESTK (Vilardell et al. 2022, Algorithm 2) restricts maxk so that sample moments do not
  underestimate. For a sample of size n (e = floor(log10 n)) the three test probabilities
  are 10^-(e-1), 10^-(e-2), 10^-(e-3) and their quantiles are estimated empirically. In each
  of `n_sims` simulations, a bootstrap resample of size 10^(e-3) is drawn and k is
  increased from 1 until the bound from the resample falls below the empirical quantile.
  The last k before that point is kept, and maxk(p) is the minimum over simulations. A line
  in log10(p) through the three points extrapolates maxk to every target p. If |correlation|
  < 0.95, no estimate is made. When all three maxk are equal, the line is flat and accepted.
  The ATAN variant uses the ATAN bound (minimised over d) in the same scan, as Toba et al.
  apply RESTK to every method.

Bounds are computed on the raw values, as in both papers (execution times are positive),
and without any finite-sample correction (neither paper has one). All arithmetic is in logs:
cycle counts of about 2e5 raised to k = 150 overflow float64.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.special import logsumexp

K_MAX = 150                     # Vilardell et al.: k in [1, 150], step 1
D_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
RESTK_SIMS = 100                # Vilardell et al., Figure 10: nsims = 100
RESTK_CORR = 0.95
METHODS = ("memik", "atan")


@dataclass(frozen=True)
class InequalityFit:
    method: str
    applicable: bool
    n: int
    ds: tuple = ()
    log_moments: np.ndarray = field(default=None, repr=False)  # (n_d, K_MAX): log E[f_d(X)^k]
    restk_ps: tuple = ()
    restk_maxk: tuple = ()
    restk_corr: float = float("nan")
    slope: float = float("nan")
    intercept: float = float("nan")
    reason: str = ""

    def maxk(self, p) -> int:
        """RESTK cap on k at probability p (linear in log10 p, clipped to [1, K_MAX])."""
        if not self.applicable:
            raise ValueError(f"{self.method} not applicable: {self.reason}")
        return int(np.clip(np.floor(self.intercept + self.slope * np.log10(p)), 1, K_MAX))


def _log_f(x, method, d):
    """log of the base function g(x) whose k-th power is f: log x or log arctan(x / d)."""
    x = np.asarray(x, float)
    return np.log(x) if method == "memik" else np.log(np.arctan(x / d))


def _log_moments(x, method, ds, k_max=K_MAX):
    """log E[g(X)^k] for k = 1..k_max and each d: array (len(ds), k_max)."""
    ks = np.arange(1, k_max + 1)[:, None]
    out = np.empty((len(ds), k_max))
    for i, d in enumerate(ds):
        lg = _log_f(x, method, d)
        out[i] = logsumexp(ks * lg[None, :], axis=1) - np.log(lg.size)
    return out


def _bounds(log_m, method, ds, p):
    """Smallest b with E[g^k] / g(b)^k <= p, for every (d, k): array (len(ds), k_max)."""
    ks = np.arange(1, log_m.shape[1] + 1)[None, :]
    lg_b = (log_m - np.log(p)) / ks            # log g(b)
    if method == "memik":
        return np.exp(lg_b)
    t = np.exp(lg_b)                           # arctan(b / d)
    with np.errstate(over="ignore", invalid="ignore"):
        b = np.asarray(ds, float)[:, None] * np.tan(t)
    return np.where(t < np.pi / 2, b, np.inf)  # arctan never reaches pi/2: no finite bound


def _best_over_d(log_m, method, ds, p):
    """min over d of the bound, per k: array (k_max,)."""
    return np.min(_bounds(log_m, method, ds, p), axis=0)


def restk(x, method, ds, n_sims=RESTK_SIMS, k_max=K_MAX, seed=0):
    """maxk at the three test probabilities (Vilardell et al. 2022, Algorithm 2)."""
    x = np.asarray(x, float)
    e = int(np.floor(np.log10(x.size)))
    m = 10 ** (e - 3)
    if m < 10:
        raise ValueError("RESTK needs at least 10,000 values (bootstrap size n / 1000)")
    ps = tuple(10.0 ** -(e - j) for j in (1, 2, 3))
    rng = np.random.default_rng(seed)
    maxk = []
    for p in ps:
        q = float(np.quantile(x, 1 - p))
        best = k_max
        for _ in range(n_sims):
            bs = rng.choice(x, m, replace=True)
            vals = _best_over_d(_log_moments(bs, method, ds, k_max), method, ds, p) / q
            under = np.flatnonzero(vals < 1)
            k_last = k_max if under.size == 0 else int(under[0])  # last k before the first underestimate
            best = min(best, k_last)
        maxk.append(best)
    return ps, tuple(maxk)


def fit_inequality(x, method="atan", d_factors=D_FACTORS, n_sims=RESTK_SIMS, k_max=K_MAX,
                   seed=0) -> InequalityFit:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    x = np.asarray(x, float)
    if np.any(x <= 0):
        raise ValueError("execution times must be positive")
    ds = (1.0,) if method == "memik" else tuple(float(np.median(x)) * f for f in d_factors)
    try:
        ps, mk = restk(x, method, ds, n_sims, k_max, seed)
    except ValueError as err:
        return InequalityFit(method, False, x.size, ds, reason=str(err))
    lp = np.log10(ps)
    if mk[0] == mk[1] == mk[2]:
        slope, icpt, corr = 0.0, float(mk[0]), 1.0
    else:
        slope, icpt = np.polyfit(lp, mk, 1)
        corr = float(np.corrcoef(lp, mk)[0, 1])
    common = dict(n=x.size, ds=ds, restk_ps=ps, restk_maxk=mk, restk_corr=corr,
                  slope=float(slope), intercept=float(icpt))
    if min(mk) < 1:
        return InequalityFit(method, False, reason="RESTK: even k = 1 underestimates", **common)
    if abs(corr) < RESTK_CORR:
        return InequalityFit(method, False,
                             reason=f"RESTK: maxk not linear in log p (|r| = {abs(corr):.2f} < 0.95)",
                             **common)
    return InequalityFit(method, True, log_moments=_log_moments(x, method, ds, k_max), **common)


def inequality_pwcet(fit: InequalityFit, p) -> np.ndarray:
    """min over d and k = 1 .. maxk(p) of the inequality bound at exceedance probability p."""
    if not fit.applicable:
        raise ValueError(f"{fit.method} not applicable: {fit.reason}")
    ps = np.atleast_1d(np.asarray(p, float))
    out = np.array([np.min(_best_over_d(fit.log_moments, fit.method, fit.ds, q)[: fit.maxk(q)])
                    for q in ps])
    return out if np.ndim(p) else out[0]
