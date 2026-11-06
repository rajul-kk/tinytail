"""Applicability gates run before any tail fit (PROTOCOL.md section 5).

A trace that fails a gate gets a verdict label and **no** EVT or MBPTA bound. The gates are
run in a fixed order and every test result is reported, including the ones after the first
failure, so a trace can be both dependent and non-stationary.
"""
from dataclasses import dataclass, field
import numpy as np
from .evt import is_degenerate
from .iid import TestResult, runs_test, ljung_box
from .stationarity import kpss_test, ks_halves

N_GATE_TESTS = 4
#: Per-test level: Bonferroni over the 4 gate tests, so the family-wise chance of refusing
#: a truly i.i.d. trace is at most 0.05.
GATE_ALPHA = 0.05 / N_GATE_TESTS
LJUNG_BOX_LAGS = 20
MIN_DISTINCT = 20
MIN_CV = 1e-4

DEGENERATE = "degenerate"
INVALID_DEPENDENT = "EVT-invalid-dependent"
INVALID_NONSTATIONARY = "EVT-invalid-nonstationary"
INVALID_POOR_FIT = "EVT-invalid-poor-fit"
VALID = "EVT-valid"


@dataclass
class GateReport:
    verdict: str  # DEGENERATE, INVALID_DEPENDENT, INVALID_NONSTATIONARY, or "passed"
    tests: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    n_distinct: int = 0
    cv: float = float("nan")

    @property
    def passed(self) -> bool:
        return self.verdict == "passed"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "failed": list(self.failed),
            "n_distinct": self.n_distinct,
            "cv": self.cv,
            "tests": [{"name": t.name, "statistic": t.statistic, "pvalue": t.pvalue,
                       "alpha": t.alpha, "reject": t.reject} for t in self.tests],
        }


def run_gates(x, alpha=GATE_ALPHA, lags=LJUNG_BOX_LAGS) -> GateReport:
    """Degeneracy, then independence (runs, Ljung-Box), then identical distribution /
    stationarity (KPSS, KS first-vs-second half)."""
    x = np.asarray(x, float)
    n_distinct = int(np.unique(x).size)
    mean = float(np.mean(x))
    cv = float(np.std(x) / mean) if mean != 0 else float("nan")
    if is_degenerate(x, MIN_DISTINCT) or not cv >= MIN_CV:
        return GateReport(DEGENERATE, [], ["degenerate"], n_distinct, cv)
    indep: list[TestResult] = [runs_test(x, alpha), ljung_box(x, lags, alpha)]
    stat: list[TestResult] = [kpss_test(x, alpha), ks_halves(x, alpha)]
    failed = [t.name for t in indep + stat if t.reject]
    if any(t.reject for t in indep):
        verdict = INVALID_DEPENDENT
    elif any(t.reject for t in stat):
        verdict = INVALID_NONSTATIONARY
    else:
        verdict = "passed"
    return GateReport(verdict, indep + stat, failed, n_distinct, cv)
