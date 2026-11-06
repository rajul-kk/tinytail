"""Held-out validation, GEV block maxima and baselines on synthetic traces with a known tail."""
import numpy as np
import pytest
from pwcet import synthetic as syn
from pwcet.evt import fit_pot, pwcet
from pwcet.gates import run_gates
from pwcet import validate as v


# ------------------------------------------------------------- exceedance check rules

def test_exceedance_check_exact_interval_and_rules():
    h = syn.iid_gpd(100_000, shape=0.0, seed=31)
    # true 1e-2 quantile claimed as a 1e-4 bound: about 100x too many exceedances
    bad = v.exceedance_check(syn.true_quantile(1e-2), h, 1e-4)
    assert bad.verdict == v.FAIL and bad.ci_low > 2e-4
    # true 1e-5 quantile claimed as a 1e-3 bound: safe but loose
    loose = v.exceedance_check(syn.true_quantile(1e-5), h, 1e-3)
    assert loose.verdict == v.PASS_PESSIMISTIC
    # the true quantile itself (this seed draws 66 hits for 100 expected: sampling noise
    # that the tolerance factor absorbs)
    ok = v.exceedance_check(syn.true_quantile(1e-3), h, 1e-3)
    assert ok.verdict == v.PASS
    # 1e-7 cannot be confirmed with 1e5 runs
    assert v.exceedance_check(syn.true_quantile(1e-7), h, 1e-7).verdict == v.UNTESTABLE


def test_untestable_level_can_still_fail():
    h = syn.iid_gpd(100_000, shape=0.0, seed=32)
    r = v.exceedance_check(syn.true_quantile(1e-3), h, 1e-7)  # ~100 hits, 0.01 expected
    assert r.expected < v.MIN_EXPECTED and r.verdict == v.FAIL


def test_zero_exceedances_interval_starts_at_zero():
    r = v.exceedance_check(1e12, np.arange(1000.0), 1e-3)
    assert r.exceed == 0 and r.ci_low == 0.0 and r.ci_high > 0


def test_split_keeps_time_order():
    fit, hold = v.split_trace(np.arange(9.0), 1 / 3)
    assert fit.tolist() == [0, 1, 2] and hold[0] == 3 and hold.size == 6


# ------------------------------------------------------------- end-to-end holdout

def test_gpd_bound_passes_holdout_on_iid_trace_with_known_tail():
    x = syn.iid_gpd(150_000, shape=0.0, seed=33)
    fit_x, hold = v.split_trace(x, 1 / 3)
    assert run_gates(fit_x).passed
    fit = fit_pot(fit_x, 0.95, gof_reps=0)
    ps = [1e-3, 1e-4, 1e-6]
    bounds = dict(zip(ps, pwcet(fit, ps)))
    for p in ps:
        assert bounds[p] == pytest.approx(syn.true_quantile(p), rel=0.03)
    res = {c.p: c.verdict for c in v.validate_bounds(bounds, hold)}
    assert res[1e-3] in (v.PASS, v.PASS_PESSIMISTIC)
    assert res[1e-4] in (v.PASS, v.PASS_PESSIMISTIC)
    assert res[1e-6] == v.UNTESTABLE


def test_shift_after_fit_window_passes_gates_but_fails_holdout():
    """The gates only see the fit window; a later mean shift must be caught by validation."""
    x = syn.regime_shift(60_000, at=0.5, delta=20.0, seed=34)
    fit_x, hold = v.split_trace(x, 1 / 3)  # shift happens inside the held-out part
    assert run_gates(fit_x).passed
    fit = fit_pot(fit_x, 0.95, gof_reps=0)
    checks = v.validate_bounds({1e-3: float(pwcet(fit, 1e-3))}, hold)
    assert checks[0].verdict == v.FAIL and checks[0].rate > 0.1


# ------------------------------------------------------------- gates refuse

def test_gates_pass_iid_and_refuse_mode_switching():
    assert run_gates(syn.iid_gpd(20_000, seed=35)).passed
    g = run_gates(syn.mode_switching(20_000, seed=36))
    assert g.verdict == "EVT-invalid-dependent"
    assert {"runs", "ljung_box"} <= set(g.failed)


def test_gates_refuse_regime_shift_and_ar1_and_constant():
    g = run_gates(syn.regime_shift(20_000, delta=20.0, seed=37))
    assert not g.passed and "kpss" in g.failed and "ks_halves" in g.failed
    assert run_gates(syn.ar1(20_000, seed=38)).verdict == "EVT-invalid-dependent"
    assert run_gates(np.full(5000, 1234.0)).verdict == "degenerate"
    # many distinct values but CV below 1e-4 is degenerate too
    assert run_gates(1e6 + np.arange(5000) * 1e-3).verdict == "degenerate"


# ------------------------------------------------------------- GEV block maxima

def test_block_maxima_shape():
    m = v.block_maxima(np.arange(10.0), 3)
    assert m.tolist() == [2.0, 5.0, 8.0]


def test_gev_recovers_exponential_tail():
    x = syn.iid_gpd(100_000, shape=0.0, seed=39)
    fit = v.fit_gev(x, gof_reps=30)
    assert fit.block_size == 1000 and fit.n_blocks == 100
    assert abs(fit.shape) < 0.2  # Gumbel domain; 100 maxima give a wide shape estimate
    assert fit.gof_pvalue > 0.05
    for p in (1e-4, 1e-6):
        assert v.gev_pwcet(fit, p) == pytest.approx(syn.true_quantile(p), rel=0.05)
    vals = v.gev_pwcet(fit, [1e-4, 1e-6, 1e-9])
    assert np.all(np.diff(vals) > 0)


# ------------------------------------------------------------- MBPTA-CV and HWM

def test_cv_plot_matches_brute_force_with_ties():
    rng = np.random.default_rng(7)
    x = np.round(rng.exponential(5.0, 400))  # integer values: many ties
    k, th, mean, cv, valid = v.cv_plot(x)
    assert k[-1] == 200  # never more than half of the sample (Rule 4)
    for kk in (3, 10, 57, 200):
        u = np.sort(x)[::-1][kk]
        exc = x[x > u] - u
        if valid[kk - 1]:
            assert exc.size == kk
            assert mean[kk - 1] == pytest.approx(exc.mean())
            assert cv[kk - 1] == pytest.approx(exc.std(ddof=1) / exc.mean())
        else:  # tie across the threshold: fewer than kk strict exceedances
            assert exc.size < kk
    assert not valid.all() and valid.any()


def test_mbpta_cv_exponential_tail_matches_truth():
    x = syn.iid_gpd(100_000, shape=0.0, seed=40)
    f = v.mbpta_cv(x)
    assert f.applicable and abs(f.cv - 1) < 0.05
    assert v.CV_MIN_EXCEED <= f.n_exceed <= x.size // 2
    assert v.cv_pwcet(f, 1e-6) == pytest.approx(syn.true_quantile(1e-6), rel=0.05)


def test_mbpta_cv_follows_the_paper_selection_rules():
    x = syn.iid_gpd(100_000, shape=0.0, seed=44)
    f = v.mbpta_cv(x)
    k, th, mean, cv, valid = v.cv_plot(x)
    band = 1 + v.CV_Z / np.sqrt(k)
    upto = valid & (k >= v.CV_MIN_TAIL) & (k <= f.n_exceed)
    assert np.all(cv[upto] <= band[upto])  # Const2: every smaller tail down to 10 passes
    elig = valid & (k >= v.CV_MIN_EXCEED) & (k <= f.n_exceed)
    assert abs(f.cv - 1) == pytest.approx(np.min(np.abs(cv[elig] - 1)))  # Rule 3: closest to 1
    assert f.threshold == th[f.n_exceed - 1]


def test_mbpta_cv_declines_when_the_top_values_look_heavy():
    """Rule 2: a rejection anywhere among the 10-49 largest values means no estimate."""
    x = np.concatenate([syn.iid_gpd(20_000, shape=0.0, seed=45), [1e5, 2e5, 4e5]])
    f = v.mbpta_cv(x)
    assert not f.applicable and f.n_exceed < v.CV_MIN_EXCEED
    assert "10-49 largest" in f.reason
    with pytest.raises(ValueError):
        v.cv_pwcet(f, 1e-6)


def test_mbpta_cv_heavy_tails_declined_or_optimistic_and_light_tail_pessimistic():
    very_heavy = v.mbpta_cv(syn.iid_gpd(100_000, shape=0.5, seed=41))
    assert not very_heavy.applicable and very_heavy.cv > 1
    with pytest.raises(ValueError):
        v.cv_pwcet(very_heavy, 1e-6)
    # Known weakness of the method, kept visible: the 95 % band at small k is wide
    # (1.28 at k = 50), so a shape-0.3 tail is often accepted and its bound is optimistic.
    heavy = v.mbpta_cv(syn.iid_gpd(100_000, shape=0.3, seed=41))
    assert heavy.applicable and heavy.n_exceed < 100
    assert v.cv_pwcet(heavy, 1e-6) < syn.true_quantile(1e-6, shape=0.3)
    light = v.mbpta_cv(syn.iid_gpd(100_000, shape=-0.2, seed=42))
    assert light.applicable and light.cv < 1
    assert v.cv_pwcet(light, 1e-6) > syn.true_quantile(1e-6, shape=-0.2)


def test_mbpta_cv_short_or_constant_trace():
    assert v.mbpta_cv(np.arange(60.0)).reason == "trace too short"
    assert not v.mbpta_cv(np.full(1000, 7.0)).applicable


def test_max_plus_margin():
    assert v.max_plus_margin([1.0, 10.0, 5.0]) == pytest.approx(12.0)
