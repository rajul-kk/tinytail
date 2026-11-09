import numpy as np
import pytest
from pwcet import synthetic as syn
from pwcet.evt import fit_pot, choose_threshold, pwcet, is_degenerate

N = 100_000
PS = [1e-6, 1e-7, 1e-8, 1e-9]


def test_gpd_shape_recovered_within_0p08():
    # ~5000 exceedances; standard error of the shape is about 0.02, tolerance is 4 sigma.
    for shape in (0.0, 0.2):
        x = syn.iid_gpd(N, shape=shape, seed=11)
        assert abs(fit_pot(x, 0.95, gof_reps=0).shape - shape) < 0.08


def test_pwcet_matches_truth_exponential_tail():
    x = syn.iid_gpd(N, shape=0.0, seed=12)
    fit = fit_pot(x, 0.95, gof_reps=0)
    for p in (1e-4, 1e-6):
        assert pwcet(fit, p) == pytest.approx(syn.true_quantile(p, shape=0.0), rel=0.05)


def test_pwcet_monotone_in_exceedance_probability():
    fit = fit_pot(syn.iid_gpd(N, shape=0.1, seed=13), 0.95, gof_reps=0)
    v = pwcet(fit, PS)  # decreasing p -> increasing bound
    assert np.all(np.diff(v) > 0)


def test_gof_accepts_true_gpd_and_rejects_wrong_tail():
    good = fit_pot(syn.iid_gpd(N, shape=0.1, seed=14), 0.95, gof_reps=100)
    assert good.gof_pvalue > 0.05
    # lognormal body+tail is not GPD at a low threshold
    x = np.random.default_rng(15).lognormal(0, 1, N)
    bad = fit_pot(x, 0.5, gof_reps=100)
    assert bad.gof_pvalue < 0.05


def test_choose_threshold_returns_fit_and_pwcet_above_max_of_sample():
    x = syn.iid_gpd(N, shape=0.0, seed=16)
    fit = choose_threshold(x, gof_reps=50)
    assert fit.gof_pvalue >= 0.05
    assert pwcet(fit, 1e-9) > x.max()


def test_degenerate_trace_refused():
    x = np.full(1000, 5000.0)
    assert is_degenerate(x)
    with pytest.raises(ValueError):
        fit_pot(x)


def test_pwcet_rejects_p_above_zeta():
    fit = fit_pot(syn.iid_gpd(N, seed=17), 0.95, gof_reps=0)
    with pytest.raises(ValueError):
        pwcet(fit, 0.5)


def _tied_top(n, seed):
    """Quantised timings: 25 levels, the top one holding 8% of the sample."""
    rng = np.random.default_rng(seed)
    p = np.full(25, 0.92 / 24)
    p[-1] = 0.08
    return rng.choice(np.arange(50.0, 75.0), size=n, p=p)


def test_tied_top_refused_with_reason():
    with pytest.raises(ValueError, match="exceedances"):
        fit_pot(_tied_top(20_000, 3), 0.95, gof_reps=0)


def test_analyse_skips_a_quantile_with_no_exceedances():
    from pwcet import analysis as an
    x = _tied_top(40_000, 4)
    r = an.analyse(x[:20_000], x[20_000:], an.Config(gof_reps=20), origin="synthetic")
    if not r["gates"]["passed"]:
        pytest.skip("gates failed on this draw: " + str(r["gates"]["failed"]))
    cands = {c["quantile"]: c for c in r["estimates"]["gpd"]["candidates"]}
    assert "error" in cands[0.95] and "error" not in cands[0.90]
