import numpy as np
from pwcet import synthetic as syn
from pwcet.iid import runs_test, ljung_box
from pwcet.stationarity import kpss_test

N = 20000


def test_iid_passes_all():
    x = syn.iid_gpd(N, seed=1)
    assert not runs_test(x).reject
    assert not ljung_box(x).reject
    assert not kpss_test(x).reject


def test_ar1_flagged_as_dependent():
    x = syn.ar1(N, phi=0.8, seed=2)
    assert runs_test(x).reject
    assert ljung_box(x).reject


def test_regime_shift_flagged():
    x = syn.regime_shift(N, delta=20.0, seed=3)
    assert kpss_test(x).reject
    assert ljung_box(x).reject


def test_periodic_spikes_flagged_by_ljung_box():
    x = syn.periodic_spikes(N, period=10, height=200.0, seed=4)
    assert ljung_box(x, lags=20).reject


def test_runs_test_constant_trace_is_nan_not_crash():
    r = runs_test(np.full(100, 7.0))
    assert np.isnan(r.pvalue) and not r.reject
