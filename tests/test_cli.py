"""End-to-end pipeline on synthetic traces: interference generator, loader, gates, report."""
import json
import numpy as np
from pwcet import synthetic as syn
from pwcet import analysis as an
from pwcet.cli import main_analyse, main_synth


def test_interference_generator_is_bimodal_and_bounded():
    x = syn.interference(20_000, seed=51)
    assert np.all(x == np.round(x))
    cold = x > 200_000 + 3_000  # misses add about 6,000 cycles
    assert 0.07 < cold.mean() < 0.13  # miss_prob 0.1
    # every run is preempted by 1 or 2 ticks of 250 cycles (1.6 ms run, 1 ms period)
    hot = x[~cold]
    assert 200_250 - 300 < np.median(hot) < 200_500 + 300
    assert x.max() < 200_000 + 6_000 + 5 * 400 + 3 * 250 + 300


def test_interference_dependence_depends_on_gap_randomisation():
    from pwcet.gates import run_gates
    assert run_gates(syn.interference(30_000, seed=52)).passed
    fixed = run_gates(syn.interference(30_000, gap_random=False, seed=53))
    assert fixed.verdict == "EVT-invalid-dependent"
    sticky = run_gates(syn.interference(30_000, miss_stickiness=0.5, seed=54))
    assert sticky.verdict == "EVT-invalid-dependent"


def test_parse_serial_format_excludes_wrap_rows_and_counts_gaps():
    text = "# board=rp2040\nT,0,100,0,0\nT,1,105,1,1\nT,3,99,2,2\n"
    t = an.parse_trace(text)
    assert t.cycles.tolist() == [100.0, 99.0]
    assert t.n_wrap_excluded == 1 and t.n_seq_missing == 1 and t.header == ["# board=rp2040"]
    assert any("timer-wrap" in r for r in an.measurement_failures(t))


def test_parse_plain_csv_with_and_without_header():
    assert an.parse_trace("idx,cycles\n0,10\n1,12\n").cycles.tolist() == [10.0, 12.0]
    assert an.parse_trace("10\n12\n13\n").cycles.tolist() == [10.0, 12.0, 13.0]


def test_cli_end_to_end_valid_trace(tmp_path, capsys):
    trace = tmp_path / "iid.csv"
    assert main_synth(["iid", "--n", "90000", "--seed", "55", "--out", str(trace)]) == 0
    out = tmp_path / "r.json"
    rc = main_analyse([str(trace), "--origin", "synthetic", "--out", str(out),
                       "--gof-reps", "20", "--gev-gof-reps", "10"])
    assert rc == 0
    r = json.loads(out.read_text())
    assert r["origin"] == "synthetic" and r["gates"]["passed"]
    assert r["verdict"] == "EVT-valid"
    assert set(r["estimates"]) == {"gpd", "gev", "mbpta_cv", "memik", "atan"}
    for m in ("memik", "atan"):  # inequality baselines: either a bound or a stated reason
        e = r["estimates"][m]
        assert ("pwcet" in e) == e["applicable"] and (e["applicable"] or e["reason"])
    v = {c["p"]: c["verdict"] for c in r["validation"]["gpd"]}
    assert v[1e-3] in ("pass", "pass-pessimistic") and v[1e-9] == "untestable"
    assert r["primary_outcome"].startswith("validated")
    assert "primary outcome" in capsys.readouterr().out


def test_cli_refuses_mode_switching_trace_and_computes_no_bound(tmp_path):
    trace = tmp_path / "ms.csv"
    main_synth(["mode-switching", "--n", "30000", "--seed", "56", "--out", str(trace)])
    out = tmp_path / "r.json"
    assert main_analyse([str(trace), "--origin", "synthetic", "--out", str(out), "--quiet"]) == 0
    r = json.loads(out.read_text())
    assert r["verdict"] == "EVT-invalid-dependent"
    assert r["estimates"] == {} and r["validation"] == {}
    assert r["primary_outcome"].startswith("not-applicable")
    assert r["baseline_max_plus_margin"]["probabilistic"] is False


def test_cli_hardware_short_session_is_measurement_failure(tmp_path):
    trace = tmp_path / "hw.csv"
    main_synth(["iid", "--n", "1000", "--out", str(trace)])
    out = tmp_path / "r.json"
    assert main_analyse([str(trace), "--origin", "hardware", "--out", str(out)]) == 3
    assert json.loads(out.read_text())["status"] == "measurement-failure"


def test_cli_bad_input_returns_2(tmp_path):
    assert main_analyse([str(tmp_path / "missing.csv"), "--origin", "synthetic"]) == 2
