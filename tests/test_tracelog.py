"""Firmware log formats: binary frames (firmware/common/tlog.h), overhead rows, format detection,
and the C encoder cross-checked against the Python parser when a native gcc is available."""
import json
import pathlib
import shutil
import subprocess
import numpy as np
import pytest
from pwcet import analysis as an
from pwcet import tracelog as tl
from pwcet.cli import main_analyse

FW = pathlib.Path(__file__).resolve().parents[1] / "firmware"


def test_frame_layout_and_checksum():
    f = tl.encode_frame(0x01020304, 200_000, 3, 0x0A)
    assert len(f) == tl.FRAME == 13 and f[:2] == b"\xA5\x5A"
    assert f[2:6] == bytes([4, 3, 2, 1])               # seq, little endian
    assert int.from_bytes(f[6:10], "little") == 200_000
    assert f[10] == 3 and f[11] == 0x0A and f[12] == sum(f[2:12]) & 0xFF


def test_binary_round_trip_with_header_wrap_and_overhead_rows():
    rows = [(0, 3, 255, 0), (1, 4, 255, 0), (2, 200_100, 0, 0), (3, 200_350, 1, 2),
            (4, 999, 2, 1), (6, 200_200, 3, 4)]            # seq 5 missing, one wrap row
    data = tl.encode_binary(rows, header=["board=test", "# scenario=S1"])
    assert tl.is_binary(data)
    t = an.parse_bytes(data)
    assert t.header == ["#board=test", "# scenario=S1"]
    assert t.cycles.tolist() == [200_100.0, 200_350.0, 200_200.0]
    assert t.overhead.tolist() == [3.0, 4.0]               # reported apart, never analysed
    assert t.n_rows == 4 and t.n_wrap_excluded == 1 and t.n_seq_missing == 1
    assert t.bad_frames == 0


def test_corrupt_and_truncated_frames_are_skipped_and_counted():
    good = [tl.encode_frame(i, 1000 + i, i % 8, 0) for i in range(5)]
    bad = bytearray(good[2])
    bad[7] ^= 0xFF                                         # payload corrupted: checksum fails
    data = tl.MAGIC + good[0] + good[1] + bytes(bad) + b"\x00\x13" + good[3] + good[4][:7]
    header, rows, n_bad = tl.decode_binary(data)
    # the corrupt frame and the stray bytes after it are skipped in one resync; plus truncation
    assert rows[:, 0].tolist() == [0, 1, 3] and n_bad == 2
    t = an.parse_bytes(data)
    assert t.n_seq_missing == 1 and t.bad_frames == 2


def test_csv_overhead_rows_are_kept_apart():
    t = an.parse_trace("# h\nT,0,3,255,0\nT,1,100,0,0\nT,2,101,1,2\n")
    assert t.cycles.tolist() == [100.0, 101.0] and t.overhead.tolist() == [3.0]
    assert t.n_rows == 2 and t.n_seq_missing == 0


def test_cli_reads_binary_log_and_reports_overhead(tmp_path):
    rng = np.random.default_rng(61)
    n = 30_000
    cyc = np.round(200_000 + rng.exponential(300.0, n)).astype(int)
    rows = [(i, 7, 255, 0) for i in range(100)]
    rows += [(100 + i, int(c), i % 8, 0) for i, c in enumerate(cyc)]
    path = tmp_path / "s1.bin"
    path.write_bytes(tl.encode_binary(rows, ["board=synthetic (not a measurement)"]))
    out = tmp_path / "r.json"
    assert main_analyse([str(path), "--origin", "synthetic", "--out", str(out), "--quiet",
                         "--gof-reps", "10", "--gev-gof-reps", "10"]) == 0
    r = json.loads(out.read_text())
    assert r["fit_set"]["n"] + r["holdout_set"]["n"] == n
    assert r["meta"]["counter_overhead"] == {"n": 100, "min": 7.0, "median": 7.0, "max": 7.0}
    assert r["meta"]["bad_frames"] == 0


@pytest.mark.skipif(shutil.which("gcc") is None, reason="no native gcc to build the host harness")
@pytest.mark.parametrize("fmt", ["csv", "bin"])
def test_firmware_harness_host_build_matches_parser(tmp_path, fmt):
    """Compiles firmware/common/{harness,tlog}.c with a fake board and parses its output.
    This checks the C encoders and the harness loop only; it measures nothing."""
    exe = tmp_path / "host_harness.exe"
    src = [FW / "host" / "host_main.c", FW / "common" / "harness.c", FW / "common" / "tlog.c"]
    subprocess.run(["gcc", "-std=c99", "-O1", "-Wall", "-Werror", f"-I{FW / 'common'}",
                    *map(str, src), "-o", str(exe)], check=True, capture_output=True)
    log = tmp_path / f"log.{fmt}"
    runs = 2500
    subprocess.run([str(exe), str(log), fmt, str(runs)], check=True, timeout=60)
    t = an.load_trace(log)
    assert t.header == ["# board=host-fake (not a measurement)"]
    assert t.overhead.size == 100 and np.all(t.overhead == 3)   # fake counter-read cost
    assert t.n_rows == runs and t.n_seq_missing == 0 and t.bad_frames == 0
    assert t.cycles.size + t.n_wrap_excluded == runs
    base = t.cycles[t.cycles < 200_200]                        # runs without a fake interrupt
    assert set(np.unique(base).astype(int)) <= {200_003 + 10 * k for k in range(8)}
    assert 0.05 < np.mean(t.cycles >= 200_250) < 0.15          # about 10 % fake ISR runs
    if fmt == "bin":                                           # identical frames from C and Python
        raw = log.read_bytes()
        first = raw[raw.find(tl.MAGIC) + len(tl.MAGIC):][: tl.FRAME]
        assert first == tl.encode_frame(0, 3, 255, 0)
