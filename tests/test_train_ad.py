"""Feature extraction (numpy only) and a training/export smoke test on random features
(skipped when TensorFlow is not installed)."""
import importlib.util
import json
import pathlib
import numpy as np
import pytest

_P = pathlib.Path(__file__).resolve().parents[1] / "models" / "train_ad.py"
_spec = importlib.util.spec_from_file_location("train_ad", _P)
ta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ta)


def test_log_mel_tone_peaks_in_matching_band_and_vector_shape():
    sr = ta.SR
    t = np.arange(sr) / sr  # 1 s
    lm = ta.log_mel(np.sin(2 * np.pi * 1000.0 * t))
    assert lm.shape == (1 + sr // ta.HOP, ta.N_MELS)
    band = int(np.argmax(lm[10]))
    hz = ta._mel_to_hz(np.linspace(ta._hz_to_mel(0), ta._hz_to_mel(sr / 2), ta.N_MELS + 2))
    assert hz[band] <= 1000.0 <= hz[band + 2]  # 1 kHz inside the peak filter's support
    v = ta.stack_frames(lm)
    assert v.shape == (lm.shape[0] - 4, 640) and v.dtype == np.float32
    assert np.array_equal(v[0, 128:256], lm[1].astype(np.float32))


def test_wav_loading_int16_and_resampling(tmp_path):
    from scipy.io import wavfile
    t = np.arange(32_000) / 32_000  # 1 s at 32 kHz, resampled to 16 kHz
    y = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(tmp_path / "normal_id_00_00000000.wav", 32_000, y)
    v = ta.wav_vectors(str(tmp_path / "normal_id_00_00000000.wav"))
    assert v.shape == (1 + ta.SR // ta.HOP - 4, 640) and np.isfinite(v).all()
    x = ta.load_dir([str(tmp_path / "normal_id_00_00000000.wav")] * 3, vectors_per_file=10)
    assert x.shape == (30, 640)
    with pytest.raises(MemoryError):
        ta.load_dir([str(tmp_path / "normal_id_00_00000000.wav")], max_gb=1e-9)


def test_mel_conversion_roundtrip_and_auc():
    f = np.array([0.0, 500.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(ta._mel_to_hz(ta._hz_to_mel(f)), f)
    assert ta.auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert ta.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_train_quantise_export_smoke(tmp_path):
    tf = pytest.importorskip("tensorflow")
    meta = ta.main(["--synthetic", "600", "--epochs", "1", "--batch", "128",
                    "--out", str(tmp_path)])
    assert 260_000 < meta["params"] < 280_000  # about 270k as in the reference
    assert meta["input"]["dtype"] == "int8" and meta["input"]["shape"] == [1, 640]
    assert meta["output"]["dtype"] == "int8"
    blob = (tmp_path / "ad_int8.tflite").read_bytes()
    assert blob[4:8] == b"TFL3"  # flatbuffer identifier
    c_src = (tmp_path / "model_data.c").read_text()
    assert f"g_ad_model_len = {len(blob)}u" in c_src and "aligned(16)" in c_src
    assert json.loads((tmp_path / "ad_meta.json").read_text())["data"].startswith("synthetic")
    # every op in the int8 graph is an integer op that TFLM/CMSIS-NN provides
    it = tf.lite.Interpreter(  # no XNNPACK delegate, so the graph's own ops are listed
        model_content=blob,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    ops = {d["op_name"] for d in it._get_ops_details()}
    assert ops <= {"FULLY_CONNECTED", "QUANTIZE", "DEQUANTIZE"}, ops
