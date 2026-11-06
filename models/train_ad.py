"""Train, int8-quantise and export the MLPerf Tiny anomaly-detection autoencoder.

Model (as recalled from the MLPerf Tiny / DCASE 2020 Task 2 baseline; still to be checked
against mlcommons/tiny, see docs/scoop_check.md): input 640 = 5 stacked frames of 128
log-mel bins; 4 x [Dense(128) BatchNorm ReLU], bottleneck [Dense(8) BatchNorm ReLU],
4 x [Dense(128) BatchNorm ReLU], Dense(640). Loss: mean squared reconstruction error, which
is also the anomaly score (averaged over a clip's vectors).

Features follow the DCASE 2020 baseline settings: 16 kHz mono, n_fft 1024, hop 512, 128
mel bands (Slaney mel scale and area normalisation, like librosa's defaults), power 2,
log-mel = 10 * log10(mel + eps), 5 consecutive frames concatenated. They are computed
here with numpy/scipy because librosa is not installed; the STFT uses a periodic Hann
window and reflect padding (center=True), as in older librosa. Small numeric differences
from librosa are possible and have not been measured.

Framework choice: TensorFlow/Keras, because it was already installed on the development
machine (TF 2.21, CPU), the MLPerf Tiny reference is a Keras model, and the TFLite
converter produces the full-integer int8 flatbuffer that TensorFlow Lite Micro with
CMSIS-NN kernels runs directly. No PyTorch export path is needed. The flatbuffer is also
written as a C array (model_data.c / model_data.h) for the firmware. On random features the
exported graph is 10 int8 FULLY_CONNECTED ops (BatchNorm folded, ReLU fused), all of which
CMSIS-NN implements. The host-side check uses tf.lite.Interpreter, which TF 2.21 marks as
deprecated in favour of LiteRT (ai_edge_litert, not installed); it still runs here.

Data: the script never downloads anything. Pass --data DIR pointing at one machine type of
the DCASE 2020 Task 2 development set laid out as DIR/train/*.wav (normal clips) and,
optionally, DIR/test/*.wav (file names containing "anomaly" or "normal"; the test clips give
a clip-level ROC AUC for the float and the int8 model). --synthetic N trains on N random
feature vectors instead: that exercises the pipeline and gives a model of the right shape
and size for timing work, but its accuracy means nothing and is reported as such.

Memory: 1,000 ToyCar clips give about 300,000 vectors (790 MB as float32). The machine
this was written on often has less than 3 GB free, so feature extraction stops with an
error once --max-gb is exceeded; use --vectors-per-file to subsample (a stated deviation
from the reference, which trains on all vectors).

Usage:
    python models/train_ad.py --synthetic 20000 --epochs 5 --out out/ad
    python models/train_ad.py --data D:/data/dcase2020/dev_data/ToyCar --out out/ad
"""
import argparse
import glob
import json
import os
import sys
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly

SR = 16_000
N_FFT = 1024
HOP = 512
N_MELS = 128
FRAMES = 5
DIM = N_MELS * FRAMES


# ------------------------------------------------------------------ features (no TF needed)

def _hz_to_mel(f):
    f = np.asarray(f, float)
    f_sp, min_hz, min_mel, step = 200.0 / 3, 1000.0, 15.0, np.log(6.4) / 27.0
    return np.where(f >= min_hz, min_mel + np.log(np.maximum(f, 1e-10) / min_hz) / step, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, float)
    f_sp, min_hz, min_mel, step = 200.0 / 3, 1000.0, 15.0, np.log(6.4) / 27.0
    return np.where(m >= min_mel, min_hz * np.exp(step * (m - min_mel)), f_sp * m)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Slaney-scale, area-normalised triangular filters (librosa.filters.mel defaults)."""
    fmax = sr / 2 if fmax is None else fmax
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    hz = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(hz)
    ramps = hz[:, None] - fft_f[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0, np.minimum(lower, upper))
    return w * (2.0 / (hz[2:] - hz[:-2]))[:, None]


_FB = None


def log_mel(y, sr=SR):
    """(frames, 128) log-mel power spectrogram of a mono signal."""
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    if sr != SR:
        y = resample_poly(y, SR, sr)
    pad = N_FFT // 2
    y = np.pad(np.asarray(y, float), pad, mode="reflect")
    n_frames = 1 + (y.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(y[idx] * get_window("hann", N_FFT), axis=1)) ** 2
    mel = spec @ _FB.T
    return 10.0 * np.log10(mel + np.finfo(float).eps)


def stack_frames(lm, frames=FRAMES):
    """Concatenate `frames` consecutive log-mel frames into 640-dim vectors."""
    n = lm.shape[0] - frames + 1
    if n < 1:
        return np.empty((0, lm.shape[1] * frames), np.float32)
    return np.concatenate([lm[t: t + n] for t in range(frames)], axis=1).astype(np.float32)


def wav_vectors(path):
    sr, y = wavfile.read(path)
    if y.ndim > 1:
        y = y[:, 0]  # DCASE files are mono; keep channel 0 otherwise
    if np.issubdtype(y.dtype, np.integer):
        y = y / float(np.iinfo(y.dtype).max)
    return stack_frames(log_mel(y.astype(float), sr))


def load_dir(files, vectors_per_file=None, max_gb=1.5, seed=0):
    rng = np.random.default_rng(seed)
    out, total = [], 0
    for f in files:
        v = wav_vectors(f)
        if vectors_per_file and v.shape[0] > vectors_per_file:
            v = v[np.sort(rng.choice(v.shape[0], vectors_per_file, replace=False))]
        total += v.nbytes
        if total > max_gb * 2**30:
            raise MemoryError(f"features exceed --max-gb {max_gb}; use --vectors-per-file "
                              "or --max-files")
        out.append(v)
    return np.concatenate(out) if out else np.empty((0, DIM), np.float32)


# ------------------------------------------------------------------ model (TF imported lazily)

def build_model(dim=DIM, width=128, bottleneck=8):
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input(shape=(dim,))
    h = inp
    for units in [width] * 4 + [bottleneck] + [width] * 4:
        h = L.Dense(units)(h)
        h = L.BatchNormalization()(h)
        h = L.Activation("relu")(h)
    out = L.Dense(dim)(h)
    return tf.keras.Model(inp, out, name="ad_autoencoder")


def quantise_int8(model, rep_data, n_rep=500):
    """Full-integer post-training quantisation (int8 weights, activations, input, output).
    BatchNorm folds into the preceding Dense during conversion."""
    import tensorflow as tf
    rep = rep_data[np.random.default_rng(0).permutation(len(rep_data))[:n_rep]]

    def gen():
        for r in rep:
            yield [r[None, :].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = gen
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def int8_reconstruction_mse(tflite_bytes, x):
    """Per-vector MSE of the int8 model (input quantised, output dequantised)."""
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=tflite_bytes)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i["quantization"]
    so, zo = o["quantization"]
    err = np.empty(len(x))
    for k, v in enumerate(x):
        q = np.clip(np.round(v / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q[None, :])
        it.invoke()
        y = (it.get_tensor(o["index"])[0].astype(np.float32) - zo) * so
        err[k] = np.mean((y - v) ** 2)
    return err


def write_c_array(tflite_bytes, out_dir, name="g_ad_model"):
    """model_data.h / model_data.c, 16-byte aligned as TFLite Micro requires."""
    h = os.path.join(out_dir, "model_data.h")
    c = os.path.join(out_dir, "model_data.c")
    with open(h, "w", newline="\n") as f:
        f.write("/* Generated by models/train_ad.py: int8 MLPerf Tiny AD autoencoder. */\n"
                "#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n#include <stdint.h>\n"
                f"extern const uint8_t {name}[];\nextern const uint32_t {name}_len;\n#endif\n")
    with open(c, "w", newline="\n") as f:
        f.write('#include "model_data.h"\n')
        f.write(f"__attribute__((aligned(16))) const uint8_t {name}[] = {{\n")
        for k in range(0, len(tflite_bytes), 12):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in tflite_bytes[k:k + 12]) + ",\n")
        f.write(f"}};\nconst uint32_t {name}_len = {len(tflite_bytes)}u;\n")
    return h, c


def auc(scores, labels):
    """ROC AUC via the Mann-Whitney statistic (ties count one half)."""
    from scipy.stats import rankdata
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="machine-type dir with train/*.wav (and test/*.wav)")
    src.add_argument("--synthetic", type=int, metavar="N", help="N random feature vectors")
    ap.add_argument("--out", default="out/ad")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--vectors-per-file", type=int)
    ap.add_argument("--max-gb", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    test = None
    if a.synthetic:
        rng = np.random.default_rng(a.seed)
        # roughly log-mel-like range (dB); meaningless for accuracy
        x = (-40 + 10 * rng.standard_normal((a.synthetic, DIM))).astype(np.float32)
        origin = "synthetic random features (no accuracy claim)"
    else:
        files = sorted(glob.glob(os.path.join(a.data, "train", "*.wav")))[: a.max_files]
        if not files:
            sys.exit(f"no wav files under {a.data}/train")
        x = load_dir(files, a.vectors_per_file, a.max_gb, a.seed)
        tfiles = sorted(glob.glob(os.path.join(a.data, "test", "*.wav")))[: a.max_files]
        test = [(f, "anomaly" in os.path.basename(f)) for f in tfiles]
        origin = f"wav files from {a.data} ({len(files)} train clips)"
    print(f"{x.shape[0]:,} training vectors of {x.shape[1]} ({origin})")

    model = build_model(x.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr), loss="mse")
    hist = model.fit(x, x, epochs=a.epochs, batch_size=a.batch, validation_split=0.1,
                     shuffle=True, verbose=2)
    model.save(os.path.join(a.out, "ad_float.keras"))

    tfl = quantise_int8(model, x)
    tfl_path = os.path.join(a.out, "ad_int8.tflite")
    with open(tfl_path, "wb") as f:
        f.write(tfl)
    write_c_array(tfl, a.out)

    # float vs int8 agreement on up to 512 training vectors
    sub = x[:512]
    f_err = np.mean((model.predict(sub, verbose=0) - sub) ** 2, axis=1)
    q_err = int8_reconstruction_mse(tfl, sub)
    it = tf.lite.Interpreter(model_content=tfl)
    inq, outq = it.get_input_details()[0], it.get_output_details()[0]
    meta = {
        "data": origin,
        "params": int(model.count_params()),
        "tflite_bytes": len(tfl),
        "input": {"dtype": str(np.dtype(inq["dtype"])), "shape": inq["shape"].tolist(),
                  "scale": float(inq["quantization"][0]), "zero_point": int(inq["quantization"][1])},
        "output": {"dtype": str(np.dtype(outq["dtype"])),
                   "scale": float(outq["quantization"][0]), "zero_point": int(outq["quantization"][1])},
        "epochs": a.epochs, "final_loss": float(hist.history["loss"][-1]),
        "final_val_loss": float(hist.history["val_loss"][-1]),
        "float_vs_int8_mse": {"float_mean": float(f_err.mean()), "int8_mean": float(q_err.mean()),
                              "pearson_r": float(np.corrcoef(f_err, q_err)[0, 1])
                              if f_err.std() > 0 and q_err.std() > 0 else None},
        "train_seconds": round(time.time() - t0, 1),
        "tensorflow": tf.__version__,
    }
    if test:
        fs, qs, ys = [], [], []
        for f, y in test:
            v = wav_vectors(f)
            fs.append(float(np.mean((model.predict(v, verbose=0) - v) ** 2)))
            qs.append(float(np.mean(int8_reconstruction_mse(tfl, v))))
            ys.append(y)
        if 0 < sum(ys) < len(ys):
            meta["test_auc"] = {"float": auc(fs, ys), "int8": auc(qs, ys), "clips": len(ys)}
    with open(os.path.join(a.out, "ad_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    main()
