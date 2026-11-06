# tinyml-pwcet

Planned empirical study: do extreme-value-theory / MBPTA assumptions (i.i.d., stationarity, tail fit) hold for int8
anomaly-detection inference on RP2040 (cacheless Cortex-M0+) versus STM32F446 (cached Cortex-M4), with interrupt, DMA and
RTOS interference as the independent variable?

## Status

Early skeleton. **No hardware has been bought and no measurements exist.**

- `docs/scoop_check.md`: literature check (abstract/README/snippet level only; verdict partly open, unverified).
- `docs/bom.md`: bill of materials (about USD 38, unverified prices, nothing purchased).
- `docs/design.md`: toolchain, model, measurement and statistics plan with rejected alternatives and risks.
- `protocol/PROTOCOL.md`: pre-registered analysis protocol (gates and thresholds, sample sizes, target probabilities,
  held-out validation rule, baselines, handling of crashed runs), frozen 2026-11-06 before any hardware data.
- `src/pwcet/`: host-side stats, tested on synthetic traces only:
  - `synthetic`: traces with known tails, invalid inputs (AR(1), regime shift, periodic spikes, mode switching), and
    an `interference` timeline model (periodic ISR preemption, bimodal cache hit/miss latency, optional co-running RTOS
    task, random or fixed gap between runs) with made-up, loosely RP2040-like parameters;
  - `iid`, `stationarity`: runs test, Ljung-Box, KPSS, first-vs-second-half Kolmogorov-Smirnov;
  - `gates`: the protocol's applicability gates (degeneracy, then independence, then stationarity, each test at
    0.05/4) with the verdict labels;
  - `evt`: GPD peaks over threshold, threshold choice, bootstrap Anderson-Darling goodness of fit, pWCET;
  - `validate`: GEV block maxima, an MBPTA-CV-style exponential baseline, max observed + 20 %, and held-out
    validation (exact Clopper-Pearson interval, tolerance factor 2, verdicts pass / pass-pessimistic / fail / untestable);
  - `analysis`, `cli`: the end-to-end pipeline and the `pwcet-analyse` / `pwcet-synth` commands.
- `models/train_ad.py`: trains the MLPerf Tiny anomaly-detection autoencoder (640-128x4-8-128x4-640, Keras), applies
  full-integer int8 post-training quantisation and exports `ad_int8.tflite` plus `model_data.c/.h` for TFLite Micro
  with CMSIS-NN. Run so far **only on random features**.
- `firmware/`: empty.

Not done: firmware, Renode, any hardware measurement.

## Analyse a trace

    pwcet-synth interference --n 300000 --seed 1 --out intf.csv      # synthetic, serial format
    pwcet-analyse intf.csv --origin synthetic                         # writes intf.csv.report.json
    pwcet-analyse s1.csv --holdout s2.csv s3.csv --origin hardware    # protocol layout (later)

Input: the firmware serial format `T,<seq>,<cycles>,<input_id>,<flags>` (timer-wrap rows dropped and counted), a CSV
with a `cycles` column, or one number per line. Without `--holdout` the first third is fitted and the rest held out.
The gates run on the fit set; on any gate failure no EVT or MBPTA bound is computed. Defaults are the protocol's
(B = 1000 bootstrap replicates for GPD, 500 for GEV): about 4 minutes for 300,000 runs on the development laptop.
Hardware traces that fail the measurement-quality rules exit with code 3 and are not analysed.

## Results so far: synthetic traces only

All numbers below come from host-generated traces (`pwcet-synth`, seeds 1 and 2, 300,000 runs: 100,000 fitted,
200,000 held out). They test the pipeline, not any hardware.

| Trace | Gates | Outcome |
|---|---|---|
| interference, random gap (i.i.d. by construction) | pass (runs p=0.70, Ljung-Box p=0.34, KPSS p≥0.10, KS p=0.73) | GPD (q=0.975, shape -0.23, AD p=0.95): 1e-3 bound exceeded 179 times vs 200 expected, 1e-4: 33 vs 20, both `pass`. Bounds at 1e-6 to 1e-9 are refuted (1e-6: 3 held-out runs above, 0.2 expected); the 1e-9 bound, 207,994, is below the held-out maximum, 207,997. GEV (shape -0.26, AD p=0.026, so its fit would be rejected) agrees at 1e-3/1e-4 and is refuted from 1e-5. MBPTA-CV-style (CV 0.34): `pass-pessimistic` (0 exceedances), 1e-9 bound 9.5 % above the held-out max. Max + 20 %: 249,338, never exceeded. |
| interference, fixed gap (ISR phase drifts) | refused: `EVT-invalid-dependent` (runs p≈0) | no bound |
| interference + co-running RTOS task | refused: dependent (runs, Ljung-Box) | no bound |
| interference, sticky cache misses | refused: dependent (runs, Ljung-Box) | no bound |
| two-mode switching | refused: dependent (runs, Ljung-Box) | no bound |
| mean shift at run 150,000 (inside the held-out part) | pass (the fit window is stationary) | every method `fail`s held out: GPD 1e-3 bound exceeded by 145,913 of 200,000 runs |

Reading: the negative-shape GPD fits the observed body of a bounded-looking tail well, but the miss latency here has
Gaussian jitter, so the true tail is unbounded and the extrapolations are optimistic. This is the H5 risk in
`docs/design.md`, and the held-out check catches it only down to about 1e-6. The gates do not see changes after the
fit window; the held-out sessions are what catches those.

## Model

    python models/train_ad.py --synthetic 20000 --epochs 5 --out out/ad          # pipeline check, no accuracy meaning
    python models/train_ad.py --data <DCASE2020>/dev_data/ToyCar --out out/ad     # real data (not downloaded here)

TensorFlow 2.21 (CPU) was already installed, so Keras and the TFLite converter are used directly. Nothing is downloaded.
With `--data`, `test/*.wav` clips give a clip-level ROC AUC for the float and the int8 model. Measured on random
features (20,000 vectors, 5 epochs, 34 s on the laptop CPU): 269,992 parameters, a 318,616-byte int8 flatbuffer whose
graph has only `FULLY_CONNECTED` ops with int8 input and output, and int8-vs-float per-vector reconstruction error
correlation 0.998. No accuracy or AUC has been measured; that needs the DCASE 2020 data.

## Test

    pip install -e .[dev]
    python -m pytest -q      # about 2-3 minutes (bootstrap goodness-of-fit tests)
