# Scoop check

First version 2026-11-06 (morning, abstract level only); rewritten 2026-11-06 (evening) after a second, deeper pass.
Each source is tagged with the level at which it was actually read:

- **full text**: the paper body was read in this repository's sessions (PDF or arXiv HTML);
- **full text (summary)**: full text read and summarised in an earlier session today; the summary is used here, the paper
  was not re-opened;
- **slides**: the conference talk slides, not the paper;
- **abstract**: abstract or catalogue page only;
- **snippet**: search-result text only; content not verified;
- **code**: source files of a repository.

Searches used a general web search plus a few direct fetches. dblp's API returned HTTP 429 and its HTML pages were
behind a bot check, and Google Scholar was not queried, so this is still **not** a systematic search of RTSS, RTAS,
ECRTS, EMSOFT or the WCET workshop. If no hit was found, that is weak evidence that none exists.

## 1. Closest work: EVT or tail analysis of ML inference latency

| Source | Read level | What it does | Overlap with this study |
|---|---|---|---|
| arXiv 2606.16106, "Beyond CPU–GPU Frequency: Memory-Clock and Tail Effects in Edge Inference Latency Estimation" | full text (summary) | Jetson Orin Nano (Cortex-A78AE + GPU) running MobileNetV2, ViT-S and Qwen2.5-1.5B. Fits a GPD above p99 but calls it "a marginal quantile heuristic", because the i.i.d. conditions do not hold. Runs no formal i.i.d. tests; burst continuation probability 0.74, against 0.001 under independence. A streaming-write adversary runs on 0/2/4 cores. Temporal holdout: the first 50k runs predict the last 50k, with p99.9 within ±2.7 % and p99.99 within ±5.7 %. Gaussian margins overshoot by 13-29x. Cites only general MBPTA works (Edgar and Burns 2001; Cucu-Grosjean et al. 2012; Davis and Cucu-Grosjean 2019). | **High on method.** Holdout validation of tail quantiles and an interference sweep are already published, so neither is a contribution here. Differences: an application-class SoC with a GPU, not an MCU; no gate that refuses to fit; holdout within one session at p99.9/p99.99, not across power cycles at fixed target probabilities. |
| Tizpaz-Niari and Sankaranarayanan, arXiv 2404.07170, "Worst-Case Convergence Time of ML Algorithms via Extreme Value Theory" (CAIN 2024) | full text (arXiv HTML v1, read today) | EVT (GPD peaks over threshold; threshold the mean + 2 sd, lowered towards the mean + 1 sd; R `extRemes`) for the convergence time of scikit-learn training and the settling time of 3 DNN controllers (ball-and-beam, inverted pendulum, TORA), plus a SnapBuddy web-server example. Runs on an Intel Haswell server. No formal i.i.d. or stationarity tests (inputs are randomised by fuzzing instead). Validation: a prediction counts as accurate if the later observed maximum lies in its 95 % CI; 75 % accurate for inference (9 of 12), 57 % for training. The introduction claims 44 algorithms and 33 DNNs, but the body reports 4 and 3. | **Low.** Same tool (EVT for ML timing), but desktop-class hardware and the "time" is convergence or settling time, not per-inference execution cycles. No MCU, no interference variable, no cache contrast. The earlier worry that a Cortex-M experiment was hidden in it is now excluded. |
| Kumar and Jain, pWCET for GoogleNet/CaffeNet from measured inference times | snippet | pWCET of large CNNs from measurements. | Large networks on non-MCU hardware (from the snippet). Full text still not read. |

## 2. pWCET and MBPTA methods (baselines and statistical practice)

| Source | Read level | Content | Use here |
|---|---|---|---|
| Abella, Padilla, del Castillo, Cazorla, "Measurement-Based WCET Estimation Using the Coefficient of Variation" (MBPTA-CV), ACM TODAES 22(4), 2017 | full text (UPC postprint) | CV-plot tail classification. Discard the lower half of the sample; candidate tails have N+th from 50 to N/2 exceedances; the exponentiality test (95 %, band from del Castillo et al. 2014) must pass for **every** tail of 10 to N+th values; among those, choose the N+th whose CV is **closest to 1**; light tails (CV below the band) are accepted as conservative; exponential fit with rate 1 / mean excess, shifted by th. Ljung-Box and two-sample KS are its i.i.d. tests. | Baseline. Our first re-implementation did not follow these rules (40-point grid, at most 10 % of the sample, largest passing tail first, no check of smaller tails). It now does (PROTOCOL.md deviation D1). The paper does not print the band formula; we use the asymptotic 1 ± 1.96 / sqrt(k). |
| Vilardell, Serra, Mezzetti, Abella, Cazorla, del Castillo, "Using Markov's Inequality with Power-Of-k Function for Probabilistic WCET Estimation" (MEMIK, RESTK), ECRTS 2022, LIPIcs 231 | full text (sections 4-6) | P(X ≥ b) ≤ E[X^k] / b^k, minimised over k in [1, maxk]. RESTK caps maxk via bootstrap: for three probabilities it finds the largest k that does not underestimate the empirical quantile, takes the minimum over simulations, and extrapolates linearly to smaller p. It gives no estimate if the correlation is below 0.95. Uses k in [1, 150], nboot = 2000. | Basis of the inequality baseline. |
| Toba, Yano, Azumi, arXiv 2511.11682, "Generalized Inequality-based Approach for Probabilistic WCET Estimation" (Nov 2025) | full text (arXiv HTML v1, two passes) | Replaces x^k with saturating functions: P(X ≥ b) ≤ E[f(X)] / f(b), with f = arctan(x/d)^k (ATAN) or tanh(x/d)^k (TANH), minimised over (d, k) by a coarse grid that is not specified. Applies the RESTK restriction. No finite-sample correction and no stated i.i.d. assumption. Synthetic distributions (n = 1e6) and Autoware callbacks (n = 1e4, p = 1e-5; tightness 1.09-1.37). Compared only with MEMIK, not with EVT or MBPTA-CV. Hardware not named. | Candidate third baseline (ATAN, with MEMIK alongside), to be added as a dated protocol deviation. The (d, k) grid will have to be ours, because the paper gives none. |
| Manau, Vilardell, Serra, Mezzetti, Abella, Cazorla, "Detecting Low-Density Mixtures in High-Quantile Tails for pWCET Estimation" (TailID), ECRTS 2025 | slides | Low-density mixtures in the tail pass whole-sample i.i.d. tests. Adds KPSS and an extremogram on the tail. On an NVIDIA Orin AGX, **2,751 of 2,790 workloads were rejected** by standard tests (KPSS/BDS/Hurst on the whole sample, KPSS and extremogram on the tail). | Shows that refusing to fit after i.i.d. tests is established MBPTA practice, and that such tests reject most traces on a GPU SoC. Our gating is therefore **not** novel in itself. TailID is a candidate extra gate (not adopted: the protocol is frozen). |
| Reghenzani et al., "Probabilistic-WCET reliability: Statistical testing of EVT hypotheses", Microprocessors and Microsystems 77, 2020 | snippet (cited in TailID slides) | Statistical testing of EVT hypotheses for pWCET. | Prior art for formal gating; full text to be read. |
| Lima, Dias, Barros, "Extreme Value Theory for Estimating Task Execution Time Bounds: A Careful Look", ECRTS 2016, pp. 200-211, doi 10.1109/ECRTS.2016.20 | snippet (title, venue and DOI confirmed by two independent listings) | Per a citing summary: a critical look at EVT for pWCET on time-predictable and time-randomised platforms. | Identified as the "careful look" paper. **Whether it uses a Cortex-M4 and a Cortex-A53 is not confirmed**: no source seen here names its platforms. The one hint (Cortex-M4 and A53) is unverified, so no Cortex-M EVT study is confirmed. |
| Lima and Bate, "Valid Application of EVT in Timing Analysis by Randomising Execution Time Measurements", RTAS 2017 | full text (partly read) | IESTA: randomises the *measurements* (re-sampling) rather than the hardware to make EVT applicable. Case studies: a binary-search benchmark from Lima et al. 2016 and Rolls-Royce engine-control tasks measured on a cycle-accurate simulator. | A software-only alternative to our S7 phase randomisation; no MCU hardware. |
| Santinelli et al., "On the Sustainability of EVT for WCET Estimation", WCET 2014 | snippet | Robustness of EVT assumptions and parameters. | Background for gating. |
| Fedotova, Krause, Siemens, MBPTA with EVT on ARM Cortex-A (IFIP AICT 499, 2017) | snippet | MBPTA on Cortex-A boards. | Cortex-A, not Cortex-M. |
| arXiv 2607.23227 (Cortex-A on Raspberry Pi; output equivalence) | abstract | About output equivalence, with no timing tails. | Mention only. |

## 3. TinyML benchmarks and MCU inference timing

| Source | Read level | Content | Overlap |
|---|---|---|---|
| mlcommons/tiny, `benchmark/training/anomaly_detection` | code (`baseline.yaml`, `keras_model.py`, `common.py`, `00_train.py`, `02_convert.py`, `requirements.txt`, `trained_models/README.md`) | Reference AD model and pipeline; see section 5. | Reference to reproduce. |
| MLPerf Tiny, arXiv 2106.07597 | abstract | Benchmark suite: accuracy, latency and energy per inference. | Reports mean latency, not tails. |
| Bartoli et al., arXiv 2505.15622 (IJCNN 2025) | abstract | Energy and latency on an STM32N6 (NPU), 1,000 runs per model. | Averages only. |
| Search: WCET or pWCET of TFLite Micro or CMSIS-NN kernels | snippets | Found MCU latency benchmarks that report the maximum observed time (e.g. a Chalmers TFLM/CMSIS-NN/uTensor comparison), an OTAWA workshop paper on Keras-generated FC networks on a single-core ARM target, FPGA DPU static WCET, decision-tree WCET with aiT and OTAWA (arXiv 2501.17428), and IRIT/ANR work on NN WCET on GPUs. | No measurement-based pWCET or EVT validity study of TFLM/CMSIS-NN inference on Cortex-M found. The OTAWA workshop paper (static analysis, FC networks, ARM) is the nearest MCU-side item and should be read. |
| Breier et al., arXiv 2602.16309 | abstract | EMFI sensitivity of number formats. | None for timing. |

## 4. Updated verdict

**Narrower than first stated; still open on the specific combination, unverified at venue level.**

What is *not* new, given the sources above:
- EVT/GPD on ML inference latency (2606.16106, 2404.07170).
- Holdout validation of tail quantiles and an interference sweep (2606.16106).
- Running i.i.d. or stationarity tests and refusing to fit when they fail: standard MBPTA practice (MBPTA-CV, Reghenzani et
  al. 2020, TailID 2025, whose GPU SoC rejected 98.6 % of workloads).

What still appears open (no counter-example found):
1. The **platform**: int8 TFLite Micro / CMSIS-NN inference on Cortex-M, contrasting a cacheless M0+ (RP2040, SRAM
   build) with a flash-cached M4 (STM32F446). No EVT, MBPTA or inequality-based pWCET study of TinyML on Cortex-M was found.
2. **Pre-registered gating combined with held-out testing at fixed target probabilities** across power-cycle sessions
   (1e-3 and 1e-4 tested, 1e-5 to 1e-9 labelled as extrapolation), with interrupt, DMA and RTOS interference as
   the controlled variable.
3. A **head-to-head of EVT (GPD, GEV), MBPTA-CV and inequality bounds (MEMIK, ATAN)** on the same MCU traces. Toba et al.
   compare only against MEMIK.

Honest framing for a write-up: "an MCU measurement study of when EVT-based pWCET is applicable to TinyML inference,
and how it compares with inequality bounds", not "the first to test EVT assumptions". The cacheless-core result may be
trivial (degenerate traces), as `design.md` R1 states.

## 5. Reference check: MLPerf Tiny anomaly-detection model (code read)

| Item | Reference (`mlcommons/tiny`) | `models/train_ad.py` before | Action |
|---|---|---|---|
| Architecture | `keras_model.py`: 4x [Dense(128) BN ReLU], [Dense(8) BN ReLU], 4x [Dense(128) BN ReLU], Dense(640) | same | none |
| Input | n_mels 128 x frames 5 = 640 | same | none |
| STFT / mel | librosa 0.6.0 `melspectrogram(n_fft=1024, hop_length=512, n_mels=128, power=2.0)`, native rate (`sr=None`) | same settings; Slaney mel, reflect padding as librosa 0.6 | none |
| Log | `20 / power * log10(mel + eps)` = 10 log10 for power 2 | same | none |
| Frame crop | **`log_mel_spectrogram[:, 50:250]`**: only frames 50-249 of each clip (MLPerf Tiny addition, not in DCASE) | missing: all frames used | **fixed** |
| Sample scaling | librosa: int16 / 32768 | int16 / 32767 | **fixed** (about 0.0003 dB) |
| Training | Adam (default lr 1e-3), MSE, 100 epochs, batch 512, shuffle, validation split 0.1 | same | none |
| BN folding | explicit fold before conversion | left to the TFLite converter (graph checked: FC only) | documented |
| Quantisation | int8 full-integer, `TFLITE_BUILTINS_INT8`, int8 input and output (`ad01_int8.tflite`); representative set **every 5th training vector** (`train_data[::5]`) | 500 random vectors | **fixed** (stride 5; optional `--rep-max` cap, recorded in `ad_meta.json` as a deviation) |
| Data | ToyCar only | any machine type folder | documented |

## 6. Remaining follow-ups

1. Read the full text of Lima, Dias and Barros 2016 (IEEE; not open access here) and confirm its platforms.
2. Read Reghenzani et al. 2020 and the TailID paper (full text, not slides).
3. Run a systematic dblp search (when the API is not rate-limited) of RTSS, RTAS, ECRTS, EMSOFT and WCET for "Cortex-M" with
   "pWCET", "MBPTA" or "EVT", and for "neural network" with "WCET".
4. Read the OTAWA workshop paper on Keras-generated FC networks, and Kumar and Jain.
