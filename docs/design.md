# Design: do MBPTA / EVT assumptions hold for int8 anomaly-detection inference on RP2040 vs STM32?

Status: written before any hardware measurement exists. No hardware has been purchased. Every number in this repo that
comes from a device must be labelled by its origin: `hardware`, `renode` (functional simulation, not cycle-accurate) or
`synthetic` (host-generated test trace). Nothing in this repository is a hardware result as of this commit.

## 1. Research question and hypotheses

**Question.** Measurement-based probabilistic timing analysis (MBPTA) fits an extreme-value distribution to measured
execution times and reads off a probabilistic WCET (pWCET) at a tiny exceedance probability. It is only valid if the
measurements are (a) independent and identically distributed (i.i.d.) or at least stationary and weakly dependent, and
(b) the tail belongs to a domain where a GEV/GPD fit is meaningful. Do those assumptions hold for int8 TinyML inference on a
cacheless Cortex-M0+ (RP2040) and on a cached Cortex-M4 (STM32F446), with interference as the independent variable?

**Hypotheses (stated before data; decision rules are in PROTOCOL.md).**
- H1. With no interference, the RP2040 inference time is almost constant (a few distinct values). The pipeline reports a
  degenerate trace and EVT is not applicable; pWCET is the observed maximum plus a margin. (Expected trivial result.)
- H2. On the STM32 baseline, small run-to-run variation from flash wait states, ART cache state and bus arbitration gives a
  light-tailed distribution that passes i.i.d. tests only when cache state is controlled.
- H3. Periodic interrupts and DMA produce multimodal, serially dependent traces that fail the runs test and Ljung-Box, so
  a naive pWCET at 1e-9 is invalid; declustering or per-scenario modelling is required.
- H4. Under an RTOS with a tick and a competing task, execution time becomes non-stationary (KPSS) when scheduling phase drifts.
- H5. Tails bounded by construction (a finite worst case from a bounded number of interrupts) give GPD shape estimates
  negative or near zero; a pWCET far above the observed maximum is then pessimistic.

Negative or trivial outcomes are acceptable results and will be reported as such.

## 2. Toolchain

| Decision | Chosen | Alternatives considered | Why not the alternatives |
|---|---|---|---|
| Inference runtime | TensorFlow Lite Micro with CMSIS-NN kernels, as in the MLPerf Tiny reference | Hand-written C only; microTVM; ExecuTorch+CMSIS-NN; Edge Impulse | Hand-written C alone does not reproduce the reference. microTVM and ExecuTorch add toolchains and are not the benchmark reference. Edge Impulse is a hosted, account-bound service. |
| Kernel library | CMSIS-NN on both boards (DSP SIMD on M4; plain C path on M0+) | TFLM reference kernels on the Pico | Same library keeps source constant; the ISA path differs, an unavoidable documented confound. |
| Fallback kernel | Small hand-written int8 fully-connected reference kernel (`firmware/common/ref_infer.c`) generated from the same `.tflite` | - | Lets the skeletons compile and run (including in Renode) without fetching the TFLM tree (hundreds of MB). It is the "hand-written C kernel" of the proposal. It is not the headline measurement path. |
| RTOS | **FreeRTOS** (kernel only) | Zephyr; bare-metal only; RTX | Zephyr supports both boards with one codebase, which is attractive, but its west workspace is several GB, its devicetree/driver layers add timing paths that are hard to attribute, and the laptop often has under 3 GB free. FreeRTOS is about 10 source files, identical on both ports, and the tick source is easy to control. Bare-metal is retained as scenario S0 (the control). |
| Build | CMake; Pico SDK for RP2040; CMSIS device headers plus minimal startup and linker script for STM32F446 | STM32CubeIDE; PlatformIO; Zephyr west | CubeIDE and PlatformIO are GUI or large installs; CMake runs in WSL. |
| Host stack | Python 3.10+, numpy, scipy, statsmodels; TensorFlow for training only | R `evd`/`ismev`; MATLAB | Already installed. R is a reasonable later cross-check. |
| Compiler | Arm GNU Toolchain 14.2.rel1 in WSL under `~/tools`, outside the repo | apt `gcc-arm-none-eabi` | apt needs sudo, unavailable non-interactively. |

Core usage on RP2040: all measurement code runs on core 0; core 1 is parked except in scenario S5, where it is the interference source.

## 3. Model and dataset

**Model.** The MLPerf Tiny anomaly-detection reference: a fully-connected autoencoder, input 640 (5 stacked frames of 128 log-mel
bins), 4x Dense(128)+BatchNorm+ReLU, bottleneck Dense(8)+BN+ReLU, 4x Dense(128)+BN+ReLU, output Dense(640). About 270k parameters,
about 270 KB as int8, which fits flash on both boards and SRAM for activations. Anomaly score is the mean squared reconstruction error.
Architecture, features (including the reference's crop to frames 50-249 of each clip) and int8 calibration were checked
against the `mlcommons/tiny` code on 2026-11-06 (scoop_check.md section 5).

**Dataset: ToyADMOS (DCASE 2020 Task 2, toy car) vs CWRU bearing vibration.**

| Option | For | Against |
|---|---|---|
| ToyADMOS toy car (MLPerf Tiny reference) | Exact reproduction target, published reference AUC, mel-spectrogram input | Multi-GB download; microphone audio, not vibration; availability to be checked |
| CWRU bearing data | Vibration, matches the project proposal; small files; simple windowing | Not the reference; no published MLPerf baseline; known leakage pitfalls across load conditions |
| Synthetic spectral frames | No download; fine for timing and CI | Not a valid accuracy claim |

**Decision.** Architecture follows MLPerf Tiny, so the reproduction concerns the model and the latency/energy method. Timing for dense int8
layers does not depend on data values (no data-dependent branches except saturation), so the dataset choice affects the accuracy claim, not
the timing claim. That is itself a checked assumption: the harness cycles through several input classes and tests input-dependence of latency.
The training script `models/train_ad.py` takes `--synthetic N` (random feature vectors; tests and timing harness) or `--data DIR`
(a DCASE 2020 Task 2 machine-type folder with `train/*.wav` and optional `test/*.wav`, downloaded by the user; log-mel
features computed with numpy/scipy). A CWRU path is not implemented. **No accuracy reproduction is claimed unless a real dataset is used and the AUC is reported.**

## 4. Measurement method

Primary timer per board, cross-checked by a GPIO toggle seen on a logic analyser.

| Board | Primary | Resolution | Cross-check | Notes |
|---|---|---|---|---|
| STM32F446 (M4) | DWT->CYCCNT, 32-bit, core clock 180 MHz | 1 cycle (5.6 ns) | PA5 toggle on logic analyser at 24 MHz (about 42 ns) | Wraps after 23.8 s, irrelevant per inference. Enable via DEMCR.TRCENA. |
| RP2040 (M0+) | No DWT exists on Cortex-M0+. Use the **SysTick** down-counter (24-bit, clk_sys 125 MHz), reading `SYST_CVR` | 1 cycle (8 ns) | GP15 toggle on logic analyser | Wraps at 134 ms; inference is a few ms; wraps detected and flagged. Under FreeRTOS the RP2040 port uses SysTick for the tick, so there the tick is moved to a TIMER alarm interrupt or the TIMER at 1 MHz is the coarse fallback (risk R5). |

Rejected: RP2040 64-bit TIMER at 1 MHz as primary (1 microsecond is 125 cycles; the discrete, tie-heavy trace damages tail fitting); PIO or PWM-slice
cycle counters (extra moving parts); pin toggling alone (capture length and USB limits make 100k+ runs awkward; used only as a sampled cross-check).

Timed region: the model invoke call only; pre- and post-processing are timed separately (phase split as in Bartoli et al.). Counter-read overhead is measured by timing an empty region 100k times and reported, not silently subtracted.

Serial format, one line per inference: `T,<seq>,<cycles>,<input_id>,<flags>`; `flags` is a hex bitmask (bit0 timer wrap, bit1 interference ISR fired in the run, bit2 DMA active, bit3 RTOS preemption). Results are buffered in RAM and flushed between batches so UART traffic does not perturb the timed region. Header lines start with `#` and carry board, scenario, clock and build id.

**Energy.** INA219 average power over a long back-to-back loop divided by inference rate (see bom.md for resolution limits), idle baseline subtracted. INA219 at about 2 kHz cannot resolve single inferences, so energy is a per-inference average, not a distribution. Stated limitation.

**Sample size.** At least 100,000 inferences per scenario per board, so the 1e-4 tail has about 10 exceedances for held-out checks. Probabilities of 1e-6 to 1e-9 cannot be validated empirically at any feasible sample size; they are extrapolations and are reported as such.

## 5. Interference scenarios (independent variable)

| ID | Scenario | Mechanism |
|---|---|---|
| S0 | Baseline, bare-metal, interrupts off; variants S0-flash and S0-ram (code and model in SRAM) | Control |
| S1 | Periodic timer interrupt at 1 kHz, ISR of about 2 microseconds | Phase uncorrelated with inference start |
| S2 | Periodic interrupt with a period harmonic to the inference period | Phase locking; expected to violate independence |
| S3 | Background DMA: memory-to-memory SRAM copy plus peripheral DMA to UART | Bus-matrix contention on data fetches |
| S4 | FreeRTOS, 1 kHz tick, inference task plus a lower- and a higher-priority periodic task | Scheduler and tick interference |
| S5 | RP2040: core 1 runs a memory-bandwidth loop on the same SRAM bank. STM32 analogue: DMA2 mem-to-mem stress | Shared fabric contention |
| S6 | STM32 only: invalidate ART caches before each run vs warm | Cache state as source of bimodality |
| S7 | Randomised inter-inference gap (uniform 0-1 ms) | MBPTA-style phase randomisation; tests whether it restores i.i.d. |

Scenario order within a session is randomised; each session is repeated in at least 3 separate power cycles. A 60 s warm-up precedes recording.

## 6. Statistical pipeline

Implemented in `src/pwcet/` and tested on synthetic traces with known answers.

1. **Pre-checks.** Degeneracy: fewer than 20 distinct values or coefficient of variation below 1e-4 flags the trace `degenerate`. EVT is then not applicable; the pipeline reports the observed maximum and the distinct-value count instead of a fitted pWCET.
2. **Independence.** Wald-Wolfowitz runs test about the median; Ljung-Box (statsmodels) at lag 20; reject at alpha 0.05.
3. **Identical distribution and stationarity.** KPSS (level, `nlags="auto"`) plus a two-sample Kolmogorov-Smirnov test between first and second halves (the standard MBPTA identical-distribution check).
4. **EVT, two estimators.** Block maxima with GEV (maximum likelihood, scipy `genextreme`), block size giving at least 50 blocks, with conversion between per-block and per-run exceedance. Peaks over threshold with GPD above a threshold at a high quantile (default 95th; 97.5th and 99th as sensitivity), pWCET `x_u + GPD.ppf(1 - p/zeta_u)` with `zeta_u` the exceedance rate.
5. **Goodness of fit.** Anderson-Darling statistic with a parametric-bootstrap p-value (standard tables are invalid when parameters are estimated) for GEV and GPD. The MBPTA-CV exponential-tail test (CV of excesses near 1) is a second opinion.
6. **pWCET.** At per-run exceedance 1e-6, 1e-7, 1e-8, 1e-9, and at 1e-3 and 1e-4 where checkable.
7. **Held-out validation.** Fit on the earlier part of a session, count exceedances in the later part (preferably another power cycle); binomial test of observed against expected count; report pessimism ratio (pWCET / observed max).
8. **Verdict labels:** `EVT-valid`, `EVT-invalid-dependent`, `EVT-invalid-nonstationary`, `EVT-invalid-poor-fit`, `degenerate`. Rules are in PROTOCOL.md.

The frozen values in `protocol/PROTOCOL.md` supersede the numbers in this section. Main differences: each gate test is run
at 0.05/4 (Bonferroni over the four tests), the GEV uses 100 blocks, the GPD threshold is the lowest of the 0.90 / 0.95 /
0.975 / 0.99 quantiles whose fit passes, and held-out validation uses an exact interval with a tolerance factor of 2.

Not implemented in v0.1 (stated, not hidden): declustering with an extremal index, time-varying GPD, Bayesian EVT, bootstrap confidence intervals on pWCET, and a conservative analytic WCET comparison.

## 7. Decisions and rejected alternatives (summary)

| Decision | Rejected | Reason |
|---|---|---|
| MLE for EVT fits | Probability-weighted moments | MLE is standard with enough exceedances; PWM is a possible robustness check |
| AD bootstrap goodness of fit | KS with estimated parameters | KS p-values invalid with estimated parameters |
| Both GEV and GPD | One only | Disagreement is informative |
| 100k+ inferences per scenario | 1,000 (as in Bartoli et al.) | 1,000 samples cannot support a tail claim beyond about 1e-2 |
| Reference-kernel fallback in firmware | Waiting for TFLM | Keeps the repo buildable and testable offline |
| Renode as functional check only | Treating Renode times as results | Not cycle-accurate (section 9) |

## 8. Risks

- R1. Cacheless cores may be trivially well behaved. The RP2040 baseline is probably near-constant; the degenerate case is then the honest finding. Interference scenarios mitigate.
- R2. RP2040 is not fully cacheless: code runs from QSPI flash through a 16 KB XIP cache. Placing code and model in SRAM is an explicit factor (S0-flash vs S0-ram); "cacheless" applies only to the SRAM-resident build.
- R3. The F446 "cached" side is weak (small ART accelerator). A clean cache contrast needs a Cortex-M7 board (upgrade path in bom.md).
- R4. Autocorrelation is the norm under periodic interference. The tests may reject almost every interesting trace; that is a result, not a bug.
- R5. RP2040 timer under FreeRTOS: SysTick contention with the RTOS port is unvalidated; fallback is TIMER at 1 MHz plus the logic analyser.
- R6. Serial output perturbs timing; mitigated by RAM buffering.
- R7. INA219 resolution; Pico energy includes regulator losses.
- R8. Scoop risk from a shallow search (scoop_check.md).
- R9. No hardware yet: only the pipeline on synthetic traces is validated.
- R10. Input-dependent latency would add a mixture component; the harness records `input_id` and checks.
- R11. A negative-shape GPD (or GEV) can fit the observed tail well and still be optimistic far below 1e-4 when the
  true tail is light but unbounded. On a synthetic interference trace with Gaussian miss jitter, the 1e-6 to 1e-9 GPD
  bounds were exceeded by held-out runs (README). Held-out data can expose this only down to about 1e-6.

## 9. Renode fallback and its fidelity

Renode (open source, Antmicro) models the instruction set functionally and runs firmware on emulated peripherals. It is **not cycle-accurate**: virtual time advances at a configured instructions-per-second rate, and it does not model flash wait states, the ART accelerator, bus-matrix arbitration, DMA contention or the RP2040 XIP cache. It can show that firmware boots, runs the inference loop, emits the serial format and that the host parser works. It cannot produce latency distributions, tail behaviour or interference effects, and nothing measured in Renode may be presented as timing evidence. Any "cycle" count seen in Renode is an artefact of its own timing model.

## 10. State at this commit

Done on the host: the statistical pipeline (gates, GPD, GEV, MBPTA-CV and MEMIK/ATAN baselines, held-out validation, analysis CLI) with tests
on synthetic traces, the pre-registered protocol, and the training/int8-export script (run only on random features). Not done:
firmware on any board (`firmware/` holds an untested skeleton: timing harness, interference generators and log format,
built only for the host with a fake board; no ARM toolchain installed; the workload is a placeholder FC stack, and
`ref_infer.c` above is not written), Renode runs, any hardware measurement, any energy measurement, any real-dataset accuracy.
