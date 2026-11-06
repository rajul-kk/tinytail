# Pre-registered analysis protocol

Written before any hardware measurement exists (no board has been bought). It fixes the questions, the measurement
plan, the sample sizes and every decision rule of the analysis, so that results cannot steer the analysis. Rationale and
rejected alternatives are in `docs/design.md`; where the two differ, this file governs. Changes after the freeze date are
allowed only as dated entries in section 14.

Tooling: the rules below are implemented in `src/pwcet/` (`gates.py`, `evt.py`, `validate.py`, `analysis.py`, added in
the commits that follow this file) and run with `pwcet-analyse`. Default parameters in the code must equal the values here; a mismatch is a bug, fixed in favour of this text.

## 1. Research questions

- **RQ1 (applicability).** For each board and interference scenario, does the measured inference-time trace pass the
  i.i.d. and stationarity gates that EVT-based pWCET estimation requires?
- **RQ2 (validity).** When the gates pass, are the GPD (primary) and GEV (secondary) pWCET estimates at the testable
  exceedance probabilities (1e-3, 1e-4) confirmed on held-out sessions?
- **RQ3 (comparison).** How do EVT estimates compare with the high-water mark plus 20 % and with an MBPTA-CV-style
  exponential bound (pessimism ratios, held-out exceedances)?
- **RQ4 (mitigation).** Does randomising the inter-inference gap (S7) turn gate-failing periodic-interrupt traces (S1)
  into gate-passing ones?

## 2. Hypotheses and predicted outcomes

Each hypothesis names the observable that decides it. Every outcome, including trivial and negative ones, is reported.

| ID | Hypothesis | Supported if |
|---|---|---|
| H1 | RP2040, S0-ram (no interference, code and model in SRAM) is near-constant | verdict `degenerate` in all 3 sessions |
| H2 | STM32F446, S0 with warm ART cache gives an EVT-usable trace | verdict `EVT-valid` and GPD passes held-out checks at 1e-3 and 1e-4 |
| H3 | Periodic interrupts make traces serially dependent; harmonic periods (S2) more than S1 | S2 verdict `EVT-invalid-dependent` on both boards; S1 runs-test or Ljung-Box p-value smaller than in S7 |
| H4 | A co-running FreeRTOS task with a 1 kHz tick (S4) makes timing non-stationary or dependent | S4 verdict is `EVT-invalid-dependent` or `EVT-invalid-nonstationary` |
| H5 | Tails are bounded by construction (finite preemptions, finite misses) | GPD shape estimate below 0 in every `EVT-valid` trace |
| H6 | Phase randomisation restores applicability | S7 passes the gates on a board where S1 fails |

## 3. Measurement setup

- **Platforms.** Raspberry Pi Pico (RP2040, Cortex-M0+, clk_sys 125 MHz) and NUCLEO-F446RE (Cortex-M4F, 180 MHz). Clocks
  fixed in firmware and printed in the trace header. USB-powered from the same laptop port, room temperature.
- **Timer.** STM32: `DWT->CYCCNT`. RP2040: SysTick current-value register (24 bit, wrap flagged). Counter-read overhead is
  measured with 100,000 empty regions per session and reported, never subtracted.
- **Workload.** The MLPerf Tiny anomaly-detection autoencoder (640-128x4-8-128x4-640), int8, exported by
  `models/train_ad.py`, run with TFLite Micro and CMSIS-NN (fallback: the hand-written int8 reference kernel, reported as
  such). The input cycles through 8 fixed vectors (`input_id` 0..7) to test input dependence.
- **Timed region.** The invoke call only. Results are buffered in RAM and flushed between batches of 1,000.
- **Software.** FreeRTOS kernel (scenario S4 only), otherwise bare metal. Compiler, flags and build id in the header.

**Interference scenarios** (definitions in `docs/design.md` section 5; all are run on both boards unless marked):

| ID | Interference |
|---|---|
| S0-flash / S0-ram | None. Code and model in flash (RP2040: through the 16 KB XIP cache; STM32: through ART) vs copied to SRAM |
| S1 | Periodic timer ISR, 1 kHz, about 2 us of work |
| S2 | Periodic ISR whose period is harmonic to the inference period (phase-locked) |
| S3 | Background DMA (memory-to-memory plus UART TX) |
| S4 | FreeRTOS: 1 kHz tick, inference task plus one higher- and one lower-priority periodic task |
| S5 | Shared-fabric contention: RP2040 core 1 memory loop; STM32 DMA2 memory-to-memory stress |
| S6 | Cache flushed before every run: STM32 ART invalidate; RP2040 XIP cache flush (code in flash) |
| S7 | As S1, plus a random idle gap before each run, uniform on [0, 2 ms] (two ISR periods) |

The RP2040 has a 16 KB XIP flash cache, so it is "cacheless" only in the S0-ram configuration; S0-flash and S6 are where
cache-miss latency is studied on that board. Scenario order within a measurement day is randomised by a script with a
recorded seed.

## 4. Sample sizes and data handling

- Per board and scenario: **3 sessions**, each in a separate power cycle, each **100,000** recorded inferences after a
  60 s warm-up whose data are discarded. Fixed N; no optional stopping and no early look.
- **Fit set: session 1. Held-out set: sessions 2 and 3** (200,000 runs). Expected held-out exceedances at the target:
  200 at 1e-3, 20 at 1e-4, 2 at 1e-5. Below 1e-4 nothing can be confirmed empirically (section 9).
- When only one file is available (synthetic traces), the first third is the fit set and the last two thirds are held out.
- Trace format `T,<seq>,<cycles>,<input_id>,<flags>`. Rows with the timer-wrap flag (bit 0) are excluded and counted.
  Interference flags (bits 1-3) are kept: interference is the variable under study.
- **No outlier removal of any kind.**
- A session is a **measurement failure** (section 11) if more than 0.1 % of rows carry the wrap flag, more than 0.1 % of
  sequence numbers are missing, or it has fewer than 100,000 valid rows.

## 5. Applicability gates (run on the fit set, in this order)

| Gate | Test | Rule |
|---|---|---|
| Degeneracy | distinct values and coefficient of variation | fewer than 20 distinct values **or** CV below 1e-4 → `degenerate` |
| Independence | Wald-Wolfowitz runs test about the median (ties with the median dropped), two-sided | reject if p < 0.0125 |
| Independence | Ljung-Box at lag 20 | reject if p < 0.0125 |
| Stationarity | KPSS, level, `nlags="auto"` (statsmodels; p-value clipped to [0.01, 0.10]) | reject if p < 0.0125, i.e. statistic beyond the 1 % critical value |
| Identical distribution | two-sample Kolmogorov-Smirnov, first half vs second half | reject if p < 0.0125 |

The level 0.0125 is 0.05 / 4 (Bonferroni), so a truly i.i.d. trace is refused with probability at most 0.05.

Any independence rejection gives `EVT-invalid-dependent`; otherwise any stationarity or KS rejection gives
`EVT-invalid-nonstationary`. All four tests are always run and reported.

**On any gate failure the analysis stops.** No GPD, GEV or MBPTA-CV bound is computed or reported for that trace, not
even "for comparison" or "exploratory". The report contains the verdict, every test statistic and p-value, and
descriptive statistics (min, median, empirical 0.999 and 0.9999 quantiles, maximum, distinct-value count) plus the
high-water mark plus 20 %, explicitly labelled as having no exceedance probability. There is no override flag.

## 6. Threshold selection (GPD, peaks over threshold)

Candidate thresholds are the empirical **0.90, 0.95, 0.975 and 0.99** quantiles of the fit set. The chosen threshold is
the **lowest candidate whose fit passes the goodness-of-fit criterion** of section 7 (most data, least variance). If no
candidate passes, the verdict is `EVT-invalid-poor-fit` and no GPD bound is reported. As a sensitivity analysis, the
report lists shape and pWCET for every candidate together with its p-value.

Known limitation, accepted in advance: trying four thresholds and keeping the first that passes raises the chance of
accepting a mis-specified tail above the nominal level.

## 7. Goodness-of-fit criterion

- **GPD (primary).** Maximum-likelihood fit of the excesses (location fixed at 0). Anderson-Darling A^2 against the fitted
  GPD, p-value from a parametric bootstrap with the parameters re-estimated in each of **B = 1,000** replicates. Accept if
  **p ≥ 0.05**.
- **GEV (secondary).** Block maxima over **100 blocks** of the fit set (block size 1,000 for 100,000 runs), MLE, the same
  bootstrap Anderson-Darling test with **B = 500**, accept if p ≥ 0.05. A per-run probability p is converted to the
  per-block probability 1 - (1 - p)^b. A GEV fit failure is reported but does not change the primary verdict.
- GEV/GPD disagreement is reported as the ratio of their pWCET values at every target probability.

## 8. Target exceedance probabilities

Per-run exceedance probabilities **1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9**. Only 1e-3 and 1e-4 can be checked on the
held-out data; 1e-5 to 1e-9 are model extrapolations and are always reported as such, never as validated.

## 9. Validation method (held-out exceedance test)

For every method (GPD, GEV, MBPTA-CV) and every target p, count held-out runs strictly above the bound: k of n. Compute
the exact two-sided 95 % Clopper-Pearson interval [lo, hi] for the exceedance rate. With tolerance factor **T = 2**:

| Verdict | Rule (applied in this order) |
|---|---|
| `fail` | lo > T p: the bound is optimistic beyond sampling error and the tolerance (unsafe) |
| `untestable` | n p < 5: a pass would carry no evidence (a `fail` is still reported if it occurs) |
| `pass-pessimistic` | hi < p / T: safe, but looser than claimed |
| `pass` | otherwise |

Why a tolerance: a bound fitted on 100,000 runs carries estimation error, and 200,000 held-out runs resolve it. In an
exploratory synthetic check before this freeze (i.i.d. exponential tail, fit on 100,000, held out 200,000), the correctly
specified GPD bound at 1e-3 was exceeded 239 times against 200 expected, which an exact-equality criterion would call a
failure. T = 2 treats an order-of-magnitude error as a failure while not penalising ordinary estimation error.

**Primary validity outcome per trace:** the GPD bound is `pass` or `pass-pessimistic` at both 1e-3 and 1e-4. The
pessimism ratio pWCET(1e-9) / held-out maximum is reported for every method.

## 10. Baselines

- **High-water mark plus margin:** the fit-set maximum times **1.20**. Its held-out exceedance count is reported (it
  has no target probability, so no pass/fail).
- **MBPTA-CV (simplified re-implementation, Abella et al. 2017):** tail sizes k from 10 % of the fit set down to 50
  excesses on a 40-point geometric grid, largest first; the first tail with coefficient of variation of excesses
  CV ≤ 1 + 1.96 / sqrt(k) is modelled as exponential (bound u + mean excess × ln(k / (n p))). If no tail qualifies the
  method declines (heavier than exponential). It is run only on traces that pass the gates, since it also assumes i.i.d.
  data. It is not checked against the authors' code, and it is reported as "MBPTA-CV-style".

## 11. Crashed, aborted and failed runs

- A session that crashes (hard fault, watchdog reset, USB disconnect), is aborted, or meets a measurement-failure rule
  of section 4 is logged with date, time, board, scenario and cause. Its raw file is kept in the archive, unanalysed.
- **Partial data are never analysed or plotted.** The logger writes raw lines only, and the analysis runs only on
  complete sessions.
- A re-run is allowed only after a **dated deviation note** (section 14) has been written, stating the cause and any
  fix, **before** the re-run starts. The re-run is a fresh power cycle and replaces the whole session.
- A gate failure, a poor fit or a validation failure is a **result, not a crash**, and never justifies a re-run.

## 12. Reporting

Every trace gets one JSON report (`pwcet-analyse`) and one row in the results table: board, scenario, session ids,
data origin (`hardware`, `renode` or `synthetic`), verdict, failed gates, chosen threshold, shape, pWCET per p and
method, and held-out verdicts. Results from Renode are never presented as timing evidence. Until hardware data exist,
every number produced by this pipeline is synthetic and is labelled as such.

## 13. Freeze

Frozen on 2026-11-06, before any hardware has been bought and before any measurement exists.

## 14. Deviations

Each entry gives the date, the section changed, what changed, why, and whether any data had been seen at the time.

None yet.
