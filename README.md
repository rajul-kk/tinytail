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
  - `synthetic`: traces with known tails, plus invalid inputs (AR(1), regime shift, periodic spikes, mode switching);
  - `iid`, `stationarity`: runs test, Ljung-Box, KPSS, first-vs-second-half Kolmogorov-Smirnov;
  - `gates`: the protocol's applicability gates (degeneracy, then independence, then stationarity, each test at
    0.05/4) with the verdict labels;
  - `evt`: GPD peaks over threshold, threshold choice, bootstrap Anderson-Darling goodness of fit, pWCET;
  - `validate`: GEV block maxima, an MBPTA-CV-style exponential baseline, max observed + 20 %, and held-out
    validation (exact Clopper-Pearson interval, tolerance factor 2, verdicts pass / pass-pessimistic / fail / untestable).
- `firmware/`: empty.

Not done: firmware, Renode, any hardware measurement.

## Test

    pip install -e .[dev]
    python -m pytest -q      # about 1-2 minutes (bootstrap goodness-of-fit tests)
