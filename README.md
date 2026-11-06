# tinyml-pwcet

Planned empirical study: do extreme-value-theory / MBPTA assumptions (i.i.d., stationarity, tail fit) hold for int8
anomaly-detection inference on RP2040 (cacheless Cortex-M0+) versus STM32F446 (cached Cortex-M4), with interrupt, DMA and
RTOS interference as the independent variable?

## Status

Early skeleton. **No hardware has been bought and no measurements exist.**

- `docs/scoop_check.md`: literature check (abstract/README/snippet level only; verdict partly open, unverified).
- `docs/bom.md`: bill of materials (about USD 38, unverified prices, nothing purchased).
- `docs/design.md`: toolchain, model, measurement and statistics plan with rejected alternatives and risks.
- `src/pwcet/`: placeholder modules only (statistics pipeline not yet implemented).
- `firmware/`: empty.

Not done: protocol pre-registration, stats pipeline, model training, firmware, Renode.

## Test

    pip install -e .[dev]
    pytest
