/* Board-independent measurement loop. UNTESTED ON HARDWARE.
 *
 * Each board provides the hooks below; harness_run() does warm-up, the overhead calibration,
 * and the recorded runs, and logs through tlog. The timed region is the inference call only.
 */
#ifndef HARNESS_H
#define HARNESS_H
#include <stdint.h>

typedef enum {
    SCEN_S0 = 0,   /* no interference (control) */
    SCEN_S1,       /* periodic timer ISR, 1 kHz, about 2 us of work */
    SCEN_S3,       /* background DMA memory-to-memory traffic */
    SCEN_S4,       /* FreeRTOS: 1 kHz tick plus a higher- and a lower-priority task */
    SCEN_S7,       /* as S1 plus a random idle gap of 0..2 ms before each run */
} scenario_t;

typedef struct {
    /* Cycle counter. `start` returns an opaque stamp; `stop` returns elapsed cycles and sets
     * *wrapped when the measurement may have wrapped (RP2040 24-bit SysTick). */
    uint32_t (*cycles_start)(void);
    uint32_t (*cycles_stop)(uint32_t start, int *wrapped);
    /* Interference event counters, incremented by the generators (read before and after a run). */
    volatile uint32_t *isr_count;   /* bit1: ISR ran during the run */
    volatile uint32_t *dma_active;  /* bit2: non-zero while DMA traffic is running */
    volatile uint32_t *ctx_switches;/* bit3: RTOS context switches */
    void (*busy_wait_cycles)(uint32_t cycles); /* used for the random gap (S7) */
    uint32_t (*rand32)(void);       /* random gap source; seeded and printed in the header */
} board_hooks_t;

/* The workload: runs one inference on input vector `input_id` (0..7). */
typedef void (*workload_fn)(uint8_t input_id);

typedef struct {
    uint32_t warmup_runs;    /* PROTOCOL.md: 60 s of warm-up; the board converts seconds to runs */
    uint32_t recorded_runs;  /* 100,000 per session */
    uint32_t overhead_runs;  /* 100,000 empty regions, logged with input_id 255 */
    uint32_t gap_max_cycles; /* S7: uniform gap on [0, gap_max_cycles], 0 = no gap */
} harness_cfg_t;

void harness_run(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work);

#endif
