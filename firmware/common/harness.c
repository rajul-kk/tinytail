/* Board-independent measurement loop. UNTESTED ON HARDWARE. */
#include "harness.h"
#include "tlog.h"

#define OVERHEAD_INPUT_ID 255u /* rows of the empty-region calibration; filtered by the host */

static uint8_t run_flags(const board_hooks_t *b, uint32_t isr0, uint32_t ctx0, int wrapped,
                         uint32_t dma_seen)
{
    uint8_t f = 0;
    if (wrapped)
        f |= TLOG_FLAG_WRAP;
    if (b->isr_count && *b->isr_count != isr0)
        f |= TLOG_FLAG_ISR;
    if (dma_seen)
        f |= TLOG_FLAG_DMA;
    if (b->ctx_switches && *b->ctx_switches != ctx0)
        f |= TLOG_FLAG_RTOS;
    return f;
}

static uint32_t timed(const board_hooks_t *b, workload_fn work, uint8_t id, uint8_t *flags)
{
    uint32_t isr0 = b->isr_count ? *b->isr_count : 0;
    uint32_t ctx0 = b->ctx_switches ? *b->ctx_switches : 0;
    uint32_t dma0 = b->dma_active ? *b->dma_active : 0;
    int wrapped = 0;
    uint32_t t0 = b->cycles_start();
    if (work)
        work(id);
    uint32_t dt = b->cycles_stop(t0, &wrapped);
    uint32_t dma1 = b->dma_active ? *b->dma_active : 0;
    *flags = run_flags(b, isr0, ctx0, wrapped, dma0 | dma1);
    return dt;
}

void harness_run(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work)
{
    uint8_t flags;
    uint32_t seq = 0;

    for (uint32_t i = 0; i < cfg->warmup_runs; i++) /* discarded (PROTOCOL.md section 4) */
        (void)timed(b, work, (uint8_t)(i % 8u), &flags);

    tlog_begin();
    /* Counter-read overhead: reported, never subtracted. Logged with its own input id. */
    for (uint32_t i = 0; i < cfg->overhead_runs; i++) {
        uint32_t dt = timed(b, 0, 0, &flags);
        if (tlog_push(seq++, dt, OVERHEAD_INPUT_ID, flags))
            tlog_flush();
    }
    tlog_flush();

    for (uint32_t i = 0; i < cfg->recorded_runs; i++) {
        if (cfg->gap_max_cycles && b->rand32 && b->busy_wait_cycles)
            b->busy_wait_cycles(b->rand32() % (cfg->gap_max_cycles + 1u));
        uint8_t id = (uint8_t)(i % 8u);
        uint32_t dt = timed(b, work, id, &flags);
        if (tlog_push(seq++, dt, id, flags))
            tlog_flush(); /* between batches only: UART traffic never overlaps a timed run */
    }
    tlog_flush();
}
