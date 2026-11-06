/* Host build of the board-independent firmware code (harness.c, tlog.c) with a fake board.
 * It only checks the control flow and the log formats against the Python parser
 * (tests/test_tracelog.py). It measures nothing: the "cycles" come from a simulated
 * counter, and the workload is a fixed fake cost plus a fake interrupt.
 *
 *   gcc -std=c99 -O1 -I../common host_main.c ../common/harness.c ../common/tlog.c -o host_harness
 *   ./host_harness <out-file> <csv|bin> <runs>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "harness.h"
#include "tlog.h"

static FILE *out;
static uint32_t fake_now;       /* simulated cycle counter */
static volatile uint32_t isr_count, dma_active, ctx_switches;
static uint32_t rng = 1u;

static uint32_t rand32(void)
{
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

static uint32_t cyc_start(void) { return fake_now; }
static uint32_t cyc_stop(uint32_t t0, int *wrapped)
{
    fake_now += 3; /* counter-read overhead */
    *wrapped = (fake_now - t0) > 1000000u; /* simulated ambiguity threshold */
    return fake_now - t0;
}
static void busy_wait(uint32_t c) { fake_now += c; }

static void fake_work(uint8_t id)
{
    fake_now += 200000u + 10u * id;   /* input-dependent cost */
    if (rand32() % 10u == 0) {        /* a fake interrupt in 10 % of runs */
        fake_now += 250u;
        isr_count++;
    }
    if (rand32() % 100000u == 0)      /* a rare run that the "timer" flags as wrapped */
        fake_now += 2000000u;
}

static void sink(const uint8_t *p, size_t n) { fwrite(p, 1, n, out); }

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <out-file> <csv|bin> <runs>\n", argv[0]);
        return 2;
    }
    out = fopen(argv[1], "wb");
    if (!out)
        return 1;
    board_hooks_t b = {cyc_start, cyc_stop, &isr_count, &dma_active, &ctx_switches,
                       busy_wait, rand32};
    uint32_t runs = (uint32_t)strtoul(argv[3], 0, 10);
    harness_cfg_t cfg = {.warmup_runs = 10u, .recorded_runs = runs, .overhead_runs = 100u,
                         .gap_max_cycles = 0u};
    tlog_init(sink, strcmp(argv[2], "bin") == 0);
    tlog_header(" board=host-fake (not a measurement)");
    harness_run(&b, &cfg, fake_work);
    fclose(out);
    return 0;
}
