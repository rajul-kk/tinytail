/* Raspberry Pi Pico (RP2040) timing harness, pico-sdk. UNTESTED: never compiled (no
 * arm-none-eabi-gcc or pico-sdk on the development machine) and never run (no board bought).
 *
 * Timer: SysTick current-value register, 24 bit at clk_sys = 125 MHz (8 ns resolution). The
 *        modular difference is correct for runs shorter than one period (2^24 cycles = 134 ms).
 *        Bit 0 (wrap) is set only when the 1 MHz TIMER says the run may be that long, i.e. when the
 *        cycle count is ambiguous. COUNTFLAG is not used: with a few-ms inference it would fire
 *        on several percent of runs that are measured correctly.
 *        S4 (FreeRTOS) uses SysTick for the kernel tick, so there the 1 MHz TIMER is the
 *        timer (125-cycle resolution, header says timer=timer_us; design.md risk R5).
 * Log:   UART0 on GP0 at 921600 baud, through a USB-UART adapter or a picoprobe. USB CDC stdio
 *        is not used: its periodic 1 ms service interrupt would be interference in S0.
 * Core 1 is parked (S5, the core-1 memory loop, is not implemented in this skeleton).
 */
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/structs/systick.h"
#include "hardware/timer.h"
#include "hardware/uart.h"
#include "../common/harness.h"
#include "../common/tlog.h"

#ifndef SCENARIO
#define SCENARIO SCEN_S0
#endif
#ifndef GAP_SEED
#define GAP_SEED 12345u
#endif
#define CPU_HZ 125000000u
#define RUNS 100000u
#define SYSTICK_PERIOD_US 134217u /* 2^24 / 125 MHz */

void workload_init(uint32_t seed);
void workload_run(uint8_t input_id);
void rtos_scenario_start(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work);
volatile uint32_t g_ctx_switches; /* incremented by traceTASK_SWITCHED_IN (FreeRTOSConfig.h) */

static volatile uint32_t isr_count, dma_active;
static uint32_t us_at_start;
static uint32_t rng_state = GAP_SEED;
static uint32_t dma_src[1024] __attribute__((aligned(4096)));
static volatile uint32_t dma_sink;
static int dma_chan;

/* ---------------------------------------------------------------- SysTick cycle counter */
static void systick_init(void)
{
    systick_hw->csr = 0;
    systick_hw->rvr = 0x00FFFFFFu;
    systick_hw->cvr = 0;
    systick_hw->csr = 0x5u; /* CLKSOURCE = processor clock, ENABLE, no interrupt */
}

static uint32_t cyc_start(void)
{
    us_at_start = time_us_32();
    return systick_hw->cvr;
}

static uint32_t cyc_stop(uint32_t t0, int *wrapped)
{
    uint32_t t1 = systick_hw->cvr;
    *wrapped = (time_us_32() - us_at_start) >= SYSTICK_PERIOD_US - 1000u; /* ambiguous */
    return (t0 - t1) & 0x00FFFFFFu; /* down-counter */
}

/* S4 fallback: 1 MHz TIMER scaled to cycles (resolution 125 cycles). */
static uint32_t us_start(void) { return time_us_32(); }
static uint32_t us_stop(uint32_t t0, int *wrapped) { *wrapped = 0; return (time_us_32() - t0) * 125u; }

static void busy_wait(uint32_t c) { busy_wait_us_32(c / 125u); }
static uint32_t rand32(void)
{
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

/* ---------------------------------------------------------------- UART log sink */
static void uart_write(const uint8_t *p, size_t n) { uart_write_blocking(uart0, p, n); }

/* ---------------------------------------------------------------- S1/S7: periodic ISR */
static bool periodic_cb(repeating_timer_t *t)
{
    (void)t;
    busy_wait_at_least_cycles(250); /* about 2 us of work */
    isr_count++;
    return true;
}
static repeating_timer_t periodic;

/* ---------------------------------------------------------------- S3: background DMA traffic */
static void dma_irq(void)
{
    dma_hw->ints0 = 1u << dma_chan;
    dma_channel_set_trans_count(dma_chan, 0xFFFFFFu, true); /* restart: never idle */
}

static void dma_traffic_start(void)
{
    dma_chan = dma_claim_unused_channel(true);
    dma_channel_config c = dma_channel_get_default_config(dma_chan);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, true);
    channel_config_set_ring(&c, false, 12); /* read address wraps within the 4 KB source */
    channel_config_set_write_increment(&c, false);
    dma_channel_set_irq0_enabled(dma_chan, true);
    irq_set_exclusive_handler(DMA_IRQ_0, dma_irq);
    irq_set_enabled(DMA_IRQ_0, true);
    dma_active = 1;
    dma_channel_configure(dma_chan, &c, &dma_sink, dma_src, 0xFFFFFFu, true);
}

/* ---------------------------------------------------------------- main */
int main(void)
{
    set_sys_clock_khz(CPU_HZ / 1000u, true);
    uart_init(uart0, 921600);
    gpio_set_function(0, GPIO_FUNC_UART);
    systick_init();
    workload_init(1u);

    board_hooks_t b = {cyc_start, cyc_stop, &isr_count, &dma_active, &g_ctx_switches,
                       busy_wait, rand32};
    if (SCENARIO == SCEN_S4) {
        b.cycles_start = us_start;
        b.cycles_stop = us_stop;
    }
    harness_cfg_t cfg = {.warmup_runs = 2000u, .recorded_runs = RUNS, .overhead_runs = RUNS,
                         .gap_max_cycles = (SCENARIO == SCEN_S7) ? 2u * CPU_HZ / 1000u : 0u};

    tlog_init(uart_write, TLOG_BINARY_DEFAULT);
    tlog_header(" board=rp2040 core=cortex-m0plus clk_hz=125000000");
    tlog_header(SCENARIO == SCEN_S4 ? " timer=timer_us (x125, 125-cycle resolution)"
                                    : " timer=systick_cvr (24 bit)");
    tlog_header(" workload=placeholder-fc in SRAM (not the AD model; see common/workload.c)");
    tlog_header(" build=" __DATE__ " " __TIME__ " gcc=" __VERSION__);
    {
        static const char *names[] = {" scenario=S0-ram", " scenario=S1", " scenario=S3",
                                      " scenario=S4", " scenario=S7"};
        tlog_header(names[SCENARIO]);
    }

    if (SCENARIO == SCEN_S1 || SCENARIO == SCEN_S7)
        add_repeating_timer_us(-1000, periodic_cb, NULL, &periodic); /* 1 kHz, start-to-start */
    if (SCENARIO == SCEN_S3)
        dma_traffic_start();
    if (SCENARIO == SCEN_S4)
        rtos_scenario_start(&b, &cfg, workload_run); /* does not return */
    else
        harness_run(&b, &cfg, workload_run);

    tlog_header(" end");
    for (;;) tight_loop_contents();
}
