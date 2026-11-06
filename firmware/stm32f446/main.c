/* NUCLEO-F446RE timing harness. UNTESTED: never compiled (no arm-none-eabi-gcc on the
 * development machine) and never run (no board bought). Register-level CMSIS code.
 *
 * Timer: DWT->CYCCNT (32 bit, 180 MHz: wraps after 23.8 s, so the wrap flag is never set).
 * Log:   USART2 on PA2 (ST-LINK virtual COM port), 921600 baud, flushed between batches.
 * Scenario is chosen at build time: -DSCENARIO=SCEN_S1 (see common/harness.h).
 *   S1/S7 TIM2 update interrupt at 1 kHz with about 2 us (360 cycles) of work
 *   S3    DMA2 Stream0 memory-to-memory copies of 1 KB, restarted from its interrupt
 * SRAM budget (128 KB): placeholder weights and inputs about 104 KB, log batch 12 KB, DMA 2 KB,
 * FreeRTOS heap 6 KB. Tight; with TFLM the weights move to flash.
 *   S4    FreeRTOS (common/rtos_scenario.c); needs the FreeRTOS kernel with the ARM_CM4F port
 */
#include <stdint.h>
#include "stm32f4xx.h"
#include "../common/harness.h"
#include "../common/tlog.h"

#ifndef SCENARIO
#define SCENARIO SCEN_S0
#endif
#ifndef GAP_SEED
#define GAP_SEED 12345u
#endif
#define CPU_HZ 180000000u
#define RUNS 100000u

void workload_init(uint32_t seed);
void workload_run(uint8_t input_id);
void rtos_scenario_start(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work);
volatile uint32_t g_ctx_switches; /* incremented by traceTASK_SWITCHED_IN (FreeRTOSConfig.h) */

static volatile uint32_t isr_count, dma_active;
static uint32_t dma_src[256], dma_dst[256]; /* SRAM is tight: 128 KB in total */
static uint32_t rng_state = GAP_SEED;

/* ---------------------------------------------------------------- clock: HSI 16 MHz -> PLL 180 MHz */
static void clock_180mhz(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_PWREN;
    PWR->CR |= PWR_CR_VOS; /* voltage scale 1 */
    RCC->PLLCFGR = (16u << RCC_PLLCFGR_PLLM_Pos) | (360u << RCC_PLLCFGR_PLLN_Pos) |
                   (0u << RCC_PLLCFGR_PLLP_Pos) /* /2 */ | RCC_PLLCFGR_PLLSRC_HSI |
                   (8u << RCC_PLLCFGR_PLLQ_Pos) | (2u << RCC_PLLCFGR_PLLR_Pos);
    RCC->CR |= RCC_CR_PLLON;
    while (!(RCC->CR & RCC_CR_PLLRDY)) {}
    PWR->CR |= PWR_CR_ODEN; /* over-drive is required above 168 MHz */
    while (!(PWR->CSR & PWR_CSR_ODRDY)) {}
    PWR->CR |= PWR_CR_ODSWEN;
    while (!(PWR->CSR & PWR_CSR_ODSWRDY)) {}
    /* 5 wait states; ART accelerator on (S0-flash). S6 would invalidate ICEN/DCEN per run. */
    FLASH->ACR = FLASH_ACR_LATENCY_5WS | FLASH_ACR_ICEN | FLASH_ACR_DCEN | FLASH_ACR_PRFTEN;
    RCC->CFGR = RCC_CFGR_PPRE1_DIV4 | RCC_CFGR_PPRE2_DIV2 | RCC_CFGR_SW_PLL; /* APB1 45, APB2 90 MHz */
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) {}
    SystemCoreClock = CPU_HZ;
}

/* ---------------------------------------------------------------- cycle counter */
static uint32_t cyc_start(void) { return DWT->CYCCNT; }
static uint32_t cyc_stop(uint32_t t0, int *wrapped) { *wrapped = 0; return DWT->CYCCNT - t0; }
static void busy_wait(uint32_t c) { uint32_t t0 = DWT->CYCCNT; while (DWT->CYCCNT - t0 < c) {} }
static uint32_t rand32(void)
{
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}

static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

/* ---------------------------------------------------------------- UART log sink */
static void uart_init(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_GPIOAEN;
    RCC->APB1ENR |= RCC_APB1ENR_USART2EN;
    GPIOA->MODER = (GPIOA->MODER & ~GPIO_MODER_MODER2) | GPIO_MODER_MODER2_1; /* PA2 alternate */
    GPIOA->AFR[0] = (GPIOA->AFR[0] & ~(0xFu << 8)) | (7u << 8);               /* AF7 = USART2 */
    USART2->BRR = (45000000u + 921600u / 2) / 921600u;                       /* OVER16 */
    USART2->CR1 = USART_CR1_TE | USART_CR1_UE;
}

static void uart_write(const uint8_t *p, size_t n)
{
    while (n--) {
        while (!(USART2->SR & USART_SR_TXE)) {}
        USART2->DR = *p++;
    }
    while (!(USART2->SR & USART_SR_TC)) {}
}

/* ---------------------------------------------------------------- S1/S7: periodic ISR */
void TIM2_IRQHandler(void)
{
    TIM2->SR = 0;
    busy_wait(360); /* about 2 us of work */
    isr_count++;
}

static void periodic_isr_start(void)
{
    RCC->APB1ENR |= RCC_APB1ENR_TIM2EN;
    TIM2->PSC = 90u - 1u;   /* timer clock 90 MHz -> 1 MHz */
    TIM2->ARR = 1000u - 1u; /* 1 kHz */
    TIM2->DIER = TIM_DIER_UIE;
    TIM2->CR1 = TIM_CR1_CEN;
    NVIC_EnableIRQ(TIM2_IRQn);
}

/* ---------------------------------------------------------------- S3: background DMA traffic */
static void dma_kick(void)
{
    DMA2_Stream0->PAR = (uint32_t)dma_src;  /* memory-to-memory: PAR is the source */
    DMA2_Stream0->M0AR = (uint32_t)dma_dst;
    DMA2_Stream0->NDTR = 256;
    DMA2_Stream0->FCR = DMA_SxFCR_DMDIS | DMA_SxFCR_FTH; /* memory-to-memory needs the FIFO */
    DMA2_Stream0->CR = DMA_SxCR_DIR_1 | DMA_SxCR_PINC | DMA_SxCR_MINC | DMA_SxCR_PSIZE_1 |
                       DMA_SxCR_MSIZE_1 | DMA_SxCR_TCIE | DMA_SxCR_EN;
}

void DMA2_Stream0_IRQHandler(void)
{
    DMA2->LIFCR = DMA_LIFCR_CTCIF0 | DMA_LIFCR_CHTIF0 | DMA_LIFCR_CTEIF0 | DMA_LIFCR_CDMEIF0 |
                  DMA_LIFCR_CFEIF0;
    dma_kick(); /* keep the bus busy continuously */
}

static void dma_traffic_start(void)
{
    RCC->AHB1ENR |= RCC_AHB1ENR_DMA2EN;
    NVIC_EnableIRQ(DMA2_Stream0_IRQn);
    dma_active = 1;
    dma_kick();
}

/* ---------------------------------------------------------------- main */
int main(void)
{
    clock_180mhz();
    dwt_init();
    uart_init();
    workload_init(1u);

    board_hooks_t b = {cyc_start, cyc_stop, &isr_count, &dma_active, &g_ctx_switches,
                       busy_wait, rand32};
    harness_cfg_t cfg = {.warmup_runs = 2000u, .recorded_runs = RUNS, .overhead_runs = RUNS,
                         .gap_max_cycles = (SCENARIO == SCEN_S7) ? 2u * CPU_HZ / 1000u : 0u};

    tlog_init(uart_write, TLOG_BINARY_DEFAULT);
    tlog_header(" board=stm32f446 core=cortex-m4f clk_hz=180000000 timer=dwt_cyccnt");
    tlog_header(" workload=placeholder-fc (not the AD model; see common/workload.c)");
    tlog_header(" build=" __DATE__ " " __TIME__ " gcc=" __VERSION__);
    {
        static const char *names[] = {"S0-ram", "S1", "S3", "S4", "S7"};
        char line[48] = " scenario=";
        const char *s = names[SCENARIO];
        int i = 10;
        while (*s && i < 46) line[i++] = *s++;
        line[i] = '\0';
        tlog_header(line);
    }

    if (SCENARIO == SCEN_S1 || SCENARIO == SCEN_S7)
        periodic_isr_start();
    if (SCENARIO == SCEN_S3)
        dma_traffic_start();
    if (SCENARIO == SCEN_S4)
        rtos_scenario_start(&b, &cfg, workload_run); /* does not return */
    else
        harness_run(&b, &cfg, workload_run);

    tlog_header(" end");
    for (;;) {}
}
