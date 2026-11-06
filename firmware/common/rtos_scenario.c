/* S4: co-running FreeRTOS tasks (PROTOCOL.md section 3). UNTESTED.
 *
 * The measurement loop runs in a middle-priority task. A higher-priority task wakes every 10 ms
 * and computes for about 200 us; a lower-priority task runs a memory loop whenever the CPU is
 * free (it can only run between inferences). The 1 kHz kernel tick is the third source.
 * Context switches are counted by traceTASK_SWITCHED_IN (see FreeRTOSConfig.h) and set bit 3.
 * Build with -DUSE_FREERTOS and the FreeRTOS kernel (ARM_CM4F port on the STM32,
 * ThirdParty/GCC/RP2040 port on the Pico).
 */
#include "harness.h"

#ifdef USE_FREERTOS
#include "FreeRTOS.h"
#include "task.h"

static const board_hooks_t *hooks;
static const harness_cfg_t *config;
static workload_fn workload;
static volatile uint32_t scratch[256];

static void measure_task(void *arg)
{
    (void)arg;
    harness_run(hooks, config, workload);
    vTaskSuspend(NULL);
}

static void high_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(10));
        hooks->busy_wait_cycles(configCPU_CLOCK_HZ / 5000u); /* about 200 us */
    }
}

static void low_task(void *arg)
{
    (void)arg;
    for (uint32_t i = 0;; i++)
        scratch[i & 255u] += i;
}

void rtos_scenario_start(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work)
{
    hooks = b;
    config = cfg;
    workload = work;
    xTaskCreate(low_task, "low", 128, 0, tskIDLE_PRIORITY + 1, 0);
    xTaskCreate(measure_task, "meas", 512, 0, tskIDLE_PRIORITY + 2, 0);
    xTaskCreate(high_task, "high", 128, 0, tskIDLE_PRIORITY + 3, 0);
    vTaskStartScheduler();
    for (;;) {}
}
#else
void rtos_scenario_start(const board_hooks_t *b, const harness_cfg_t *cfg, workload_fn work)
{
    (void)b; (void)cfg; (void)work;
    for (;;) {} /* built without FreeRTOS: S4 unavailable */
}
#endif
