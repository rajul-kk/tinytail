/* Minimal FreeRTOS configuration for scenario S4. UNTESTED. */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H
#include <stdint.h>

extern volatile uint32_t g_ctx_switches;
#define traceTASK_SWITCHED_IN() (g_ctx_switches++)

#if defined(BOARD_RP2040)
#define configCPU_CLOCK_HZ 125000000u
#define configNUMBER_OF_CORES 1 /* measurement on core 0 only */
#define configSUPPORT_PICO_SYNC_INTEROP 1
#define configSUPPORT_PICO_TIME_INTEROP 1
#else /* STM32F446 */
#define configCPU_CLOCK_HZ 180000000u
#define configPRIO_BITS 4
#define configKERNEL_INTERRUPT_PRIORITY (15 << (8 - configPRIO_BITS))
#define configMAX_SYSCALL_INTERRUPT_PRIORITY (5 << (8 - configPRIO_BITS))
#define vPortSVCHandler SVC_Handler
#define xPortPendSVHandler PendSV_Handler
#define xPortSysTickHandler SysTick_Handler
#endif

#define configTICK_RATE_HZ 1000u /* 1 kHz tick (PROTOCOL.md S4) */
#define configUSE_PREEMPTION 1
#define configUSE_TIME_SLICING 1
#define configMAX_PRIORITIES 5
#define configMINIMAL_STACK_SIZE 128
#define configTOTAL_HEAP_SIZE (6 * 1024)
#define configSUPPORT_DYNAMIC_ALLOCATION 1
#define configSUPPORT_STATIC_ALLOCATION 0
#define configUSE_IDLE_HOOK 0
#define configUSE_TICK_HOOK 0
#define configUSE_16_BIT_TICKS 0
#define configUSE_MUTEXES 1
#define configCHECK_FOR_STACK_OVERFLOW 0
#define configUSE_MALLOC_FAILED_HOOK 0
#define INCLUDE_vTaskDelayUntil 1
#define INCLUDE_xTaskDelayUntil 1
#define INCLUDE_vTaskSuspend 1
#define INCLUDE_vTaskDelay 1

#endif
