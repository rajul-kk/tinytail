# UNTESTED (pico-sdk is not installed on the development machine).
#   export PICO_SDK_PATH=~/pico-sdk
#   cmake -B build -DSCENARIO=SCEN_S1 [-DFREERTOS_KERNEL_PATH=~/FreeRTOS-Kernel] && cmake --build build
cmake_minimum_required(VERSION 3.13)
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(tinyml_pwcet_rp2040 C CXX ASM)
pico_sdk_init()

set(SCENARIO "SCEN_S0" CACHE STRING "SCEN_S0 | SCEN_S1 | SCEN_S3 | SCEN_S4 | SCEN_S7")
add_executable(harness main.c ../common/harness.c ../common/tlog.c ../common/workload.c
               ../common/rtos_scenario.c)
target_include_directories(harness PRIVATE ../common)
target_compile_definitions(harness PRIVATE SCENARIO=${SCENARIO} BOARD_RP2040)
target_compile_options(harness PRIVATE -O2 -Wall)
target_link_libraries(harness pico_stdlib hardware_dma hardware_timer hardware_uart)
pico_set_binary_type(harness copy_to_ram)   # S0-ram: code runs from SRAM, not through XIP
pico_enable_stdio_usb(harness 0)
pico_enable_stdio_uart(harness 0)           # the harness writes UART0 directly

if(SCENARIO STREQUAL "SCEN_S4")
  include($ENV{FREERTOS_KERNEL_PATH}/portable/ThirdParty/GCC/RP2040/FreeRTOS_Kernel_import.cmake)
  target_compile_definitions(harness PRIVATE USE_FREERTOS)
  target_link_libraries(harness FreeRTOS-Kernel FreeRTOS-Kernel-Heap4)
endif()
pico_add_extra_outputs(harness)
