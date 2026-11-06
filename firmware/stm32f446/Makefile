# UNTESTED (arm-none-eabi-gcc is not installed on the development machine).
# Needs ST's CMSIS device pack (github.com/STMicroelectronics/cmsis_device_f4) and CMSIS core:
#   make CMSIS=~/cmsis CMSIS_F4=~/cmsis_device_f4 LDSCRIPT=STM32F446RETx_FLASH.ld SCENARIO=SCEN_S1
# S4 also needs FREERTOS=~/FreeRTOS-Kernel.
CC       = arm-none-eabi-gcc
SCENARIO ?= SCEN_S0
CMSIS    ?= $(HOME)/cmsis
CMSIS_F4 ?= $(HOME)/cmsis_device_f4
LDSCRIPT ?= STM32F446RETx_FLASH.ld
CFLAGS   = -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16 -O2 -Wall \
           -DSTM32F446xx -DSCENARIO=$(SCENARIO) \
           -I../common -I$(CMSIS)/CMSIS/Core/Include -I$(CMSIS_F4)/Include
SRC      = main.c ../common/harness.c ../common/tlog.c ../common/workload.c \
           ../common/rtos_scenario.c $(CMSIS_F4)/Source/Templates/system_stm32f4xx.c \
           $(CMSIS_F4)/Source/Templates/gcc/startup_stm32f446xx.s

ifeq ($(SCENARIO),SCEN_S4)
CFLAGS  += -DUSE_FREERTOS -I$(FREERTOS)/include -I$(FREERTOS)/portable/GCC/ARM_CM4F
SRC     += $(FREERTOS)/tasks.c $(FREERTOS)/list.c $(FREERTOS)/queue.c \
           $(FREERTOS)/portable/GCC/ARM_CM4F/port.c $(FREERTOS)/portable/MemMang/heap_4.c
endif

harness.elf: $(SRC)
	$(CC) $(CFLAGS) -T$(LDSCRIPT) --specs=nano.specs --specs=nosys.specs -o $@ $(SRC)

clean:
	rm -f harness.elf
