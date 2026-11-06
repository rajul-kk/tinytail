# Firmware timing harness: UNTESTED skeleton

**Nothing here has been cross-compiled or run on a board.** The development machine has no `arm-none-eabi-gcc`
and no pico-sdk (checked 2026-11-06; nothing large was installed), and no board has been bought. The only code that
has been compiled and run is the board-independent part (`common/harness.c`, `common/tlog.c`,
`common/workload.c`), built for the host with MinGW gcc 6.3 and a fake board (`host/host_main.c`).
`tests/test_tracelog.py` builds that host program and parses its output with the Python reader. Board files use
register names from memory of the CMSIS and pico-sdk headers and may not compile as written.

| Path | What |
|---|---|
| `common/tlog.[ch]` | RAM buffer of 1,000 records, flushed between batches; CSV or 13-byte binary frames |
| `common/harness.[ch]` | warm-up, 100,000 counter-overhead rows (`input_id` 255), 100,000 timed runs, flags, optional random gap |
| `common/workload.c` | **placeholder** int8 FC stack with the AD model's layer shapes (not the AD model; TFLM + CMSIS-NN not integrated) |
| `common/rtos_scenario.c`, `FreeRTOSConfig.h` | S4: measurement task plus a higher-priority (10 ms, 200 us) and a lower-priority task, 1 kHz tick |
| `stm32f446/main.c`, `Makefile` | 180 MHz from HSI, `DWT->CYCCNT`, USART2 (ST-LINK VCP) 921600 baud, TIM2 1 kHz ISR (S1/S7), DMA2 memory-to-memory (S3) |
| `rp2040/main.c`, `CMakeLists.txt` | 125 MHz, SysTick CVR (24 bit), UART0 on GP0 at 921600 baud, repeating-timer ISR (S1/S7), looping DMA channel (S3), `copy_to_ram` build |
| `host/host_main.c` | fake board for the host build (simulated counter; measures nothing) |

Scenarios are chosen at build time (`SCENARIO=SCEN_S0|S1|S3|S4|S7`). S2 (harmonic ISR), S5 (core-1 or DMA2 stress) and S6
(cache flush per run) are not implemented yet.

## Log format (read by `pwcet-analyse`)

- Header lines start with `#` (board, clock, timer, workload, scenario, build date and compiler).
- CSV: `T,<seq>,<cycles>,<input_id>,<flags-hex>`, one line per run.
- Binary: the line `#BIN1`, then 13-byte frames `A5 5A | seq u32 LE | cycles u32 LE | input_id u8 | flags u8 |
  checksum u8` (sum of bytes 2-11 mod 256). That is 13 bytes per run against about 20 in CSV. Corrupt frames are skipped
  and counted, and their sequence numbers show up as missing.
- `flags`: bit 0 timer wrap (row excluded), bit 1 interference ISR ran, bit 2 DMA active, bit 3 RTOS context switch.
- `input_id` 255 marks the empty-region overhead rows. The host reports them (`counter_overhead` in the JSON) and never
  analyses or subtracts them.

## Design points found while writing the skeleton

- **RP2040 wrap flag.** SysTick's COUNTFLAG would mark every run during which the 24-bit counter passed zero: several
  percent of runs for a few-ms inference. That would trip the 0.1 % wrap rule of PROTOCOL.md section 4 although those runs
  are measured correctly, because the modular difference is exact below 134 ms. The flag is therefore set only when
  the 1 MHz TIMER shows the run may be longer than one counter period, i.e. when the count is ambiguous.
- **RP2040 log channel.** USB CDC stdio needs a periodic 1 ms service interrupt, which would be interference in S0.
  The harness writes UART0 directly. That needs a USB-UART adapter or a second Pico as picoprobe (about USD 4).
  `docs/bom.md` had listed it as optional and now marks it required.
- **RP2040 under FreeRTOS (S4).** The kernel tick uses SysTick, so S4 times with the 1 MHz TIMER (125-cycle
  resolution, stated in the header). This is design.md risk R5.
- **STM32F446 SRAM.** 128 KB in total. The placeholder weights (about 104 KB), the log batch (12 KB), the DMA buffers
  (2 KB) and the FreeRTOS heap (6 KB) only just fit. With TFLM the weights move to flash, which is also the S0-flash case.
