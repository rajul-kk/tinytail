# Bill of materials (target: under USD 40, nothing purchased)

Prices are approximate list prices from memory of typical 2025-26 retail and were **not verified** in this session.
Links are vendor product pages; check stock and price before ordering. Nothing here has been bought.

| # | Item | Why | Approx. USD | Link |
|---|---|---|---|---|
| 1 | Raspberry Pi Pico (RP2040), with pre-soldered headers ("Pico H") if available | Cacheless Cortex-M0+ (dual core, 125 MHz default, 264 KB SRAM, XIP flash with a small 16 KB cache that must be accounted for, see design.md R2) | 4-5 | https://www.raspberrypi.com/products/raspberry-pi-pico/ |
| 2 | ST NUCLEO-F446RE | Cortex-M4F at 180 MHz with DWT cycle counter, FPU, DSP SIMD (CMSIS-NN fast path), ART flash accelerator (instruction cache and prefetch), on-board ST-LINK/V2-1 (SWD debug, virtual COM port), IDD jumper (JP6) for current measurement | 14-17 | https://www.st.com/en/evaluation-tools/nucleo-f446re.html |
| 3 | INA219 breakout (I2C, 0.1 ohm shunt) | Current/bus-voltage sensing for energy per inference | 3-6 (generic) or about 10 (Adafruit 904) | https://www.adafruit.com/product/904 |
| 4 | 8-channel 24 MHz USB logic analyser clone (Saleae-compatible, works with sigrok/PulseView) | Independent cross-check of GPIO-toggle timing; optional but recommended | 8-10 | generic marketplace listings |
| 5 | Breadboard (half size) plus 20 male-male and 10 male-female jumper wires | Wiring | 4-5 | generic |
| 6 | USB cables: micro-USB for Pico, mini-USB for Nucleo (assumed already owned) | Power and serial | 0 (assumed) | - |

**Total, core (items 1, 2, 3 generic, 5): about USD 25-33. With logic analyser (item 4): about USD 33-43.**
Realistic target with a generic INA219: about **USD 38**. If the total exceeds 40, drop item 4 first: the DWT counter (STM32)
and SysTick (RP2040) already give cycle-resolution timing, and the logic analyser is only a cross-check.

## Board choice justification

The study contrasts a cacheless core with a cached one. Alternatives considered:

- **NUCLEO-F446RE (chosen).** Cheapest board with a DWT cycle counter, Cortex-M4 DSP instructions (so CMSIS-NN uses its SIMD
  path, a different code path from the Pico), and the ART accelerator, which provides an instruction cache, data cache
  lines and prefetch on flash fetches. Honest caveat: this is a small flash cache, not the large L1 I/D caches of a
  Cortex-M7, so "cached" here means "flash accelerator cache" and any cache-driven jitter will be modest.
- **NUCLEO-H743ZI / NUCLEO-F767ZI (Cortex-M7, 16 KB I-cache and 16 KB D-cache).** The scientifically stronger "cached"
  platform, but about USD 25-30 alone, which breaks the USD 40 budget. Recorded as the **upgrade path** once the budget allows;
  the firmware is written so the board is a build-time parameter.
- **STM32F4 Discovery / Blue Pill.** Blue Pill clones (Cortex-M3, often fake STM32 parts) lack DWT on some silicon revisions
  and have no ST-LINK; rejected.
- **ESP32.** Cached, but Xtensa, Wi-Fi/FreeRTOS background activity cannot be switched off, and no DWT; rejected because the
  interference would not be controlled.
- **Pico 2 (RP2350, Cortex-M33).** Not cacheless in the same sense and out of scope for the stated RP2040 contrast; note as future work.

## Wiring

INA219 (I2C address 0x40). Use one sensor and move it between boards; do not leave both powered through it at once.

**Nucleo-F446RE (measure MCU current only):**
1. Remove the JP6 (IDD) jumper. Connect INA219 `VIN+` to JP6 pin 1 (supply side) and `VIN-` to JP6 pin 2 (MCU side),
   so the shunt is in series with the MCU supply. The ST-LINK stays powered from USB.
2. INA219 `VCC` to 3V3, `GND` to GND, `SDA`/`SCL` to the I2C pins used by the INA219 reader. The INA219 reader is a
   separate small board (the Pico is the natural choice, using I2C0 GP4/GP5) so that I2C traffic never runs on the device
   under test. Pico GND must be common with the Nucleo GND.
3. Timing pin: Nucleo PA5 (LD2 LED pin; also visible on D13) toggles around the inference. Logic analyser channel 0 on PA5, GND to GND.

**Pico (measure board current):**
1. Power the Pico from a 5 V source through the INA219: `VIN+` from USB 5 V rail (or a separate USB breakout), `VIN-` to Pico `VSYS`.
   This includes the Pico's on-board regulator losses; report energy for the Pico as "board-level".
2. INA219 read by a second Pico or by the Nucleo? Use the same method as above: a third controller is not available, so for the
   Pico measurement the INA219 is read by the Nucleo (I2C1 PB8/PB9) while the Nucleo is idle. Document which board read which.
3. Timing pin: Pico GP15 toggles around the inference; logic analyser channel 0 on GP15.

INA219 limits: 12-bit conversion takes about 532 microseconds per sample (default averaging off), so per-inference current traces are not
resolved for inferences of a few milliseconds. Energy per inference is therefore computed from the **average power during a long
back-to-back inference loop** divided by the measured inference rate, and cross-checked against an idle baseline.
Calibrate with the 0.1 ohm shunt, PGA /1 (40 mV, 400 mA full scale) or PGA /2 for the Nucleo at 180 MHz. Verify full scale against
a multimeter before trusting absolute numbers.

## Not needed / not bought

No oscilloscope, no debug probe (ST-LINK on the Nucleo can flash it; the Pico can be flashed over USB BOOTSEL; a second Pico as a
picoprobe is optional and costs about USD 4).
