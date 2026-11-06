/* Trace log: RAM-buffered timing records, flushed between batches (never inside a timed region).
 *
 * UNTESTED ON HARDWARE. The encoder is plain C99 with no board dependencies.
 *
 * Two output formats, both read by the host (`pwcet-analyse`, src/pwcet/tracelog.py):
 *
 * CSV (default, human-readable), one line per inference:
 *     T,<seq>,<cycles>,<input_id>,<flags-hex>\n
 *
 * Binary (TLOG_BINARY=1, about 3x fewer UART bytes). After the '#' header lines, the line
 * "#BIN1\n" starts a stream of 13-byte little-endian frames:
 *     0  0xA5  sync
 *     1  0x5A  sync
 *     2  seq     uint32
 *     6  cycles  uint32
 *     10 input_id uint8
 *     11 flags    uint8
 *     12 checksum uint8 = sum of bytes 2..11 modulo 256
 *
 * flags: bit0 timer wrap (row excluded by the host), bit1 interference ISR ran during the run,
 * bit2 DMA active, bit3 RTOS context switch during the run.
 */
#ifndef TLOG_H
#define TLOG_H
#include <stddef.h>
#include <stdint.h>

#define TLOG_FLAG_WRAP 0x01u
#define TLOG_FLAG_ISR  0x02u
#define TLOG_FLAG_DMA  0x04u
#define TLOG_FLAG_RTOS 0x08u

#ifndef TLOG_BINARY_DEFAULT
#define TLOG_BINARY_DEFAULT 0 /* 1 = binary frames, 0 = CSV lines */
#endif

#define TLOG_FRAME_BYTES 13
#define TLOG_BATCH 1000 /* records buffered before a flush (PROTOCOL.md section 3) */

typedef struct {
    uint32_t seq;
    uint32_t cycles;
    uint8_t input_id;
    uint8_t flags;
} tlog_rec_t;

/* Byte sink supplied by the board (blocking UART write). */
typedef void (*tlog_write_fn)(const uint8_t *buf, size_t len);

void tlog_init(tlog_write_fn write, int binary);
/* Header line without the leading '#' or trailing newline, e.g. "board=stm32f446 clk=180000000". */
void tlog_header(const char *text);
/* Ends the header; in binary mode writes "#BIN1\n". Call once before the first record. */
void tlog_begin(void);
/* Stores one record in RAM. Returns 1 when the batch is full and tlog_flush() must be called. */
int tlog_push(uint32_t seq, uint32_t cycles, uint8_t input_id, uint8_t flags);
/* Writes all buffered records to the sink. Call only outside timed regions. */
void tlog_flush(void);

/* Encoders, exposed for host-side tests. Return the number of bytes written. */
size_t tlog_encode_frame(const tlog_rec_t *r, uint8_t out[TLOG_FRAME_BYTES]);
size_t tlog_format_csv(const tlog_rec_t *r, char *out, size_t cap);

#endif
