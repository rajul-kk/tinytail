/* Trace log encoder. See tlog.h for the formats. UNTESTED ON HARDWARE. */
#include "tlog.h"

static tlog_rec_t buf[TLOG_BATCH];
static size_t used;
static tlog_write_fn sink;
static int bin_mode;

void tlog_init(tlog_write_fn write, int binary)
{
    sink = write;
    bin_mode = binary;
    used = 0;
}

static size_t put_str(char *out, size_t pos, size_t cap, const char *s)
{
    while (*s && pos + 1 < cap)
        out[pos++] = *s++;
    return pos;
}

static size_t put_u32(char *out, size_t pos, size_t cap, uint32_t v, int hex)
{
    char tmp[10];
    int n = 0;
    unsigned base = hex ? 16u : 10u;
    do {
        unsigned d = v % base;
        tmp[n++] = (char)(d < 10 ? '0' + d : 'a' + d - 10);
        v /= base;
    } while (v);
    while (n && pos + 1 < cap)
        out[pos++] = tmp[--n];
    return pos;
}

void tlog_header(const char *text)
{
    char line[128];
    size_t n = put_str(line, 0, sizeof line, "#");
    n = put_str(line, n, sizeof line, text);
    n = put_str(line, n, sizeof line, "\n");
    sink((const uint8_t *)line, n);
}

void tlog_begin(void)
{
    if (bin_mode)
        sink((const uint8_t *)"#BIN1\n", 6);
}

int tlog_push(uint32_t seq, uint32_t cycles, uint8_t input_id, uint8_t flags)
{
    if (used < TLOG_BATCH) {
        buf[used].seq = seq;
        buf[used].cycles = cycles;
        buf[used].input_id = input_id;
        buf[used].flags = flags;
        used++;
    }
    return used >= TLOG_BATCH;
}

size_t tlog_encode_frame(const tlog_rec_t *r, uint8_t out[TLOG_FRAME_BYTES])
{
    uint8_t sum = 0;
    out[0] = 0xA5;
    out[1] = 0x5A;
    for (int i = 0; i < 4; i++) {
        out[2 + i] = (uint8_t)(r->seq >> (8 * i));
        out[6 + i] = (uint8_t)(r->cycles >> (8 * i));
    }
    out[10] = r->input_id;
    out[11] = r->flags;
    for (int i = 2; i < 12; i++)
        sum = (uint8_t)(sum + out[i]);
    out[12] = sum;
    return TLOG_FRAME_BYTES;
}

size_t tlog_format_csv(const tlog_rec_t *r, char *out, size_t cap)
{
    size_t n = put_str(out, 0, cap, "T,");
    n = put_u32(out, n, cap, r->seq, 0);
    n = put_str(out, n, cap, ",");
    n = put_u32(out, n, cap, r->cycles, 0);
    n = put_str(out, n, cap, ",");
    n = put_u32(out, n, cap, r->input_id, 0);
    n = put_str(out, n, cap, ",");
    n = put_u32(out, n, cap, r->flags, 1);
    n = put_str(out, n, cap, "\n");
    if (cap)
        out[n < cap ? n : cap - 1] = '\0';
    return n;
}

void tlog_flush(void)
{
    for (size_t i = 0; i < used; i++) {
        if (bin_mode) {
            uint8_t f[TLOG_FRAME_BYTES];
            sink(f, tlog_encode_frame(&buf[i], f));
        } else {
            char line[48];
            size_t n = tlog_format_csv(&buf[i], line, sizeof line);
            sink((const uint8_t *)line, n);
        }
    }
    used = 0;
}
