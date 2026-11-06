/* Workload called inside the timed region. UNTESTED ON HARDWARE.
 *
 * PLACEHOLDER: until TFLite Micro + CMSIS-NN are integrated (they need the model_data.c that
 * models/train_ad.py exports and the TFLM tree, neither of which is in this repository), this
 * runs a hand-written int8 fully-connected stack with the same layer shapes as the MLPerf Tiny
 * AD autoencoder (640-128-128-128-128-8-128-128-128-128-640: about 264k multiply-accumulates).
 * One 640x128 and one 128x128 weight block are reused for all layers and generated in SRAM at
 * start-up (about 104 KB with inputs and activations; this is the "S0-ram" placement). Timings from this placeholder are NOT the AD model's
 * timings and must be labelled as such in the trace header (workload=placeholder-fc).
 */
#include <stdint.h>

#define IN_DIM 640
#define HID 128
#define BOT 8

static int8_t w_big[HID * IN_DIM];  /* 640 -> 128 (also read transposed for 128 -> 640) */
static int8_t w_hid[HID * HID];     /* 128 -> 128, 128 -> 8 and 8 -> 128 use sub-blocks */
static int8_t inputs[8][IN_DIM];    /* 8 fixed input vectors (input_id 0..7) */
static int8_t act_a[IN_DIM], act_b[IN_DIM];

static int8_t requant(int32_t acc)
{
    acc >>= 8; /* fixed shift instead of a per-layer multiplier: placeholder only */
    if (acc > 127)
        acc = 127;
    if (acc < 0)
        acc = 0; /* ReLU */
    return (int8_t)acc;
}

static void fc(const int8_t *x, int n_in, const int8_t *w, int stride, int transposed,
               int8_t *y, int n_out)
{
    for (int o = 0; o < n_out; o++) {
        int32_t acc = 0;
        for (int i = 0; i < n_in; i++)
            acc += (int32_t)x[i] * (transposed ? w[i * stride + o] : w[o * stride + i]);
        y[o] = requant(acc);
    }
}

void workload_init(uint32_t seed)
{
    uint32_t s = seed ? seed : 1u;
    for (int i = 0; i < HID * IN_DIM; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        w_big[i] = (int8_t)(s >> 24);
    }
    for (int i = 0; i < HID * HID; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        w_hid[i] = (int8_t)(s >> 24);
    }
    for (int k = 0; k < 8; k++)
        for (int i = 0; i < IN_DIM; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            inputs[k][i] = (int8_t)(s >> 24);
        }
}

void workload_run(uint8_t input_id)
{
    const int8_t *x = inputs[input_id & 7u];
    fc(x, IN_DIM, w_big, IN_DIM, 0, act_a, HID);          /* 640 -> 128 */
    for (int l = 0; l < 3; l++) {                          /* 3 x 128 -> 128 */
        fc(act_a, HID, w_hid, HID, 0, act_b, HID);
        for (int i = 0; i < HID; i++) act_a[i] = act_b[i];
    }
    fc(act_a, HID, w_hid, HID, 0, act_b, BOT);             /* 128 -> 8 */
    fc(act_b, BOT, w_hid, HID, 0, act_a, HID);             /* 8 -> 128 */
    for (int l = 0; l < 3; l++) {                          /* 3 x 128 -> 128 */
        fc(act_a, HID, w_hid, HID, 0, act_b, HID);
        for (int i = 0; i < HID; i++) act_a[i] = act_b[i];
    }
    fc(act_a, HID, w_big, IN_DIM, 1, act_b, IN_DIM);       /* 128 -> 640 */
}
