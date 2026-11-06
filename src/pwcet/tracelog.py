"""Firmware trace logs: the binary frame format of firmware/common/tlog.h, plus format detection.

CSV lines (`T,<seq>,<cycles>,<input_id>,<flags-hex>`) are parsed by `analysis.parse_trace`.
The binary format is '#' header lines, then the line "#BIN1", then 13-byte frames:

    0xA5 0x5A | seq u32 LE | cycles u32 LE | input_id u8 | flags u8 | checksum u8

where checksum = sum of bytes 2..11 modulo 256. A frame with a bad checksum, or bytes that are
not a frame, are skipped by scanning for the next sync pair and counted as `bad_frames`. The
records they carried show up as missing sequence numbers, which the measurement-failure
rule of PROTOCOL.md section 4 already covers.

Rows with input_id 255 are the counter-read overhead calibration (empty timed regions,
firmware/common/harness.c). They are kept apart from the inference cycles: reported, never
subtracted, never analysed as execution times.
"""
import struct
import numpy as np

MAGIC = b"#BIN1\n"
SYNC = b"\xA5\x5A"
FRAME = 13
OVERHEAD_INPUT_ID = 255
_BODY = struct.Struct("<IIBB")


def encode_frame(seq: int, cycles: int, input_id: int, flags: int) -> bytes:
    body = _BODY.pack(seq & 0xFFFFFFFF, cycles & 0xFFFFFFFF, input_id, flags)
    return SYNC + body + bytes([sum(body) & 0xFF])


def encode_binary(rows, header=()) -> bytes:
    """rows: iterable of (seq, cycles, input_id, flags). header: lines with or without '#'."""
    head = b"".join(("#" + h.lstrip("#") + "\n").encode() for h in header)
    return head + MAGIC + b"".join(encode_frame(*r) for r in rows)


def is_binary(data: bytes) -> bool:
    return data.startswith(MAGIC) or (b"\n" + MAGIC) in data


def decode_binary(data: bytes):
    """Returns (header lines, rows as an int64 array of shape (k, 4), bad frame count)."""
    i = data.find(MAGIC)
    if i < 0:
        raise ValueError("no #BIN1 marker: not a binary trace")
    header = [h.strip() for h in data[:i].decode("utf-8", "replace").splitlines() if h.strip()]
    pos, end = i + len(MAGIC), len(data)
    rows, bad = [], 0
    while pos < end:
        if pos + FRAME > end:  # truncated last frame
            bad += 1
            break
        if data[pos: pos + 2] == SYNC:
            body = data[pos + 2: pos + 12]
            if sum(body) & 0xFF == data[pos + 12]:
                rows.append(_BODY.unpack(body))
                pos += FRAME
                continue
        bad += 1  # corrupt frame or stray bytes: resynchronise on the next sync pair
        nxt = data.find(SYNC, pos + 1)
        if nxt < 0:
            break
        pos = nxt
    arr = np.array(rows, dtype=np.int64).reshape(-1, 4)
    return header, arr, bad


def split_rows(seq, cycles, input_id, flags):
    """Applies the protocol rules to decoded rows: overhead rows (input_id 255) are set
    apart, wrap-flagged rows (bit 0) are excluded and counted, missing sequence numbers are
    counted over the whole stream. Returns a dict of arrays and counts."""
    seq, cycles = np.asarray(seq, np.int64), np.asarray(cycles, float)
    input_id, flags = np.asarray(input_id, np.int64), np.asarray(flags, np.int64)
    over = input_id == OVERHEAD_INPUT_ID
    wrap = (flags & 1).astype(bool)
    run = ~over
    span = int(seq.max() - seq.min() + 1) if seq.size else 0
    return {
        "cycles": cycles[run & ~wrap],
        "flags": flags[run & ~wrap],
        "input_id": input_id[run & ~wrap],
        "n_rows": int(run.sum()),
        "n_wrap_excluded": int((run & wrap).sum()),
        "n_seq_missing": span - int(np.unique(seq).size),
        "overhead": cycles[over & ~wrap],
    }
