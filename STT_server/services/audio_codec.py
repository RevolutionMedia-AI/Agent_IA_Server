"""G.711 mu-law codec — pure-Python drop-in for the deprecated ``audioop`` stdlib module.

Only ``width=2`` (16-bit signed PCM LE) is supported, matching the only callsite
used by the Agent_IA voice pipeline. The decode table is byte-for-byte identical
to CPython ``audioop.ulaw2lin`` so the two implementations can be swapped without
behavioural drift.
"""
import math
import struct


# ── lookup tables ────────────────────────────────────────────────────────
# CPython 3.12 audioop `_st_ulaw2linear16[256]`, packed as signed int16 LE pairs.
_MULAW_TO_LINEAR = b''.join(struct.pack('<h', x) for x in (
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364,  -9852,  -9340,  -8828,  -8316,
    -7932,  -7676,  -7420,  -7164,  -6908,  -6652,  -6396,  -6140,
    -5884,  -5628,  -5372,  -5116,  -4860,  -4604,  -4348,  -4092,
    -3900,  -3772,  -3644,  -3516,  -3388,  -3260,  -3132,  -3004,
    -2876,  -2748,  -2620,  -2492,  -2364,  -2236,  -2108,  -1980,
    -1884,  -1820,  -1756,  -1692,  -1628,  -1564,  -1500,  -1436,
    -1372,  -1308,  -1244,  -1180,  -1116,  -1052,   -988,   -924,
    -876,   -844,   -812,   -780,   -748,   -716,   -684,   -652,
    -620,   -588,   -556,   -524,   -492,   -460,   -428,   -396,
    -372,   -356,   -340,   -324,   -308,   -292,   -276,   -260,
    -244,   -228,   -212,   -196,   -180,   -164,   -148,   -132,
    -120,   -112,   -104,    -96,    -88,    -80,    -72,    -64,
    -56,    -48,    -40,    -32,    -24,    -16,     -8,      0,
    32124,  31100,  30076,  29052,  28028,  27004,  25980,  24956,
    23932,  22908,  21884,  20860,  19836,  18812,  17788,  16764,
    15996,  15484,  14972,  14460,  13948,  13436,  12924,  12412,
    11900,  11388,  10876,  10364,   9852,   9340,   8828,   8316,
    7932,   7676,   7420,   7164,   6908,   6652,   6396,   6140,
    5884,   5628,   5372,   5116,   4860,   4604,   4348,   4092,
    3900,   3772,   3644,   3516,   3388,   3260,   3132,   3004,
    2876,   2748,   2620,   2492,   2364,   2236,   2108,   1980,
    1884,   1820,   1756,   1692,   1628,   1564,   1500,   1436,
    1372,   1308,   1244,   1180,   1116,   1052,    988,    924,
    876,    844,    812,    780,    748,    716,    684,    652,
    620,    588,    556,    524,    492,    460,    428,    396,
    372,    356,    340,    324,    308,    292,    276,    260,
    244,    228,    212,    196,    180,    164,    148,    132,
    120,    112,    104,     96,     88,     80,     72,     64,
    56,     48,     40,     32,     24,     16,      8,      0,
))

_SEG_UEND = (0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)


# ── public API ───────────────────────────────────────────────────────────
def ulaw2lin(audio: bytes, width: int) -> bytes:
    """Decode mu-law bytes to signed PCM LE. Only ``width=2`` supported."""
    if width != 2:
        raise ValueError(f"width must be 2, got {width}")
    n = len(audio)
    out = bytearray(n * 2)
    tab = _MULAW_TO_LINEAR
    for i, b in enumerate(audio):
        out[i * 2:i * 2 + 2] = tab[b * 2:b * 2 + 2]
    return bytes(out)


def lin2ulaw(audio: bytes, width: int) -> bytes:
    """Encode signed PCM LE to mu-law. Only ``width=2`` supported."""
    if width != 2:
        raise ValueError(f"width must be 2, got {width}")
    if len(audio) % 2 != 0:
        raise ValueError("audio length not a multiple of width")
    n = len(audio) // 2
    out = bytearray(n)
    for i in range(n):
        s = struct.unpack_from('<h', audio, i * 2)[0]
        out[i] = _encode_sample(s)
    return bytes(out)


def rms(audio: bytes, width: int) -> int:
    """Root-mean-square of signed PCM samples. Only ``width=2`` supported."""
    if width != 2:
        raise ValueError(f"width must be 2, got {width}")
    if len(audio) == 0:
        return 0
    n = len(audio) // width
    if n == 0:
        return 0
    total = 0.0
    for i in range(0, len(audio), width):
        s = struct.unpack_from('<h', audio, i)[0]
        total += s * s
    return int(math.sqrt(total / n))


# ── helpers ──────────────────────────────────────────────────────────────
def _encode_sample(s: int) -> int:
    """G.711 mu-law encode for one 16-bit signed sample. Matches CPython audioop."""
    # CPython audioop.lin2ulaw feeds st_14linear2ulaw(val >> 18), where val is int16 << 16,
    # so net effect is "arithmetic right shift of int16 by 2" (floor for negatives).
    s14 = s >> 2
    if s14 < 0:
        sign_mask = 0x7F
        s14 = -s14
    else:
        sign_mask = 0xFF
    if s14 > 32635:
        s14 = 32635
    s14 += 33  # BIAS >> 2
    seg = 8
    for i, end in enumerate(_SEG_UEND):
        if s14 <= end:
            seg = i
            break
    if seg >= 8:
        return 0x7F ^ sign_mask
    code = (seg << 4) | ((s14 >> (seg + 1)) & 0x0F)
    return code ^ sign_mask


# ── self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import audioop
    import random

    # 1. ulaw 0xFF decodes to silent zero (positive silence in PCMU).
    assert ulaw2lin(bytes([0xFF] * 40), 2) == bytes([0x00] * 80), "0xFF should be silence"

    # 2. Round-trip stability for 256 single-byte inputs.
    random.seed(0)
    failures = 0
    for _ in range(256):
        x = bytes([random.randint(0, 255)])
        rt = lin2ulaw(ulaw2lin(x, 2), 2)
        if rt != x:
            failures += 1
    assert failures == 0, f"roundtrip failures: {failures}"

    # 3. rms of [1, 1, 1, 1] (int16 LE) > 0  (test as specified; bytes [0x01,0x00]*4)
    assert rms(bytes([0x01, 0x00]) * 4, 2) > 0, "rms of nonzero should be > 0"

    # 4. width=3 must raise ValueError on every function.
    for fn in (ulaw2lin, lin2ulaw, rms):
        try:
            fn(b"\x00" * 12, 3)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} did not raise for width=3")

    # 5. Drift check vs audioop across all 256 mu-law inputs (warning if mismatched).
    drift = 0
    for b in range(256):
        mine = ulaw2lin(bytes([b]), 2)
        theirs = audioop.ulaw2lin(bytes([b]), 2)
        if mine != theirs:
            drift += 1
    assert drift == 0, f"decode drift vs audioop: {drift}/256 entries"

    print("audio_codec: OK")
