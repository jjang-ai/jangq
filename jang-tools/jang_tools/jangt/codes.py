"""JANGT code families: 16-bit trellis state -> approximately Gaussian value, built only from Metal-cheap ALU ops
(32-bit multiply by an odd constant, masks, popcount, shifts, fp add/fma). Every family is parameterized; constants are
found by jangt.codesearch, never copied from published codes.

Each family returns float64 values for all 65536 states (numpy reference); the Metal kernels must reproduce these
bit-exactly in fp16/fp32 (checked by the kernel tests). `ops` = estimated Metal ALU ops per decoded weight."""
from __future__ import annotations
import numpy as np

STATES = np.arange(1 << 16, dtype=np.uint64)
M32 = np.uint64(0xFFFFFFFF)


def _hash(mult: int, add: int = 0) -> np.ndarray:
    return (STATES * np.uint64(mult) + np.uint64(add)) & M32


def _popcount32(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros_like(x)
    for b in range(32):
        c += (x >> np.uint64(b)) & np.uint64(1)
    return c.astype(np.float64)


def popmix(mult: int, m1: int, m2: int, w2: float = 2.0) -> tuple[np.ndarray, int]:
    """pop(h & m1) + w2 * pop(h & m2), centered and normalized: binomial sums -> near-Gaussian. ~6 ops."""
    h = _hash(mult)
    v = _popcount32(h & np.uint64(m1)) + w2 * _popcount32(h & np.uint64(m2))
    return _norm(v), 6


def nibble_sum(mult: int, add: int = 0) -> tuple[np.ndarray, int]:
    """Irwin-Hall: sum of the eight 4-bit fields of a 32-bit hash (two masked multiply-accumulates on Metal). ~6 ops."""
    h = _hash(mult, add)
    v = np.zeros(len(h))
    for i in range(8):
        v += ((h >> np.uint64(4 * i)) & np.uint64(0xF)).astype(np.float64)
    return _norm(v), 6


def byte_pair(mult: int, add: int = 0) -> tuple[np.ndarray, int]:
    """Signed sum of the four bytes split into two pairs with different weights: finer level grid. ~7 ops."""
    h = _hash(mult, add)
    b = [((h >> np.uint64(8 * i)) & np.uint64(0xFF)).astype(np.float64) for i in range(4)]
    v = (b[0] + b[1]) + 0.5 * (b[2] + b[3])
    return _norm(v), 7


def random_gaussian(seed: int = 0) -> tuple[np.ndarray, int]:
    """Reference only (needs a 64K-entry LUT): an ideal unconstrained Gaussian codebook."""
    return _norm(np.random.default_rng(seed).standard_normal(1 << 16)), -1


def _norm(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    return v / v.std()


FAMILIES = {"popmix": popmix, "nibble_sum": nibble_sum, "byte_pair": byte_pair, "random_gaussian": random_gaussian}


def pop1(mult: int, add: int = 0) -> tuple[np.ndarray, int]:
    """popcount of a 32-bit hash: Binomial(32, 1/2), 33 levels. IMUL(+IADD) + POPCNT + convert/FMA: ~3 ops."""
    h = _hash(mult, add)
    return _norm(_popcount32(h)), 3


def pop_wsum(mult: int, m: int, w: float = 0.5) -> tuple[np.ndarray, int]:
    """popcount(h) + w*popcount(h & m): finer grid than pop1 for one extra AND+POPCNT+FMA (~5 ops)."""
    h = _hash(mult)
    return _norm(_popcount32(h) + w * _popcount32(h & np.uint64(m))), 5


def byte2(mult: int, add: int = 0) -> tuple[np.ndarray, int]:
    """Sum of the top two bytes of the hash (triangular, 511 levels). IMUL + 2 SHR/AND + IADD + convert: ~5 ops."""
    h = _hash(mult, add)
    return _norm(((h >> np.uint64(24)) + ((h >> np.uint64(16)) & np.uint64(0xFF))).astype(np.float64)), 5


FAMILIES.update({"pop1": pop1, "pop_wsum": pop_wsum, "byte2": byte2})


def pop16(mult: int, add: int = 0) -> tuple[np.ndarray, int]:
    """16-bit variant for packed short2 math: popcount((s*M + A) mod 2^16) -> Binomial(16), 17 levels."""
    h = (STATES * np.uint64(mult & 0xFFFF) + np.uint64(add & 0xFFFF)) & np.uint64(0xFFFF)
    return _norm(_popcount32(h)), 2


def pop16x2(m1: int, m2: int) -> tuple[np.ndarray, int]:
    """Two 16-bit hashes of the same state (one short2 multiply): pop(h1) + 0.5*pop(h2), finer grid."""
    h1 = (STATES * np.uint64(m1 & 0xFFFF)) & np.uint64(0xFFFF)
    h2 = (STATES * np.uint64(m2 & 0xFFFF)) & np.uint64(0xFFFF)
    return _norm(_popcount32(h1) + 0.5 * _popcount32(h2)), 3


FAMILIES.update({"pop16": pop16, "pop16x2": pop16x2})


# ---- vector codes (V=2): one 16-bit state -> TWO values, so extraction + hash are paid once per 2 weights ----
def _f16(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


def v2_halfbits(mult: int, mask: int, orbits: int) -> tuple[np.ndarray, int]:
    """h = s*M; the two 16-bit halves become fp16 numbers by bit pattern ((half & mask) | orbits) — no int->float
    convert, no popcount. Sign bit randomised by the hash, exponent partly fixed by `orbits`. Per PAIR on Metal:
    extract (2) + IMUL (1) + AND/OR on the 32-bit word (2) + one half2 FMA (1) => ~3 ops/weight."""
    h = _hash(mult)
    lo = (h & np.uint64(0xFFFF)); hi = (h >> np.uint64(16))
    m = np.uint64(mask & 0xFFFF); o = np.uint64(orbits & 0xFFFF)
    a = _f16((lo & m) | o); b = _f16((hi & m) | o)
    c = np.stack([a, b], axis=1); c = c - c.mean(axis=0); return c / c.std(), 3


def v2_pop(mult: int) -> tuple[np.ndarray, int]:
    """Two popcounts from one hash: pop(low 16) and pop(high 16) (Binomial(16) each). ~4 ops/weight."""
    h = _hash(mult)
    a = _popcount32(h & np.uint64(0xFFFF)); b = _popcount32(h >> np.uint64(16))
    c = np.stack([a, b], axis=1); c = c - c.mean(axis=0); return c / c.std(), 4


def v2_random(seed: int = 0) -> tuple[np.ndarray, int]:
    """Reference: ideal unconstrained 2-D Gaussian codebook (LUT)."""
    return np.random.default_rng(seed).standard_normal((1 << 16, 2)), -1
