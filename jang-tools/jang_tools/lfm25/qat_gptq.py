"""Fixed-grid GPTQ codes-only QAT for LFM2.5 dense tensors.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-04.

Ports the DSV4-0731 learned-rounding approach (GPTQ error-compensated
rounding on the FIXED min-max grid, per-tensor best-of-RTN guard, byte
parity with ``mx.quantize`` storage) from routed MoE experts to plain dense
linears, and extends it with an MXFP8 grid:

  AffineGrid — f16 min-max scales/biases exactly as ``mx.quantize`` stores
               them; codes are uint32 bit-packed little-endian streams.
  Mxfp8Grid  — e8m0 uint8 group scales (2^(u8-127)) + e4m3fn byte codes via
               ``mx.to_fp8``/``mx.from_fp8`` (the exact kernels MLX uses),
               packed 4 codes per uint32.

Both grids pass a pack self-test against ``mx.quantize`` output before any
codes are emitted; a failed self-test aborts the conversion (never silently
falls back to a mismatched layout).

The GPTQ objective is plain per-tensor output reconstruction on captured
input activations: H = XᵀX (float64, escalating damping), Cholesky of H⁻¹,
blocked column sweep, and a hard best-of guard — a tensor never ships codes
worse than RTN on the calibration batch.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

__all__ = ["AffineGrid", "Mxfp8Grid", "gptq_codes", "pack_self_test"]


# ── packing ───────────────────────────────────────────────────────────────


def pack_rows_affine(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack integer codes (rows, cols) into MLX-affine uint32 words
    (continuous little-endian bitstream per row). Ported verbatim from the
    DSV4-0731 QAT script (pack self-test proven there for 2/3/8-bit)."""
    rows, cols = q.shape
    total_bits = cols * bits
    assert total_bits % 32 == 0, (cols, bits)
    out_words = total_bits // 32
    packed = np.zeros((rows, out_words), dtype=np.uint32)
    chunk = 65536
    weights32 = np.uint32(1) << np.arange(32, dtype=np.uint32)
    bit_idx = np.arange(bits, dtype=np.uint32)
    for r0 in range(0, rows, chunk):
        qq = q[r0 : r0 + chunk].astype(np.uint32)
        bits_arr = ((qq[:, :, None] >> bit_idx) & 1).astype(np.uint32)
        stream = bits_arr.reshape(qq.shape[0], total_bits)
        words = stream.reshape(qq.shape[0], out_words, 32) * weights32
        packed[r0 : r0 + chunk] = words.sum(axis=2, dtype=np.uint32)
    return packed


def pack_rows_fp8(codes_u8: np.ndarray) -> np.ndarray:
    """Pack e4m3 byte codes (rows, cols) into uint32 words, 4 bytes/word,
    little-endian byte order (col 0 = lowest byte)."""
    rows, cols = codes_u8.shape
    assert cols % 4 == 0, cols
    c = np.ascontiguousarray(codes_u8)
    return c.reshape(rows, cols // 4, 4).view(np.uint32).reshape(rows, cols // 4)


# ── grids ─────────────────────────────────────────────────────────────────


class AffineGrid:
    """Fixed affine grid: f16 min-max scales/biases from mx.quantize."""

    mode = "affine"

    def __init__(self, w_f32: np.ndarray, group_size: int, bits: int):
        self.gs = group_size
        self.bits = bits
        self.qmax = (1 << bits) - 1
        _qw, s, b = mx.quantize(mx.array(w_f32), group_size=group_size, bits=bits)
        self.s16 = np.asarray(s.astype(mx.float16))
        self.b16 = np.asarray(b.astype(mx.float16))
        del _qw
        s_exp = np.repeat(self.s16.astype(np.float32), group_size, axis=-1)
        self.b_exp = np.repeat(self.b16.astype(np.float32), group_size, axis=-1)
        self.s_safe = np.where(np.abs(s_exp) < 1e-12, 1.0, s_exp)

    def snap(self, w: np.ndarray, col0: int, col1: int) -> tuple[np.ndarray, np.ndarray]:
        """Round columns [col0:col1) of w onto the fixed grid.
        Returns (codes float32, dequantized float32)."""
        s = self.s_safe[:, col0:col1]
        b = self.b_exp[:, col0:col1]
        codes = np.clip(np.rint((w - b) / s), 0, self.qmax)
        return codes, s * codes + b

    def rtn_codes(self, w_f32: np.ndarray) -> np.ndarray:
        codes, _ = self.snap(w_f32, 0, w_f32.shape[1])
        return codes

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        return self.s_safe * codes + self.b_exp

    def emit(self, base: str, codes: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"{base}.weight": pack_rows_affine(codes.astype(np.uint32), self.bits),
            f"{base}.scales": self.s16,
            f"{base}.biases": self.b16,
        }


class Mxfp8Grid:
    """Fixed MXFP8 grid: e8m0 uint8 group scales + e4m3fn byte codes."""

    mode = "mxfp8"
    bits = 8

    def __init__(self, w_f32: np.ndarray, group_size: int = 32, bits: int = 8):
        assert group_size == 32 and bits == 8, "MXFP8 is gs32/8-bit"
        self.gs = group_size
        _qw, sc = mx.quantize(mx.array(w_f32), group_size=32, bits=8, mode="mxfp8")
        self.scales_u8 = np.asarray(sc)
        del _qw
        scale = np.exp2(self.scales_u8.astype(np.float32) - 127.0)
        self.scale_exp = np.repeat(scale, 32, axis=-1)

    def snap(self, w: np.ndarray, col0: int, col1: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (codes uint8, dequantized float32) for columns [col0:col1)."""
        s = self.scale_exp[:, col0:col1]
        codes = np.asarray(mx.to_fp8(mx.array(w / s)))
        deq = np.asarray(mx.from_fp8(mx.array(codes), dtype=mx.float32)) * s
        return codes.astype(np.float32), deq

    def rtn_codes(self, w_f32: np.ndarray) -> np.ndarray:
        codes, _ = self.snap(w_f32, 0, w_f32.shape[1])
        return codes

    def dequant(self, codes: np.ndarray) -> np.ndarray:
        vals = np.asarray(
            mx.from_fp8(mx.array(codes.astype(np.uint8)), dtype=mx.float32)
        )
        return vals * self.scale_exp

    def emit(self, base: str, codes: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"{base}.weight": pack_rows_fp8(codes.astype(np.uint8)),
            f"{base}.scales": self.scales_u8,
        }


def make_grid(w_f32: np.ndarray, mode: str, group_size: int, bits: int):
    if mode == "mxfp8":
        return Mxfp8Grid(w_f32, group_size, bits)
    return AffineGrid(w_f32, group_size, bits)


# ── parity self-test ──────────────────────────────────────────────────────


def _extract_codes_from_mx(qw, s, b, gs: int, bits: int) -> np.ndarray:
    """Recover integer codes from an mx.quantize result (DSV4 method)."""
    deq = mx.dequantize(qw, s, b, group_size=gs, bits=bits)
    s_exp = np.repeat(np.asarray(s, dtype=np.float32), gs, axis=-1)
    b_exp = np.repeat(np.asarray(b, dtype=np.float32), gs, axis=-1)
    d = np.asarray(deq, dtype=np.float32)
    s_safe = np.where(s_exp == 0.0, 1.0, s_exp)
    codes = np.rint((d - b_exp) / s_safe)
    return np.clip(codes, 0, (1 << bits) - 1).astype(np.uint32)


def pack_self_test() -> None:
    """Both grid emitters must reproduce mx.quantize byte-for-byte."""
    rng = np.random.default_rng(7)
    for bits, gs in ((6, 32), (8, 32), (8, 64), (6, 64)):
        w = rng.standard_normal((8, 512)).astype(np.float32)
        # (a) bit-layout parity: pack_rows must reproduce mx.quantize words
        qw, s, b = mx.quantize(mx.array(w), group_size=gs, bits=bits)
        codes_mx = _extract_codes_from_mx(qw, s, b, gs, bits)
        if not np.array_equal(pack_rows_affine(codes_mx, bits), np.asarray(qw)):
            raise SystemExit(f"AFFINE PACK SELF-TEST FAILED (layout) bits={bits} gs={gs}")
        # (b) runtime parity: emitted arrays must dequantize (via the MLX
        # kernel) to exactly the values the grid promised.
        grid = AffineGrid(w, gs, bits)
        codes = grid.rtn_codes(w)
        mine = grid.emit("t", codes)
        deq_mine = grid.dequant(codes)
        deq_mx = np.asarray(
            mx.dequantize(
                mx.array(mine["t.weight"]),
                mx.array(mine["t.scales"]),
                mx.array(mine["t.biases"]),
                group_size=gs,
                bits=bits,
            )
        )
        # The MLX kernel dequantizes at the scales' dtype (f16); the grid
        # models it in f32 — agreement is only expected to f16 precision.
        if not np.allclose(deq_mine, np.asarray(deq_mx, dtype=np.float32),
                           atol=1e-3, rtol=2e-3):
            raise SystemExit(f"AFFINE PACK SELF-TEST FAILED (runtime) bits={bits} gs={gs}")
    w = rng.standard_normal((8, 512)).astype(np.float32)
    qw, sc = mx.quantize(mx.array(w), group_size=32, bits=8, mode="mxfp8")
    grid = Mxfp8Grid(w)
    codes = grid.rtn_codes(w)
    mine = grid.emit("t", codes)
    if not np.array_equal(mine["t.weight"], np.asarray(qw)):
        raise SystemExit("MXFP8 PACK SELF-TEST FAILED: packed words differ")
    if not np.array_equal(mine["t.scales"], np.asarray(sc)):
        raise SystemExit("MXFP8 PACK SELF-TEST FAILED: scales differ")
    print("PACK_SELF_TEST_OK (affine 6/8-bit gs32/64 + mxfp8)", flush=True)


# ── GPTQ ──────────────────────────────────────────────────────────────────


def gptq_codes(
    w_f32: np.ndarray,
    grid,
    x_f32: np.ndarray,
    label: str = "",
    blk: int = 128,
    verbose: bool = True,
):
    """GPTQ error-compensated rounding of one dense tensor on a fixed grid.

    w_f32: (out, in) weight in the FOLDED domain.
    x_f32: (K, in) captured inputs in the same domain.

    Returns (codes, stats). Codes are best-of: GPTQ only ships if it beats
    RTN on the calibration reconstruction error.
    """
    out_dim, in_dim = w_f32.shape
    assert x_f32.shape[1] == in_dim, (x_f32.shape, w_f32.shape)
    t0 = time.time()

    base_codes = grid.rtn_codes(w_f32)

    def recon_err(codes: np.ndarray) -> float:
        e = grid.dequant(codes) - w_f32
        y = e @ x_f32.T
        return float(np.mean(np.square(y)))

    base_err = recon_err(base_codes)

    # Hessian in float64 (K < in_dim is common; damping escalation makes the
    # factorization PD instead of silently falling back — DSV4 lesson).
    H = (x_f32.T @ x_f32).astype(np.float64)
    diag_idx = np.arange(in_dim)
    diag = H[diag_idx, diag_idx]
    H[diag_idx, diag_idx] = np.where(diag <= 0, 1.0, diag)
    mean_diag = max(float(diag.mean()), 1e-10)
    U = None
    for damp_frac in (0.01, 0.05, 0.2, 1.0):
        try:
            Hd = H.copy()
            Hd[diag_idx, diag_idx] += damp_frac * mean_diag
            Hinv = np.linalg.inv(Hd)
            Hinv = 0.5 * (Hinv + Hinv.T)
            U = np.ascontiguousarray(np.linalg.cholesky(Hinv).T).astype(np.float32)
            if damp_frac != 0.01 and verbose:
                print(f"    [{label}] WARN: damping escalated to {damp_frac}",
                      flush=True)
            break
        except np.linalg.LinAlgError:
            continue
    del H

    if U is None:
        if verbose:
            print(f"    [{label}] WARN: Cholesky failed — keeping RTN codes",
                  flush=True)
        return base_codes, {
            "base_recon": base_err, "qat_recon": base_err,
            "improvement": 0.0, "flipped_frac": 0.0, "used": "rtn",
            "seconds": round(time.time() - t0, 1),
        }

    wq_work = w_f32.astype(np.float32).copy()
    codes = np.zeros_like(base_codes)
    u_diag = U[diag_idx, diag_idx]
    u_diag = np.where(np.abs(u_diag) < 1e-10, 1.0, u_diag)
    for b0 in range(0, in_dim, blk):
        b1 = min(b0 + blk, in_dim)
        err_blk = np.zeros((out_dim, b1 - b0), dtype=np.float32)
        for i in range(b0, b1):
            wcol = wq_work[:, i : i + 1]
            q, deq = grid.snap(wcol, i, i + 1)
            codes[:, i] = q[:, 0]
            err = (wcol[:, 0] - deq[:, 0]) / u_diag[i]
            err_blk[:, i - b0] = err
            if i + 1 < b1:
                wq_work[:, i + 1 : b1] -= err[:, None] * U[i, i + 1 : b1][None, :]
        if b1 < in_dim:
            wq_work[:, b1:] -= err_blk @ U[b0:b1, b1:]
    del wq_work, U

    gptq_err = recon_err(codes)
    use_gptq = gptq_err < base_err
    final = codes if use_gptq else base_codes
    flipped = float((final != base_codes).mean())
    stats = {
        "base_recon": base_err,
        "qat_recon": gptq_err if use_gptq else base_err,
        "improvement": 0.0 if base_err == 0 else 1.0 - (gptq_err if use_gptq else base_err) / base_err,
        "flipped_frac": flipped,
        "used": "gptq" if use_gptq else "rtn",
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"    [{label}] {stats['used']}: recon {base_err:.5e} -> "
            f"{stats['qat_recon']:.5e} ({stats['improvement']:+.1%}) "
            f"flipped {flipped:.2%} in {stats['seconds']}s",
            flush=True,
        )
    return final, stats
