"""GPTQ on MLX's own affine grid — the affine baseline that minitest compares JANGT against."""
from __future__ import annotations

import mlx.core as mx


def _hinv_upper(H, damp):
    """Upper Cholesky factor U of H^-1 with relative damping, computed ENTIRELY
    on the CPU stream (every op inside `with mx.stream(mx.cpu)`). Any GPU op that waits on the multi-second CPU
    factorisation (even a transpose) trips the Metal command-buffer watchdog: reproduced on the 17408-wide Qwen3.8-27B
    down_proj Hessian."""
    with mx.stream(mx.cpu):
        B, K, _ = H.shape
        d = mx.mean(mx.diagonal(H, axis1=-2, axis2=-1), axis=-1, keepdims=True)
        Hd = H + (damp * d)[..., None] * mx.eye(K)[None]
        L = mx.linalg.cholesky(Hd)
        Linv = mx.linalg.inv(L)
        Hinv = mx.matmul(mx.swapaxes(Linv, -1, -2), Linv)
        U = mx.swapaxes(mx.linalg.cholesky(Hinv), -1, -2)
        mx.eval(U)
    return U


def pack_affine_lsb(q, bits):
    """MLX affine packing for ANY width (2/3/4/5/6/8): continuous LSB-first bitstream per row. Verified bit-identical to
    mx.quantize's packed output at every width."""
    *lead, K = q.shape
    g = q.astype(mx.uint32).reshape(*lead, K // 32, 32); words = []
    for w in range(bits):
        acc = mx.zeros(g.shape[:-1], dtype=mx.uint32)
        for j in range(32):
            st = j * bits; en = st + bits
            if en <= w * 32 or st >= (w + 1) * 32:
                continue
            off = st - w * 32; v = g[..., j]
            acc = acc | (v << off) if off >= 0 else acc | (v >> (-off))
        words.append(acc)
    return mx.stack(words, -1).reshape(*lead, K * bits // 32)


def affine_gptq(W, H, bits, gs, damp=0.01, block=128, row_chunk=1024):
    """Row-chunked wrapper: GPTQ rows are independent given chol(H^-1), so W is processed row_chunk rows at a time
    (identical codes / grid; each Metal command buffer stays far below the GPU watchdog). U is factored once."""
    B, N, Kd = W.shape
    U = _hinv_upper(H, damp); mx.eval(U)
    outs = [_affine_gptq_rows(W[:, r0:r0 + row_chunk], U, bits, gs, block) for r0 in range(0, N, row_chunk)]
    return (mx.concatenate([o[0] for o in outs], axis=1), mx.concatenate([o[1] for o in outs], axis=1),
            mx.concatenate([o[2] for o in outs], axis=1))


def _affine_gptq_rows(W, U, bits, gs, block=128):
    """Batched GPTQ on MLX's per-group affine grid. W (B,N,K) f32, H (B,K,K) f32 -> (packed, scales, biases) whose
    mx.dequantize is exactly the GPTQ solution. The grid (bf16 scale/bias per group) is MLX's own RTN grid of the
    ORIGINAL weights (static groups), codes chosen column by column with error feedback through chol(H^-1)."""
    B, N, Kd = W.shape
    _, s, b = mx.quantize(W.astype(mx.bfloat16), group_size=gs, bits=bits)       # (B,N,K/gs) bf16
    sf, bf = s.astype(mx.float32), b.astype(mx.float32)
    qmax = (1 << bits) - 1
    Q = mx.zeros((B, N, Kd), dtype=mx.uint8)
    Wc = W.astype(mx.float32)
    for b0 in range(0, Kd, block):
        b1 = min(b0 + block, Kd)
        Wb = Wc[..., b0:b1]; Ub = U[:, b0:b1, b0:b1]; errs, qs = [], []
        for j in range(b1 - b0):
            g = (b0 + j) // gs
            w = Wb[..., j]
            q = mx.clip(mx.round((w - bf[..., g]) / sf[..., g]), 0, qmax)
            e = (w - (q * sf[..., g] + bf[..., g])) / Ub[:, j, j][:, None]
            Wb = Wb - e[..., None] * Ub[:, j, :][:, None, :] * (mx.arange(b1 - b0) > j)[None, None, :]
            errs.append(e); qs.append(q.astype(mx.uint8))
            if (j + 1) % 32 == 0:
                mx.eval(Wb, *errs[-32:], *qs[-32:])
        Q[..., b0:b1] = mx.stack(qs, axis=-1)
        if b1 < Kd:
            Wc = mx.concatenate([Wc[..., :b1], Wc[..., b1:] - mx.matmul(mx.stack(errs, -1), U[:, b0:b1, b1:])], axis=-1)
        mx.eval(Q, Wc)
    packed = pack_affine_lsb(Q, bits)                                              # MLX affine packing == LSB-first
    mx.eval(packed)
    return packed, s, b
