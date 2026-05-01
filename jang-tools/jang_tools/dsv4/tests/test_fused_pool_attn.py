"""A7 — fused_pool_attention vs inline path equivalence.

Synthesize realistic shapes, run both paths, assert bit-equivalent
output. This is the test surface the future Metal kernel must pass.
"""
from __future__ import annotations
import sys
import mlx.core as mx
import numpy as np

sys.path.insert(0, "/Users/eric/jang/jang-tools")
from jang_tools.dsv4.fused_pool_attn import fused_pool_attention
from jang_tools.dsv4.mlx_model import _build_window_mask, _compressed_visibility


def inline_path(q, k_win, v_win, k_pool, v_pool, topk_idx, sink, win_mask, comp_mask, scale):
    """Reference implementation — what attention.__call__ does today."""
    from mlx.core.fast import scaled_dot_product_attention as sdpa
    if k_pool is None or k_pool.shape[2] == 0:
        return sdpa(q, k_win, v_win, scale=scale, mask=win_mask, sinks=sink)
    if q.shape[2] == 1 and topk_idx is not None:
        B, _, P, hd = k_pool.shape
        K = topk_idx.shape[-1]
        idx = mx.broadcast_to(topk_idx[:, None, :, :, None], (B, 1, 1, K, hd))
        k_e = mx.broadcast_to(k_pool[:, None, :, :, :], (B, 1, 1, P, hd))
        v_e = mx.broadcast_to(v_pool[:, None, :, :, :], (B, 1, 1, P, hd))
        kg = mx.take_along_axis(k_e, idx, axis=3).reshape(B, K, hd)
        vg = mx.take_along_axis(v_e, idx, axis=3).reshape(B, K, hd)
        return sdpa(q, mx.concatenate([k_win, kg[:, None]], axis=2),
                    mx.concatenate([v_win, vg[:, None]], axis=2),
                    scale=scale, mask=None, sinks=sink)
    full_k = mx.concatenate([k_win, k_pool], axis=2)
    full_v = mx.concatenate([v_win, v_pool], axis=2)
    full_m = mx.concatenate([win_mask, comp_mask], axis=-1)
    return sdpa(q, full_k, full_v, scale=scale, mask=full_m, sinks=sink)


def case(label, *, B, n_heads, L, W, P, ratio, K, head_dim, has_sink):
    rng = np.random.default_rng(hash(label) & 0xffffffff)
    dtype = mx.bfloat16
    q = mx.array(rng.standard_normal((B, n_heads, L, head_dim)).astype(np.float32)).astype(dtype)
    k_win = mx.array(rng.standard_normal((B, 1, W, head_dim)).astype(np.float32)).astype(dtype)
    v_win = mx.array(rng.standard_normal((B, 1, W, head_dim)).astype(np.float32)).astype(dtype)
    sink = mx.array(rng.standard_normal((n_heads,)).astype(np.float32)).astype(dtype) if has_sink else None
    if P > 0:
        k_pool = mx.array(rng.standard_normal((B, 1, P, head_dim)).astype(np.float32)).astype(dtype)
        v_pool = mx.array(rng.standard_normal((B, 1, P, head_dim)).astype(np.float32)).astype(dtype)
    else:
        k_pool = v_pool = None

    offset = 0 if L > 1 else (W // 2)
    win_mask = _build_window_mask(B, L, offset, 128, W) if L > 1 or W > 0 else None
    if P > 0:
        comp_mask = _compressed_visibility(B, L, offset, P, ratio)
    else:
        comp_mask = None

    if K and K > 0 and P > 0 and L == 1:
        # Synthesize topk_idx: top-K of permuted indices (deterministic)
        topk_idx = mx.array(np.tile(rng.permutation(P)[:K], (B, L, 1)).astype(np.uint32))
    else:
        topk_idx = None

    scale = head_dim ** -0.5
    out_fused = fused_pool_attention(q, k_win, v_win, k_pool, v_pool,
                                      topk_idx, sink, win_mask, comp_mask, scale)
    out_inline = inline_path(q, k_win, v_win, k_pool, v_pool,
                              topk_idx, sink, win_mask, comp_mask, scale)
    mx.eval(out_fused, out_inline)

    diff = mx.abs(out_fused.astype(mx.float32) - out_inline.astype(mx.float32))
    max_d = float(mx.max(diff).item())
    rms = float(mx.sqrt(mx.mean(diff * diff)).item())
    ok = max_d < 1e-4
    flag = "OK" if ok else "FAIL"
    print(f"  {label:<55s} max_abs={max_d:.2e}  rms={rms:.2e}  {flag}")
    return ok


print("=== A7 fused vs inline equivalence ===")
ALL = True
ALL &= case("SWA only (P=0)",                      B=1, n_heads=64, L=8,   W=64,  P=0,    ratio=0,   K=0,   head_dim=64,  has_sink=True)
ALL &= case("HSA prefill L=8 P=2",                 B=1, n_heads=64, L=8,   W=64,  P=2,    ratio=128, K=0,   head_dim=64,  has_sink=True)
ALL &= case("HSA decode L=1 P=4",                  B=1, n_heads=64, L=1,   W=64,  P=4,    ratio=128, K=0,   head_dim=64,  has_sink=True)
ALL &= case("CSA decode L=1 P=8 K=4 (gather)",     B=1, n_heads=64, L=1,   W=64,  P=8,    ratio=4,   K=4,   head_dim=64,  has_sink=True)
ALL &= case("CSA prefill L=4 P=8 (mask path)",     B=1, n_heads=64, L=4,   W=32,  P=8,    ratio=4,   K=0,   head_dim=64,  has_sink=True)
ALL &= case("no sink",                             B=1, n_heads=64, L=8,   W=64,  P=2,    ratio=128, K=0,   head_dim=64,  has_sink=False)

print(f"\n=== {'ALL OK' if ALL else 'FAIL'} ===")
sys.exit(0 if ALL else 1)
