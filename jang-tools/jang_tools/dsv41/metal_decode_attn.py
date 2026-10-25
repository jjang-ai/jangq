"""Fused Metal decode attention for DeepSeek-V4.1-Flash (JANG runtime).

Translated from the vMLX engine's ``glm5_mla_decode_attn`` (ddalcu/mlx-serve PR #911 fused
serial decode, re-expressed as MLX ``mx.fast.metal_kernel``). V4.1's decode attention is the
same geometry GLM-5.3 uses: 64 query heads at rank/head_dim 512 attending a single shared
latent cache (MQA), optionally over DSA/window-selected rows. The stock Python path
(``_attn_core``) gathers the selected rows with ``mx.take`` and materialises
``[C, K, D]`` + an fp32 copy of the latent every step; here the bf16 latent rows are read in
place by index and accumulated in fp32 (bf16 accumulation over the 512-dim contraction drifts
into repetition loops — fp32 accumulation is required, the fp32 *copies* are not).

The one V4.1-specific addition over the GLM kernel is the learned per-head attention sink
``attn_sink``: an extra softmax denominator term ``exp(sink[h] - M)`` with no matching value
row. A sink is a per-head constant, not a key row, so it enters only the *combine* pass — the
row/partial pass is byte-for-byte the GLM kernel. Equivalence to ``_attn_core`` is to fp32
summation order (cos ~1, not bit-identical), exactly as the engine kernel is to stock SDPA.

Decode only (one query token, S==1). Prefill keeps the chunked path. ``VMLX_DSV41_DECODE_ATTN=0``
restores the stock gather path.
"""
from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

SPLIT = 128


def _split() -> int:
    return int(os.environ.get("VMLX_DSV41_MLA_SPLIT", str(SPLIT)))


def decode_attn_requested() -> bool:
    # Default OFF: validated bit-equivalent (cos 0.999999) but a wall-clock wash at decode
    # (31.08 vs 31.42 tok/s) — decode is dispatch/occupancy-bound, not attention-compute-bound,
    # so one fused kernel does not move the critical path. Kept as a building block for a full
    # fused per-layer decode chain (the only thing that closes the idle gaps). Set =1 to enable.
    return os.environ.get("VMLX_DSV41_DECODE_ATTN", "0").strip().lower() not in {"", "0", "false", "off", "no"}


@lru_cache(maxsize=8)
def _partial(rank: int, split_rows: int = SPLIT):
    """Per-(split, head) online-softmax partial over indexed latent rows. Identical to the GLM
    kernel's indexed variant: row = latent[indices[r]], kept when valid[r] && indices[r] <= q_pos."""
    assert rank % 32 == 0
    per = rank // 32
    src = f"""
    const uint split = threadgroup_position_in_grid.x;
    const uint hg = threadgroup_position_in_grid.y;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint h = hg * 8u + sg;
    const uint n_heads = (uint)meta[0], n_rows = (uint)meta[1];
    if (h >= n_heads) return;
    const float scale = sc[0];
    float qv[{per}], acc[{per}];
    for (uint j = 0; j < {per}u; ++j) {{ qv[j] = (float)q[(size_t)h * {rank}u + lane * {per}u + j] * scale; acc[j] = 0.0f; }}
    float m = -INFINITY, l = 0.0f;
    const uint r0 = split * {split_rows}u, r1 = min(r0 + {split_rows}u, n_rows);
    for (uint r = r0; r < r1; ++r) {{
        if (!(valid[r] && (int)indices[r] <= q_pos[0])) continue;
        device const T* lr = latent + (size_t)((uint)indices[r]) * {rank}u + lane * {per}u;
        float kv[{per}];
        float d = 0.0f;
        for (uint j = 0; j < {per}u; ++j) {{ kv[j] = (float)lr[j]; d += qv[j] * kv[j]; }}
        d = simd_sum(d);
        float mn = max(m, d);
        float c = metal::exp(m - mn), p = metal::exp(d - mn);
        l = l * c + p;
        for (uint j = 0; j < {per}u; ++j) acc[j] = acc[j] * c + p * kv[j];
        m = mn;
    }}
    const size_t o = (size_t)split * n_heads + h;
    for (uint j = 0; j < {per}u; ++j) pacc[o * {rank}u + lane * {per}u + j] = acc[j];
    if (lane == 0u) {{ pm[o] = m; pl[o] = l; }}
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_dsv41_mla_decode_partial_r{rank}_s{split_rows}",
        input_names=["q", "latent", "sc", "meta", "indices", "valid", "q_pos"],
        output_names=["pm", "pl", "pacc"],
        header="#include <metal_stdlib>\nusing namespace metal;\n", source=src)


@lru_cache(maxsize=4)
def _combine(rank: int):
    """Merge the split partials and fold in the per-head sink. M includes sink[h]; L carries the
    extra exp(sink[h]-M) denominator term that has no value row (so acc is unchanged by the sink)."""
    src = f"""
    const uint h = threadgroup_position_in_grid.x;
    const uint d = thread_index_in_threadgroup;
    const uint n_heads = (uint)meta[0], n_splits = (uint)meta[2];
    float M = sink[h];
    for (uint s = 0; s < n_splits; ++s) M = max(M, pm[(size_t)s * n_heads + h]);
    float L = metal::exp(sink[h] - M), a = 0.0f;
    for (uint s = 0; s < n_splits; ++s) {{
        const size_t o = (size_t)s * n_heads + h;
        float w = (pm[o] == -INFINITY) ? 0.0f : metal::exp(pm[o] - M);
        L += pl[o] * w;
        a += pacc[o * {rank}u + d] * w;
    }}
    out[(size_t)h * {rank}u + d] = (T)(a / L);
"""
    return mx.fast.metal_kernel(
        name=f"vmlx_dsv41_mla_decode_combine_r{rank}", input_names=["pm", "pl", "pacc", "meta", "sink"],
        output_names=["out"], header="#include <metal_stdlib>\nusing namespace metal;\n", source=src)


def dsv41_decode_attn(q, raw, sink, widx, scale: float):
    """Decode-step sparse attention, drop-in for ``_attn_core`` when S==1.

    q    (1, 1, H, D)   query for the single new token (bf16/fp16)
    raw  (1, N, D)      shared latent cache, keys == values (bf16/fp16, read in place)
    sink (H,)           learned per-head attention sink (fp32)
    widx (1, 1, K)      row indices into ``raw`` for this query; -1 = no row
    Returns (1, 1, H, D) in q's dtype, or None if the shapes aren't the decode geometry (caller
    then falls back to the chunked gather path)."""
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 1 or raw.ndim != 3 or raw.shape[0] != 1:
        return None
    H, D = int(q.shape[2]), int(q.shape[3])
    N = int(raw.shape[1])
    if raw.shape[2] != D or D % 32 or D > 1024 or N == 0:
        return None
    qd = q.dtype if q.dtype in (mx.bfloat16, mx.float16) else mx.bfloat16
    latent = raw.reshape(N, D).astype(qd)
    K = int(widx.shape[-1])
    if K == 0:
        return None
    idx = widx.reshape(K).astype(mx.int32)
    valid = (idx >= 0)
    idx = mx.where(valid, idx, 0)
    sr = _split()
    splits = (K + sr - 1) // sr
    meta = mx.array([H, K, splits], dtype=mx.int32)
    inputs = [q.reshape(H, D).astype(qd), latent, mx.array([scale], mx.float32), meta,
              idx, valid, mx.array([N], mx.int32)]  # q_pos = N-? ; causality already baked into widx (-1), so admit all rows
    pm, pl, pacc = _partial(D, sr)(
        inputs=inputs, template=[("T", qd)], grid=(splits * 256, (H + 7) // 8, 1), threadgroup=(256, 1, 1),
        output_shapes=[(splits, H), (splits, H), (splits, H, D)], output_dtypes=[mx.float32] * 3)
    out = _combine(D)(
        inputs=[pm, pl, pacc, meta, sink.astype(mx.float32)], template=[("T", qd)],
        grid=(H * D, 1, 1), threadgroup=(D, 1, 1), output_shapes=[(H, D)], output_dtypes=[qd])[0]
    return out.reshape(1, 1, H, D)


__all__ = ["dsv41_decode_attn", "decode_attn_requested"]
