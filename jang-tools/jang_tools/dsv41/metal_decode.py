# SPDX-License-Identifier: MIT
# The GEMV / quantized-GEMV instruction patterns below are transcribed from MLX v0.32 (gemv.h,
# quantized.h: load_vector, qdot, qmv_fast_impl, qmv_impl) so each product keeps MLX's
# accumulation order. Copyright (c) 2023-2026 Apple Inc., MIT licensed.
"""Fused Metal kernels for DSV4.1 single-stream decode and DSpark verify (<= MAX_ROWS rows).

Each entry point returns None when its inputs are not eligible; the caller then takes the stock path.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

MAX_ROWS = 8


def _gpu() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


# ----------------------------------------------------------------------------- router GEMV
# MLX's f32 GEMV (BM4/BN1/SM1/SN32/TM4/TN4: 4 rows per simdgroup, each lane 4 consecutive k per
# 128-wide block, simd_shuffle_down reduction) reading the bf16 weight directly, so the per-call
# f32 copy of the 384 x 5120 gate weight (~7.9 MB written + read per layer) is never made. Grid y =
# input row: every row is reduced exactly like a 1-row decode. At 1 row the logits are bit-identical
# to `x.astype(f32) @ w.astype(f32).T`; at 2-8 rows stock MLX switches to a GEMM, and these logits
# then equal the 1-row decode of each row instead (<= ~2e-6 absolute difference from the GEMM).

_ROUTER_SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u;
    uint r_in = threadgroup_position_in_grid.y;
    const device T* xr = x + (size_t)r_in * K;
    float accum[4] = {0};
    for (uint base = 0u; base < (uint)K; base += 128u) {
        uint k = base + lane * 4u;
        float v[4];
        #pragma clang loop unroll(full)
        for (uint j = 0u; j < 4u; ++j) v[j] = float(xr[k + j]);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 4u; ++r) {
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 4u; ++j)
                accum[r] += float(weight[(size_t)(row + r) * K + k + j]) * v[j];
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 4u; ++r) {
        #pragma clang loop unroll(full)
        for (ushort offset = 16u; offset >= 1u; offset >>= 1u)
            accum[r] += simd_shuffle_down(accum[r], offset);
        if (lane == 0u) out[(size_t)r_in * N + row + r] = accum[r];
    }
"""


@lru_cache(maxsize=1)
def _router_kernel():
    return mx.fast.metal_kernel(name="dsv41_router_gemv_bf16w_f32", input_names=["x", "weight"],
                                output_names=["out"], source=_ROUTER_SRC)


def router_logits(x: mx.array, weight: mx.array):
    """f32 logits [R, E] = x [R, K] (bf16) . weight [E, K] (bf16)^T with f32 products, or None."""
    if x.ndim != 2 or weight.ndim != 2 or not _gpu():
        return None
    R, K = x.shape
    E = weight.shape[0]
    if (R > MAX_ROWS or K != weight.shape[1] or K % 128 or E % 16
            or x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16):
        return None
    return _router_kernel()(inputs=[x, weight], template=[("T", x.dtype), ("K", K), ("N", E)],
                            grid=(E // 16 * 128, R, 1), threadgroup=(128, 1, 1),
                            output_shapes=[(R, E)], output_dtypes=[mx.float32])[0]
