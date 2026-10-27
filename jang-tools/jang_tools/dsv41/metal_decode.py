# SPDX-License-Identifier: MIT
# The GEMV / quantized-GEMV instruction patterns below are transcribed from MLX v0.32 (gemv.h,
# quantized.h: load_vector, qdot, qmv_fast_impl, qmv_impl) so each product keeps MLX's
# accumulation order. Copyright (c) 2023-2026 Apple Inc., MIT licensed.
"""Fused Metal kernels for DSV4.1 single-stream decode and DSpark verify (<= MAX_ROWS rows).

Why decode was slow: a 1-row step is not limited by the 10.6 GB of weights it reads (~330 GB/s of
~800) but by dispatch count. Stock MLX ops ran ~150 dispatches per layer -- the router's f32 recast of
the 384 x 5120 gate every call, separate gate/up/down gather_qmm plus three shared-expert linears
(~25), and ~14 small ops per hyper-connection sublayer -- each too short to fill the GPU, with the host
encoding all 40 layers before the GPU started. These kernels collapse the per-layer work:

  router_logits     bf16 gate weight read in place with MLX's f32 GEMV order (no per-call f32 copy)
  moe_decode        6 routed + shared expert in 2 dispatches (gate+up+SwiGLU, then down+sum)
  hc_stage_decode   hyper-connection collapse + mixes + RMSNorm + Sinkhorn split in 2 dispatches
  hc_post_decode    the expand back into the 4 hc streams in 1 dispatch

Together with per-layer mx.async_eval (model.LAYER_ASYNC) they take AR decode from 29.0 to 39.2 tok/s
at 2k context (26.0 -> 34.3 at 100k), greedy tokens unchanged: every kernel reproduces MLX's
accumulation order (and its TF32 rounding of f32 matmuls), so outputs are bit-equal to the stock ops.

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
    # MLX takes this GEMV tiling only while K < 16 * E (a split-K variant otherwise) and E < 4096
    # (bm = 8 above); measured bit-equal for E 128..512 in that range (tests/dsv41_decode_fusion_test.py).
    if (R > MAX_ROWS or K != weight.shape[1] or K % 128 or E % 16 or K >= 16 * E or E > 2048
            or x.dtype != mx.bfloat16 or weight.dtype != mx.bfloat16):
        return None
    return _router_kernel()(inputs=[x, weight], template=[("T", x.dtype), ("K", K), ("N", E)],
                            grid=(E // 16 * 128, R, 1), threadgroup=(128, 1, 1),
                            output_shapes=[(R, E)], output_dtypes=[mx.float32])[0]


# ----------------------------------------------------------------------------- fused MoE decode
# Two dispatches replace the ~25 of Block.moe (two gather_qmm + SwiGLU elementwise + gather_qmm +
# reshape-sum + three shared-expert qlinear + their elementwise), mlx-serve MOE_GU / MOE_DN style:
#   moe_gu: grid (F/8 tiles, TOPK+1 slots, R rows). Slot s < TOPK is routed expert idx[r, s]
#           (2/3-bit), slot TOPK is the shared expert (w1/w3, 8-bit). Each simdgroup computes 4 gate and
#           4 up rows from one read of x, rounds each to bf16 as gather_qmm / qlinear do, and applies the
#           clamped SwiGLU (and the route weight for routed slots) in f32 -> h [R, TOPK+1, F] bf16.
#   moe_dn: grid (D/4 tiles, R rows), TOPK+1 simdgroups (one per slot) each project 4 output rows of
#           its slot (down_proj / w2), round to bf16, then sum in f32 in slot order (MLX's sequential
#           axis reduction) plus the shared expert -> y [R, D] bf16.
# Every projection runs MLX's qmv_fast or qmv lane walk and reduction (whichever MLX picks for that K
# and bit width, see _fast) for its (row, slot), as a 1-row gather_qmm / quantized_matmul does. So decode is
# bit-identical to the stock path, and each verify row is computed exactly as its 1-row decode.

_QHEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

template <int bits> constexpr int dsv_pack_factor() { return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits); }
template <int bits> constexpr int dsv_bytes_per_pack() { return ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3); }

template <typename T, int vpt, int bits>
inline float dsv_load_vector(const device T* x, thread float* x_thread) {
  float sum = 0;
  if (bits == 2) {
    for (int i = 0; i < vpt; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  } else if (bits == 3) {
    for (int i = 0; i < vpt; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) +
          float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  } else if (bits == 4) {
    for (int i = 0; i < vpt; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < vpt; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  return sum;
}

template <typename T, int vpt, int bits>
inline float dsv_load_vector_safe(const device T* x, thread float* x_thread, int N) {
  float sum = 0;
  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  } else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]) + float(x[i + 4]) +
          float(x[i + 5]) + float(x[i + 6]) + float(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  } else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += float(x[i]) + float(x[i + 1]) + float(x[i + 2]) + float(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }
  for (int i = N; i < vpt; i++) x_thread[i] = 0;
  return sum;
}

template <int vpt, int bits>
inline float dsv_qdot(const device uint8_t* w, const thread float* x_thread, float scale, float bias, float sum) {
  float accum = 0;
  if (bits == 2) {
    for (int i = 0; i < (vpt / 4); i++) {
      accum += (x_thread[4 * i] * (w[i] & 0x03) + x_thread[4 * i + 1] * (w[i] & 0x0c) +
                x_thread[4 * i + 2] * (w[i] & 0x30) + x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  } else if (bits == 3) {
    for (int i = 0; i < (vpt / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;
      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);
      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);
      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (vpt / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) + x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) + x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < vpt; i++) accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

template <int vpt, int bits>
inline float dsv_qdot_safe(const device uint8_t* w, const thread float* x_thread, float scale, float bias,
                           float sum, int N) {
  float accum = 0;
  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum += (x_thread[4 * i] * (w[i] & 0x03) + x_thread[4 * i + 1] * (w[i] & 0x0c) +
                x_thread[4 * i + 2] * (w[i] & 0x30) + x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  } else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;
      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);
      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);
      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  } else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum += (x_thread[4 * i] * (ws[i] & 0x000f) + x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
                x_thread[4 * i + 2] * (ws[i] & 0x0f00) + x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  } else if (bits == 8) {
    for (int i = 0; i < N; i++) accum += x_thread[i] * w[i];
  }
  return scale * accum + sum * bias;
}

// NP projections (sharing x) x 4 output rows, MLX qmv_fast_impl (FAST, K % 512 == 0) or qmv_impl
// lane walk. w*/s*/b* point at the simdgroup's first output row. acc[p * 4 + row] (not yet reduced).
template <typename T, int GS, int bits, bool FAST, int NP>
inline void dsv_qmv4(const device uint8_t* const* w, const device T* const* sc, const device T* const* bi,
                     const device T* x, int K, uint lane, thread float* acc) {
  constexpr int packs = FAST ? (bits <= 2 ? 1 : 2) : 1;
  constexpr int pf = dsv_pack_factor<bits>();
  constexpr int bpp = dsv_bytes_per_pack<bits>();
  constexpr int vpt = pf * packs;
  constexpr int block = vpt * 32;
  constexpr int scale_step = GS / vpt;
  const int in_w = K * bpp / pf;
  const int in_g = K / GS;
  const device uint8_t* ws[NP];
  const device T* ss[NP];
  const device T* bs[NP];
  for (int p = 0; p < NP; p++) {
    ws[p] = w[p] + lane * packs * bpp;
    ss[p] = sc[p] + lane / scale_step;
    bs[p] = bi[p] + lane / scale_step;
  }
  x += lane * vpt;
  thread float x_thread[vpt];
  int k = 0;
  const int full_end = FAST ? K : K - block;
  for (; k < full_end; k += block) {
    float sum = dsv_load_vector<T, vpt, bits>(x, x_thread);
    for (int p = 0; p < NP; p++) {
      for (int row = 0; row < 4; row++) {
        float s = ss[p][row * in_g];
        float b = bs[p][row * in_g];
        acc[p * 4 + row] += dsv_qdot<vpt, bits>(ws[p] + row * in_w, x_thread, s, b, sum);
      }
      ws[p] += block * bpp / pf;
      ss[p] += block / GS;
      bs[p] += block / GS;
    }
    x += block;
  }
  if (!FAST) {
    const int remaining = clamp(int(K - k - int(lane) * vpt), 0, vpt);
    if (remaining > 0) {
      float sum = dsv_load_vector_safe<T, vpt, bits>(x, x_thread, remaining);
      for (int p = 0; p < NP; p++) {
        for (int row = 0; row < 4; row++) {
          float s = ss[p][row * in_g];
          float b = bs[p][row * in_g];
          acc[p * 4 + row] += dsv_qdot_safe<vpt, bits>(ws[p] + row * in_w, x_thread, s, b, sum, remaining);
        }
      }
    }
  }
}

inline float dsv_minimum(float x, float y) { return metal::isnan(x) ? x : (x < y ? x : y); }
inline float dsv_maximum(float x, float y) { return metal::isnan(x) ? x : (x > y ? x : y); }
inline float dsv_silu(float x) {  // nn.silu = x * sigmoid(x), MLX Sigmoid
  float y = 1 / (1 + metal::precise::exp(metal::abs(x)));
  return x * ((x < 0) ? y : 1 - y);
}
"""

_MOE_GU_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint slot = threadgroup_position_in_grid.y;
    const uint r = threadgroup_position_in_grid.z;
    const uint out_row = threadgroup_position_in_grid.x * (4u * NSG) + sg * 4u;
    const device T* xr = x + (size_t)r * D;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float rw = 1.0f;
    if (slot < (uint)TOPK) {
      const size_t row0 = (size_t)idx[r * TOPK + slot] * F + out_row;
      const int in_w = D * dsv_bytes_per_pack<BR>() / dsv_pack_factor<BR>();
      const device uint8_t* w[2] = {(const device uint8_t*)gw + row0 * in_w, (const device uint8_t*)uw + row0 * in_w};
      const device T* s[2] = {gs + row0 * (D / GS), us + row0 * (D / GS)};
      const device T* b[2] = {gb + row0 * (D / GS), ub + row0 * (D / GS)};
      dsv_qmv4<T, GS, BR, FAST_D, 2>(w, s, b, xr, D, lane, acc);
      rw = route_w[r * TOPK + slot];
    } else {
      const size_t row0 = out_row;
      const int in_w = D * dsv_bytes_per_pack<BS>() / dsv_pack_factor<BS>();
      const device uint8_t* w[2] = {(const device uint8_t*)w1 + row0 * in_w, (const device uint8_t*)w3 + row0 * in_w};
      const device T* s[2] = {s1 + row0 * (D / GS), s3 + row0 * (D / GS)};
      const device T* b[2] = {b1 + row0 * (D / GS), b3 + row0 * (D / GS)};
      dsv_qmv4<T, GS, BS, FAST_DS, 2>(w, s, b, xr, D, lane, acc);
    }
    for (int row = 0; row < 4; row++) {
      float g = simd_sum(acc[row]);
      float u = simd_sum(acc[4 + row]);
      if (lane == 0) {
        g = float(static_cast<T>(g));
        u = float(static_cast<T>(u));
        if (HAS_LIM) {
          u = dsv_minimum(dsv_maximum(u, -LIM), LIM);
          g = dsv_minimum(g, LIM);
        }
        float h = dsv_silu(g) * u;
        if (slot < (uint)TOPK) h = h * rw;
        out[((size_t)r * (TOPK + 1) + slot) * F + out_row + row] = static_cast<T>(h);
      }
    }
"""

_MOE_DN_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint slot = simdgroup_index_in_threadgroup;
    const uint r = threadgroup_position_in_grid.y;
    const uint out_row = threadgroup_position_in_grid.x * (4u * RG);
    const device T* hr = h + ((size_t)r * (TOPK + 1) + slot) * F;
    float acc[4 * RG];
    for (int i = 0; i < 4 * RG; i++) acc[i] = 0;
    if (slot < (uint)TOPK) {
      const size_t row0 = (size_t)idx[r * TOPK + slot] * D + out_row;
      const int in_w = F * dsv_bytes_per_pack<BD>() / dsv_pack_factor<BD>();
      const device uint8_t* w[RG];
      const device T* s[RG];
      const device T* b[RG];
      for (int p = 0; p < RG; p++) {
        w[p] = (const device uint8_t*)dw + (row0 + 4 * p) * in_w;
        s[p] = ds + (row0 + 4 * p) * (F / GS);
        b[p] = db + (row0 + 4 * p) * (F / GS);
      }
      dsv_qmv4<T, GS, BD, FAST_F, RG>(w, s, b, hr, F, lane, acc);
    } else {
      const size_t row0 = out_row;
      const int in_w = F * dsv_bytes_per_pack<BS>() / dsv_pack_factor<BS>();
      const device uint8_t* w[RG];
      const device T* s[RG];
      const device T* b[RG];
      for (int p = 0; p < RG; p++) {
        w[p] = (const device uint8_t*)w2 + (row0 + 4 * p) * in_w;
        s[p] = s2 + (row0 + 4 * p) * (F / GS);
        b[p] = b2 + (row0 + 4 * p) * (F / GS);
      }
      dsv_qmv4<T, GS, BS, FAST_FS, RG>(w, s, b, hr, F, lane, acc);
    }
    threadgroup float part[(TOPK + 1) * 4 * RG];
    for (int i = 0; i < 4 * RG; i++) {
      float v = simd_sum(acc[i]);
      if (lane == 0) part[slot * 4 * RG + i] = float(static_cast<T>(v));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane < 4 * RG) {
      float y = 0.0f;
      for (int s = 0; s < TOPK; s++) y = part[s * 4 * RG + lane] + y;
      y = y + part[TOPK * 4 * RG + lane];
      out[(size_t)r * D + out_row + lane] = static_cast<T>(y);
    }
"""


# (simdgroups per threadgroup in moe_gu, 4-row groups per simdgroup in moe_dn): occupancy only, the
# per-row arithmetic does not depend on it.
MOE_TILE = (2, 1)


def _fast(bits: int, K: int) -> bool:
    """MLX runs qmv_fast when K is a multiple of its block (32 lanes x values per thread: 512 for
    2/3/4-bit, 256 for 8-bit), else qmv. Measured on f32 outputs against quantized_matmul / gather_qmm
    (tests/dsv41_decode_fusion_test.py)."""
    return K % (256 if bits == 8 else 512) == 0


@lru_cache(maxsize=None)
def _moe_kernels(lim: float):
    has = lim > 0
    src = _MOE_GU_SRC.replace("HAS_LIM", "true" if has else "false").replace("LIM", f"{lim if has else 0.0:.9e}f")
    gu = mx.fast.metal_kernel(
        name=f"dsv41_moe_gu_{'lim' if has else 'nolim'}", header=_QHEADER, source=src,
        input_names=["x", "idx", "route_w", "gw", "gs", "gb", "uw", "us", "ub", "w1", "s1", "b1", "w3", "s3", "b3"],
        output_names=["out"])
    dn = mx.fast.metal_kernel(
        name="dsv41_moe_dn", header=_QHEADER, source=_MOE_DN_SRC,
        input_names=["h", "idx", "dw", "ds", "db", "w2", "s2", "b2"], output_names=["out"])
    return gu, dn


def _moe_eligible(x, idx, experts, shared, group_size) -> bool:
    if x.ndim != 2 or idx.ndim != 2 or not _gpu():
        return False
    R, D = x.shape
    g, u, d = experts.gate_proj, experts.up_proj, experts.down_proj
    F = g.weight.shape[1]
    sw = (shared.w1, shared.w3, shared.w2)
    return not (R > MAX_ROWS or x.dtype != mx.bfloat16 or g.bits != u.bits or g.bits not in (2, 3, 4, 8)
                or d.bits not in (2, 3, 4, 8) or any(m.bits != shared.w1.bits or m.group_size != group_size for m in sw)
                or shared.w1.bits not in (2, 3, 4, 8) or F % 8 or D % 8 or F % group_size or D % group_size
                or g.scales.dtype != mx.bfloat16 or shared.w1.scales.dtype != mx.bfloat16
                or d.weight.shape[1] != D or shared.w1.weight.shape[0] != F or shared.w2.weight.shape[0] != D)


def moe_decode(x, idx, w, experts, shared, swiglu_limit: float, group_size: int = 64):
    """Routed + shared expert MoE for R <= MAX_ROWS rows in two dispatches, or None if not eligible.

    x [R, D] bf16; idx [R, TOPK] expert ids (argpartition order = summation order); w [R, TOPK] f32
    route weights (already normalized and scaled). experts: SwitchMLP (gate_proj/up_proj/down_proj
    SwitchProj); shared: SharedExpert (w1/w2/w3 QuantizedLinear). Returns y [R, D] bf16."""
    if not _moe_eligible(x, idx, experts, shared, group_size):
        return None
    R, D = x.shape
    topk = idx.shape[1]
    g, u, d = experts.gate_proj, experts.up_proj, experts.down_proj
    F = g.weight.shape[1]
    lim = float(swiglu_limit)
    nsg, rg = MOE_TILE
    if F % (4 * nsg) or D % (4 * rg):
        return None
    gu, dn = _moe_kernels(lim)
    idx = idx.astype(mx.uint32)
    w = w.astype(mx.float32)
    tmpl = [("T", mx.bfloat16), ("D", D), ("F", F), ("TOPK", topk), ("GS", group_size),
            ("BS", shared.w1.bits)]
    h = gu(inputs=[x, idx, w, g.weight, g.scales, g.biases, u.weight, u.scales, u.biases,
                   shared.w1.weight, shared.w1.scales, shared.w1.biases,
                   shared.w3.weight, shared.w3.scales, shared.w3.biases],
           template=tmpl + [("BR", g.bits), ("FAST_D", _fast(g.bits, D)), ("FAST_DS", _fast(shared.w1.bits, D)),
                            ("NSG", nsg)],
           grid=(F // (4 * nsg) * 32 * nsg, topk + 1, R), threadgroup=(32 * nsg, 1, 1),
           output_shapes=[(R, topk + 1, F)], output_dtypes=[mx.bfloat16])[0]
    y = dn(inputs=[h, idx, d.weight, d.scales, d.biases, shared.w2.weight, shared.w2.scales, shared.w2.biases],
           template=tmpl + [("BD", d.bits), ("FAST_F", _fast(d.bits, F)), ("FAST_FS", _fast(shared.w1.bits, F)),
                            ("RG", rg)],
           grid=(D // (4 * rg) * 32 * (topk + 1), R, 1), threadgroup=(32 * (topk + 1), 1, 1),
           output_shapes=[(R, D)], output_dtypes=[mx.bfloat16])[0]
    return y


# ----------------------------------------------------------------------------- fused hyper-connections
# Per sublayer the stock decode runs hc_mixes (astype, square, mean = 2-pass all-reduce, rsqrt, f32
# GEMV 24 x 20480, multiply, Sinkhorn kernel), hc_pre (compiled multiply + axis reduction + cast),
# RMSNorm and hc_post (compiled multiply, einsum GEMM, add, cast): ~14 dispatches. Here (mlx-serve
# HC_STAGE / HC_SPLIT, oMLX K1/K2) three:
#   hc_stage: tg 0 collapses the stream with the previous pre and RMS-norms it (MLX rms_single_row /
#             rms_looped order); tg 1 sums x^2 in MLX's row_reduce_looped order; tgs 2-7 run MLX's gemv
#             (BM1 BN8 SM1 SN32 TM4 TN4, its choice for 24 x 20480) for the 24 mixes.
#   hc_split: mean, rsqrt, mixes * rs and the release Sinkhorn (the code of _hc_split_metal)
#             -> pre / post / comb.
#   hc_post:  post * h + comb^T . residual -> next stream (bf16). MLX's f32 GEMM for the einsum runs on
#             the M5 matrix units in TF32 (MLX_ENABLE_TF32=1, the default): comb truncated to TF32, the
#             four products summed pairwise; reproduced exactly here.
# Each row (decode or verify) is computed as its own 1-row decode. Bit-identity with the stock ops is
# checked once at first use (hc_selfcheck); any mismatch (another MLX, TF32 off) disables the path.

# Collapse with the previous pre + RMSNorm of one row (1024 threads; x, pre, nw, xin, tmem, lid, lane, sg in
# scope): shared by hc_stage (threadgroup 0) and hc_boundary.
_HC_COLLAPSE_NORM_SRC = r"""      // hc_pre: s = sum_j pre_j * x_j in f32 (sequential axis reduction), cast to T; then RMSNorm.
      float p[HC];
      for (int j = 0; j < HC; j++) p[j] = pre[r * HC + j];
      float acc = 0;
      if (NORM_LOOPED) {
        for (uint b = 0; b < (uint)D; b += 1024u * 4u) {
          if (b + lid * 4u + 4u <= (uint)D) {
            for (int i = 0; i < 4; i++) {
              uint d = b + lid * 4u + i;
              float s = 0.0f;
              for (int j = 0; j < HC; j++) { float t = p[j] * float(xr[j * D + d]); s = t + s; }
              float xi = float(static_cast<T>(s));
              acc += xi * xi;
            }
          } else {
            for (int i = 0; i < 4; i++) {
              uint d = b + lid * 4u + i;
              if (d < (uint)D) {
                float s = 0.0f;
                for (int j = 0; j < HC; j++) { float t = p[j] * float(xr[j * D + d]); s = t + s; }
                float xi = float(static_cast<T>(s));
                acc += xi * xi;
              }
            }
          }
        }
      } else {
        float tx[4];
        for (int i = 0; i < 4; i++) {
          uint d = lid * 4u + i;
          float s = 0.0f;
          if (d < (uint)D) {
            for (int j = 0; j < HC; j++) { float t = p[j] * float(xr[j * D + d]); s = t + s; }
          }
          tx[i] = (d < (uint)D) ? float(static_cast<T>(s)) : 0.0f;
          acc += tx[i] * tx[i];
        }
      }
      acc = simd_sum(acc);
      if (sg == 0) tmem[lane] = 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0) tmem[sg] = acc;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg == 0) {
        acc = simd_sum(tmem[lane]);
        if (lane == 0) tmem[32] = metal::precise::rsqrt(acc / (float)D + NORM_EPS);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float inv = tmem[32];
      for (uint d = lid; d < (uint)D; d += 1024u) {
        float s = 0.0f;
        for (int j = 0; j < HC; j++) { float t = p[j] * float(xr[j * D + d]); s = t + s; }
        T xv = static_cast<T>(s);
        xin[(size_t)r * D + d] = nw[d] * static_cast<T>(xv * inv);
      }
"""

_HC_STAGE_SRC = r"""
    const uint tg = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device T* xr = x + (size_t)r * (HC * D);
    threadgroup float tmem[1024];
    if (tg == 0) {
""" + _HC_COLLAPSE_NORM_SRC + r"""    } else if (tg == 1) {
      // sum(x^2) over the HC * D values: MLX row_reduce_looped, one threadgroup of RED_TGS threads.
      float total = 0.0f;
      if (lid < (uint)RED_TGS) {
        const int n = HC * D;
        const int blocks = n / (RED_TGS * 4);
        const int extra = n - blocks * (RED_TGS * 4);
        long off = lid * 4;
        for (int b = 0; b < blocks; b++) {
          for (int i = 0; i < 4; i++) { float v = float(xr[off + i]); total = (v * v) + total; }
          off += RED_TGS * 4;
        }
        const int index = lid * 4;
        if (index + 4 <= extra) {
          for (int i = 0; i < 4; i++) { float v = float(xr[off + i]); total = (v * v) + total; }
        } else {
          for (int i = 0; index + i < extra; i++) { float v = float(xr[off + i]); total = (v * v) + total; }
        }
      }
      total = simd_sum(total);
      if (lane == 0) tmem[sg] = total;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint nsg = (RED_TGS + 31) / 32;
      total = lid < nsg ? tmem[lid] : 0.0f;
      total = simd_sum(total);
      if (lid == 0) part[r] = total;
    } else {
      // MLX gemv BM1 BN8 SM1 SN32 TM4 TN4: one MLX threadgroup (8 simdgroups) per real one, so the
      // 6 output tiles stream fn on 6 cores as MLX does; simdgroups 8.. idle.
      const uint v = sg / 8u, sgN = sg % 8u;
      const int out_row = (v == 0) ? (int)(tg - 2) * 4 : MIX;
      float res[4] = {0, 0, 0, 0};
      if (out_row < MIX) {
        int bn = (int)(32u * sgN + lane) * 4;
        const device float* mat = fn + (size_t)out_row * (HC * D);
        const int n_iter = (HC * D) / 1024;
        for (int i = 0; i < n_iter; ++i) {
          float vc[4];
          for (int tn = 0; tn < 4; tn++) vc[tn] = float(xr[bn + tn]);
          int mo = 0;
          for (int tm = 0; tm < 4; tm++) {
            float inter[4];
            for (int tn = 0; tn < 4; tn++) inter[tn] = mat[mo + bn + tn];
            for (int tn = 0; tn < 4; tn++) res[tm] += inter[tn] * vc[tn];
            mo += HC * D;
          }
          bn += 1024;
        }
        const int leftover = (HC * D) - n_iter * 1024;
        if (leftover > 0) {
          float vc[4];
          for (int tn = 0; tn < 4; tn++) vc[tn] = (bn + tn < HC * D) ? float(xr[bn + tn]) : 0.0f;
          for (int tm = 0; tm < 4; tm++) {
            float inter[4];
            for (int tn = 0; tn < 4; tn++) inter[tn] = (bn + tn < HC * D) ? mat[tm * (HC * D) + bn + tn] : 0.0f;
            for (int tn = 0; tn < 4; tn++) res[tm] += inter[tn] * vc[tn];
          }
        }
        for (int tm = 0; tm < 4; tm++)
          for (ushort sn = 16; sn >= 1; sn >>= 1) res[tm] += simd_shuffle_down(res[tm], sn);
      }
      // tgp reduction across the 8 K-split simdgroups (tgp_results = base + sgN * (blockM + TM))
      threadgroup float* tr = tmem + v * 64u + sgN * 8u;
      if (lane == 0) for (int tm = 0; tm < 4; tm++) tr[tm] = res[tm];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0 && sgN == 0 && out_row < MIX) {
        for (int sgn = 1; sgn < 8; sgn++)
          for (int tm = 0; tm < 4; tm++) res[tm] += tmem[v * 64u + sgn * 8u + tm];
        for (int tm = 0; tm < 4; tm++) dots[(size_t)r * MIX + out_row + tm] = res[tm];
      }
    }
"""

_HC_SPLIT2_SRC = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint r = threadgroup_position_in_grid.y;
    const float total = part[r];
    if (lane != 0) return;
    const float mean = total * INV_N;
    const float rs = metal::precise::rsqrt(mean + RMS_EPS);
    float m[24];
    for (int j = 0; j < 24; j++) m[j] = dots[(size_t)r * 24 + j] * rs;
    const uint t = r;
    float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    for (int j = 0; j < 4; j++) {
        pre[t * 4 + j] = 1.0f / (1.0f + metal::precise::exp(-(m[j] * s0 + base[j]))) + EPS;
        post[t * 4 + j] = 2.0f / (1.0f + metal::precise::exp(-(m[4 + j] * s1 + base[4 + j])));
    }
    float c[16];
    for (int i = 0; i < 16; i++) c[i] = m[8 + i] * s2 + base[8 + i];
    for (int rr = 0; rr < 4; rr++) {
        float mx_ = c[rr * 4];
        for (int k = 1; k < 4; k++) mx_ = metal::max(mx_, c[rr * 4 + k]);
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) { c[rr * 4 + k] = metal::precise::exp(c[rr * 4 + k] - mx_); sum += c[rr * 4 + k]; }
        for (int k = 0; k < 4; k++) c[rr * 4 + k] = c[rr * 4 + k] / sum + EPS;
    }
    for (int k = 0; k < 4; k++) {
        float cs = c[k] + c[4 + k] + c[8 + k] + c[12 + k];
        for (int rr = 0; rr < 4; rr++) c[rr * 4 + k] = c[rr * 4 + k] / (cs + EPS);
    }
    for (int it = 0; it < ITERS - 1; it++) {
        for (int rr = 0; rr < 4; rr++) {
            float rsum = c[rr * 4] + c[rr * 4 + 1] + c[rr * 4 + 2] + c[rr * 4 + 3];
            for (int k = 0; k < 4; k++) c[rr * 4 + k] = c[rr * 4 + k] / (rsum + EPS);
        }
        for (int k = 0; k < 4; k++) {
            float cs = c[k] + c[4 + k] + c[8 + k] + c[12 + k];
            for (int rr = 0; rr < 4; rr++) c[rr * 4 + k] = c[rr * 4 + k] / (cs + EPS);
        }
    }
    for (int i = 0; i < 16; i++) comb[t * 16 + i] = c[i];
"""

_HC_POST_SRC = r"""
    const uint d = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (d >= (uint)D) return;
    float hv = float(h[(size_t)r * D + d]);
    float rv[4];
    for (int j = 0; j < 4; j++) rv[j] = float(res[((size_t)r * 4 + j) * D + d]);
    for (int k = 0; k < 4; k++) {
      float c[4];
      for (int j = 0; j < 4; j++) c[j] = TF32 ? as_type<float>(as_type<uint>(comb[r * 16 + j * 4 + k]) & 0xFFFFE000u)
                                              : comb[r * 16 + j * 4 + k];
      float p0 = c[0] * rv[0], p1 = c[1] * rv[1], p2 = c[2] * rv[2], p3 = c[3] * rv[3];
      float e = (p0 + p1) + (p2 + p3);
      float t = post[r * 4 + k] * hv;
      out[((size_t)r * 4 + k) * D + d] = static_cast<T>(t + e);
    }
"""


def _red_tgs(n: int) -> int:
    """MLX threadgroup_size_from_row_size: the row_reduce_looped width for a 1-row sum of n values."""
    if n <= 512:
        return 32
    if n <= 1024:
        return 128
    return min(1024, ((n + 3) // 4 + 31) // 32 * 32)


@lru_cache(maxsize=None)
def _hc_kernels(norm_eps: float, rms_eps: float, hc_eps: float, iters: int, n: int):
    lit = lambda v: f"{float(v):.9e}f"  # noqa: E731
    stage_src = _HC_STAGE_SRC.replace("NORM_EPS", lit(norm_eps)).replace("RED_TGS", str(_red_tgs(n)))
    stage = mx.fast.metal_kernel(name=f"dsv41_hc_stage_{n}", input_names=["x", "pre", "fn", "nw"],
                                 output_names=["xin", "part", "dots"], source=stage_src)
    split_src = (_HC_SPLIT2_SRC.replace("INV_N", lit(1.0 / n)).replace("RMS_EPS", lit(rms_eps))
                 .replace("ITERS", str(int(iters))).replace("EPS", lit(hc_eps)))
    split = mx.fast.metal_kernel(name=f"dsv41_hc_split2_{iters}", input_names=["part", "dots", "scale", "base"],
                                 output_names=["pre", "post", "comb"], source=split_src)
    post = mx.fast.metal_kernel(name="dsv41_hc_post", input_names=["h", "res", "post", "comb"],
                                output_names=["out"], source=_HC_POST_SRC)
    return stage, split, post


@lru_cache(maxsize=1)
def _gemm_rounds_to_tf32() -> bool:
    """Whether this device's f32 GEMMs truncate operands to TF32. M5 matrix units do (MLX default);
    other GPUs accumulate in another order, and the hc expand kernel then cannot be bit-equal (the
    runtime self-check keeps the stock ops there). Probed with the expand einsum's own shape."""
    comb = mx.full((1, 1, 4, 4), 1.0 + 2.0 ** -12, mx.float32)  # not representable in TF32
    res = mx.ones((1, 1, 4, 256), mx.float32)
    y = mx.einsum("bsjk,bsjd->bskd", comb, res)
    return bool((y[0, 0, 0, 0] == 4.0).item())


def _tf32() -> bool:
    import os
    if os.environ.get("MLX_ENABLE_TF32", "1").strip() in {"0", "false", "False"}:
        return False
    return _gpu() and _gemm_rounds_to_tf32()


def hc_stage_decode(x, pre, fn, scale, base, norm_w, norm_eps, rms_eps, hc_eps, iters):
    """x [R, 4, D] bf16 stream; pre [R, 4] f32 (the previous sublayer's pre). Returns (xin [R, D] bf16 =
    RMSNorm(hc_pre(x, pre)), pre', post', comb' of hc_mixes(x)), or None if not eligible."""
    if x.ndim != 3 or x.shape[1] != 4 or x.dtype != mx.bfloat16 or not _gpu():
        return None
    R, hc, D = x.shape
    n = hc * D
    if (R > MAX_ROWS or fn.shape != (24, n) or fn.dtype != mx.float32 or norm_w.dtype != mx.bfloat16
            or norm_w.shape != (D,) or n % 4 or n < 16 * 24 or D > 1024 * 4 * 64):
        return None
    stage, split, _ = _hc_kernels(float(norm_eps), float(rms_eps), float(hc_eps), int(iters), n)
    xin, part, dots = stage(inputs=[x, pre.astype(mx.float32), fn, norm_w], template=[("T", mx.bfloat16), ("D", D),
                            ("HC", hc), ("MIX", 24), ("NORM_LOOPED", D > 4096)],
                            grid=(1024 * (2 + 24 // 4), R, 1), threadgroup=(1024, 1, 1),
                            output_shapes=[(R, D), (R, 1), (R, 24)],
                            output_dtypes=[mx.bfloat16, mx.float32, mx.float32], init_value=0)
    p, po, cb = split(inputs=[part, dots, scale.astype(mx.float32), base.astype(mx.float32)],
                      grid=(32, R, 1), threadgroup=(32, 1, 1),
                      output_shapes=[(R, 4), (R, 4), (R, 16)], output_dtypes=[mx.float32] * 3)
    return xin, p, po, cb.reshape(R, 4, 4)


def hc_post_decode(h, residual, post, comb):
    """h [R, D] bf16, residual [R, 4, D] bf16, post [R, 4], comb [R, 4, 4] -> [R, 4, D] bf16, or None."""
    if h.ndim != 2 or residual.ndim != 3 or residual.shape[1] != 4 or not _gpu() or h.dtype != mx.bfloat16:
        return None
    R, D = h.shape
    _, _, post_k = _hc_kernels(0.0, 0.0, 0.0, 1, 4 * D)
    return post_k(inputs=[h, residual, post.astype(mx.float32), comb.reshape(R, 16).astype(mx.float32)],
                  template=[("T", mx.bfloat16), ("D", D), ("TF32", _tf32())],
                  grid=(D, R, 1), threadgroup=(256, 1, 1), output_shapes=[(R, 4, D)], output_dtypes=[mx.bfloat16])[0]


# ----------------------------------------------------------------------------- hc boundary (2 dispatches)
# The boundary layout of TensorFold's DeepSeek-V4.1 Metal family (BobClawblaw/TensorFold, deepseek-v41-main,
# zig/kernels/metal/deepseek/v41.metal ds_hc_expand_mix / ds_hc_boundary; Apache-2.0): per sublayer
#   hc_mix:      (optionally) the pending branch expanded into the streams, then the 24 mixes' and the sum of
#                squares' partial sums, a threadgroup a 1024-value slice (20 for D = 5120): fn is read by
#                20 cores instead of hc_stage's 6.
#   hc_boundary: one threadgroup of 1024 a row: the partials' totals; the collapse with the previous pre and
#                its RMSNorm (_HC_COLLAPSE_NORM_SRC, as hc_stage); the split (pre, post, Sinkhorn comb) on
#                simdgroup 0, a comb cell a lane.
# Bit-identical to hc_stage + hc_split + hc_post except the 24 mix dot products and the sum of squares,
# whose partial sums run in another order (MLX's gemv / row reduction cannot be split across cores):
# the expand is hc_post's arithmetic, the collapse + norm is hc_stage's code, and the split sums every row
# and column of the Sinkhorn in _HC_SPLIT2_SRC's order (shuffles feed each lane its row / column in order).

# The pending branch expanded into the streams (hc_post's arithmetic: comb TF32-truncated, products summed
# pairwise, post * h added, cast to T), its values kept for the mixes; or the streams as they are.
_HC_MIX_EXPAND = r"""
    {
      const uint k = col / (uint)D, d0 = col % (uint)D;
      float c[4];
      for (int jj = 0; jj < 4; jj++) c[jj] = TF32 ? as_type<float>(as_type<uint>(comb[r * 16 + jj * 4 + k]) & 0xFFFFE000u)
                                                  : comb[r * 16 + jj * 4 + k];
      const float pk = post[r * 4 + k];
      for (int i = 0; i < 4; i++) {
        const uint d = d0 + i;
        const float hv = float(h[(size_t)r * D + d]);
        float rv[4];
        for (int jj = 0; jj < 4; jj++) rv[jj] = float(x[((size_t)r * 4 + jj) * D + d]);
        float p0 = c[0] * rv[0], p1 = c[1] * rv[1], p2 = c[2] * rv[2], p3 = c[3] * rv[3];
        float e = (p0 + p1) + (p2 + p3);
        float tt = pk * hv;
        const T o = static_cast<T>(tt + e);
        xo[(size_t)r * (4 * D) + col + i] = o;
        v[i] = float(o);
      }
    }
"""
_HC_MIX_LOAD = r"""
    for (int i = 0; i < 4; i++) v[i] = float(x[(size_t)r * (4 * D) + col + i]);
"""

_HC_MIX_SRC = r"""
    const uint j = threadgroup_position_in_grid.x, r = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[8][MIX + 1];
    const uint col = 1024u * j + 4u * t;
    float v[4];
""" + "{BRANCH}" + r"""
    float acc[MIX + 1];
    acc[MIX] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    for (int m = 0; m < MIX; m++) {
      const device float* f = fn + (size_t)m * (4 * D) + col;
      acc[m] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
    }
    for (int m = 0; m <= MIX; m++) {
      const float s = simd_sum(acc[m]);
      if (lane == 0) red[sg][m] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t <= (uint)MIX) {
      float s = 0.0f;
      for (int q = 0; q < 8; q++) s += red[q][t];
      part[((size_t)r * SLICES + j) * (MIX + 1) + t] = s;
    }
"""

_HC_BOUNDARY_SRC = r"""
    const uint r = threadgroup_position_in_grid.y;
    const uint lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const device T* xr = x + (size_t)r * (HC * D);
    threadgroup float tmem[1024];
    threadgroup float tot[MIX + 1];
    if (lid <= (uint)MIX) {
      float s = 0.0f;
      for (int j = 0; j < SLICES; j++) s += part[((size_t)r * SLICES + j) * (MIX + 1) + lid];
      tot[lid] = s;
    }
""" + _HC_COLLAPSE_NORM_SRC + r"""
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      const float rs = metal::precise::rsqrt(tot[MIX] * INV_N + RMS_EPS);
      const float s0 = scale[0], s1 = scale[1], s2 = scale[2];
      if (lane < 4u) {
        pre_out[r * 4 + lane] = 1.0f / (1.0f + metal::precise::exp(-((tot[lane] * rs) * s0 + base[lane]))) + EPS;
        post_out[r * 4 + lane] = 2.0f / (1.0f + metal::precise::exp(-((tot[4 + lane] * rs) * s1 + base[4 + lane])));
      }
      const ushort cell = ushort(lane & 15u), row = cell & ~ushort(3), colk = cell & ushort(3);
      float c = (tot[8 + cell] * rs) * s2 + base[8 + cell];
      // row softmax: max and sum over the row in order
      float mx_ = simd_shuffle(c, row);
      for (ushort k = 1; k < 4; k++) mx_ = metal::max(mx_, simd_shuffle(c, ushort(row + k)));
      c = metal::precise::exp(c - mx_);
      float sum = 0.0f;
      for (ushort k = 0; k < 4; k++) sum += simd_shuffle(c, ushort(row + k));
      c = c / sum + EPS;
      float cs = simd_shuffle(c, colk);
      for (ushort k = 1; k < 4; k++) cs = cs + simd_shuffle(c, ushort(colk + 4 * k));
      c = c / (cs + EPS);
      for (int it = 0; it < ITERS - 1; it++) {
        float rsum = simd_shuffle(c, row);
        for (ushort k = 1; k < 4; k++) rsum = rsum + simd_shuffle(c, ushort(row + k));
        c = c / (rsum + EPS);
        cs = simd_shuffle(c, colk);
        for (ushort k = 1; k < 4; k++) cs = cs + simd_shuffle(c, ushort(colk + 4 * k));
        c = c / (cs + EPS);
      }
      if (lane < 16u) comb[r * 16 + lane] = c;
    }
"""


@lru_cache(maxsize=None)
def _hc_boundary_kernels(norm_eps: float, rms_eps: float, hc_eps: float, iters: int, n: int):
    lit = lambda v: f"{float(v):.9e}f"  # noqa: E731
    mix = {b: mx.fast.metal_kernel(name=f"dsv41_hc_mix_{int(b)}",
                                   input_names=["x", "fn", "h", "post", "comb"] if b else ["x", "fn"],
                                   output_names=["xo", "part"] if b else ["part"],
                                   source=_HC_MIX_SRC.replace("{BRANCH}", _HC_MIX_EXPAND if b else _HC_MIX_LOAD))
           for b in (False, True)}
    src = (_HC_BOUNDARY_SRC.replace("NORM_EPS", lit(norm_eps)).replace("INV_N", lit(1.0 / n))
           .replace("RMS_EPS", lit(rms_eps)).replace("ITERS", str(int(iters))).replace("EPS", lit(hc_eps)))
    boundary = mx.fast.metal_kernel(name=f"dsv41_hc_boundary_{iters}",
                                    input_names=["x", "part", "pre", "scale", "base", "nw"],
                                    output_names=["xin", "pre_out", "post_out", "comb"], source=src)
    return mix, boundary


def hc_mix_decode(x, fn, h=None, post=None, comb=None):
    """x [R, 4, D] bf16 streams (with h [R, D], post [R, 4], comb [R, 4, 4]: the pending branch expanded into
    them first, hc_post_decode's arithmetic). Returns (streams, part [R, slices, 25]) or None if not eligible."""
    if x.ndim != 3 or x.shape[1] != 4 or x.dtype != mx.bfloat16 or not _gpu():
        return None
    R, hc, D = x.shape
    n = hc * D
    if R > MAX_ROWS or D % 256 or fn.shape != (24, n) or fn.dtype != mx.float32:
        return None
    if h is not None and (h.shape != (R, D) or h.dtype != mx.bfloat16):
        return None
    mix, _ = _hc_boundary_kernels(0.0, 0.0, 0.0, 1, n)
    slices = n // 1024
    tmpl = [("T", mx.bfloat16), ("D", D), ("MIX", 24), ("SLICES", slices), ("TF32", _tf32())]
    shapes, dtypes = [(R, slices, 25)], [mx.float32]
    if h is None:
        return x, mix[False](inputs=[x, fn], template=tmpl, grid=(256 * slices, R, 1), threadgroup=(256, 1, 1),
                             output_shapes=shapes, output_dtypes=dtypes)[0]
    xo, part = mix[True](inputs=[x, fn, h, post.astype(mx.float32), comb.reshape(R, 16).astype(mx.float32)],
                         template=tmpl, grid=(256 * slices, R, 1), threadgroup=(256, 1, 1),
                         output_shapes=[(R, 4, D)] + shapes, output_dtypes=[mx.bfloat16] + dtypes)
    return xo, part


def hc_boundary_decode(x, part, pre, scale, base, norm_w, norm_eps, rms_eps, hc_eps, iters):
    """x [R, 4, D] bf16 streams, part from hc_mix_decode, pre [R, 4] the previous sublayer's pre. Returns
    (xin [R, D] = RMSNorm(hc_pre(x, pre)), pre', post', comb' [R, 4, 4]) as hc_stage_decode does."""
    R, hc, D = x.shape
    n = hc * D
    if norm_w.shape != (D,) or norm_w.dtype != mx.bfloat16:
        return None
    _, boundary = _hc_boundary_kernels(float(norm_eps), float(rms_eps), float(hc_eps), int(iters), n)
    xin, p, po, cb = boundary(
        inputs=[x, part, pre.astype(mx.float32), scale.astype(mx.float32), base.astype(mx.float32), norm_w],
        template=[("T", mx.bfloat16), ("D", D), ("HC", hc), ("MIX", 24), ("SLICES", n // 1024),
                  ("NORM_LOOPED", D > 4096)],
        grid=(1024, R, 1), threadgroup=(1024, 1, 1),
        output_shapes=[(R, D), (R, 4), (R, 4), (R, 16)], output_dtypes=[mx.bfloat16] + [mx.float32] * 3)
    return xin, p, po, cb.reshape(R, 4, 4)
