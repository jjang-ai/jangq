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


def moe_decode(x, idx, w, experts, shared, swiglu_limit: float, group_size: int = 64):
    """Routed + shared expert MoE for R <= MAX_ROWS rows in two dispatches, or None if not eligible.

    x [R, D] bf16; idx [R, TOPK] expert ids (argpartition order = summation order); w [R, TOPK] f32
    route weights (already normalized and scaled). experts: SwitchMLP (gate_proj/up_proj/down_proj
    SwitchProj); shared: SharedExpert (w1/w2/w3 QuantizedLinear). Returns y [R, D] bf16."""
    if x.ndim != 2 or idx.ndim != 2 or not _gpu():
        return None
    R, D = x.shape
    topk = idx.shape[1]
    g, u, d = experts.gate_proj, experts.up_proj, experts.down_proj
    F = g.weight.shape[1]
    sw = (shared.w1, shared.w3, shared.w2)
    if (R > MAX_ROWS or x.dtype != mx.bfloat16 or g.bits != u.bits or g.bits not in (2, 3, 4, 8)
            or d.bits not in (2, 3, 4, 8) or any(m.bits != shared.w1.bits or m.group_size != group_size for m in sw)
            or shared.w1.bits not in (2, 3, 4, 8) or F % 8 or D % 4 or F % group_size or D % group_size or g.scales.dtype != mx.bfloat16 or shared.w1.scales.dtype != mx.bfloat16
            or d.weight.shape[1] != D or shared.w1.weight.shape[0] != F or shared.w2.weight.shape[0] != D):
        return None
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
