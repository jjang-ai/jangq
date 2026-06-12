"""Opt-in MPP/NAX TensorOps lane for TurboQuant dense matmul.

Unlike ``mpp_dense_kernel``, this path does not materialize a dense weight
matrix. It unpacks JANGTQ codebook values directly into cooperative TensorOps
fragments and runs a 16x32x16 MPP matmul tile.
"""

from __future__ import annotations

from functools import lru_cache
import os
import weakref

import mlx.core as mx
import numpy as np

from .hadamard_kernel import hadamard_rotate_metal


_MPP_NAX_ON_MODES = {"1", "true", "yes", "on"}
_MPP_NAX_OFF_MODES = {"0", "false", "no", "off", "disable", "disabled"}


def _env_mode(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower()


def mpp_nax_prefill_mode() -> str:
    """Return the effective MPP/NAX mode for sorted routed prefill.

    Decode and non-sorted gather remain opt-in through ``JANGTQ_MPP_NAX``.
    Sorted routed prefill is the shape that can fill 16-row TensorOps tiles, so
    it defaults to ``auto`` unless explicitly disabled.
    """
    explicit = _env_mode("JANGTQ_MPP_NAX_PREFILL")
    if explicit is not None:
        return "" if explicit in _MPP_NAX_OFF_MODES else explicit

    global_mode = _env_mode("JANGTQ_MPP_NAX")
    if global_mode in _MPP_NAX_OFF_MODES:
        return ""
    if global_mode:
        return global_mode
    return "auto"


def mpp_nax_mode_allows(
    mode: str,
    dispatches: int,
    in_features: int,
    out_features: int,
) -> bool:
    """Return whether an MPP/NAX mode should run for this routed shape."""
    if mode in _MPP_NAX_ON_MODES:
        return True
    if mode != "auto":
        return False
    if dispatches >= 512:
        return True
    return dispatches >= 256 and (int(in_features) * int(out_features)) >= (2048 * 2048)


_MPP_NAX_HEADER = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
"""


@lru_cache(maxsize=1)
def mpp_nax_tensorops_available() -> bool:
    """Return whether the MPP/NAX TensorOps lane is allowed to try dispatch.

    This must not run a smoke Metal command. vMLX queries it from /health after
    loading large mmap-backed models, and executing an unrelated probe on the
    live MLX stream can wedge the server before the first user generation. The
    real kernels still compile/dispatch on demand; callers catch failures and
    fall back unless JANGTQ_MPP_NAX_STRICT=1 is set.
    """
    if os.environ.get("JANGTQ_MPP_NAX_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    return bool(getattr(getattr(mx, "fast", None), "metal_kernel", None))


@lru_cache(maxsize=64)
def _make_tq_mpp_nax_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    bits: int,
):
    # Tile accesses stay in-bounds via the bounds-checked fills below (each
    # lane writes 0 into the threadgroup tile for any (k, m)/(n, k) coordinate
    # past in_features/batch_size/out_features), so no caller-side padding of
    # x_rot or out is required.
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    # Each SIMD group handles one (tile_m, tile_n) output tile.
    # Dequantize the 16x32 weight K-strip into threadgroup half memory, then
    # use tensor() with MPP coordinate order (K,M),(N,K),(N,M).
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_m = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint m0 = tile_m * 16u;
uint lane = thread_index_in_threadgroup;

threadgroup half tg_a[16 * 16];
threadgroup half tg_b[16 * 32];
threadgroup float tg_c[32 * 16];

// Zero-init C tile in threadgroup
for (uint i = lane; i < 32u * 16u; i += 32u)
  tg_c[i] = 0.0f;

const uint vals_per_u32 = 32u / {bits}u;
const uint mask = (1u << {bits}u) - 1u;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  // Fill A tile: layout tg_a[k + m*16] matches tensor stride {1, 16} = (K,M)
  // 32 lanes fill 256 elements: each lane fills 8 (256/32)
  for (uint idx = lane; idx < 16u * 16u; idx += 32u) {{
    uint k_local = idx / 16u;
    uint m_local = idx % 16u;
    uint k = k0 + k_local;
    uint m = m0 + m_local;
    half val = half(0);
    if (k < {in_features}u && m < {batch_size}u)
      val = x_rot[m * {in_features}u + k];
    tg_a[k_local + m_local * 16u] = val;
  }}

  // Fill B tile: tg_b[k_local * 32 + n_local] = dequant(expert, k0+k_local, n0+n_local)
  // 32 lanes fill 512 elements: each lane fills 16
  for (uint k_local = 0u; k_local < 16u; k_local++) {{
    uint k = k0 + k_local;
    uint nc = n0 + lane;
    half val = half(0);
    if (k < {in_features}u && nc < {out_features}u) {{
      uint pack_idx = k / vals_per_u32;
      uint bit_offset = (k % vals_per_u32) * {bits}u;
      uint pv = packed[nc * {packed_cols}u + pack_idx];
      uint cb_idx = (pv >> bit_offset) & mask;
      val = static_cast<half>(codebook[cb_idx] * static_cast<float>(norms[nc]));
    }}
    tg_b[k_local * 32u + lane] = val;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // tensor() coordinate order: (K,M), (N,K), (N,M) — same as mpp_dense_kernel
  constexpr auto desc = tensor_ops::matmul2d_descriptor(
      16, 32, 16, false, false, false,
      tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  tensor_ops::matmul2d<desc, execution_simdgroup> op;
  auto X = tensor(tg_a, extents<int, 16, 16>(), array<int, 2>{{1, 16}});
  auto W = tensor(tg_b, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto O = tensor(tg_c, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  op.run(X, W, O);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// Scatter C tile from threadgroup to device output
for (uint idx = lane; idx < 32u * 16u; idx += 32u) {{
  uint n_local = idx / 16u;
  uint m_local = idx % 16u;
  uint m = m0 + m_local;
  uint nc = n0 + n_local;
  if (m < {batch_size}u && nc < {out_features}u)
    out[m * {out_features}u + nc] = tg_c[n_local + m_local * 32u];
}}
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_mpp_nax_b{batch_size}_i{in_features}_o{out_features}_q{bits}",
        input_names=["x_rot", "packed", "norms", "codebook"],
        output_names=["out"],
        header=_MPP_NAX_HEADER,
        source=source,
    )


def tq_matmul_mpp_nax(
    x: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    signs: mx.array,
    in_features: int,
    bits: int,
) -> mx.array:
    """Run TQ matmul by unpacking codebook values into MPP NAX fragments."""
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported JANGTQ bits for MPP NAX: {bits}")
    if not mpp_nax_tensorops_available():
        raise RuntimeError("MPP NAX tensor_ops unavailable for MLX custom kernels")

    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True

    orig_shape = x.shape
    if x.ndim > 2:
        x_flat = x.reshape(-1, in_features)
    else:
        x_flat = x

    batch_size = int(x_flat.shape[0])
    out_features = int(packed.shape[0])
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs).astype(mx.float16)
    kernel = _make_tq_mpp_nax_kernel(
        batch_size, in_features, out_features, int(bits)
    )
    n_tiles = (out_features + 31) // 32
    m_tiles = (batch_size + 15) // 16
    out = kernel(
        inputs=[x_rot, packed, norms, codebook],
        output_shapes=[(batch_size, out_features)],
        output_dtypes=[mx.float32],
        grid=(n_tiles * 32, m_tiles, 1),
        threadgroup=(32, 1, 1),
    )[0]

    if x.ndim > 2:
        out = out.reshape(*orig_shape[:-1], out_features)
    if squeeze:
        out = out.squeeze(0)
    if out.dtype != x.dtype:
        out = out.astype(x.dtype)
    return out


@lru_cache(maxsize=64)
def _make_gather_tq_mpp_nax_kernel(
    in_features: int,
    out_features: int,
    bits: int,
):
    # Single-row dispatch: M=1 live row per tile, padded to M=16 with zeros.
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint dispatch_idx = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint lane = thread_index_in_threadgroup;
uint expert = rhs_indices[dispatch_idx];
uint expert_base = expert * {out_features}u * {packed_cols}u;
uint norm_base = expert * {out_features}u;

threadgroup half tg_a[16 * 16];
threadgroup half tg_b[16 * 32];
threadgroup float tg_c[32 * 16];

// Zero-init C tile
for (uint i = lane; i < 32u * 16u; i += 32u)
  tg_c[i] = 0.0f;

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, execution_simdgroup> op;

const uint vals_per_u32 = 32u / {bits}u;
const uint mask = (1u << {bits}u) - 1u;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  // Fill A tile: only row 0 (m_local=0) has real data; rest are zero
  for (uint idx = lane; idx < 16u * 16u; idx += 32u) {{
    uint k_local = idx / 16u;
    uint m_local = idx % 16u;
    uint k = k0 + k_local;
    half val = half(0);
    if (m_local == 0u && k < {in_features}u)
      val = x_rot[dispatch_idx * {in_features}u + k];
    tg_a[k_local + m_local * 16u] = val;
  }}

  // Fill B tile: dequantize expert weights for this K-strip x N-strip
  for (uint k_local = 0u; k_local < 16u; k_local++) {{
    uint k = k0 + k_local;
    uint nc = n0 + lane;
    half val = half(0);
    if (k < {in_features}u && nc < {out_features}u) {{
      uint pack_idx = k / vals_per_u32;
      uint bit_offset = (k % vals_per_u32) * {bits}u;
      uint pv = packed[expert_base + nc * {packed_cols}u + pack_idx];
      uint cb_idx = (pv >> bit_offset) & mask;
      val = static_cast<half>(codebook[cb_idx] * static_cast<float>(norms[norm_base + nc]));
    }}
    tg_b[k_local * 32u + lane] = val;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto X = tensor(tg_a, extents<int, 16, 16>(), array<int, 2>{{1, 16}});
  auto W = tensor(tg_b, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto O = tensor(tg_c, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  op.run(X, W, O);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// Scatter row 0 of C tile to device output (only m_local=0 row is valid)
for (uint n_local = lane; n_local < 32u; n_local += 32u) {{
  uint nc = n0 + n_local;
  if (nc < {out_features}u)
    out[dispatch_idx * {out_features}u + nc] = tg_c[n_local + 0u * 32u];
}}
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_gather_mpp_nax_i{in_features}_o{out_features}_q{bits}",
        input_names=["x_rot", "packed", "norms", "codebook", "rhs_indices"],
        output_names=["out"],
        header=_MPP_NAX_HEADER,
        source=source,
    )


def gather_tq_matmul_mpp_nax_from_rot(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    rhs_indices: mx.array,
    in_features: int,
    out_features: int,
    bits: int,
) -> mx.array:
    """Run routed TQ gather using one 16x32x16 NAX tile per dispatch row.

    Each dispatch row can select a different expert, so this intentionally uses
    only one live M row in the 16-row TensorOps tile. Later optimized variants
    can bucket rows by expert to use all M rows.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported JANGTQ bits for routed MPP NAX: {bits}")
    if not mpp_nax_tensorops_available():
        raise RuntimeError("MPP NAX tensor_ops unavailable for MLX custom kernels")

    n_dispatches = int(x_rot.shape[0])
    kernel = _make_gather_tq_mpp_nax_kernel(in_features, out_features, int(bits))
    n_tiles = (out_features + 31) // 32
    out = kernel(
        inputs=[x_rot.astype(mx.float16), packed, norms, codebook, rhs_indices.astype(mx.uint32)],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(n_tiles * 32, n_dispatches, 1),
        threadgroup=(32, 1, 1),
    )[0]
    return out


@lru_cache(maxsize=64)
def _make_grouped_gather_tq_mpp_nax_kernel(
    in_features: int,
    out_features: int,
    bits: int,
):
    # Grouped sorted dispatch: up to M=16 rows per tile, same expert.
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_id = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint lane = thread_index_in_threadgroup;
uint tile_start = tile_starts[tile_id];
uint tile_count = tile_counts[tile_id];
uint expert = tile_experts[tile_id];
uint expert_base = expert * {out_features}u * {packed_cols}u;
uint norm_base = expert * {out_features}u;

threadgroup half tg_a[16 * 16];
threadgroup half tg_b[16 * 32];
threadgroup float tg_c[32 * 16];

for (uint i = lane; i < 32u * 16u; i += 32u)
  tg_c[i] = 0.0f;

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, execution_simdgroup> op;

const uint vals_per_u32 = 32u / {bits}u;
const uint mask = (1u << {bits}u) - 1u;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  for (uint idx = lane; idx < 16u * 16u; idx += 32u) {{
    uint k_local = idx / 16u;
    uint m_local = idx % 16u;
    uint k = k0 + k_local;
    half val = half(0);
    if (k < {in_features}u && m_local < tile_count)
      val = x_rot[(tile_start + m_local) * {in_features}u + k];
    tg_a[k_local + m_local * 16u] = val;
  }}

  for (uint k_local = 0u; k_local < 16u; k_local++) {{
    uint k = k0 + k_local;
    uint nc = n0 + lane;
    half val = half(0);
    if (k < {in_features}u && nc < {out_features}u) {{
      uint pack_idx = k / vals_per_u32;
      uint bit_offset = (k % vals_per_u32) * {bits}u;
      uint pv = packed[expert_base + nc * {packed_cols}u + pack_idx];
      uint cb_idx = (pv >> bit_offset) & mask;
      val = static_cast<half>(codebook[cb_idx] * static_cast<float>(norms[norm_base + nc]));
    }}
    tg_b[k_local * 32u + lane] = val;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto X = tensor(tg_a, extents<int, 16, 16>(), array<int, 2>{{1, 16}});
  auto W = tensor(tg_b, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto O = tensor(tg_c, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  op.run(X, W, O);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// Scatter valid rows from C tile to device output
for (uint idx = lane; idx < 32u * 16u; idx += 32u) {{
  uint n_local = idx / 16u;
  uint m_local = idx % 16u;
  uint nc = n0 + n_local;
  if (m_local < tile_count && nc < {out_features}u)
    out[(tile_start + m_local) * {out_features}u + nc] = tg_c[n_local + m_local * 32u];
}}
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_grouped_gather_mpp_nax_i{in_features}_o{out_features}_q{bits}",
        input_names=[
            "x_rot",
            "packed",
            "norms",
            "codebook",
            "tile_starts",
            "tile_counts",
            "tile_experts",
        ],
        output_names=["out"],
        header=_MPP_NAX_HEADER,
        source=source,
    )


def _build_sorted_group_tiles_cpu(rhs_indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Build same-expert M=16 tile metadata for an already-sorted index vector."""
    idx = np.array(rhs_indices, dtype=np.uint32).reshape(-1)
    starts: list[int] = []
    counts: list[int] = []
    experts: list[int] = []
    i = 0
    while i < len(idx):
        expert = int(idx[i])
        j = i + 1
        while j < len(idx) and int(idx[j]) == expert:
            j += 1
        for start in range(i, j, 16):
            starts.append(start)
            counts.append(min(16, j - start))
            experts.append(expert)
        i = j
    return (
        mx.array(np.array(starts, dtype=np.uint32)),
        mx.array(np.array(counts, dtype=np.uint32)),
        mx.array(np.array(experts, dtype=np.uint32)),
    )


_GROUP_TILE_CACHE = {"ref": None, "tiles": None}


def _build_sorted_group_tiles_cached(
    rhs_indices: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Single-entry cache for same-forward sorted expert tile metadata."""
    cached_ref = _GROUP_TILE_CACHE["ref"]
    cached_obj = cached_ref() if cached_ref is not None else None
    if cached_obj is rhs_indices and _GROUP_TILE_CACHE["tiles"] is not None:
        return _GROUP_TILE_CACHE["tiles"]

    tiles = _build_sorted_group_tiles_cpu(rhs_indices)
    try:
        _GROUP_TILE_CACHE["ref"] = weakref.ref(rhs_indices)
        _GROUP_TILE_CACHE["tiles"] = tiles
    except TypeError:
        _GROUP_TILE_CACHE["ref"] = None
        _GROUP_TILE_CACHE["tiles"] = None
    return tiles


def build_sorted_group_tiles(rhs_indices_sorted: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Build same-expert M=16 tile metadata for proof/benchmark callers."""
    return _build_sorted_group_tiles_cached(rhs_indices_sorted)


def gather_tq_matmul_mpp_nax_grouped_from_rot_with_tiles(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    tile_starts: mx.array,
    tile_counts: mx.array,
    tile_experts: mx.array,
    in_features: int,
    out_features: int,
    bits: int,
) -> mx.array:
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported JANGTQ bits for grouped MPP NAX: {bits}")
    if not mpp_nax_tensorops_available():
        raise RuntimeError("MPP NAX tensor_ops unavailable for MLX custom kernels")

    n_dispatches = int(x_rot.shape[0])
    kernel = _make_grouped_gather_tq_mpp_nax_kernel(
        in_features, out_features, int(bits)
    )
    n_tiles = int(tile_starts.shape[0])
    n_output_tiles = (out_features + 31) // 32
    out = kernel(
        inputs=[
            x_rot.astype(mx.float16),
            packed,
            norms,
            codebook,
            tile_starts,
            tile_counts,
            tile_experts,
        ],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(n_output_tiles * 32, n_tiles, 1),
        threadgroup=(32, 1, 1),
    )[0]
    return out


def gather_tq_matmul_mpp_nax_grouped_from_rot(
    x_rot: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    rhs_indices_sorted: mx.array,
    in_features: int,
    out_features: int,
    bits: int,
) -> mx.array:
    """Run routed TQ gather using same-expert M=16 NAX tiles.

    This proof helper currently builds tile metadata on CPU from a sorted expert
    vector. It is correct for validating the kernel shape, but a production
    prefill path should build/reuse equivalent metadata without per-layer CPU
    synchronization.
    """
    tile_starts, tile_counts, tile_experts = build_sorted_group_tiles(
        rhs_indices_sorted
    )
    return gather_tq_matmul_mpp_nax_grouped_from_rot_with_tiles(
        x_rot,
        packed,
        norms,
        codebook,
        tile_starts,
        tile_counts,
        tile_experts,
        in_features,
        out_features,
        bits,
    )


@lru_cache(maxsize=64)
def _make_grouped_fused_gate_up_swiglu_mpp_nax_kernel(
    in_features: int,
    out_features: int,
    bits: int,
):
    # Fused gate+up matmuls with SwiGLU activation, grouped sorted tiles.
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_id = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint lane = thread_index_in_threadgroup;
uint tile_start = tile_starts[tile_id];
uint tile_count = tile_counts[tile_id];
uint expert = tile_experts[tile_id];
float swiglu_limit = static_cast<float>(meta[0]) * 0.001f;
uint expert_base = expert * {out_features}u * {packed_cols}u;
uint norm_base = expert * {out_features}u;

threadgroup half tg_a[16 * 16];
threadgroup half tg_bg[16 * 32];
threadgroup half tg_bu[16 * 32];
threadgroup float tg_cg[32 * 16];
threadgroup float tg_cu[32 * 16];

for (uint i = lane; i < 32u * 16u; i += 32u) {{
  tg_cg[i] = 0.0f;
  tg_cu[i] = 0.0f;
}}

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, execution_simdgroup> op;

const uint vals_per_u32 = 32u / {bits}u;
const uint mask = (1u << {bits}u) - 1u;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  // Fill A tile
  for (uint idx = lane; idx < 16u * 16u; idx += 32u) {{
    uint k_local = idx / 16u;
    uint m_local = idx % 16u;
    uint k = k0 + k_local;
    half val = half(0);
    if (k < {in_features}u && m_local < tile_count)
      val = x_rot[(tile_start + m_local) * {in_features}u + k];
    tg_a[k_local + m_local * 16u] = val;
  }}

  // Fill B_gate and B_up tiles
  for (uint k_local = 0u; k_local < 16u; k_local++) {{
    uint k = k0 + k_local;
    uint nc = n0 + lane;
    half vg = half(0);
    half vu = half(0);
    if (k < {in_features}u && nc < {out_features}u) {{
      uint pack_idx = k / vals_per_u32;
      uint bit_offset = (k % vals_per_u32) * {bits}u;
      uint pvg = packed_gate[expert_base + nc * {packed_cols}u + pack_idx];
      uint pvu = packed_up[expert_base + nc * {packed_cols}u + pack_idx];
      uint cbg = (pvg >> bit_offset) & mask;
      uint cbu = (pvu >> bit_offset) & mask;
      float ng = static_cast<float>(norms_gate[norm_base + nc]);
      float nu = static_cast<float>(norms_up[norm_base + nc]);
      vg = static_cast<half>(codebook[cbg] * ng);
      vu = static_cast<half>(codebook[cbu] * nu);
    }}
    tg_bg[k_local * 32u + lane] = vg;
    tg_bu[k_local * 32u + lane] = vu;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto X  = tensor(tg_a,  extents<int, 16, 16>(), array<int, 2>{{1, 16}});
  auto Wg = tensor(tg_bg, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto Wu = tensor(tg_bu, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto Og = tensor(tg_cg, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  auto Ou = tensor(tg_cu, extents<int, 32, 16>(), array<int, 2>{{1, 32}});
  op.run(X, Wg, Og);
  op.run(X, Wu, Ou);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}}

// Scatter SwiGLU(gate, up) to device output
for (uint idx = lane; idx < 32u * 16u; idx += 32u) {{
  uint n_local = idx / 16u;
  uint m_local = idx % 16u;
  uint nc = n0 + n_local;
  if (m_local < tile_count && nc < {out_features}u) {{
    float gate = tg_cg[n_local + m_local * 32u];
    float up   = tg_cu[n_local + m_local * 32u];
    if (swiglu_limit > 0.0f) {{
      gate = metal::min(gate, swiglu_limit);
      up   = metal::min(metal::max(up, -swiglu_limit), swiglu_limit);
    }}
    float act = (gate / (1.0f + metal::exp(-gate))) * up;
    out[(tile_start + m_local) * {out_features}u + nc] = act;
  }}
}}
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_grouped_fused_gate_up_swiglu_mpp_nax_i{in_features}_o{out_features}_q{bits}",
        input_names=[
            "x_rot",
            "packed_gate",
            "norms_gate",
            "packed_up",
            "norms_up",
            "codebook",
            "tile_starts",
            "tile_counts",
            "tile_experts",
            "meta",
        ],
        output_names=["out"],
        header=_MPP_NAX_HEADER,
        source=source,
    )


def fused_gate_up_swiglu_mpp_nax_grouped_from_rot_with_tiles(
    x_rot: mx.array,
    packed_gate: mx.array,
    norms_gate: mx.array,
    packed_up: mx.array,
    norms_up: mx.array,
    codebook: mx.array,
    tile_starts: mx.array,
    tile_counts: mx.array,
    tile_experts: mx.array,
    in_features: int,
    out_features: int,
    bits: int,
    swiglu_limit: float = 0.0,
) -> mx.array:
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported JANGTQ bits for grouped fused MPP NAX: {bits}")
    if not mpp_nax_tensorops_available():
        raise RuntimeError("MPP NAX tensor_ops unavailable for MLX custom kernels")

    n_dispatches = int(x_rot.shape[0])
    kernel = _make_grouped_fused_gate_up_swiglu_mpp_nax_kernel(
        in_features, out_features, int(bits)
    )
    n_tiles = int(tile_starts.shape[0])
    n_output_tiles = (out_features + 31) // 32
    meta = mx.array([max(0, int(round(float(swiglu_limit or 0.0) * 1000.0)))], dtype=mx.uint32)
    out = kernel(
        inputs=[
            x_rot.astype(mx.float16),
            packed_gate,
            norms_gate,
            packed_up,
            norms_up,
            codebook,
            tile_starts,
            tile_counts,
            tile_experts,
            meta,
        ],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(n_output_tiles * 32, n_tiles, 1),
        threadgroup=(32, 1, 1),
    )[0]
    return out


def fused_gate_up_swiglu_mpp_nax_grouped_from_rot(
    x_rot: mx.array,
    packed_gate: mx.array,
    norms_gate: mx.array,
    packed_up: mx.array,
    norms_up: mx.array,
    codebook: mx.array,
    rhs_indices_sorted: mx.array,
    in_features: int,
    out_features: int,
    bits: int,
    swiglu_limit: float = 0.0,
) -> mx.array:
    """Run sorted routed gate/up/SwiGLU using same-expert M=16 NAX tiles."""
    tile_starts, tile_counts, tile_experts = build_sorted_group_tiles(
        rhs_indices_sorted
    )
    return fused_gate_up_swiglu_mpp_nax_grouped_from_rot_with_tiles(
        x_rot,
        packed_gate,
        norms_gate,
        packed_up,
        norms_up,
        codebook,
        tile_starts,
        tile_counts,
        tile_experts,
        in_features,
        out_features,
        bits,
        swiglu_limit=swiglu_limit,
    )
