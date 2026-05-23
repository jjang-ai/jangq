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
#include <metal_simdgroup>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct JangTQFrag16 {
  template <typename U>
  using vec8 = typename metal::vec<U, 8>;

  static short2 get_coord() {
    const ushort lane = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = lane >> 2;
    const short fm = ((qid & 4) | ((lane >> 1) & 3));
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtr>
  static void fill_a(
      thread vec8<T>& dst,
      SrcPtr x,
      uint batch_size,
      uint in_features,
      uint m0,
      uint k0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m0 + static_cast<uint>(i * 8 + sc.y);
      uint col = k0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint c = col + static_cast<uint>(j);
        if (row < batch_size && c < in_features) {
          dst[i * 4 + j] = static_cast<T>(x[row * in_features + c]);
        } else {
          dst[i * 4 + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename SrcPtr>
  static void fill_a_single_row(
      thread vec8<T>& dst,
      SrcPtr x,
      uint in_features,
      uint row_idx,
      uint k0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint local_row = static_cast<uint>(i * 8 + sc.y);
      uint col = k0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint c = col + static_cast<uint>(j);
        if (local_row == 0u && c < in_features) {
          dst[i * 4 + j] = static_cast<T>(x[row_idx * in_features + c]);
        } else {
          dst[i * 4 + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename SrcPtr>
  static void fill_a_group(
      thread vec8<T>& dst,
      SrcPtr x,
      uint in_features,
      uint tile_start,
      uint tile_count,
      uint k0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint local_row = static_cast<uint>(i * 8 + sc.y);
      uint col = k0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint c = col + static_cast<uint>(j);
        if (local_row < tile_count && c < in_features) {
          dst[i * 4 + j] =
              static_cast<T>(x[(tile_start + local_row) * in_features + c]);
        } else {
          dst[i * 4 + j] = T(0);
        }
      }
    }
  }

  template <typename PackedPtr, typename NormPtr, typename CodebookPtr>
  static void fill_b(
      thread vec8<half>& dst,
      PackedPtr packed,
      NormPtr norms,
      CodebookPtr codebook,
      uint packed_cols,
      uint out_features,
      uint in_features,
      uint bits,
      uint k0,
      uint n0) {
    const short2 sc = get_coord();
    const uint vals_per_u32 = 32u / bits;
    const uint mask = (1u << bits) - 1u;
    for (short i = 0; i < 2; i++) {
      uint k = k0 + static_cast<uint>(i * 8 + sc.y);
      uint col = n0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint out_col = col + static_cast<uint>(j);
        if (k < in_features && out_col < out_features) {
          uint pack_idx = k / vals_per_u32;
          uint bit_offset = (k % vals_per_u32) * bits;
          uint pv = packed[out_col * packed_cols + pack_idx];
          uint cb_idx = (pv >> bit_offset) & mask;
          float w = codebook[cb_idx] * static_cast<float>(norms[out_col]);
          dst[i * 4 + j] = static_cast<half>(w);
        } else {
          dst[i * 4 + j] = half(0);
        }
      }
    }
  }

  template <typename PackedPtr, typename NormPtr, typename CodebookPtr>
  static void fill_b_expert(
      thread vec8<half>& dst,
      PackedPtr packed,
      NormPtr norms,
      CodebookPtr codebook,
      uint expert,
      uint packed_cols,
      uint out_features,
      uint in_features,
      uint bits,
      uint k0,
      uint n0) {
    const short2 sc = get_coord();
    const uint vals_per_u32 = 32u / bits;
    const uint mask = (1u << bits) - 1u;
    const uint expert_base = expert * out_features * packed_cols;
    const uint norm_base = expert * out_features;
    for (short i = 0; i < 2; i++) {
      uint k = k0 + static_cast<uint>(i * 8 + sc.y);
      uint col = n0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint out_col = col + static_cast<uint>(j);
        if (k < in_features && out_col < out_features) {
          uint pack_idx = k / vals_per_u32;
          uint bit_offset = (k % vals_per_u32) * bits;
          uint pv = packed[expert_base + out_col * packed_cols + pack_idx];
          uint cb_idx = (pv >> bit_offset) & mask;
          float w = codebook[cb_idx] * static_cast<float>(norms[norm_base + out_col]);
          dst[i * 4 + j] = static_cast<half>(w);
        } else {
          dst[i * 4 + j] = half(0);
        }
      }
    }
  }

  template <typename T, typename DstPtr>
  static void store_c(
      thread vec8<T>& src,
      DstPtr out,
      uint batch_size,
      uint out_features,
      uint m0,
      uint n0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint row = m0 + static_cast<uint>(i * 8 + sc.y);
      uint col = n0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint out_col = col + static_cast<uint>(j);
        if (row < batch_size && out_col < out_features) {
          out[row * out_features + out_col] = src[i * 4 + j];
        }
      }
    }
  }

  template <typename T, typename DstPtr>
  static void store_c_single_row(
      thread vec8<T>& src,
      DstPtr out,
      uint out_features,
      uint row_idx,
      uint n0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint local_row = static_cast<uint>(i * 8 + sc.y);
      uint col = n0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint out_col = col + static_cast<uint>(j);
        if (local_row == 0u && out_col < out_features) {
          out[row_idx * out_features + out_col] = src[i * 4 + j];
        }
      }
    }
  }

  template <typename T, typename DstPtr>
  static void store_c_group(
      thread vec8<T>& src,
      DstPtr out,
      uint out_features,
      uint tile_start,
      uint tile_count,
      uint n0) {
    const short2 sc = get_coord();
    for (short i = 0; i < 2; i++) {
      uint local_row = static_cast<uint>(i * 8 + sc.y);
      uint col = n0 + static_cast<uint>(sc.x);
      for (short j = 0; j < 4; j++) {
        uint out_col = col + static_cast<uint>(j);
        if (local_row < tile_count && out_col < out_features) {
          out[(tile_start + local_row) * out_features + out_col] =
              src[i * 4 + j];
        }
      }
    }
  }
};
"""


_MPP_NAX_SMOKE_SOURCE = r"""
uint m0 = threadgroup_position_in_grid.y * 16u;
uint n0 = threadgroup_position_in_grid.x * 32u;
constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_c = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

JangTQFrag16::vec8<half> fa;
JangTQFrag16::vec8<half> fb;
JangTQFrag16::fill_a(fa, A, 16u, 16u, m0, 0u);
for (short i = 0; i < 8; i++) ct_a[i] = fa[i];
for (short nn = 0; nn < 2; nn++) {
  JangTQFrag16::fill_a(fb, B, 16u, 32u, 0u, n0 + static_cast<uint>(16 * nn));
  for (short i = 0; i < 8; i++) ct_b[nn * 8 + i] = fb[i];
}
for (short i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;
op.run(ct_a, ct_b, ct_c);
for (short nn = 0; nn < 2; nn++) {
  JangTQFrag16::vec8<float> fc;
  for (short i = 0; i < 8; i++) fc[i] = ct_c[nn * 8 + i];
  JangTQFrag16::store_c(fc, C, 16u, 32u, m0, n0 + static_cast<uint>(16 * nn));
}
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
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_m = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint m0 = tile_m * 16u;

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_c = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

for (short i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  JangTQFrag16::vec8<half> fa;
  JangTQFrag16::fill_a(
      fa, x_rot, {batch_size}u, {in_features}u, m0, k0);
  for (short i = 0; i < 8; i++) ct_a[i] = fa[i];

  for (short nn = 0; nn < 2; nn++) {{
    JangTQFrag16::vec8<half> fb;
    JangTQFrag16::fill_b(
        fb,
        packed,
        norms,
        codebook,
        {((in_features + (32 // bits) - 1) // (32 // bits))}u,
        {out_features}u,
        {in_features}u,
        {bits}u,
        k0,
        n0 + static_cast<uint>(16 * nn));
    for (short i = 0; i < 8; i++) ct_b[nn * 8 + i] = fb[i];
  }}

  op.run(ct_a, ct_b, ct_c);
}}

for (short nn = 0; nn < 2; nn++) {{
  JangTQFrag16::vec8<float> fc;
  for (short i = 0; i < 8; i++) fc[i] = ct_c[nn * 8 + i];
  JangTQFrag16::store_c(
      fc, out, {batch_size}u, {out_features}u, m0,
      n0 + static_cast<uint>(16 * nn));
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
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint dispatch_idx = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint expert = rhs_indices[dispatch_idx];

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_c = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

for (short i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  JangTQFrag16::vec8<half> fa;
  JangTQFrag16::fill_a_single_row(
      fa, x_rot, {in_features}u, dispatch_idx, k0);
  for (short i = 0; i < 8; i++) ct_a[i] = fa[i];

  for (short nn = 0; nn < 2; nn++) {{
    JangTQFrag16::vec8<half> fb;
    JangTQFrag16::fill_b_expert(
        fb,
        packed,
        norms,
        codebook,
        expert,
        {packed_cols}u,
        {out_features}u,
        {in_features}u,
        {bits}u,
        k0,
        n0 + static_cast<uint>(16 * nn));
    for (short i = 0; i < 8; i++) ct_b[nn * 8 + i] = fb[i];
  }}

  op.run(ct_a, ct_b, ct_c);
}}

for (short nn = 0; nn < 2; nn++) {{
  JangTQFrag16::vec8<float> fc;
  for (short i = 0; i < 8; i++) fc[i] = ct_c[nn * 8 + i];
  JangTQFrag16::store_c_single_row(
      fc, out, {out_features}u, dispatch_idx,
      n0 + static_cast<uint>(16 * nn));
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
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_id = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint tile_start = tile_starts[tile_id];
uint tile_count = tile_counts[tile_id];
uint expert = tile_experts[tile_id];

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_c = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_b), float>();

for (short i = 0; i < ct_c.get_capacity(); i++) ct_c[i] = 0.0f;

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  JangTQFrag16::vec8<half> fa;
  JangTQFrag16::fill_a_group(
      fa, x_rot, {in_features}u, tile_start, tile_count, k0);
  for (short i = 0; i < 8; i++) ct_a[i] = fa[i];

  for (short nn = 0; nn < 2; nn++) {{
    JangTQFrag16::vec8<half> fb;
    JangTQFrag16::fill_b_expert(
        fb,
        packed,
        norms,
        codebook,
        expert,
        {packed_cols}u,
        {out_features}u,
        {in_features}u,
        {bits}u,
        k0,
        n0 + static_cast<uint>(16 * nn));
    for (short i = 0; i < 8; i++) ct_b[nn * 8 + i] = fb[i];
  }}

  op.run(ct_a, ct_b, ct_c);
}}

for (short nn = 0; nn < 2; nn++) {{
  JangTQFrag16::vec8<float> fc;
  for (short i = 0; i < 8; i++) fc[i] = ct_c[nn * 8 + i];
  JangTQFrag16::store_c_group(
      fc, out, {out_features}u, tile_start, tile_count,
      n0 + static_cast<uint>(16 * nn));
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
    packed_cols = (in_features + (32 // bits) - 1) // (32 // bits)
    source = f"""
uint tile_n = threadgroup_position_in_grid.x;
uint tile_id = threadgroup_position_in_grid.y;
uint n0 = tile_n * 32u;
uint tile_start = tile_starts[tile_id];
uint tile_count = tile_counts[tile_id];
uint expert = tile_experts[tile_id];
float swiglu_limit = static_cast<float>(meta[0]) * 0.001f;

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    16, 32, 16, false, false, true,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
auto ct_bg = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_bu = op.template get_right_input_cooperative_tensor<half, half, float>();
auto ct_cg = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_bg), float>();
auto ct_cu = op.template get_destination_cooperative_tensor<
    decltype(ct_a), decltype(ct_bu), float>();

for (short i = 0; i < ct_cg.get_capacity(); i++) {{
  ct_cg[i] = 0.0f;
  ct_cu[i] = 0.0f;
}}

for (uint k0 = 0u; k0 < {in_features}u; k0 += 16u) {{
  JangTQFrag16::vec8<half> fa;
  JangTQFrag16::fill_a_group(
      fa, x_rot, {in_features}u, tile_start, tile_count, k0);
  for (short i = 0; i < 8; i++) ct_a[i] = fa[i];

  for (short nn = 0; nn < 2; nn++) {{
    JangTQFrag16::vec8<half> fbg;
    JangTQFrag16::vec8<half> fbu;
    JangTQFrag16::fill_b_expert(
        fbg,
        packed_gate,
        norms_gate,
        codebook,
        expert,
        {packed_cols}u,
        {out_features}u,
        {in_features}u,
        {bits}u,
        k0,
        n0 + static_cast<uint>(16 * nn));
    JangTQFrag16::fill_b_expert(
        fbu,
        packed_up,
        norms_up,
        codebook,
        expert,
        {packed_cols}u,
        {out_features}u,
        {in_features}u,
        {bits}u,
        k0,
        n0 + static_cast<uint>(16 * nn));
    for (short i = 0; i < 8; i++) {{
      ct_bg[nn * 8 + i] = fbg[i];
      ct_bu[nn * 8 + i] = fbu[i];
    }}
  }}

  op.run(ct_a, ct_bg, ct_cg);
  op.run(ct_a, ct_bu, ct_cu);
}}

const short2 sc = JangTQFrag16::get_coord();
for (short nn = 0; nn < 2; nn++) {{
  for (short i = 0; i < 2; i++) {{
    uint local_row = static_cast<uint>(i * 8 + sc.y);
    uint row = tile_start + local_row;
    uint col = n0 + static_cast<uint>(16 * nn) + static_cast<uint>(sc.x);
    for (short j = 0; j < 4; j++) {{
      uint out_col = col + static_cast<uint>(j);
      if (local_row < tile_count && out_col < {out_features}u) {{
        uint frag_idx = nn * 8 + i * 4 + j;
        float gate = ct_cg[frag_idx];
        float up = ct_cu[frag_idx];
        if (swiglu_limit > 0.0f) {{
          gate = metal::min(gate, swiglu_limit);
          up = metal::min(metal::max(up, -swiglu_limit), swiglu_limit);
        }}
        float act = (gate / (1.0f + metal::exp(-gate))) * up;
        out[row * {out_features}u + out_col] = act;
      }}
    }}
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
