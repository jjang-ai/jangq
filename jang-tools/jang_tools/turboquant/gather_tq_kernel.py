"""
Batched gather TQ matmul — Metal kernel for MoE inference.
Created by Jinho Jang (eric@jangq.ai)

Single Metal dispatch that does:
  gather + unpack + codebook_lookup + dot_product + norm_scale

For MoE switch layers — matches mx.gather_qmm semantics.

Input is pre-rotated (Hadamard) once per token before the kernel.
The kernel handles batched expert dispatch internally.
"""

import os

import mlx.core as mx
import numpy as np
from typing import Optional

from .codebook import compute_codebook
from .rotation import generate_random_signs
from .hadamard_kernel import hadamard_rotate_metal


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
        if value <= 0 or value > 64:
            return default
        return value
    except Exception:
        return default


# === P17: 20 outputs per thread ===
# Swept OPT ∈ {3..48}: 4 → 43 μs, 8 → 34, 16 → 30, 20 → 29, 24 → 29, 32 → 32, 48 → 48
# OPT=20 is the sweet spot (24 ties, 32 starts to spill).
# Previous P12 picked OPT=4 on M4; M3 Ultra has a very different register profile.
_GATHER_OPT = _env_int("JANGTQ_GATHER_OPT", 20)
_GATHER_TQ_SOURCE = f'''
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * {_GATHER_OPT}u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc[{_GATHER_OPT}];
    #pragma unroll
    for (uint o = 0; o < {_GATHER_OPT}; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset = token_idx * in_features;

    uint n_outs = {_GATHER_OPT}u;
    if (out_idx_0 + {_GATHER_OPT}u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {{
        uint i_base = pack_idx * vals_per_u32;
        uint pv[{_GATHER_OPT}];
        #pragma unroll
        for (uint o = 0; o < {_GATHER_OPT}; o++) {{
            pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
        }}
        if (bits == 2u) {{
            #pragma unroll
            for (uint k = 0; k < 16u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 2u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 3u) {{
            #pragma unroll
            for (uint k = 0; k < 10u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 3u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 4u) {{
            #pragma unroll
            for (uint k = 0; k < 8u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 4u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else if (bits == 8u) {{
            #pragma unroll
            for (uint k = 0; k < 4u; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * 8u;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }} else {{
            #pragma unroll
            for (uint k = 0; k < vals_per_u32; k++) {{
                uint i = i_base + k;
                if (i >= in_features) break;
                float xv = static_cast<float>(x_rot[x_offset + i]);
                uint shift = k * bits;
                #pragma unroll
                for (uint o = 0; o < {_GATHER_OPT}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }}
    }}

    #pragma unroll
    for (uint o = 0; o < {_GATHER_OPT}; o++) {{
        acc[o] = simd_sum(acc[o]);
    }}

    if (lane == 0) {{
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {{
            uint oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }}
    }}
'''


# P19: Hadamard rotation fused INTO the gather kernel.
# Each threadgroup loads its token's input row, butterflies in shmem,
# then does the matmul from shmem — eliminating one global-memory read
# and one kernel dispatch.
#
# Works only for fixed in_features 1536 = 1024 + 512 (MiniMax intermediate).
# Specialized per (in_features, N_SPLIT) in a builder below.
def _make_fused_rot_gather_source(in_features, n_split, opt=_GATHER_OPT, tgsize=256):
    if in_features == 1536:
        blocks = "uint block_sizes[2] = {1024u, 512u}; uint block_logs[2] = {10u, 9u}; uint offsets[2] = {0u, 1024u}; uint n_blocks = 2u;"
    elif in_features == 3072:
        blocks = "uint block_sizes[2] = {2048u, 1024u}; uint block_logs[2] = {11u, 10u}; uint offsets[2] = {0u, 2048u}; uint n_blocks = 2u;"
    elif in_features == 2048:
        blocks = "uint block_sizes[1] = {2048u}; uint block_logs[1] = {11u}; uint offsets[1] = {0u}; uint n_blocks = 1u;"
    elif in_features == 1024:
        blocks = "uint block_sizes[1] = {1024u}; uint block_logs[1] = {10u}; uint offsets[1] = {0u}; uint n_blocks = 1u;"
    else:
        return None  # unsupported dim → fall back to separate rotate + gather

    simds = tgsize // 32
    # outs_per_tg is computed dynamically from meta[2] in the kernel
    return f'''
    uint tid = thread_position_in_threadgroup.x;
    uint tg_y = thread_position_in_grid.y;
    uint lane = tid % 32u;
    uint simd_id = tid / 32u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    uint N_SPLIT = {n_split}u;
    uint split_id = tg_y % N_SPLIT;
    uint dispatch_idx = tg_y / N_SPLIT;
    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    threadgroup float shmem[{in_features}];

    uint tgs = {tgsize}u;
    uint elems_per = (in_features + tgs - 1u) / tgs;
    for (uint e = 0; e < elems_per; e++) {{
        uint i = tid * elems_per + e;
        if (i < in_features) {{
            shmem[i] = static_cast<float>(x[dispatch_idx * in_features + i]) * signs[i];
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {blocks}
    for (uint b = 0; b < n_blocks; b++) {{
        uint d_b = block_sizes[b];
        uint log_b = block_logs[b];
        uint offset = offsets[b];
        uint ept_b = (d_b + tgs - 1u) / tgs;
        if (ept_b == 0u) ept_b = 1u;
        for (uint stage = 0; stage < log_b; stage++) {{
            uint h = 1u << stage;
            uint two_h = 2u * h;
            float newv[64] = {{0.0f}};
            for (uint kk = 0; kk < ept_b; kk++) {{
                uint i_local = tid * ept_b + kk;
                if (i_local < d_b) {{
                    uint bs = (i_local / two_h) * two_h;
                    uint pos = i_local - bs;
                    float a = shmem[offset + bs + pos];
                    if (pos < h) {{ newv[kk] = a + shmem[offset + bs + pos + h]; }}
                    else         {{ newv[kk] = shmem[offset + bs + pos - h] - a; }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint kk = 0; kk < ept_b; kk++) {{
                uint i_local = tid * ept_b + kk;
                if (i_local < d_b) {{ shmem[offset + i_local] = newv[kk]; }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float norm_b = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint kk = 0; kk < ept_b; kk++) {{
            uint i_local = tid * ept_b + kk;
            if (i_local < d_b) {{ shmem[offset + i_local] *= norm_b; }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    uint outs_per_tg = (out_features + N_SPLIT - 1u) / N_SPLIT;
    uint out_base = split_id * outs_per_tg;
    uint out_end = out_base + outs_per_tg;
    if (out_end > out_features) out_end = out_features;
    uint expert_base = expert * out_features * packed_cols;
    uint simds_ct = {simds}u;

    for (uint pass = 0; ; pass++) {{
        uint out_group = pass * simds_ct + simd_id;
        uint out_idx_0 = out_base + out_group * {opt}u;
        if (out_idx_0 >= out_end) break;

        float acc[{opt}];
        #pragma unroll
        for (uint o = 0; o < {opt}; o++) acc[o] = 0.0f;

        uint n_outs = {opt}u;
        if (out_idx_0 + {opt}u > out_features) n_outs = out_features - out_idx_0;
        if (out_idx_0 + n_outs > out_end) n_outs = out_end - out_idx_0;

        for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {{
            uint i_base = pack_idx * vals_per_u32;
            uint pv[{opt}];
            #pragma unroll
            for (uint o = 0; o < {opt}; o++) {{
                pv[o] = (o < n_outs) ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx] : 0u;
            }}
            #pragma unroll
            for (uint kk = 0; kk < vals_per_u32; kk++) {{
                uint i = i_base + kk;
                if (i >= in_features) break;
                float xv = shmem[i];
                uint shift = kk * bits;
                #pragma unroll
                for (uint o = 0; o < {opt}; o++) {{
                    float w = codebook[(pv[o] >> shift) & mask];
                    acc[o] += xv * w;
                }}
            }}
        }}

        #pragma unroll
        for (uint o = 0; o < {opt}; o++) acc[o] = simd_sum(acc[o]);

        if (lane == 0) {{
            uint base_off = (token_idx * K + k_idx) * out_features;
            for (uint o = 0; o < n_outs; o++) {{
                uint oi = out_idx_0 + o;
                float n_v = static_cast<float>(norms[expert * out_features + oi]);
                out[base_off + oi] = acc[o] * n_v;
            }}
        }}
    }}
'''


_FUSED_ROT_GATHER_KERNEL_CACHE = {}
_FUSED_ROT_N_SPLIT = 2  # swept optimum for MiniMax shapes


def _get_fused_rot_gather_kernel(in_features):
    key = (in_features, _FUSED_ROT_N_SPLIT)
    if key in _FUSED_ROT_GATHER_KERNEL_CACHE:
        return _FUSED_ROT_GATHER_KERNEL_CACHE[key]
    src = _make_fused_rot_gather_source(in_features, _FUSED_ROT_N_SPLIT)
    if src is None:
        _FUSED_ROT_GATHER_KERNEL_CACHE[key] = None
        return None
    k = mx.fast.metal_kernel(
        name=f"fused_rot_gather_{in_features}_n{_FUSED_ROT_N_SPLIT}",
        input_names=["x", "signs", "packed", "norms", "codebook", "rhs_indices", "meta"],
        output_names=["out"],
        source=src,
    )
    _FUSED_ROT_GATHER_KERNEL_CACHE[key] = k
    return k


import weakref

_kernel_cache = {}

# P1: cache rotated x. Use weakrefs to detect when original x is freed —
# Python id() is reused after GC, so id-only caching can serve stale data.
# Cache holds ONE entry; gate/up call sequence is back-to-back so a 1-slot
# cache is sufficient.
_X_ROT_CACHE = {"key": None, "x_ref": None, "signs_ref": None, "rot": None}


def _mpp_nax_mode() -> str:
    return os.environ.get("JANGTQ_MPP_NAX", "").strip().lower()


def _auto_mpp_nax_wins(dispatches: int, in_features: int, out_features: int) -> bool:
    if dispatches >= 512:
        return True
    return dispatches >= 256 and (int(in_features) * int(out_features)) >= (2048 * 2048)


def _as_uint32_indices(indices: mx.array) -> mx.array:
    if indices.dtype == mx.uint32:
        return indices
    return indices.astype(mx.uint32)


def _gather_tq_mpp_nax_grouped_from_rot(
    x_rot,
    packed,
    norms,
    codebook,
    idx_flat,
    in_features,
    out_features,
    bits,
):
    from .mpp_nax_kernel import gather_tq_matmul_mpp_nax_grouped_from_rot

    return gather_tq_matmul_mpp_nax_grouped_from_rot(
        x_rot,
        packed,
        norms,
        codebook,
        idx_flat,
        in_features,
        out_features,
        bits,
    )


def _rotate_cached_by_id(x_orig, x_flat, signs):
    """Cache keyed on weak refs to (x_orig, signs). Single-slot cache."""
    cached_x = _X_ROT_CACHE["x_ref"]() if _X_ROT_CACHE["x_ref"] else None
    cached_s = _X_ROT_CACHE["signs_ref"]() if _X_ROT_CACHE["signs_ref"] else None
    if cached_x is x_orig and cached_s is signs and _X_ROT_CACHE["rot"] is not None:
        return _X_ROT_CACHE["rot"]
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs)
    try:
        _X_ROT_CACHE["x_ref"] = weakref.ref(x_orig)
        _X_ROT_CACHE["signs_ref"] = weakref.ref(signs)
        _X_ROT_CACHE["rot"] = x_rot
    except TypeError:
        # mx.array may not support weakref — fall back to no cache
        _X_ROT_CACHE["x_ref"] = None
        _X_ROT_CACHE["signs_ref"] = None
        _X_ROT_CACHE["rot"] = None
    return x_rot

def _get_kernel():
    if "gather_tq" not in _kernel_cache:
        _kernel_cache["gather_tq"] = mx.fast.metal_kernel(
            name="gather_tq_matmul",
            input_names=["x_rot", "packed", "norms", "codebook", "rhs_indices", "meta"],
            output_names=["out"],
            source=_GATHER_TQ_SOURCE,
        )
    return _kernel_cache["gather_tq"]


# P15: compile-friendly decode helper for per_row mode (down_proj path).
_DECODE_CACHE = {}


def make_fused_rot_gather_decode(in_features, out_features, bits, K):
    """Return fn(x_unrotated, signs, packed, norms, cb, idx_flat) → (K, out_f).

    x_unrotated: (K, in_features) fp32 — NOT yet Hadamard-rotated.
    The kernel does the rotation in threadgroup memory before the matmul.
    Returns None if in_features not supported (fall back to split kernels).
    """
    key = ("fused_rot_gather", in_features, out_features, bits, K)
    if key in _DECODE_CACHE:
        return _DECODE_CACHE[key]
    kernel = _get_fused_rot_gather_kernel(in_features)
    if kernel is None:
        _DECODE_CACHE[key] = None
        return None
    vals_per_u32 = 32 // bits
    packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    meta = mx.array([1, in_features, out_features, packed_cols, bits], dtype=mx.uint32)
    tgsize = 256
    n_disp = K * _FUSED_ROT_N_SPLIT

    def _fn(x_unrotated, signs, packed, norms, cb, idx_flat):
        out, = kernel(
            inputs=[x_unrotated, signs, packed, norms, cb, idx_flat, meta],
            output_shapes=[(K, out_features)],
            output_dtypes=[mx.float32],
            grid=(tgsize, n_disp, 1),
            threadgroup=(tgsize, 1, 1),
        )
        return out

    _DECODE_CACHE[key] = _fn
    return _fn


def make_gather_tq_decode_per_row(in_features, out_features, bits, K):
    """Return pure function(x_rot, packed, norms, cb, idx_flat) → (K, out_f).

    x_rot: (K, in_features) float32, already Hadamard-rotated
    idx_flat: (K,) uint32
    Per-row mode: one input row per index (n_dispatches=K, K_meta=1).
    """
    key = ("gather_per_row", in_features, out_features, bits, K)
    if key in _DECODE_CACHE:
        return _DECODE_CACHE[key]
    vals_per_u32 = 32 // bits
    packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    meta = mx.array([1, in_features, out_features, packed_cols, bits], dtype=mx.uint32)
    out_groups = (out_features + _GATHER_OPT - 1) // _GATHER_OPT
    grid_x = out_groups * 32
    tg_x = min(grid_x, 256)
    n_disp = K
    kernel = _get_kernel()

    def _fn(x_rot, packed, norms, cb, idx_flat):
        out, = kernel(
            inputs=[x_rot, packed, norms, cb, idx_flat, meta],
            output_shapes=[(n_disp, out_features)],
            output_dtypes=[mx.float32],
            grid=(grid_x, n_disp, 1),
            threadgroup=(tg_x, 1, 1),
        )
        return out

    _DECODE_CACHE[key] = _fn
    return _fn


def gather_tq_matmul(
    x: mx.array,           # (..., in_features) - will be flattened
    packed: mx.array,      # (n_experts, out_features, packed_cols) uint32
    norms: mx.array,       # (n_experts, out_features) float16
    codebook: mx.array,    # (2^bits,) float32
    signs: mx.array,       # (in_features,) float32
    rhs_indices: mx.array, # (..., K) flat indices
    bits: int,
    sorted_indices: bool = False,
) -> mx.array:
    """Fused gather + unpack + matmul for MoE.

    Returns: (..., K, out_features)
    """
    # Determine shapes
    in_features = x.shape[-1]
    n_experts, out_features, packed_cols = packed.shape

    # Three call patterns observed from mlx_lm SwitchGLU:
    #
    # 1) gate/up_proj (broadcast K):
    #    x       = (..., 1, 1, in_features)   — token dims, no K yet
    #    indices = (..., K)                   — K experts per token
    #    output  = (..., K, 1, out_features)  — K broadcast in
    #    behavior: each x[token] used K times against indices[token, 0..K-1]
    #
    # 2) down_proj (1:1):
    #    x       = (..., K, 1, in_features)   — K already split out
    #    indices = (..., K)                   — same K, same expert per slot
    #    output  = (..., K, 1, out_features)
    #    behavior: x[token, k] uses indices[token, k]
    #
    # 3) sorted (mlx_lm do_sort=True, indices.size >= 64):
    #    x       = (N, 1, in_features)        — N = batch*K rows
    #    indices = (N,)                       — flat
    #    output  = (N, 1, out_features)
    if rhs_indices.ndim == 1:
        # Sorted path
        while x.ndim > 2 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        x_flat = x.reshape(-1, in_features)
        batch = x_flat.shape[0]
        K = 1
        idx_flat = _as_uint32_indices(rhs_indices)
        assert idx_flat.shape[0] == batch, \
            f"sorted: x batch={batch}, indices={idx_flat.shape[0]}"
        n_dispatches = batch
        out_shape_kind = "sorted"
    else:
        K = rhs_indices.shape[-1]
        idx_total = 1
        for s in rhs_indices.shape:
            idx_total *= s

        x_squeezed = x
        while x_squeezed.ndim > 2 and x_squeezed.shape[-2] == 1:
            x_squeezed = x_squeezed.squeeze(-2)
        x_flat = x_squeezed.reshape(-1, in_features)
        batch = x_flat.shape[0]

        if batch == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            n_dispatches = batch
            out_shape_kind = "per_row"
        elif batch * K == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            n_dispatches = batch * K
            out_shape_kind = "broadcast"
        else:
            raise ValueError(
                f"shape mismatch: x batch={batch}, indices total={idx_total}, K={K}")

    # Pre-rotate x (cache across gate/up that share input and signs).
    # Key by id(x) (the original argument) NOT id(x_flat) — reshape returns
    # a new object each call, so using x_flat would miss the cache.
    x_rot = _rotate_cached_by_id(x, x_flat, signs)

    if (
        sorted_indices
        and rhs_indices.ndim == 1
        and (
            _mpp_nax_mode() in {"1", "true", "yes"}
            or (
                _mpp_nax_mode() == "auto"
                and _auto_mpp_nax_wins(int(idx_flat.shape[0]), in_features, out_features)
            )
        )
    ):
        try:
            out_raw = _gather_tq_mpp_nax_grouped_from_rot(
                x_rot,
                packed,
                norms,
                codebook,
                idx_flat,
                in_features,
                out_features,
                bits,
            )
            out = out_raw.reshape(batch, 1, out_features)
            if out.dtype != x.dtype:
                out = out.astype(x.dtype)
            return out
        except Exception:
            if os.environ.get("JANGTQ_MPP_NAX_STRICT", "").strip() == "1":
                raise

    # Run kernel
    kernel = _get_kernel()

    # Kernel uses token_idx = dispatch / K_meta. For per_row we set K_meta=1.
    # NOTE: meta cache (P4) caused output corruption — likely because reusing
    # the same mx.array as kernel input across calls breaks something in MLX's
    # lazy eval pipeline. Allocate fresh each call (small overhead, ~5 ints).
    K_meta = 1 if (rhs_indices.ndim == 1 or out_shape_kind == "per_row") else K
    meta = mx.array([K_meta, in_features, out_features, packed_cols, bits], dtype=mx.uint32)

    out_groups = (out_features + _GATHER_OPT - 1) // _GATHER_OPT
    grid_x = out_groups * 32
    tg_x = min(256, grid_x)
    out = kernel(
        inputs=[x_rot, packed, norms, codebook, idx_flat, meta],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(grid_x, n_dispatches, 1),
        threadgroup=(tg_x, 1, 1),
    )[0]

    # Reshape to match mlx.gather_qmm output shape
    if out_shape_kind == "sorted":
        # (N, 1, out_features)
        out = out.reshape(batch, 1, out_features)
    elif out_shape_kind == "per_row":
        # x had shape (..., K, 1, in) so output is (..., K, 1, out)
        # x_squeezed already has the K folded into batch dims
        # Reconstruct from rhs_indices.shape + (1, out_features)
        out = out.reshape(*rhs_indices.shape, 1, out_features)
    else:
        # broadcast: (..., K, 1, out_features)
        out = out.reshape(*rhs_indices.shape[:-1], K, 1, out_features)

    if out.dtype != x.dtype:
        out = out.astype(x.dtype)
    return out
