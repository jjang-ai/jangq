"""
Fused gate+up TQ matmul kernel.
Created by Jinho Jang (eric@jangq.ai)

Computes gate_proj(x) and up_proj(x) in a single Metal dispatch. Both share
the same x, signs, codebook, and indices. Only the packed weights and norms
differ. This kernel reads x_rot once and produces both outputs.
"""

import mlx.core as mx
import weakref
from .hadamard_kernel import hadamard_rotate_metal


_FUSED_SOURCE = '''
    // Grid: (out_features * 32, batch * K, 1)
    // Each thread computes ONE output for BOTH gate and up.
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_idx = global_x / 32u;
    uint lane = global_x % 32u;

    uint K = meta[0];
    uint in_features = meta[1];
    uint out_features = meta[2];
    uint packed_cols = meta[3];
    uint bits = meta[4];

    if (out_idx >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx = dispatch_idx % K;
    uint expert = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask = (1u << bits) - 1u;

    float acc_gate = 0.0f;
    float acc_up = 0.0f;

    uint packed_row_off = expert * out_features * packed_cols + out_idx * packed_cols;
    uint x_off = token_idx * in_features;

    for (uint i = lane; i < in_features; i += 32u) {
        uint pack_idx = i / vals_per_u32;
        uint bit_offset = (i % vals_per_u32) * bits;

        uint pv_g = packed_gate[packed_row_off + pack_idx];
        uint pv_u = packed_up[packed_row_off + pack_idx];
        uint cb_g = (pv_g >> bit_offset) & mask;
        uint cb_u = (pv_u >> bit_offset) & mask;

        float w_g = codebook[cb_g];
        float w_u = codebook[cb_u];
        float xv = static_cast<float>(x_rot[x_off + i]);

        acc_gate += xv * w_g;
        acc_up += xv * w_u;
    }

    acc_gate = simd_sum(acc_gate);
    acc_up = simd_sum(acc_up);

    if (lane == 0) {
        float norm_g = static_cast<float>(norms_gate[expert * out_features + out_idx]);
        float norm_u = static_cast<float>(norms_up[expert * out_features + out_idx]);
        uint out_off = (token_idx * K + k_idx) * out_features + out_idx;
        out_gate[out_off] = acc_gate * norm_g;
        out_up[out_off] = acc_up * norm_u;
    }
'''

# === P17: 10 outputs per thread ===
# Swept OPT ∈ {2..16}: 4 → 59 μs, 6 → 54, 8 → 51, 10 → 50, 12 → 50, 16 → 65
# OPT=10 is a stable sweet spot (OPT=12 same speed, OPT=16 register spill cliff).
# P12 picked OPT=4 on M4; re-measuring on M3 Ultra shows the sweet spot is much
# higher. Likely register-file sizing and scheduling differ by generation.
_FUSED_SWIGLU_OPT = 10
_FUSED_SWIGLU_SOURCE = f'''
    uint global_x = thread_position_in_grid.x;
    uint dispatch_idx = thread_position_in_grid.y;

    uint out_group = global_x / 32u;
    uint lane = global_x % 32u;
    uint out_idx_0 = out_group * {_FUSED_SWIGLU_OPT}u;

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

    float acc_g[{_FUSED_SWIGLU_OPT}];
    float acc_u[{_FUSED_SWIGLU_OPT}];
    #pragma unroll
    for (uint o = 0; o < {_FUSED_SWIGLU_OPT}; o++) {{ acc_g[o] = 0.0f; acc_u[o] = 0.0f; }}

    uint expert_base = expert * out_features * packed_cols;
    uint x_off = token_idx * in_features;

    uint n_outs = {_FUSED_SWIGLU_OPT}u;
    if (out_idx_0 + {_FUSED_SWIGLU_OPT}u > out_features) n_outs = out_features - out_idx_0;

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {{
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[{_FUSED_SWIGLU_OPT}], pvu[{_FUSED_SWIGLU_OPT}];
        #pragma unroll
        for (uint o = 0; o < {_FUSED_SWIGLU_OPT}; o++) {{
            if (o < n_outs) {{
                uint row_off = expert_base + (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = packed_gate[row_off];
                pvu[o] = packed_up[row_off];
            }} else {{
                pvg[o] = 0u;
                pvu[o] = 0u;
            }}
        }}

        #pragma unroll
        for (uint k = 0; k < vals_per_u32; k++) {{
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = static_cast<float>(x_rot[x_off + i]);
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < {_FUSED_SWIGLU_OPT}; o++) {{
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }}
        }}
    }}

    #pragma unroll
    for (uint o = 0; o < {_FUSED_SWIGLU_OPT}; o++) {{
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }}

    if (lane == 0) {{
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {{
            uint oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[expert * out_features + oi]);
            float nu = static_cast<float>(norms_up[expert * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            // §441 (2026-04-26): DSV4 swiglu_limit=10 clamp. Required per
            // DSV4 design — without it, deep MoE stacks diverge numerically.
            // Python-side _dsv4_swiglu (mlx_model.py:1280) clamps gate to
            // (-inf, +10] and up to [-10, +10] BEFORE silu*mul. The fused
            // Metal path was missing this — fixed now.
            gv = metal::min(gv, 10.0f);              // gate clamp (max only)
            uv = metal::clamp(uv, -10.0f, 10.0f);    // up clamp (symmetric)
            out_act[base_off + oi] = (gv / (1.0f + metal::fast::exp(-gv))) * uv;
        }}
    }}
'''


_kernel = None
_kernel_swiglu = None


# P15: compile-friendly decode helpers. Builds per-(in_f, out_f, bits, K)
# closures with meta/grid/tg baked as Python constants — no per-call mx.array
# allocation inside the hot path. Safe to wrap in mx.compile.
_DECODE_CACHE = {}


def make_fused_gate_up_swiglu_decode(in_features, out_features, bits, K):
    """Return a pure function(x_rot, pg, ng, pu, nu, cb, idx_flat) → (K, out_f).

    x_rot: (1, in_features) float32, already Hadamard-rotated
    idx_flat: (K,) uint32
    Assumes broadcast mode (batch=1, K experts).
    """
    key = ("gu_swiglu", in_features, out_features, bits, K)
    if key in _DECODE_CACHE:
        return _DECODE_CACHE[key]

    vals_per_u32 = 32 // bits
    packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
    meta = mx.array([K, in_features, out_features, packed_cols, bits], dtype=mx.uint32)
    # P17: OPT=10 outputs per thread (swept optimum on M3 Ultra)
    out_groups = (out_features + _FUSED_SWIGLU_OPT - 1) // _FUSED_SWIGLU_OPT
    grid_x = out_groups * 32
    tg_x = min(grid_x, 256)
    n_disp = K
    kernel = _get_kernel_swiglu()

    def _fn(x_rot, pg, ng, pu, nu, cb, idx_flat):
        out_raw, = kernel(
            inputs=[x_rot, pg, ng, pu, nu, cb, idx_flat, meta],
            output_shapes=[(n_disp, out_features)],
            output_dtypes=[mx.float32],
            grid=(grid_x, n_disp, 1),
            threadgroup=(tg_x, 1, 1),
        )
        return out_raw

    _DECODE_CACHE[key] = _fn
    return _fn


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="fused_gate_up",
            input_names=[
                "x_rot", "packed_gate", "norms_gate", "packed_up", "norms_up",
                "codebook", "rhs_indices", "meta",
            ],
            output_names=["out_gate", "out_up"],
            source=_FUSED_SOURCE,
        )
    return _kernel


def _get_kernel_swiglu():
    global _kernel_swiglu
    if _kernel_swiglu is None:
        _kernel_swiglu = mx.fast.metal_kernel(
            name="fused_gate_up_swiglu",
            input_names=[
                "x_rot", "packed_gate", "norms_gate", "packed_up", "norms_up",
                "codebook", "rhs_indices", "meta",
            ],
            output_names=["out_act"],
            source=_FUSED_SWIGLU_SOURCE,
        )
    return _kernel_swiglu


def fused_gate_up_swiglu_matmul(
    x: mx.array,
    packed_gate: mx.array, norms_gate: mx.array,
    packed_up: mx.array, norms_up: mx.array,
    codebook: mx.array, signs: mx.array,
    rhs_indices: mx.array,
    bits: int,
) -> mx.array:
    """Fused gate+up+SwiGLU activation. Returns post-activation result.

    Computes: SiLU(gate_proj(x)) * up_proj(x) in one Metal dispatch.
    Output shape matches what gather_qmm would produce, ready to feed into down_proj.
    """
    in_features = x.shape[-1]
    n_experts, out_features, packed_cols = packed_gate.shape

    if rhs_indices.ndim == 1:
        while x.ndim > 2 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        x_flat = x.reshape(-1, in_features)
        batch = x_flat.shape[0]
        K = 1
        idx_flat = rhs_indices.astype(mx.uint32)
        out_shape_kind = "sorted"
    else:
        K = rhs_indices.shape[-1]
        idx_total = 1
        for s in rhs_indices.shape:
            idx_total *= s
        x_sq = x
        while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
            x_sq = x_sq.squeeze(-2)
        x_flat = x_sq.reshape(-1, in_features)
        batch = x_flat.shape[0]
        if batch == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            out_shape_kind = "per_row"
        elif batch * K == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            out_shape_kind = "broadcast"
        else:
            raise ValueError(f"shape mismatch: x batch={batch}, idx_total={idx_total}, K={K}")

    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs)

    K_meta = 1 if out_shape_kind in ("sorted", "per_row") else K
    meta = mx.array([K_meta, in_features, out_features, packed_cols, bits], dtype=mx.uint32)
    n_dispatches = batch if out_shape_kind != "broadcast" else batch * K

    kernel = _get_kernel_swiglu()
    # P17: 10 outputs per thread (sweet spot on M3 Ultra)
    out_groups = (out_features + _FUSED_SWIGLU_OPT - 1) // _FUSED_SWIGLU_OPT
    grid_x = out_groups * 32
    out_raw, = kernel(
        inputs=[x_rot, packed_gate, norms_gate, packed_up, norms_up,
                codebook, idx_flat, meta],
        output_shapes=[(n_dispatches, out_features)],
        output_dtypes=[mx.float32],
        grid=(grid_x, n_dispatches, 1),
        threadgroup=(min(grid_x, 256), 1, 1),
    )

    if out_shape_kind == "sorted":
        out = out_raw.reshape(batch, 1, out_features)
    elif out_shape_kind == "per_row":
        out = out_raw.reshape(*rhs_indices.shape, 1, out_features)
    else:
        out = out_raw.reshape(*rhs_indices.shape[:-1], K, 1, out_features)

    if out.dtype != x.dtype:
        out = out.astype(x.dtype)
    return out


def fused_gate_up_matmul(
    x: mx.array,
    packed_gate: mx.array, norms_gate: mx.array,
    packed_up: mx.array, norms_up: mx.array,
    codebook: mx.array, signs: mx.array,
    rhs_indices: mx.array,
    bits: int,
) -> tuple:
    """Compute gate_proj(x) and up_proj(x) in one fused kernel dispatch.

    Returns: (out_gate, out_up) each with the gather_qmm-compatible shape.
    """
    assert packed_gate.shape == packed_up.shape, "gate and up must have same shape"

    in_features = x.shape[-1]
    n_experts, out_features, packed_cols = packed_gate.shape

    # Shape detection (matches gather_tq_matmul)
    if rhs_indices.ndim == 1:
        while x.ndim > 2 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        x_flat = x.reshape(-1, in_features)
        batch = x_flat.shape[0]
        K = 1
        idx_flat = rhs_indices.astype(mx.uint32)
        out_shape_kind = "sorted"
    else:
        K = rhs_indices.shape[-1]
        idx_total = 1
        for s in rhs_indices.shape:
            idx_total *= s
        x_sq = x
        while x_sq.ndim > 2 and x_sq.shape[-2] == 1:
            x_sq = x_sq.squeeze(-2)
        x_flat = x_sq.reshape(-1, in_features)
        batch = x_flat.shape[0]
        if batch == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            out_shape_kind = "per_row"
        elif batch * K == idx_total:
            idx_flat = rhs_indices.reshape(-1).astype(mx.uint32)
            out_shape_kind = "broadcast"
        else:
            raise ValueError(f"shape mismatch: x batch={batch}, idx_total={idx_total}, K={K}")

    # Rotate x (no cache — just one call per layer in fused mode)
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs)

    K_meta = 1 if out_shape_kind in ("sorted", "per_row") else K
    meta = mx.array([K_meta, in_features, out_features, packed_cols, bits], dtype=mx.uint32)
    n_dispatches = batch if out_shape_kind != "broadcast" else batch * K

    kernel = _get_kernel()
    outs = kernel(
        inputs=[x_rot, packed_gate, norms_gate, packed_up, norms_up,
                codebook, idx_flat, meta],
        output_shapes=[(n_dispatches, out_features), (n_dispatches, out_features)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(out_features * 32, n_dispatches, 1),
        threadgroup=(min(out_features * 32, 256), 1, 1),
    )
    out_gate_raw, out_up_raw = outs

    # Reshape to match mlx.gather_qmm output shape
    def reshape_out(raw):
        if out_shape_kind == "sorted":
            return raw.reshape(batch, 1, out_features)
        elif out_shape_kind == "per_row":
            return raw.reshape(*rhs_indices.shape, 1, out_features)
        else:
            return raw.reshape(*rhs_indices.shape[:-1], K, 1, out_features)

    out_gate = reshape_out(out_gate_raw)
    out_up = reshape_out(out_up_raw)

    if out_gate.dtype != x.dtype:
        out_gate = out_gate.astype(x.dtype)
        out_up = out_up.astype(x.dtype)

    return out_gate, out_up
