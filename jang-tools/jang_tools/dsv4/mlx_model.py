"""MLX model file for DeepSeek-V4 — the runtime mlx_lm plugs into.

Mirrors mlx_lm/models/deepseek_v32.py patterns with DSV4-specific changes:
- MLA with head_dim=512, o_lora_rank+o_groups grouped output projection
- mHC (Manifold-Constrained Hyper-Connections) wrapping attn + ffn
- sqrtsoftplus scoring + hash-routing for first N layers
- Full attention (no CSA/HCA yet — those are Phase 7.5B.2)
- No MTP head at inference (discarded per DSV convention)

This file is registered into mlx_lm.models at runtime via
`jang_tools.dsv4.mlx_register`, so `load_jangtq_model` works on
DSV4-Flash bundles.
"""

from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs, create_attention_mask, scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU

try:  # env-gated Metal prefill kernel (VMLX_DSV4_HEADS16_PREFILL, default on)
    from .indexed_prefill_attention import dsv4_heads16_prefill_attention
except Exception:  # pragma: no cover - optional module; stock path if absent
    dsv4_heads16_prefill_attention = None

try:  # env-gated Metal decode kernel (VMLX_DSV4_INDEXED_DECODE, default off)
    from .indexed_decode_attention import dsv4_indexed_decode_attention
except Exception:  # pragma: no cover - optional module; stock path if absent
    dsv4_indexed_decode_attention = None


logger = logging.getLogger(__name__)


_DSV4_ACTIVATION_QAT_ENV = "DSV4_ACTIVATION_QAT"
_DSV4_ACTIVATION_QAT_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _dsv4_activation_qat_requested(environ=None) -> bool:
    """Return the explicit opt-in state for source-native activation QAT.

    The runtime intentionally defaults this diagnostic-quality graph path off.
    Unknown values also fail safe to off; changing the process environment
    requires a model-process restart because the effective state is captured
    when this module is imported.
    """

    source = os.environ if environ is None else environ
    raw_value = source.get(_DSV4_ACTIVATION_QAT_ENV, "")
    return str(raw_value).strip().lower() in _DSV4_ACTIVATION_QAT_TRUE_VALUES


_DSV4_ACTIVATION_QAT_REQUESTED = _dsv4_activation_qat_requested()


def _layerwise_prefill_materialization_enabled(
    input_ids, final_context_tokens: int = 0
) -> bool:
    """Bound DSV4's lazy cross-layer graph during multi-token prefill.

    CSA layers create a query-by-compressed-pool attention graph.  Leaving all
    43 decoder layers lazy until final logits makes those graphs coexist and
    can exceed Metal's working set even when the persistent SWA/CSA/HCA cache
    itself fits.  Evaluating the hidden state at layer boundaries preserves
    the exact graph math while releasing prior-layer attention temporaries.

    The barrier costs 25-30% prefill throughput (live A/B: 267 vs 346 pp/s at
    15k), while attention sub-chunking already bounds the dominant per-layer
    score temporaries at 512 query rows.  When ``DSV4_LAYERWISE_PREFILL`` is
    unset, barriers therefore engage only once the chunk's final context
    (cache offset + chunk width) exceeds the standalone-proven safe curve
    (``DSV4_LAYERWISE_PREFILL_AUTO_TOKENS``, default 24,576) or when
    sub-chunking is disabled.  An explicit env value keeps the old absolute
    on/off semantics.
    """

    raw_enabled = os.environ.get("DSV4_LAYERWISE_PREFILL")
    if raw_enabled is not None and raw_enabled.strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    try:
        min_tokens = max(
            2,
            int(os.environ.get("DSV4_LAYERWISE_PREFILL_MIN_TOKENS", "256")),
        )
    except (TypeError, ValueError):
        min_tokens = 256
    try:
        if int(input_ids.shape[-1]) < min_tokens:
            return False
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    if raw_enabled is None and _dsv4_attn_subchunk_tokens() > 0:
        try:
            auto_threshold = int(
                os.environ.get("DSV4_LAYERWISE_PREFILL_AUTO_TOKENS", "24576")
            )
        except (TypeError, ValueError):
            auto_threshold = 24576
        if auto_threshold > 0:
            try:
                if max(0, int(final_context_tokens or 0)) <= auto_threshold:
                    return False
            except (TypeError, ValueError):
                pass
    return True


def _dsv4_attn_subchunk_tokens() -> int:
    """Attention sub-chunk length for wide-chunk prefill.

    DSV4 prefill has opposed optima: SWA/CSA attention cost grows
    super-linearly with chunk width (dense visibility masks stream
    T_q x T_kv and defeat SDPA tile skipping), while MoE gather_qmm
    throughput grows with batch (7.75 -> 21 TF going 512 -> 2048 rows).
    Running attention in 512-token sub-slices against the layer cache is
    numerically identical to top-level 512-chunk prefill (same call
    sequence per layer, causal decomposition), while MoE still sees the
    full chunk batch. 0 disables.
    """

    try:
        return max(0, int(os.environ.get("DSV4_ATTN_SUBCHUNK", "512")))
    except (TypeError, ValueError):
        return 512


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "deepseek_v4"
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_hidden_layers: int = 43
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    qk_rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    num_hash_layers: int = 3
    num_nextn_predict_layers: int = 1
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict] = None
    compress_rope_theta: float = 160000.0
    max_position_embeddings: int = 1048576
    sliding_window: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    # Unused but present in config
    hc_mult_: int = 4
    compress_ratios: Optional[List[int]] = None
    # Indexer (for compress_ratio=4 layers)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512


# ---------- Pure-MLX ops ----------

def _hc_split_sinkhorn_ops(
    mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
    hc_mult: int, iters: int = 20, eps: float = 1e-6,
):
    """Pure-MLX implementation matching mlx_lm PR #1192 deepseek_v4 reference.
    Fallback when fused Metal kernel is unavailable (CPU backend or no Metal).

    Splits mixes (shape (..., (2+mult)*mult)) into (pre, post, comb):
      pre:  (..., mult)         — sigmoid + eps, NO normalization
      post: (..., mult)         — 2 * sigmoid, NO eps (factor of 2 is critical)
      comb: (..., mult, mult)   — doubly-stochastic via Sinkhorn
                                  (softmax init + col-norm + (iters-1) row/col iterations)
    """
    mixes = mixes.astype(mx.float32)
    hc_scale = hc_scale.astype(mx.float32)
    hc_base = hc_base.astype(mx.float32)
    mh = hc_mult
    pre_scale, post_scale, comb_scale = hc_scale[0], hc_scale[1], hc_scale[2]

    pre = mx.sigmoid(mixes[..., :mh] * pre_scale + hc_base[:mh]) + eps
    post = 2 * mx.sigmoid(mixes[..., mh:2 * mh] * post_scale + hc_base[mh:2 * mh])
    comb = mx.reshape(
        mixes[..., 2 * mh:] * comb_scale,
        mixes.shape[:-1] + (mh, mh),
    ) + mx.reshape(hc_base[2 * mh:], (mh, mh))
    comb = mx.softmax(comb, axis=-1, precise=True) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(max(iters - 1, 0)):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def _make_hc_split_sinkhorn_kernel():
    """Fused Metal kernel for HC Sinkhorn. Ports mlx-lm PR #1192 latest optim
    (commit c0d9222d, 2026-04-24). Does the entire pre/post/comb compute in
    a SINGLE GPU kernel launch — avoids 40+ intermediate MLX op graphs per
    layer × 43 layers = 3000+ graph nodes saved per token.

    Returns None if Metal is unavailable (fallback to pure-ops path).
    """
    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = """
        uint idx = thread_position_in_grid.x;
        constexpr int MIX = (2 + HC) * HC;
        float epsv = static_cast<float>(eps[0]);

        auto mix = mixes + idx * MIX;
        auto pre_out = pre + idx * HC;
        auto post_out = post + idx * HC;
        auto comb_out = comb + idx * HC * HC;

        float pre_scale = static_cast<float>(scale[0]);
        float post_scale = static_cast<float>(scale[1]);
        float comb_scale = static_cast<float>(scale[2]);

        for (int i = 0; i < HC; ++i) {
            float z = static_cast<float>(mix[i]) * pre_scale
                + static_cast<float>(base[i]);
            pre_out[i] = 1.0f / (1.0f + metal::fast::exp(-z)) + epsv;
        }
        for (int i = 0; i < HC; ++i) {
            int off = HC + i;
            float z = static_cast<float>(mix[off]) * post_scale
                + static_cast<float>(base[off]);
            post_out[i] = 2.0f / (1.0f + metal::fast::exp(-z));
        }

        float c[HC * HC];
        for (int i = 0; i < HC; ++i) {
            float row_max = -INFINITY;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                int off = 2 * HC + cidx;
                float v = static_cast<float>(mix[off]) * comb_scale
                    + static_cast<float>(base[off]);
                c[cidx] = v;
                row_max = metal::max(row_max, v);
            }
            float row_sum = 0.0f;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                float v = metal::fast::exp(c[cidx] - row_max);
                c[cidx] = v;
                row_sum += v;
            }
            float inv_sum = 1.0f / row_sum;
            for (int j = 0; j < HC; ++j) {
                int cidx = i * HC + j;
                c[cidx] = c[cidx] * inv_sum + epsv;
            }
        }

        for (int j = 0; j < HC; ++j) {
            float col_sum = 0.0f;
            for (int i = 0; i < HC; ++i) {
                col_sum += c[i * HC + j];
            }
            float inv_denom = 1.0f / (col_sum + epsv);
            for (int i = 0; i < HC; ++i) {
                c[i * HC + j] *= inv_denom;
            }
        }

        for (int iter = 1; iter < ITERS; ++iter) {
            for (int i = 0; i < HC; ++i) {
                float row_sum = 0.0f;
                for (int j = 0; j < HC; ++j) {
                    row_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (row_sum + epsv);
                for (int j = 0; j < HC; ++j) {
                    c[i * HC + j] *= inv_denom;
                }
            }
            for (int j = 0; j < HC; ++j) {
                float col_sum = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    col_sum += c[i * HC + j];
                }
                float inv_denom = 1.0f / (col_sum + epsv);
                for (int i = 0; i < HC; ++i) {
                    c[i * HC + j] *= inv_denom;
                }
            }
        }

        for (int i = 0; i < HC * HC; ++i) {
            comb_out[i] = c[i];
        }
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_split_sinkhorn",
        input_names=["mixes", "scale", "base", "eps"],
        output_names=["pre", "post", "comb"],
        source=source,
    )


_hc_split_sinkhorn_kernel = _make_hc_split_sinkhorn_kernel()
_hc_eps_array_cache = None


def hc_split_sinkhorn(
    mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
    hc_mult: int, iters: int = 20, eps: float = 1e-6,
):
    """Public API — dispatches to fused Metal kernel when available.
    Same output semantics as `_hc_split_sinkhorn_ops`.
    """
    if _hc_split_sinkhorn_kernel is None:
        return _hc_split_sinkhorn_ops(
            mixes,
            hc_scale=hc_scale,
            hc_base=hc_base,
            hc_mult=hc_mult,
            iters=iters,
            eps=eps,
        )
    global _hc_eps_array_cache
    if _hc_eps_array_cache is None:
        _hc_eps_array_cache = mx.array([eps], dtype=mx.float32)
    return _hc_split_sinkhorn_kernel(
        inputs=[mixes, hc_scale, hc_base, _hc_eps_array_cache],
        template=[("HC", hc_mult), ("ITERS", iters)],
        grid=(mixes.size // ((2 + hc_mult) * hc_mult), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult),
            (*mixes.shape[:-1], hc_mult, hc_mult),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )


# ---------- Attention (simplified: full scaled_dot_product) ----------


def _activation_blocks(x: mx.array, block_size: int):
    """Pad only a non-production tail and expose last-axis QAT blocks.

    The official 0731 dimensions are exact multiples of 64 (FP8) or 32
    (FP4).  Padding keeps small architecture tests useful without changing
    any value or scale for those production shapes.
    """

    if block_size <= 0:
        raise ValueError("activation QAT block_size must be positive")
    shape = tuple(x.shape)
    width = int(shape[-1])
    if width == 0:
        return x.astype(mx.float32), shape, width
    padded_width = ((width + block_size - 1) // block_size) * block_size
    x32 = x.astype(mx.float32)
    if padded_width != width:
        pad_width = [(0, 0)] * x.ndim
        pad_width[-1] = (0, padded_width - width)
        x32 = mx.pad(x32, pad_width)
    return x32.reshape(*shape[:-1], -1, block_size), shape, width


def _round_e4m3fn(x: mx.array) -> mx.array:
    """Round FP32 values to the finite-only E4M3 grid, returning FP32.

    MLX 0.31 has no float8 dtype.  This implements the exact E4M3FN value
    lattice used by ``torch.float8_e4m3fn``: 2^-9 subnormal spacing, three
    mantissa bits per normal binade, and a maximum finite value of 448.
    ``mx.round`` is round-to-nearest-even, matching the reference cast.
    """

    clipped = mx.clip(x, -448.0, 448.0)
    magnitude = mx.abs(clipped)
    min_normal = 2.0**-6
    subnormal_step = 2.0**-9
    safe_magnitude = mx.maximum(magnitude, min_normal)
    normal_step = mx.power(
        mx.array(2.0, dtype=mx.float32),
        mx.floor(mx.log2(safe_magnitude)) - 3.0,
    )
    step = mx.where(magnitude < min_normal, subnormal_step, normal_step)
    rounded = mx.minimum(mx.round(magnitude / step) * step, 448.0)
    rounded = mx.where(clipped < 0, -rounded, rounded)
    return mx.where(mx.isnan(x), x, rounded)


def act_quant_sim(x: mx.array, block_size: int = 64) -> mx.array:
    """Official 0731 FP8-E4M3 activation QAT round-trip.

    This mirrors ``inference/kernel.py::act_quant(..., inplace=True)`` for
    the model's UE8M0 configuration: blockwise absolute maximum, power-of-two
    scale, true E4M3FN rounding, dequantization, then restoration to the input
    dtype.  It is source-native model math, not an optional runtime heuristic.
    """

    blocks, shape, width = _activation_blocks(x, block_size)
    if width == 0:
        return x
    amax = mx.maximum(
        mx.max(mx.abs(blocks), axis=-1, keepdims=True),
        1e-4,
    )
    scale = mx.power(
        mx.array(2.0, dtype=mx.float32),
        mx.ceil(mx.log2(amax / 448.0)),
    )
    result = (_round_e4m3fn(blocks / scale) * scale).reshape(
        *shape[:-1], -1
    )[..., :width]
    return result.astype(x.dtype)


def _round_e2m1fn(x: mx.array) -> mx.array:
    """Round FP32 values to the finite E2M1 grid, returning FP32."""

    clipped = mx.clip(x, -6.0, 6.0)
    magnitude = mx.abs(clipped)
    # Positive E2M1 finite values are {0,.5,1,1.5,2,3,4,6}.  Boundary
    # comparisons encode round-to-nearest-even at the unequal-width ties.
    rounded = mx.where(
        magnitude <= 0.25,
        0.0,
        mx.where(
            magnitude < 0.75,
            0.5,
            mx.where(
                magnitude <= 1.25,
                1.0,
                mx.where(
                    magnitude < 1.75,
                    1.5,
                    mx.where(
                        magnitude <= 2.5,
                        2.0,
                        mx.where(
                            magnitude < 3.5,
                            3.0,
                            mx.where(magnitude <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    rounded = mx.where(clipped < 0, -rounded, rounded)
    return mx.where(mx.isnan(x), x, rounded)


def fp4_act_quant_sim(x: mx.array, block_size: int = 32) -> mx.array:
    """Official 0731 FP4-E2M1 activation QAT round-trip."""

    blocks, shape, width = _activation_blocks(x, block_size)
    if width == 0:
        return x
    amax = mx.maximum(
        mx.max(mx.abs(blocks), axis=-1, keepdims=True),
        6.0 * (2.0**-126),
    )
    scale = mx.power(
        mx.array(2.0, dtype=mx.float32),
        mx.ceil(mx.log2(amax / 6.0)),
    )
    result = (_round_e2m1fn(blocks / scale) * scale).reshape(
        *shape[:-1], -1
    )[..., :width]
    return result.astype(x.dtype)


def hadamard_rotate_activation(x: mx.array) -> mx.array:
    """Normalized Sylvester Hadamard transform used by the 0731 indexer."""

    width = int(x.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ValueError(
            "DeepSeek-V4 indexer QAT requires a positive power-of-two head dim"
        )
    # MLX defaults to the same orthonormal 1/sqrt(width) scale as the official
    # 0731 path, while dispatching to its optimized native implementation.
    return mx.hadamard_transform(x).astype(x.dtype)


def _fp8_qat_non_rope_ops(x: mx.array, rope_dims: int) -> mx.array:
    """Apply block-64 E4M3 QAT while preserving positional dimensions."""

    split = int(x.shape[-1]) - int(rope_dims)
    if split <= 0:
        return x
    return mx.concatenate(
        [act_quant_sim(x[..., :split], 64), x[..., split:]],
        axis=-1,
    )


def _indexer_activation_roundtrip_ops(x: mx.array) -> mx.array:
    """Pure-MLX fallback for the official Hadamard-128 + E2M1 graph."""

    rotated = mx.hadamard_transform(x.astype(mx.float32))
    return fp4_act_quant_sim(rotated, 32).astype(x.dtype)


def _make_e4m3_kv_activation_roundtrip_kernel():
    """Fuse the source-native post-RoPE KV QAT graph into one Metal launch."""

    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = r"""
        const uint gid = thread_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint group = gid >> 6;
        const uint block = group % NBT;
        const uint row = group / NBT;
        const uint idx = row * N + block * 64 + lane;

        if (block >= NBQ) {
            y[idx] = static_cast<outT>(x[idx]);
        } else {
            threadgroup float scratch[64];
            const float input_value = static_cast<float>(x[idx]);
            scratch[lane] = metal::abs(input_value);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 32; stride > 0; stride >>= 1) {
                if (lane < stride) {
                    scratch[lane] = metal::max(
                        scratch[lane], scratch[lane + stride]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const float amax = metal::max(scratch[0], 1.0e-4f);
            const float raw_scale = amax / 448.0f;
            const uint raw_bits = as_type<uint>(raw_scale);
            const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
            const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
            const int scale_exp = raw_exp + int(has_mantissa);
            const float scale = as_type<float>(uint(scale_exp + 127) << 23);

            const float normalized = metal::clamp(
                input_value / scale, -448.0f, 448.0f);
            const float sign = normalized < 0.0f ? -1.0f : 1.0f;
            const float absolute = metal::min(metal::abs(normalized), 448.0f);
            int low = 0;
            int high = 126;
            while (low < high) {
                const int middle = (low + high + 1) >> 1;
                const int exponent = (middle >> 3) & 0x0f;
                const int mantissa = middle & 0x07;
                const float candidate = exponent == 0
                    ? float(mantissa) * 0.001953125f
                    : (1.0f + float(mantissa) * 0.125f)
                        * metal::fast::exp2(float(exponent - 7));
                if (candidate <= absolute) low = middle;
                else high = middle - 1;
            }

            int best = low;
            const int best_exponent = (best >> 3) & 0x0f;
            const int best_mantissa = best & 0x07;
            float best_value = best_exponent == 0
                ? float(best_mantissa) * 0.001953125f
                : (1.0f + float(best_mantissa) * 0.125f)
                    * metal::fast::exp2(float(best_exponent - 7));
            if (best < 126) {
                const int next = best + 1;
                const int next_exponent = (next >> 3) & 0x0f;
                const int next_mantissa = next & 0x07;
                const float next_value = next_exponent == 0
                    ? float(next_mantissa) * 0.001953125f
                    : (1.0f + float(next_mantissa) * 0.125f)
                        * metal::fast::exp2(float(next_exponent - 7));
                const float best_diff = metal::abs(absolute - best_value);
                const float next_diff = metal::abs(absolute - next_value);
                if (next_diff < best_diff ||
                    (next_diff == best_diff && (next & 1) == 0 && (best & 1) != 0)) {
                    best_value = next_value;
                }
            }
            y[idx] = static_cast<outT>(sign * best_value * scale);
        }
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_e4m3_kv_activation_roundtrip",
        input_names=["x"],
        output_names=["y"],
        source=source,
    )


def _make_indexer_activation_roundtrip_kernel():
    """Fuse official Hadamard-128 and block-32 E2M1 QAT on Metal."""

    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = r"""
        const uint gid = thread_position_in_grid.x;
        const uint lane = thread_position_in_threadgroup.x;
        const uint row = gid >> 7;
        const uint idx = row * 128 + lane;
        threadgroup float values[128];
        threadgroup float magnitudes[128];

        values[lane] = static_cast<float>(x[idx]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 128; stride <<= 1) {
            if (lane < 64) {
                const uint block = lane / stride;
                const uint offset = lane % stride;
                const uint low_idx = block * 2 * stride + offset;
                const uint high_idx = low_idx + stride;
                const float low = values[low_idx];
                const float high = values[high_idx];
                values[low_idx] = low + high;
                values[high_idx] = low - high;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float rotated = values[lane] * 0.08838834764831845f;
        magnitudes[lane] = metal::abs(rotated);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 16; stride > 0; stride >>= 1) {
            if ((lane & 31u) < stride) {
                magnitudes[lane] = metal::max(
                    magnitudes[lane], magnitudes[lane + stride]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint block_start = lane & ~31u;
        const float amax = metal::max(
            magnitudes[block_start], 7.052966104933725e-38f);
        const float raw_scale = amax / 6.0f;
        const uint raw_bits = as_type<uint>(raw_scale);
        const int raw_exp = int((raw_bits >> 23) & 0xffu) - 127;
        const bool has_mantissa = (raw_bits & 0x7fffffu) != 0u;
        const int scale_exp = raw_exp + int(has_mantissa);
        const float scale = as_type<float>(uint(scale_exp + 127) << 23);

        const float normalized = metal::clamp(rotated / scale, -6.0f, 6.0f);
        const float absolute = metal::abs(normalized);
        constexpr float codebook[8] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
        };
        int best = 0;
        float best_diff = absolute;
        for (int code = 1; code < 8; ++code) {
            const float diff = metal::abs(absolute - codebook[code]);
            if (diff < best_diff ||
                (diff == best_diff && (code & 1) == 0 && (best & 1) != 0)) {
                best = code;
                best_diff = diff;
            }
        }
        const float sign = normalized < 0.0f ? -1.0f : 1.0f;
        y[idx] = static_cast<outT>(sign * codebook[best] * scale);
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_indexer_hadamard128_e2m1_roundtrip",
        input_names=["x"],
        output_names=["y"],
        source=source,
    )


_e4m3_kv_activation_roundtrip_kernel = (
    _make_e4m3_kv_activation_roundtrip_kernel()
)
_indexer_activation_roundtrip_kernel = (
    _make_indexer_activation_roundtrip_kernel()
)


_DSV4_ACTIVATION_QAT_STATUS = {
    "env": _DSV4_ACTIVATION_QAT_ENV,
    "default": False,
    "requested": _DSV4_ACTIVATION_QAT_REQUESTED,
    "effective": _DSV4_ACTIVATION_QAT_REQUESTED,
    # ``observed`` is deliberately not inferred from the environment.  It is
    # populated only after both owned graph paths have actually run.
    "observed": None,
    "attested": False,
    "e4m3_kv_pool_observed": None,
    "hadamard_fp4_indexer_observed": None,
    "implementation_available": True,
    "fused_e4m3_available": _e4m3_kv_activation_roundtrip_kernel is not None,
    "fused_indexer_available": _indexer_activation_roundtrip_kernel is not None,
    "fp32_compressor_staging_unconditional": True,
    # These observations attest the two shared transform dispatch families,
    # not every individual call site that may use each helper later in a long
    # sequence (for example the deferred indexer-query call site).
    "attestation_scope": "transform_family_dispatch_not_every_call_site",
    "transform_families": [
        "e4m3_post_rope_kv_or_compressor_pool_dispatch",
        "hadamard_fp4_indexer_pool_or_query_dispatch",
    ],
}


def get_dsv4_activation_qat_status() -> Dict[str, Any]:
    """Return the current process-local activation-QAT attestation."""

    status = dict(_DSV4_ACTIVATION_QAT_STATUS)
    status["transform_families"] = list(status["transform_families"])
    return status


def _observe_dsv4_activation_qat_path(path: str) -> bool:
    """Record the actual active/bypass state at an owned graph boundary."""

    if path not in {
        "e4m3_kv_pool_observed",
        "hadamard_fp4_indexer_observed",
    }:
        raise ValueError(f"unknown DSV4 activation-QAT path: {path}")
    effective = bool(_DSV4_ACTIVATION_QAT_STATUS["effective"])
    # The process-local setting is immutable after import.  Once a path has
    # attested its branch, keep decode hot paths free of repeated dict writes.
    if _DSV4_ACTIVATION_QAT_STATUS[path] is not None:
        return effective
    _DSV4_ACTIVATION_QAT_STATUS[path] = effective
    path_states = (
        _DSV4_ACTIVATION_QAT_STATUS["e4m3_kv_pool_observed"],
        _DSV4_ACTIVATION_QAT_STATUS["hadamard_fp4_indexer_observed"],
    )
    attested = all(value is not None for value in path_states)
    _DSV4_ACTIVATION_QAT_STATUS["attested"] = attested
    _DSV4_ACTIVATION_QAT_STATUS["observed"] = (
        effective if attested and path_states[0] == path_states[1] else None
    )
    return effective


def _fp8_qat_non_rope(x: mx.array, rope_dims: int) -> mx.array:
    """Dispatch the official post-RoPE KV QAT graph without changing math."""

    split = int(x.shape[-1]) - int(rope_dims)
    width = int(x.shape[-1])
    if split <= 0:
        return x
    if not _observe_dsv4_activation_qat_path("e4m3_kv_pool_observed"):
        return x
    try:
        use_metal = mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        use_metal = False
    if (
        _e4m3_kv_activation_roundtrip_kernel is None
        or not use_metal
        or width % 64
        or split % 64
        or int(x.size) == 0
    ):
        return _fp8_qat_non_rope_ops(x, rope_dims)
    contiguous = mx.contiguous(x)
    rows = int(contiguous.size) // width
    return _e4m3_kv_activation_roundtrip_kernel(
        inputs=[contiguous],
        template=[
            ("N", width),
            ("NBQ", split // 64),
            ("NBT", width // 64),
            ("outT", contiguous.dtype),
        ],
        grid=(rows * width, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[contiguous.shape],
        output_dtypes=[contiguous.dtype],
    )[0]


def _indexer_activation_roundtrip(x: mx.array) -> mx.array:
    """Dispatch the official indexer activation graph without changing math."""

    if not _observe_dsv4_activation_qat_path(
        "hadamard_fp4_indexer_observed"
    ):
        return x
    try:
        use_metal = mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        use_metal = False
    if (
        _indexer_activation_roundtrip_kernel is None
        or not use_metal
        or int(x.shape[-1]) != 128
        or int(x.size) == 0
    ):
        return _indexer_activation_roundtrip_ops(x)
    contiguous = mx.contiguous(x)
    rows = int(contiguous.size) // 128
    return _indexer_activation_roundtrip_kernel(
        inputs=[contiguous],
        template=[("outT", contiguous.dtype)],
        grid=(rows * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[contiguous.shape],
        output_dtypes=[contiguous.dtype],
    )[0]


class DeepseekV4RoPE(nn.Module):
    """Port of PR #1192 DeepseekV4RoPE — on-the-fly cos/sin for YaRN."""
    def __init__(self, dims, base, scaling_config=None, max_position_embeddings=1048576):
        super().__init__()
        self.dims = dims
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        rope_type = None
        if scaling_config is not None:
            rope_type = scaling_config.get("type") or scaling_config.get("rope_type")
        if rope_type in ("yarn", "deepseek_yarn"):
            factor = scaling_config["factor"]
            orig = scaling_config["original_max_position_embeddings"]
            beta_fast = scaling_config.get("beta_fast", 32)
            beta_slow = scaling_config.get("beta_slow", 1)
            def correction_dim(n):
                return dims * math.log(orig / (n * 2 * math.pi)) / (2 * math.log(base))
            low = max(math.floor(correction_dim(beta_fast)), 0)
            high = min(math.ceil(correction_dim(beta_slow)), dims - 1)
            if low == high:
                high += 0.001
            ramp = (mx.arange(dims // 2, dtype=mx.float32) - low) / (high - low)
            smooth = 1 - mx.clip(ramp, 0, 1)
            inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth
        elif rope_type not in (None, "default"):
            raise ValueError(f"Unsupported DeepSeek-V4 RoPE type: {rope_type}")
        self._inv_freq = (inv_freq,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x, offset=0, inverse=False, positions=None):
        # NOTE: mx.fast.rope was tried as a fast path here but produced
        # incoherent output (likely an inv_freq layout/scale convention
        # mismatch with YaRN-modified freqs). Reverted to manual cos/sin
        # path which is verified bit-exact against PR #1192 reference.
        # Future: investigate exact mx.fast.rope freqs format requirements.
        dtype = x.dtype
        L = x.shape[-2]
        pos = (
            mx.arange(offset, offset + L, dtype=mx.float32)
            if positions is None
            else positions.astype(mx.float32)
        )
        freqs = pos[:, None] * self.inv_freq[None, :]
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        if inverse:
            sin = -sin
        broadcast_shape = (1,) * (x.ndim - 2) + cos.shape
        cos = cos.reshape(broadcast_shape).astype(dtype)
        sin = sin.reshape(broadcast_shape).astype(dtype)
        x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = x[..., 0], x[..., 1]
        out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
        return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


# Decode-path host-encoding reduction: fuse per-token elementwise glue into a
# small number of large `mx.compile` regions. DSV4 decode is host-bound (~30ms
# of a ~42ms token is CPU graph build + Metal kernel encoding), so collapsing
# hundreds of tiny ops into few compiled dispatches is the primary speed lever.
# All compiled paths are gated to single-token shapes (L == 1) so prefill keeps
# the raw graph (bounded trace count) and every region was verified bit-exact
# (max|diff| = 0.0) against the raw path before adoption.
_DSV4_COMPILE_REGIONS = os.environ.get("DSV4_COMPILE_REGIONS", "1") == "1"

# Diagnostic-only ceiling probe: freeze compressed pools during decode to
# measure token speed with zero compressor/indexer bookkeeping ops. Produces
# WRONG long-range attention — never enable outside perf diagnosis.
_DSV4_PERF_PROBE_SKIP_POOL = os.environ.get("DSV4_PERF_PROBE_SKIP_POOL", "0") == "1"

# Decode pool-bookkeeping deferral. Between compression-window boundaries the
# compressor/indexer emit ZERO pool rows, yet the per-token path still pays
# buffer concat/slice ops + front projections across 43 layers (~5.1ms/token
# of host graph-build + kernel-encode, measured via the frozen-pool probe).
# Deferral buffers raw hidden-state refs in Python lists and runs the real
# advancement in one batched call at the token where a window completes — the
# emitted rows and their visibility timing are identical to the per-token
# path, so pool contents match token-for-token. External state readers
# (anchors, prefix-cache extraction, block export, `state` property) flush
# pending tokens first via `DeepseekV4Cache.flush_pool_pending()`.
_DSV4_DEFER_MODE = int(os.environ.get("DSV4_DEFER_POOL", "1") or "1")
# 0 = off, 1 = defer + window-boundary flush (production), 9 = perf probe:
# defer path but never append/flush (stale pool, WRONG numerics — diagnostic
# for separating bookkeeping cost from live-pool growth cost).
_DSV4_DEFER_POOL = _DSV4_DEFER_MODE != 0
# One-time diagnostics proving the deferral path engages and actually batches.
_DSV4_DEFER_DIAG = {"entered": False, "batched": False}


def _dsv4_rope_pair_rotate(x, cos, sin):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    out = mx.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis=-1)
    return out.reshape(*out.shape[:-2], out.shape[-2] * 2)


def _dsv4_rope_rotate(x, pos, inv_freq, inverse):
    """Same numerics/op-order as `DeepseekV4RoPE.__call__` + partial split,
    expressed over explicit `(pos, inv_freq)` inputs so it can live inside an
    `mx.compile` region (value changes in `pos` never retrace)."""
    dtype = x.dtype
    rope_dim = inv_freq.shape[0] * 2
    freqs = pos.astype(mx.float32)[:, None] * inv_freq[None, :]
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    if inverse:
        sin = -sin
    broadcast_shape = (1,) * (x.ndim - 2) + freqs.shape
    cos = cos.reshape(broadcast_shape).astype(dtype)
    sin = sin.reshape(broadcast_shape).astype(dtype)
    if x.shape[-1] == rope_dim:
        return _dsv4_rope_pair_rotate(x, cos, sin)
    nope, pe = mx.split(x, [x.shape[-1] - rope_dim], axis=-1)
    pe = _dsv4_rope_pair_rotate(pe, cos, sin)
    return mx.concatenate([nope, pe], axis=-1)


@mx.compile
def _dsv4_rope_rot_fwd(x, pos, inv_freq):
    return _dsv4_rope_rotate(x, pos, inv_freq, inverse=False)


@mx.compile
def _dsv4_rope_rot_inv(x, pos, inv_freq):
    return _dsv4_rope_rotate(x, pos, inv_freq, inverse=True)


def _apply_partial_rope(x, rope, offset=0, inverse=False, positions=None):
    if _DSV4_COMPILE_REGIONS and x.shape[-2] == 1:
        pos = (
            mx.arange(offset, offset + 1, dtype=mx.float32)
            if positions is None
            else positions
        )
        fn = _dsv4_rope_rot_inv if inverse else _dsv4_rope_rot_fwd
        return fn(x, pos, rope.inv_freq)
    rope_dim = rope.dims
    if x.shape[-1] == rope_dim:
        return rope(x, offset=offset, inverse=inverse, positions=positions)
    nope, pe = mx.split(x, [x.shape[-1] - rope_dim], axis=-1)
    pe = rope(pe, offset=offset, inverse=inverse, positions=positions)
    return mx.concatenate([nope, pe], axis=-1)


class DeepseekV4Cache:
    """Simplified cache for DSV4: wraps a RotatingKVCache (sliding-window local
    attention) with `compressor_state` + `indexer_state` cumulative pool buffers
    (HSA + CSA cross-window compressed-global-context).

    The cache is constructed per layer with that layer's `compress_ratio`. The
    ratio is needed by `trim()` to compute proportional pool truncation —
    each compressed pool row covers `compress_ratio` underlying KV tokens, so
    truncating the local KV by `n` tokens means dropping `n // ratio`
    compressed rows from the tail (the latest, output-side rows that the
    trim is specifically meant to discard).

    For short prompts (<sliding_window), the plain rotating-cache behavior is
    equivalent to RotatingKVCache.

    `compress_ratio=None` is accepted for backward compatibility — `trim()`
    falls back to the v2.5.14 full-reset behavior in that case (correct but
    pays full pool re-derivation on every multi-turn). The `make_cache`
    factory always passes the layer's actual ratio so this fallback only
    fires for callers that constructed the cache before this signature
    extension."""
    def __init__(self, sliding_window, compress_ratio=None):
        from mlx_lm.models.cache import RotatingKVCache
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}
        self.indexer_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}
        # Deferred-decode pool bookkeeping (see _DSV4_DEFER_POOL). Kept in a
        # parallel structure so the canonical 3-key state dicts stay clean for
        # the `state` tuple, anchors, and engine-side key-by-key copies.
        self._pool_pending = None
        # `compress_ratio` is the per-layer attention compression ratio used
        # by Compressor.accumulate_windows / update_pool. Stored on the cache
        # so `trim()` can do proportional pool-row truncation
        # (row_begin = p0 / ratio, row_end = ceil(p1 / ratio)).
        # When unset, `trim()` falls back to full reset.
        self.compress_ratio = compress_ratio

    @property
    def offset(self):
        return self.local.offset

    @property
    def keys(self):
        return self.local.keys

    @keys.setter
    def keys(self, value):
        self.local.keys = value

    @property
    def state(self):
        """Cache state tuple — mlx_lm generate iterates this for pipelined evaluation."""
        self.flush_pool_pending()
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(self.compressor_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
            tuple(self.indexer_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
        )

    @state.setter
    def state(self, value):
        self._pool_pending = None
        local_state, compressor_state, indexer_state = value
        if local_state is None:
            self.local.keys = None
            self.local.values = None
        else:
            self.local.state = local_state
        self.compressor_state = dict(zip(("buffer_kv", "buffer_gate", "pooled"), compressor_state))
        self.indexer_state = dict(zip(("buffer_kv", "buffer_gate", "pooled"), indexer_state))

    @property
    def meta_state(self):
        return self.local.meta_state

    @meta_state.setter
    def meta_state(self, value):
        self.local.meta_state = value

    def update_and_fetch(self, keys, values):
        return self.local.update_and_fetch(keys, values)

    def make_mask(self, *a, **k):
        return self.local.make_mask(*a, **k)

    def is_trimmable(self):
        return self.local.is_trimmable()

    def trim(self, n):
        """Trim local KV by n tokens AND truncate compressor + indexer pool state.

        Why this matters
        ----------------
        `DeepseekV4Cache` wraps a `RotatingKVCache` (`self.local`) for the
        sliding-window local attention path PLUS two cumulative pool-state
        dicts (`compressor_state`, `indexer_state`) holding the HSA / CSA
        cross-window compressed-global-context. Every forward pass updates
        the pool via `accumulate_windows` (window-aligned KV/gate buffers)
        and `update_pool` (appended pooled vectors), driven by running KV
        positions.

        The bug `trim()` fixes: pre-2.5.14, `trim(n)` only delegated to
        `self.local.trim(n)` — truncating local KV by n tokens without
        touching the pool. The scheduler then stored the half-truncated
        cache to the prefix cache for next-turn reuse. The contaminated
        pool — built from output-side tokens that trim was meant to
        discard — got restored on the next turn, and the model's
        pool-attention path read global-context vectors derived from
        prior turns' GENERATED OUTPUT. Symptom on DSV4-Flash:
        polite-assistant attractor loops on /v1/chat/completions.
        Bench mode (SimpleEngine, no cache reuse) was unaffected — proof
        the model itself is sound, only cross-turn pool-state survival
        was broken.

        Strategy: proportional row truncation
        -------------------------------------
        Each `pooled` row covers `compress_ratio` underlying KV tokens.
        After `self.local.trim(n)` removes the latest n KV tokens, the
        latest `n // compress_ratio` pool rows correspond to those
        discarded tokens and must go. Earlier pool rows remain valid
        because they were built from KV positions that survived the
        trim.

        Row range: `row_begin = p0 / ratio`, `row_end = ceil(p1 /
        ratio)`. Multi-turn long-context
        chats now keep their compressed history across turns instead of
        re-deriving the entire pool from scratch every turn (which
        2.5.14's full-reset did).

        `buffer_kv` and `buffer_gate` are partial-window buffers — tokens
        that haven't yet filled a complete window for compression. Their
        START_POS may fall in the discarded range or at a position where
        the upstream window is now incomplete; safest to clear them
        unconditionally and let `accumulate_windows` rebuild them on the
        next forward.

        Fallback: when `compress_ratio` is None (cache constructed via
        the legacy single-arg signature), fall back to v2.5.14's full
        pool reset — still correct, just heavier.
        """
        # Deferred pool tokens are the newest unprocessed tokens; any trim
        # discards them. This is equivalent to the per-token path, which would
        # have buffered them into buffer_kv/buffer_gate and then cleared
        # those buffers unconditionally below (pending never spans a window
        # boundary, so it can never have contributed pool rows).
        self._drop_pool_pending()

        # Trim KV first so we know the new total length for proportional
        # pool truncation.
        rv = self.local.trim(n)

        # Clear partial-window buffers unconditionally. They hold
        # incompletely-filled window state keyed by start_pos which is
        # invalidated by ANY trim. `accumulate_windows` re-derives them
        # from the kept KV on the next forward (see `update_pool`
        # docstring, lines 462-471, which handles `pooled is None`).
        for state in (self.compressor_state, self.indexer_state):
            state["buffer_kv"] = None
            state["buffer_gate"] = None

        ratio = self.compress_ratio
        if ratio is None or ratio <= 0:
            # Legacy / unknown-ratio path: full reset (v2.5.14 fallback).
            for state in (self.compressor_state, self.indexer_state):
                state["pooled"] = None
            return rv

        # Proportional pool truncation. `n // ratio` is the number of
        # pool rows that became stale (latest, output-side rows). For
        # boundary safety: always discard at least one trailing row
        # when n > 0, since the most-recently-appended pool row may
        # have been computed from a window that overlapped output
        # tokens — keeping it would re-introduce the contamination.
        rows_to_drop = max(1, n // ratio) if n > 0 else 0
        if rows_to_drop == 0:
            return rv

        for state in (self.compressor_state, self.indexer_state):
            pooled = state["pooled"]
            if pooled is None:
                continue
            n_rows = pooled.shape[1]
            keep = max(0, n_rows - rows_to_drop)
            if keep == 0:
                state["pooled"] = None
            elif keep < n_rows:
                # Slice axis=1 (the window/row axis) to first `keep`
                # entries. `pooled.shape == (B, n_rows, dim)` per
                # `update_pool` line 466.
                state["pooled"] = pooled[:, :keep, :]
        return rv

    def size(self):
        return self.local.size()

    def empty(self):
        return self.local.empty()

    @property
    def nbytes(self):
        total = self.local.nbytes
        for state in (self.compressor_state, self.indexer_state):
            for value in state.values():
                if value is not None:
                    total += value.nbytes
        if self._pool_pending:
            for slot in self._pool_pending.values():
                if slot:
                    for arr in slot["xs"]:
                        total += arr.nbytes
        return total

    def _branch_state(self, key):
        return self.indexer_state if key == "indexer_state" else self.compressor_state

    def _pool_pending_slot(self, state_key):
        pending = self._pool_pending
        if pending is None:
            pending = {"compressor_state": None, "indexer_state": None}
            self._pool_pending = pending
        slot = pending.get(state_key)
        if slot is None:
            slot = {"xs": [], "base": 0, "module": None, "rope": None, "last_pool": None}
            pending[state_key] = slot
        return slot

    def note_pool_result(self, state_key, module, rope, result):
        """Memoize the last pool object returned by a real advancement so
        deferred tokens can hand the identical object to attention."""
        slot = self._pool_pending_slot(state_key)
        slot["module"] = module
        slot["rope"] = rope
        slot["last_pool"] = result

    def defer_pool_token(self, module, x, rope, start_pos, state_key):
        """Defer one decode token's compressor/indexer advancement.

        Non-boundary tokens emit zero pool rows in the per-token path, so
        buffering the raw hidden-state ref and advancing in one batched call
        at the window-completion token produces identical rows at identical
        token positions while skipping the per-token buffer bookkeeping ops.
        """
        slot = self._pool_pending_slot(state_key)
        if _DSV4_DEFER_MODE == 9 and slot["last_pool"] is not None:
            return slot["last_pool"]
        if not slot["xs"]:
            slot["base"] = int(start_pos)
        slot["module"] = module
        slot["rope"] = rope
        slot["xs"].append(x)
        ratio = int(module.compress_ratio)
        buf = self._branch_state(state_key).get("buffer_kv")
        blen = 0 if buf is None else int(buf.shape[1])
        if module.overlap and blen >= ratio:
            blen -= ratio
        if not _DSV4_DEFER_DIAG["entered"]:
            _DSV4_DEFER_DIAG["entered"] = True
            logger.info(
                "DSV4 pool deferral engaged: state_key=%s ratio=%d blen=%d "
                "last_pool=%s", state_key, ratio, blen,
                "set" if slot["last_pool"] is not None else "None",
            )
        if slot["last_pool"] is None or blen + len(slot["xs"]) >= ratio:
            # Window completes at this token (or no memoized pool yet on a
            # fresh/restored cache): run the real advancement now.
            return self._flush_pool_slot(state_key)
        return slot["last_pool"]

    def _flush_pool_slot(self, state_key):
        slot = self._pool_pending_slot(state_key)
        xs = slot["xs"]
        if not xs:
            return slot["last_pool"]
        slot["xs"] = []
        x = xs[0] if len(xs) == 1 else mx.concatenate(xs, axis=1)
        if len(xs) > 1 and not _DSV4_DEFER_DIAG["batched"]:
            _DSV4_DEFER_DIAG["batched"] = True
            logger.info(
                "DSV4 pool deferral first batched flush: state_key=%s k=%d",
                state_key, len(xs),
            )
        result = slot["module"]._advance(x, slot["rope"], self, slot["base"], state_key)
        slot["last_pool"] = result
        return result

    def flush_pool_pending(self):
        """Materialize deferred pool tokens before any external state read
        (anchors, block export, `state` tuple, prefix-cache extraction)."""
        if not self._pool_pending:
            return
        for state_key, slot in self._pool_pending.items():
            if slot and slot["xs"]:
                self._flush_pool_slot(state_key)

    def _drop_pool_pending(self):
        if not self._pool_pending:
            return
        for slot in self._pool_pending.values():
            if slot:
                slot["xs"] = []
                slot["last_pool"] = None

    @staticmethod
    def _copy_delta_tree(value, _collector=None):
        """Detach a small cache-record tree from the live mutable cache.

        ``value + mx.zeros_like(value)`` already graph-detaches each leaf: the
        term references the pre-mutation source array, not an alias, so the
        snapshot is well defined even before it is materialized. The eval only
        has to pin that snapshot before the next decode step overwrites rolling
        state. When ``_collector`` is supplied the caller batches that pinning:
        each detached leaf is appended for a single ``async_eval`` at the block
        boundary instead of one blocking ``mx.eval`` per leaf. At ~7 leaves x
        ~40 composite layers a forced-anchor boundary otherwise serializes a
        few hundred GPU round-trips onto the decode thread — the DSV4
        long-output per-256-token stall. Ordering safety is unchanged: the copy
        ops are submitted at the boundary, before the next decode's in-place
        writes, so same-stream ordering still reads pre-mutation values.
        """
        if value is None:
            return None
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            copied = value + mx.zeros_like(value)
            if _collector is None:
                mx.eval(copied)
            else:
                _collector.append(copied)
            return copied
        if isinstance(value, tuple):
            return tuple(
                DeepseekV4Cache._copy_delta_tree(item, _collector) for item in value
            )
        if isinstance(value, list):
            return [
                DeepseekV4Cache._copy_delta_tree(item, _collector) for item in value
            ]
        if isinstance(value, dict):
            return {
                key: DeepseekV4Cache._copy_delta_tree(item, _collector)
                for key, item in value.items()
            }
        return value

    def _export_pool_delta(self, state, start_row, end_row, _collector=None):
        """Export BF16 pool rows for one immutable token block."""
        from .cache_delta import DSV4_POOL_DELTA_SCHEMA

        pooled = state.get("pooled")
        if pooled is None:
            if start_row or end_row:
                raise ValueError("DSV4 pool rows are missing for a non-empty delta")
            value = None
        else:
            rows = int(pooled.shape[1])
            if start_row < 0 or end_row < start_row or end_row > rows:
                raise ValueError(
                    "DSV4 pool delta is outside retained rows: "
                    f"start={start_row} end={end_row} rows={rows}"
                )
            value = self._copy_delta_tree(pooled[:, start_row:end_row, :], _collector)
        return {
            "schema": DSV4_POOL_DELTA_SCHEMA,
            "storage": "bf16",
            "start_row": int(start_row),
            "end_row": int(end_row),
            "value": value,
        }

    def _append_pool_delta(self, state, delta):
        """Append one validated BF16 pool delta during checkpoint restore."""
        from .cache_delta import DSV4_POOL_DELTA_SCHEMA

        if not isinstance(delta, dict) or delta.get("schema") != DSV4_POOL_DELTA_SCHEMA:
            raise ValueError("unsupported DSV4 pool delta")
        if delta.get("storage") == "none":
            if int(delta.get("start_row", 0)) or int(delta.get("end_row", 0)):
                raise ValueError("absent DSV4 pool delta cannot declare rows")
            return
        if delta.get("storage") != "bf16":
            raise ValueError("base DeepseekV4Cache requires BF16 pool deltas")
        value = delta.get("value")
        expected_rows = int(delta.get("end_row", 0)) - int(delta.get("start_row", 0))
        if expected_rows < 0:
            raise ValueError("negative DSV4 pool delta row span")
        if value is None:
            if expected_rows:
                raise ValueError("non-empty DSV4 pool delta has no tensor")
            return
        if int(value.shape[1]) != expected_rows:
            raise ValueError("DSV4 pool delta tensor row count does not match metadata")
        current = state.get("pooled")
        current_rows = 0 if current is None else int(current.shape[1])
        if current_rows != int(delta.get("start_row", -1)):
            raise ValueError(
                "DSV4 pool delta is not contiguous: "
                f"current={current_rows} start={delta.get('start_row')}"
            )
        state["pooled"] = (
            value if current is None else mx.concatenate([current, value], axis=1)
        )

    def export_block_delta(
        self,
        start_token: int,
        end_token: int,
        *,
        block_size: int = 256,
        anchor_interval_blocks: int = 8,
        force_anchor: bool = False,
        _eval_collector=None,
    ):
        """Export one immutable native block record without flattening pools.

        Pool rows are emitted for every block.  Local rotating state and the
        incomplete compressor/indexer buffers are emitted only at periodic
        anchors or an explicitly requested request boundary.

        ``_eval_collector`` (optional): when the caller exports many layers at
        one block boundary it passes a shared list so every detached snapshot
        leaf is batched into a single ``async_eval`` instead of one blocking
        ``mx.eval`` per leaf. See ``_copy_delta_tree``.
        """
        from .cache_delta import DSV4_BLOCK_DELTA_SCHEMA

        self.flush_pool_pending()
        start = int(start_token)
        end = int(end_token)
        block_size = int(block_size)
        anchor_interval_blocks = int(anchor_interval_blocks)
        if start < 0 or end <= start:
            raise ValueError(f"invalid DSV4 block interval [{start}, {end})")
        if block_size <= 0 or anchor_interval_blocks <= 0:
            raise ValueError("DSV4 block and anchor intervals must be positive")
        if end - start > block_size:
            raise ValueError("DSV4 block delta exceeds configured block size")
        ratio = int(self.compress_ratio or 0)
        if ratio <= 0:
            raise ValueError("DeepseekV4Cache block deltas require a compression ratio")

        start_row = start // ratio
        end_row = end // ratio
        compressor_delta = self._export_pool_delta(
            self.compressor_state, start_row, end_row, _eval_collector
        )
        if ratio == 4:
            indexer_delta = self._export_pool_delta(
                self.indexer_state, start_row, end_row, _eval_collector
            )
        else:
            from .cache_delta import DSV4_POOL_DELTA_SCHEMA

            indexer_delta = {
                "schema": DSV4_POOL_DELTA_SCHEMA,
                "storage": "none",
                "start_row": 0,
                "end_row": 0,
            }
        anchor_interval = block_size * anchor_interval_blocks
        make_anchor = bool(force_anchor or end % anchor_interval == 0)
        anchor = None
        if make_anchor:
            from .cache_delta import canonical_rotating_window

            if self.local.empty():
                local_state = None
                local_meta_state = self.meta_state
            else:
                (
                    local_keys,
                    local_values,
                    local_max_size,
                    local_keep,
                    local_offset,
                    local_idx,
                ) = canonical_rotating_window(
                    self.local,
                    expected_offset=end,
                )
                local_state = (local_keys, local_values)
                local_meta_state = tuple(
                    map(
                        str,
                        (local_keep, local_max_size, local_offset, local_idx),
                    )
                )
            anchor = {
                "tokens": end,
                "periodic": bool(end % anchor_interval == 0),
                "terminal": bool(force_anchor),
                "local_state": self._copy_delta_tree(local_state, _eval_collector),
                "meta_state": self._copy_delta_tree(
                    local_meta_state, _eval_collector
                ),
                "compressor_buffer_kv": self._copy_delta_tree(
                    self.compressor_state.get("buffer_kv"), _eval_collector
                ),
                "compressor_buffer_gate": self._copy_delta_tree(
                    self.compressor_state.get("buffer_gate"), _eval_collector
                ),
                "indexer_buffer_kv": self._copy_delta_tree(
                    self.indexer_state.get("buffer_kv"), _eval_collector
                ),
                "indexer_buffer_gate": self._copy_delta_tree(
                    self.indexer_state.get("buffer_gate"), _eval_collector
                ),
            }
        return {
            "schema": DSV4_BLOCK_DELTA_SCHEMA,
            "class_name": type(self).__name__,
            "start_token": start,
            "end_token": end,
            "block_size": block_size,
            "anchor_interval_blocks": anchor_interval_blocks,
            "sliding_window": int(getattr(self.local, "max_size", 128) or 128),
            "compress_ratio": ratio,
            "compressor_pool": compressor_delta,
            "indexer_pool": indexer_delta,
            "anchor": anchor,
        }

    @classmethod
    def restore_anchor_from_deltas(
        cls,
        deltas,
        *,
        target_tokens: int,
        block_size: int = 256,
        anchor_interval_blocks: int = 8,
    ):
        """Restore the greatest validated anchor not after ``target_tokens``."""
        from .cache_delta import DSV4AnchorRestore, DSV4_BLOCK_DELTA_SCHEMA

        target = int(target_tokens)
        records = list(deltas or ())
        if target <= 0 or not records:
            raise ValueError("DSV4 restore requires a positive target and deltas")
        allowed_classes = {"DeepseekV4Cache", "PoolQuantizedV4Cache"}
        expected_class = cls.__name__
        if expected_class not in allowed_classes:
            raise ValueError(
                f"unsupported DSV4 delta cache class {expected_class!r}"
            )

        def _validate_pool_span(record, key, expected_row_start, expected_row_end):
            delta = record.get(key)
            if not isinstance(delta, dict):
                raise ValueError(f"DSV4 {key} delta is malformed")
            try:
                row_start = int(delta.get("start_row", -1))
                row_end = int(delta.get("end_row", -1))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"DSV4 {key} row span is not integral") from exc
            if (row_start, row_end) != (expected_row_start, expected_row_end):
                raise ValueError(
                    f"DSV4 {key} row span does not match token geometry: "
                    f"got={row_start}:{row_end} "
                    f"expected={expected_row_start}:{expected_row_end}"
                )

        def _validate_anchor(anchor, *, end, sliding_window):
            if not isinstance(anchor, dict):
                raise ValueError("DSV4 anchor is malformed")
            if int(anchor.get("tokens", -1)) != end:
                raise ValueError("DSV4 anchor token boundary mismatch")
            periodic = anchor.get("periodic")
            terminal = anchor.get("terminal")
            if not isinstance(periodic, bool) or not isinstance(terminal, bool):
                raise ValueError("DSV4 anchor kind flags must be boolean")
            if not periodic and not terminal:
                raise ValueError("DSV4 anchor is neither periodic nor terminal")
            if periodic and end % (int(block_size) * int(anchor_interval_blocks)):
                raise ValueError("DSV4 periodic anchor is misaligned")

            local_state = anchor.get("local_state")
            if not isinstance(local_state, (tuple, list)) or len(local_state) != 2:
                raise ValueError("DSV4 anchor is missing exact local K/V state")
            local_keys, local_values = local_state
            key_shape = tuple(getattr(local_keys, "shape", ()))
            value_shape = tuple(getattr(local_values, "shape", ()))
            expected_rows = min(end, sliding_window)
            if (
                len(key_shape) < 3
                or key_shape != value_shape
                or int(key_shape[-2]) != expected_rows
            ):
                raise ValueError(
                    "DSV4 anchor local K/V geometry is not canonical: "
                    f"keys={key_shape} values={value_shape} "
                    f"expected_rows={expected_rows}"
                )

            local_meta = anchor.get("meta_state")
            if not isinstance(local_meta, (tuple, list)) or len(local_meta) < 4:
                raise ValueError("DSV4 anchor is missing local rotating metadata")
            try:
                keep, max_size, offset, idx = map(int, local_meta[:4])
            except (TypeError, ValueError) as exc:
                raise ValueError("DSV4 anchor local metadata is not integral") from exc
            if (
                keep < 0
                or keep > max_size
                or max_size != sliding_window
                or offset != end
                or idx != expected_rows
            ):
                raise ValueError(
                    "DSV4 anchor local metadata is not canonical: "
                    f"keep={keep} max_size={max_size} offset={offset} "
                    f"idx={idx} expected={sliding_window}/{end}/{expected_rows}"
                )

        expected_start = 0
        selected = []
        selected_anchor = None
        chain_ratio = None
        chain_sliding_window = None
        saw_partial = False
        for record in records:
            if not isinstance(record, dict) or record.get("schema") != DSV4_BLOCK_DELTA_SCHEMA:
                raise ValueError("unsupported DSV4 block delta")
            record_class = str(record.get("class_name") or "")
            if record_class != expected_class:
                raise ValueError(
                    "DSV4 cache class changed inside delta chain: "
                    f"expected={expected_class!r} got={record_class!r}"
                )
            if int(record.get("block_size", 0)) != int(block_size):
                raise ValueError("DSV4 block-size mismatch")
            if int(record.get("anchor_interval_blocks", 0)) != int(anchor_interval_blocks):
                raise ValueError("DSV4 anchor-interval mismatch")
            start = int(record.get("start_token", -1))
            end = int(record.get("end_token", -1))
            if (
                saw_partial
                or start != expected_start
                or start % int(block_size)
                or end <= start
                or end - start > int(block_size)
            ):
                raise ValueError(
                    "non-contiguous DSV4 delta chain: "
                    f"expected={expected_start} got=[{start},{end})"
                )
            ratio = int(record.get("compress_ratio", 0))
            sliding_window = int(record.get("sliding_window", 0))
            if ratio not in (4, 128) or sliding_window <= 0:
                raise ValueError("invalid DSV4 cache geometry in delta chain")
            if chain_ratio is None:
                chain_ratio = ratio
                chain_sliding_window = sliding_window
            elif ratio != chain_ratio or sliding_window != chain_sliding_window:
                raise ValueError(
                    "DSV4 cache geometry changed inside delta chain: "
                    f"ratio={chain_ratio}->{ratio} "
                    f"window={chain_sliding_window}->{sliding_window}"
                )

            expected_row_start = start // ratio
            expected_row_end = end // ratio
            _validate_pool_span(
                record,
                "compressor_pool",
                expected_row_start,
                expected_row_end,
            )
            if ratio == 4:
                _validate_pool_span(
                    record,
                    "indexer_pool",
                    expected_row_start,
                    expected_row_end,
                )
            else:
                _validate_pool_span(record, "indexer_pool", 0, 0)

            anchor = record.get("anchor")
            if anchor is not None:
                _validate_anchor(
                    anchor,
                    end=end,
                    sliding_window=sliding_window,
                )
            is_partial = end - start < int(block_size)
            if is_partial:
                if not isinstance(anchor, dict) or anchor.get("terminal") is not True:
                    raise ValueError(
                        "DSV4 partial block must be an exact terminal anchor"
                    )
                saw_partial = True
            if end > target:
                break
            selected.append(record)
            expected_start = end
            if anchor is not None:
                selected_anchor = record
        if selected_anchor is None:
            raise ValueError("DSV4 delta chain has no anchor at or before target")

        checkpoint = int(selected_anchor["end_token"])
        selected = [record for record in selected if int(record["end_token"]) <= checkpoint]
        first = selected[0]
        ratio = int(first.get("compress_ratio", 0))
        sliding_window = int(first.get("sliding_window", 0))
        cache = cls(sliding_window=sliding_window, compress_ratio=ratio)
        compressor_rows = 0
        indexer_rows = 0
        for record in selected:
            if (
                int(record.get("compress_ratio", 0)) != ratio
                or int(record.get("sliding_window", 0)) != sliding_window
            ):
                raise ValueError("DSV4 geometry changed inside selected delta chain")

        if cls.__name__ == "PoolQuantizedV4Cache":
            # Native q8 segments are already detached and immutable. Append
            # their code/scale/min tuples directly; only the bounded initial
            # BF16 hot tier can trigger one deterministic promotion.
            # Compaction is deferred to one finalize per branch: compacting per
            # record re-concatenates the trailing slab on every append, and a
            # near-1M chain (~1.6K records x 2 branches) schedules that whole
            # burst without a blocking eval — enough to OOM Metal beside ~95GB
            # of weights. The finalize regroups once with bounded per-slab
            # evals instead.
            for record in selected:
                cache._append_pool_delta(
                    cache.compressor_state,
                    record["compressor_pool"],
                    defer_compaction=True,
                )
                cache._append_pool_delta(
                    cache.indexer_state,
                    record["indexer_pool"],
                    defer_compaction=True,
                )
                compressor_rows = int(record["compressor_pool"]["end_row"])
                indexer_rows = int(record["indexer_pool"]["end_row"])
            cache._finalize_pool_delta_appends(cache.compressor_state)
            cache._finalize_pool_delta_appends(cache.indexer_state)
        else:
            # A million-token BF16 chain contains thousands of block deltas.
            # Repeated pairwise concatenation is quadratic; validate the row
            # spans, then materialize each branch exactly once.
            def _restore_bf16_branch(state, key, *, allow_none):
                expected_start = 0
                values = []
                for item in selected:
                    delta = item[key]
                    start_row = int(delta.get("start_row", -1))
                    end_row = int(delta.get("end_row", -1))
                    if start_row != expected_start or end_row < start_row:
                        raise ValueError(
                            f"DSV4 {key} delta is not contiguous: "
                            f"expected={expected_start} got={start_row}:{end_row}"
                        )
                    storage = delta.get("storage")
                    if storage == "none":
                        if not allow_none or start_row or end_row:
                            raise ValueError(f"invalid absent DSV4 {key} delta")
                    elif storage == "bf16":
                        value = delta.get("value")
                        if value is None or int(value.shape[1]) != end_row - start_row:
                            raise ValueError(f"invalid DSV4 {key} BF16 rows")
                        if end_row > start_row:
                            values.append(value)
                    else:
                        raise ValueError(
                            f"base DeepseekV4Cache cannot restore {storage!r} {key}"
                        )
                    expected_start = end_row
                if values:
                    state["pooled"] = (
                        values[0]
                        if len(values) == 1
                        else mx.concatenate(values, axis=1)
                    )
                    mx.eval(state["pooled"])
                else:
                    state["pooled"] = None
                return expected_start

            compressor_rows = _restore_bf16_branch(
                cache.compressor_state,
                "compressor_pool",
                allow_none=False,
            )
            indexer_rows = _restore_bf16_branch(
                cache.indexer_state,
                "indexer_pool",
                allow_none=ratio != 4,
            )
        expected_rows = checkpoint // ratio
        expected_indexer_rows = expected_rows if ratio == 4 else 0
        if compressor_rows != expected_rows or indexer_rows != expected_indexer_rows:
            raise ValueError(
                "DSV4 restored pool rows do not match checkpoint: "
                f"compressor={compressor_rows} indexer={indexer_rows} "
                f"expected={expected_rows}/{expected_indexer_rows}"
            )

        anchor = selected_anchor["anchor"]
        local_state = anchor.get("local_state")
        if local_state is None:
            cache.local.keys = None
            cache.local.values = None
        else:
            local_rows = int(local_state[0].shape[-2])
            expected_local_rows = min(checkpoint, sliding_window)
            if local_rows != expected_local_rows:
                raise ValueError(
                    "DSV4 anchor local window is not canonical: "
                    f"rows={local_rows} expected={expected_local_rows}"
                )
            cache.local.state = local_state
        local_meta = anchor.get("meta_state")
        if local_meta is None or len(local_meta) < 4:
            raise ValueError("DSV4 anchor is missing local rotating metadata")
        try:
            _, _, local_offset, local_idx = map(int, local_meta[:4])
        except (TypeError, ValueError) as exc:
            raise ValueError("DSV4 anchor local metadata is not integral") from exc
        if local_offset != checkpoint or local_idx != min(checkpoint, sliding_window):
            raise ValueError(
                "DSV4 anchor local metadata is not canonical: "
                f"offset={local_offset} idx={local_idx} checkpoint={checkpoint}"
            )
        cache.meta_state = local_meta
        cache._pool_pending = None
        cache.compressor_state["buffer_kv"] = anchor.get("compressor_buffer_kv")
        cache.compressor_state["buffer_gate"] = anchor.get("compressor_buffer_gate")
        cache.indexer_state["buffer_kv"] = anchor.get("indexer_buffer_kv")
        cache.indexer_state["buffer_gate"] = anchor.get("indexer_buffer_gate")
        return DSV4AnchorRestore(
            cache=cache,
            checkpoint_tokens=checkpoint,
            replayed_tokens=max(0, target - checkpoint),
        )

    def accumulate_windows(self, kv, gate, state_key, ratio, start_pos):
        state = self._branch_state(state_key)
        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1]:
            kv = mx.concatenate([buf_kv, kv], axis=1)
            gate = mx.concatenate([buf_gate, gate], axis=1)
        usable = (kv.shape[1] // ratio) * ratio
        state["buffer_kv"] = kv[:, usable:]
        state["buffer_gate"] = gate[:, usable:]
        pool_base = max(0, start_pos) - (buf_kv.shape[1] if buf_kv is not None else 0)
        return kv[:, :usable], gate[:, :usable], pool_base

    def accumulate_overlap_windows(self, kv, gate, state_key, ratio, start_pos, head_dim):
        """Accumulate DSV4 ratio-4 overlap-compressor windows.

        Source DSV4 keeps two logical windows for overlap compression:
        ``state[:ratio]`` is the previous complete window and
        ``state[ratio:]`` is the current partial/complete window. When the
        current window completes during decode, the new compressed row uses
        previous-window first-half features plus current-window second-half
        features. A plain remainder buffer loses that previous window and
        silently emits a zero-left-half row at every decode boundary.

        ``gate`` is expected to already include the per-position APE term.
        Returns tensors shaped ``(B, rows, 2 * ratio, head_dim)`` that are
        ready for softmax over the window axis.
        """
        state = self._branch_state(state_key)
        B = kv.shape[0]
        out_dim = kv.shape[-1]

        def _empty():
            return (
                mx.zeros((B, 0, 2 * ratio, head_dim), dtype=kv.dtype),
                mx.zeros((B, 0, 2 * ratio, head_dim), dtype=gate.dtype),
            )

        def _make_row(prev_kv, prev_gate, cur_kv, cur_gate):
            row_kv = mx.zeros((B, 1, 2 * ratio, head_dim), dtype=kv.dtype)
            row_gate = mx.full(
                (B, 1, 2 * ratio, head_dim),
                -float("inf"),
                dtype=gate.dtype,
            )
            if prev_kv is not None:
                row_kv[:, 0, :ratio] = prev_kv[:, :, :head_dim]
                row_gate[:, 0, :ratio] = prev_gate[:, :, :head_dim]
            row_kv[:, 0, ratio:] = cur_kv[:, :, head_dim:]
            row_gate[:, 0, ratio:] = cur_gate[:, :, head_dim:]
            return row_kv, row_gate

        if start_pos == 0:
            usable = (kv.shape[1] // ratio) * ratio
            remainder_kv = kv[:, usable:]
            remainder_gate = gate[:, usable:]
            if usable >= ratio:
                state["buffer_kv"] = (
                    mx.concatenate([kv[:, usable - ratio:usable], remainder_kv], axis=1)
                    if remainder_kv.shape[1]
                    else kv[:, usable - ratio:usable]
                )
                state["buffer_gate"] = (
                    mx.concatenate([gate[:, usable - ratio:usable], remainder_gate], axis=1)
                    if remainder_gate.shape[1]
                    else gate[:, usable - ratio:usable]
                )
            else:
                state["buffer_kv"] = remainder_kv
                state["buffer_gate"] = remainder_gate
            if usable == 0:
                rows, gate_rows = _empty()
                return rows, gate_rows, start_pos

            W = usable // ratio
            full_kv = kv[:, :usable].reshape(B, W, ratio, out_dim)
            full_gate = gate[:, :usable].reshape(B, W, ratio, out_dim)
            rows = mx.zeros((B, W, 2 * ratio, head_dim), dtype=kv.dtype)
            gate_rows = mx.full(
                (B, W, 2 * ratio, head_dim),
                -float("inf"),
                dtype=gate.dtype,
            )
            rows[:, :, ratio:] = full_kv[:, :, :, head_dim:]
            rows[:, 1:, :ratio] = full_kv[:, :-1, :, :head_dim]
            gate_rows[:, :, ratio:] = full_gate[:, :, :, head_dim:]
            gate_rows[:, 1:, :ratio] = full_gate[:, :-1, :, :head_dim]
            return rows, gate_rows, start_pos

        buf_kv, buf_gate = state["buffer_kv"], state["buffer_gate"]
        if buf_kv is not None and buf_kv.shape[1] >= ratio:
            prev_kv, prev_gate = buf_kv[:, :ratio], buf_gate[:, :ratio]
            partial_kv, partial_gate = buf_kv[:, ratio:], buf_gate[:, ratio:]
        else:
            prev_kv = prev_gate = None
            partial_kv = buf_kv
            partial_gate = buf_gate

        prior_partial_len = partial_kv.shape[1] if partial_kv is not None else 0
        current_kv = (
            mx.concatenate([partial_kv, kv], axis=1)
            if partial_kv is not None and partial_kv.shape[1]
            else kv
        )
        current_gate = (
            mx.concatenate([partial_gate, gate], axis=1)
            if partial_gate is not None and partial_gate.shape[1]
            else gate
        )

        row_kvs = []
        row_gates = []
        while current_kv.shape[1] >= ratio:
            cur_kv = current_kv[:, :ratio]
            cur_gate = current_gate[:, :ratio]
            row_kv, row_gate = _make_row(prev_kv, prev_gate, cur_kv, cur_gate)
            row_kvs.append(row_kv)
            row_gates.append(row_gate)
            prev_kv, prev_gate = cur_kv, cur_gate
            current_kv = current_kv[:, ratio:]
            current_gate = current_gate[:, ratio:]

        if prev_kv is not None:
            state["buffer_kv"] = (
                mx.concatenate([prev_kv, current_kv], axis=1)
                if current_kv.shape[1]
                else prev_kv
            )
            state["buffer_gate"] = (
                mx.concatenate([prev_gate, current_gate], axis=1)
                if current_gate.shape[1]
                else prev_gate
            )
        else:
            state["buffer_kv"] = current_kv
            state["buffer_gate"] = current_gate

        pool_base = max(0, start_pos - prior_partial_len)
        if not row_kvs:
            rows, gate_rows = _empty()
            return rows, gate_rows, pool_base
        return mx.concatenate(row_kvs, axis=1), mx.concatenate(row_gates, axis=1), pool_base

    def update_pool(self, new_pooled, state_key):
        state = self._branch_state(state_key)
        pool = state["pooled"]
        if new_pooled.shape[1] > 0:
            pool = new_pooled if pool is None else mx.concatenate([pool, new_pooled], axis=1)
            state["pooled"] = pool
        if pool is None:
            pool = mx.zeros((new_pooled.shape[0], 0, new_pooled.shape[-1]), new_pooled.dtype)
        return pool


class Compressor(nn.Module):
    def __init__(self, config, compress_ratio, head_dim, rotate=False):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.rotate = bool(rotate)
        self.overlap = compress_ratio == 4
        self.out_dim = head_dim * (2 if self.overlap else 1)
        self.wkv = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.wgate = nn.Linear(config.hidden_size, self.out_dim, bias=False)
        self.ape = mx.zeros((compress_ratio, self.out_dim), dtype=mx.float32)
        self.norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)

    def _overlap_transform(self, x, fill_value):
        B, W, R, _ = x.shape
        out = mx.full((B, W, 2 * R, self.head_dim), fill_value, dtype=x.dtype)
        out[:, :, R:] = x[:, :, :, self.head_dim:]
        out[:, 1:, :R] = x[:, :-1, :, :self.head_dim]
        return out

    def _decode_front_overlap(self, x, pos):
        fn = self.__dict__.get("_decode_front_overlap_compiled")
        if fn is None:
            ratio = float(self.compress_ratio)

            def _impl(xin, p):
                x32 = xin.astype(mx.float32)
                kv = self.wkv(x32)
                gate = self.wgate(x32)
                idx = (p % ratio).astype(mx.int32)
                ape = mx.take(self.ape, idx, axis=0)
                return kv, gate + ape[None]

            fn = mx.compile(_impl)
            self.__dict__["_decode_front_overlap_compiled"] = fn
        return fn(x, pos)

    def _decode_front(self, x):
        fn = self.__dict__.get("_decode_front_compiled")
        if fn is None:

            def _impl(xin):
                x32 = xin.astype(mx.float32)
                return self.wkv(x32), self.wgate(x32)

            fn = mx.compile(_impl)
            self.__dict__["_decode_front_compiled"] = fn
        return fn(x)

    def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
        if (
            _DSV4_DEFER_POOL
            and cache is not None
            and x.shape[0] == 1
            and x.shape[1] == 1
            and hasattr(cache, "defer_pool_token")
        ):
            return cache.defer_pool_token(self, x, rope, start_pos, state_key)
        if cache is not None and hasattr(cache, "flush_pool_pending"):
            # Multi-token advancement (prefill chunk / tool continuation) must
            # consume any deferred decode tokens first to keep window order.
            cache.flush_pool_pending()
        result = self._advance(x, rope, cache, start_pos, state_key)
        if _DSV4_DEFER_POOL and cache is not None and hasattr(cache, "note_pool_result"):
            cache.note_pool_result(state_key, self, rope, result)
        return result

    def _advance(self, x, rope, cache, start_pos, state_key="compressor_state"):
        B, _, _ = x.shape
        dtype = x.dtype
        if _DSV4_PERF_PROBE_SKIP_POOL and x.shape[1] == 1 and cache is not None:
            frozen = self.__dict__.get("_probe_frozen_pool")
            if frozen is not None:
                return frozen
        # Compiled front path covers single-token decode AND the small batched
        # flush the pool deferral emits (k<=ratio for overlap/ratio-4 layers;
        # mx.compile shape-specializes so each k adds one trace, k<=4 total).
        # Ratio-128 flushes stay uncompiled — up to 128 distinct k shapes.
        use_compiled = (
            _DSV4_COMPILE_REGIONS
            and cache is not None
            and (
                x.shape[1] == 1
                or (self.overlap and x.shape[1] <= self.compress_ratio)
            )
        )
        # Official 0731 Compressor.forward promotes the residual stream before
        # both projections and keeps projection, softmax, and pooling math in
        # FP32.  The affine JANG bundle stores F16 q8 sidecars, so passing F16
        # directly to MLX QuantizedLinear would otherwise keep this whole path
        # in F16.  Restore the source precision boundary explicitly.
        if use_compiled:
            if self.overlap:
                if x.shape[1] == 1:
                    pos = mx.array([start_pos], dtype=mx.float32)
                else:
                    pos = mx.arange(x.shape[1], dtype=mx.float32) + float(start_pos)
                kv, gate = self._decode_front_overlap(x, pos)
            else:
                kv, gate = self._decode_front(x)
        else:
            x_fp32 = x.astype(mx.float32)
            kv = self.wkv(x_fp32)
            gate = self.wgate(x_fp32)
        if cache is not None and self.overlap:
            if not use_compiled:
                pos = start_pos + mx.arange(gate.shape[1])
                ape = mx.take(self.ape.astype(gate.dtype), pos % self.compress_ratio, axis=0)
                gate = gate + ape[None]
            kv, gate, pool_base = cache.accumulate_overlap_windows(
                kv, gate, state_key, self.compress_ratio, start_pos, self.head_dim
            )
            already_windowed = True
        elif cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
            already_windowed = False
        else:
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, state_key, self.compress_ratio, start_pos
            )
            already_windowed = False
        has_rows = kv.shape[1] > 0 if already_windowed else ready_kv.shape[1] > 0
        if not has_rows:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            if already_windowed:
                W = kv.shape[1]
            else:
                W = ready_kv.shape[1] // self.compress_ratio
                kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
                gate = ready_gate.reshape(B, W, self.compress_ratio, self.out_dim) + self.ape.astype(ready_gate.dtype)
                if self.overlap:
                    kv = self._overlap_transform(kv, 0.0)
                    gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(kv.dtype)
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(dtype))
            positions = (
                mx.arange(new_pooled.shape[1], dtype=mx.float32) * self.compress_ratio
                + pool_base
            )
            new_pooled = _apply_partial_rope(new_pooled[:, None], rope, positions=positions).squeeze(1)
            if self.rotate:
                new_pooled = _indexer_activation_roundtrip(new_pooled)
            else:
                new_pooled = _fp8_qat_non_rope(
                    new_pooled,
                    self.rope_head_dim,
                )
        if cache is not None:
            if hasattr(cache, "update_pool_view"):
                # seq_len tells the pool cache whether this forward is a
                # prefill chunk (bounded tiled view past the dense-row cap)
                # or a decode step (dense hot tier, fused-SDPA fast path).
                result = cache.update_pool_view(
                    new_pooled, state_key, seq_len=x.shape[1]
                )
            else:
                result = cache.update_pool(new_pooled, state_key)
            if _DSV4_PERF_PROBE_SKIP_POOL and x.shape[1] == 1:
                self.__dict__["_probe_frozen_pool"] = result
            return result
        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config, compress_ratio):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.compress_ratio = int(compress_ratio)
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(
            config,
            compress_ratio,
            self.head_dim,
            rotate=True,
        )
        self.scale = self.head_dim ** -0.5

    def update_pool(self, x, rope, cache, start_pos):
        """Advance the indexer's compressor state for this attention pass.

        DSV4's reference runtime advances the indexer compressor on every
        compress-ratio-4 pass, including while the retained pool is still too
        small to require sparse top-k selection.  Keeping this state update
        separate from scoring avoids the query projection/top-k cost below the
        threshold without letting the indexer pool fall behind the main pool.
        """
        return self.compressor(
            x,
            rope,
            cache,
            start_pos,
            state_key="indexer_state",
        )

    def select(self, x, q_residual, position_rope, pooled, start_pos):
        """Select sparse compressed-pool rows from already-updated state."""
        B, L, _ = x.shape
        if pooled.shape[1] == 0:
            return None
        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)
        q = _indexer_activation_roundtrip(q)
        weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads ** -0.5)
        if getattr(pooled, "is_dsv4_quantized_pool_view", False):
            return _dsv4_tiled_index_topk(
                q,
                weights,
                pooled,
                scale=self.scale,
                top_k=self.index_topk,
                offset=offset,
                ratio=self.compress_ratio,
            )
        scores = _dsv4_indexer_scores(q, pooled, weights, float(self.scale))
        return _dsv4_causal_index_topk(
            scores,
            top_k=self.index_topk,
            offset=offset,
            ratio=self.compress_ratio,
        )

    def __call__(self, x, q_residual, rope, position_rope, cache, start_pos):
        """Compatibility path that advances state and then selects rows."""
        pooled = self.update_pool(x, rope, cache, start_pos)
        return self.select(x, q_residual, position_rope, pooled, start_pos)


def _mlx_apply_rotary_cis(x: mx.array, freqs_cis_real: mx.array) -> mx.array:
    """MLX port of DSV4's apply_rotary_emb.
    x: (..., rd) where rd is even.
    freqs_cis_real: (L, rd/2, 2) packed as [cos, sin] pairs.

    Returns rotated x with same shape. Handles any leading dims for x;
    freqs_cis broadcasts on the seq-len axis.

    CAUTION: unlike the torch reference this does NOT mutate x in place.
    """
    dtype = x.dtype
    shape = x.shape
    rd = shape[-1]
    x32 = x.astype(mx.float32).reshape(*shape[:-1], rd // 2, 2)
    xa = x32[..., 0]
    xb = x32[..., 1]
    # freqs_cis_real: (L, rd/2, 2) -> cos = [...,0], sin = [...,1]
    cos = freqs_cis_real[..., 0]
    sin = freqs_cis_real[..., 1]
    # Broadcast cos/sin over leading dims of xa/xb
    ya = xa * cos - xb * sin
    yb = xa * sin + xb * cos
    out = mx.stack([ya, yb], axis=-1)
    return mx.reshape(out, shape).astype(dtype)


def _precompute_freqs_cis_real(
    dim: int, seqlen: int, original_seq_len: int,
    base: float, factor: float, beta_fast: int, beta_slow: int,
) -> mx.array:
    """Precompute (seqlen, dim/2, 2) real-valued [cos, sin] pairs for YaRN RoPE.

    Matches PR #1192 DeepseekV4RoPE YaRN formula — notably `high` is clamped
    to `dim - 1` (not `dim // 2 - 1`). Previous `dim // 2 - 1` clamp gave a
    steeper smoothing ramp, producing rotated q/k that diverged from
    reference by ~12% RMS in attention output.

    Also mirrors reference's smoothing sign: `smooth = 1 - clip(ramp)`,
    freqs = (inv_freq / factor) * (1 - smooth) + inv_freq * smooth.
    """
    import math
    idx = mx.arange(0, dim, 2).astype(mx.float32)
    freqs = 1.0 / (base ** (idx / dim))
    if original_seq_len > 0 and factor > 1:
        def correction_dim(n):
            return dim * math.log(original_seq_len / (n * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (mx.arange(dim // 2).astype(mx.float32) - low) / (high - low)
        smooth = 1 - mx.clip(ramp, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = mx.arange(seqlen).astype(mx.float32)
    theta = mx.outer(t, freqs)  # (seqlen, dim/2)
    cos = mx.cos(theta)
    sin = mx.sin(theta)
    return mx.stack([cos, sin], axis=-1)  # (seqlen, dim/2, 2)


_PHASE_MEM_LOG = os.environ.get("DSV4_PREFILL_MEM_LOG", "") == "2"


def _phase_mem(layer_id, tag, *arrays):
    arrs = [a for a in arrays if isinstance(a, mx.array)]
    if arrs:
        mx.eval(*arrs)
    logger.info(
        "DSV4 phase-mem layer=%d phase=%s active=%.2fGB peak=%.2fGB",
        layer_id,
        tag,
        mx.get_active_memory() / 2**30,
        mx.get_peak_memory() / 2**30,
    )
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()


def _dsv4_window_visibility(
    batch: int,
    seq_len: int,
    offset: int,
    window: int,
    window_len: int,
) -> mx.array:
    """Boolean visibility for the local SWA window.

    Shape is ``(B, 1, S, W)`` so it broadcasts onto SDPA scores
    ``(B, H, S, W)``. ``window_len`` is the current RotatingKVCache length
    after the chunk has been appended.
    """
    if window_len <= 0:
        return mx.zeros((batch, 1, seq_len, 0), dtype=mx.bool_)
    q_pos = offset + mx.arange(seq_len)
    k_pos = (offset + seq_len) - window_len + mx.arange(window_len)
    visible = (k_pos[None, :] <= q_pos[:, None]) & (
        k_pos[None, :] > (q_pos[:, None] - window)
    )
    return mx.broadcast_to(visible[None, None, :, :], (batch, 1, seq_len, window_len))


def _dsv4_compressed_visibility(
    batch: int,
    seq_len: int,
    offset: int,
    compressed_len: int,
    ratio: int,
) -> mx.array:
    """Boolean visibility for DSV4 compressed pool rows.

    Pool row ``k`` summarizes raw positions ``[k*ratio, (k+1)*ratio)``.
    Query position ``q`` may see that row only after the summarized raw
    window has ended: ``(k + 1) * ratio <= q + 1``.
    """
    if compressed_len <= 0:
        return mx.zeros((batch, 1, seq_len, 0), dtype=mx.bool_)
    q_pos = offset + mx.arange(seq_len)
    k_idx = mx.arange(compressed_len)
    visible = ((k_idx[None, :] + 1) * ratio) <= (q_pos[:, None] + 1)
    return mx.broadcast_to(
        visible[None, None, :, :],
        (batch, 1, seq_len, compressed_len),
    )


def _dsv4_index_visibility(
    batch: int,
    seq_len: int,
    offset: int,
    row_start: int,
    row_count: int,
    ratio: int,
) -> mx.array:
    """Return rows eligible for indexer top-k before ranking.

    A compressed row becomes causal only after all ``ratio`` source tokens in
    that row have been observed. ``offset`` is the absolute position of the
    first query in the current call, so the rule is identical for one-shot
    prefill, nonzero-offset chunked prefill, and decode.
    """
    if row_count <= 0:
        return mx.zeros((batch, seq_len, 0), dtype=mx.bool_)
    if ratio <= 0:
        raise ValueError("DSV4 indexer compression ratio must be positive")
    q_pos = int(offset) + mx.arange(seq_len)
    row_idx = int(row_start) + mx.arange(row_count)
    visible = ((row_idx[None, :] + 1) * int(ratio)) <= (
        q_pos[:, None] + 1
    )
    return mx.broadcast_to(visible[None], (batch, seq_len, row_count))


def _dsv4_causal_index_topk(
    scores: mx.array,
    *,
    top_k: int,
    offset: int,
    ratio: int,
) -> mx.array:
    """Select top-k only after excluding causally unavailable pool rows."""
    batch, seq_len, rows = map(int, scores.shape)
    if rows <= 0:
        return mx.zeros((batch, seq_len, 0), dtype=mx.int32)
    visible = _dsv4_index_visibility(
        batch,
        seq_len,
        int(offset),
        0,
        rows,
        int(ratio),
    )
    scores = mx.where(
        visible,
        scores,
        mx.full(scores.shape, -float("inf"), dtype=scores.dtype),
    )
    k = min(max(1, int(top_k)), rows)
    return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


_DSV4_POOL_TILE_TARGET_BYTES = 128 * 1024 * 1024
_DSV4_POOL_TILE_MAX_ROWS = 64 * 1024
_DSV4_POOL_TILE_MIN_ROWS = 64


def _dsv4_pool_tile_rows(q: mx.array, value_dim: int) -> int:
    """Choose a bounded row tile from query-score and BF16-view geometry."""
    batch, heads, seq_len, _ = map(int, q.shape)
    score_bytes_per_row = batch * heads * seq_len * 4
    value_bytes_per_row = batch * int(value_dim) * 2
    mask_bytes_per_row = batch * seq_len
    bytes_per_row = max(1, score_bytes_per_row + value_bytes_per_row + mask_bytes_per_row)
    rows = _DSV4_POOL_TILE_TARGET_BYTES // bytes_per_row
    rows = min(_DSV4_POOL_TILE_MAX_ROWS, max(_DSV4_POOL_TILE_MIN_ROWS, rows))
    if rows >= _DSV4_POOL_TILE_MIN_ROWS:
        rows = (rows // _DSV4_POOL_TILE_MIN_ROWS) * _DSV4_POOL_TILE_MIN_ROWS
    return max(1, rows)


def _dsv4_indexer_scores(
    q: mx.array,
    pool_tile: mx.array,
    head_weights: mx.array,
    scale: float,
) -> mx.array:
    """Head-reduced indexer scores [B, L, C] for one pool tile.

    out[b, l, c] = sum_h relu(scale * dot(q[b, h, l, :], pool[b, c, :]))
                   * head_weights[b, l, h]

    Tries the fused Metal kernel first (env-gated, self-tested, returns None
    when it declines); otherwise runs the stock op chain, which materializes a
    [B, H, L, C] fp32 intermediate.
    """
    try:
        from .indexer_scores_kernel import fused_indexer_scores as _fused

        fused = _fused(q, pool_tile, head_weights, float(scale))
        if fused is not None:
            return fused
    except Exception as exc:
        import warnings
        warnings.warn(f"DSV4 fused indexer unavailable; using reference scoring: {exc}", RuntimeWarning)
    q32 = q.astype(mx.float32)
    tile32 = pool_tile.astype(mx.float32)
    weights = head_weights.swapaxes(-1, -2)[..., None].astype(mx.float32)
    scores = q32 @ tile32[:, None].swapaxes(-1, -2)
    scores = mx.maximum(scores, 0) * float(scale)
    return (scores * weights).sum(axis=1)


def _dsv4_tiled_index_topk(
    q: mx.array,
    head_weights: mx.array,
    pooled,
    *,
    scale: float,
    top_k: int,
    offset: int,
    ratio: int,
) -> mx.array:
    """Exact global top-k over a segmented native-q8 indexer pool.

    Only one bounded BF16 tile and its score matrix exist at a time. Candidate
    scores are reduced to the global top-k before the next tile is consumed.
    """
    total_rows = int(pooled.shape[1])
    k = min(max(1, int(top_k)), total_rows)
    tile_rows = _dsv4_pool_tile_rows(q, int(pooled.shape[-1]))
    q32 = q.astype(mx.float32)
    weights = head_weights.swapaxes(-1, -2)[..., None].astype(mx.float32)
    best_scores = None
    best_indices = None
    for start, tile in pooled.iter_dequantized_tiles(max_rows=tile_rows):
        if best_scores is not None:
            # Bound the lazy graph across tiles WITHOUT a GPU round-trip stall:
            # a blocking eval here serialized decode (one stall per layer per
            # token) once pools promoted to q8. async_eval materializes the
            # accumulator so prior tile buffers free, but never blocks.
            mx.async_eval(best_scores, best_indices)
        scores = _dsv4_indexer_scores(q, tile, head_weights, float(scale))
        visible = _dsv4_index_visibility(
            int(scores.shape[0]),
            int(scores.shape[1]),
            int(offset),
            int(start),
            int(tile.shape[1]),
            int(ratio),
        )
        scores = mx.where(
            visible,
            scores,
            mx.full(scores.shape, -float("inf"), dtype=scores.dtype),
        )
        indices = mx.broadcast_to(
            (int(start) + mx.arange(int(tile.shape[1]), dtype=mx.int32))[None, None],
            scores.shape,
        )
        if best_scores is not None:
            scores = mx.concatenate([best_scores, scores], axis=-1)
            indices = mx.concatenate([best_indices, indices], axis=-1)
        keep = min(k, int(scores.shape[-1]))
        if int(scores.shape[-1]) > keep:
            selected = mx.argpartition(-scores, kth=keep - 1, axis=-1)[..., :keep]
            best_scores = mx.take_along_axis(scores, selected, axis=-1)
            best_indices = mx.take_along_axis(indices, selected, axis=-1)
        else:
            best_scores = scores
            best_indices = indices
    if best_indices is None:
        return mx.zeros((*q.shape[:1], q.shape[2], 0), dtype=mx.int32)
    return best_indices.astype(mx.int32)


def _dsv4_attention_accumulate(
    q32: mx.array,
    kv: mx.array,
    mask: mx.array,
    *,
    scale: float,
    running_max: mx.array,
    running_sum: mx.array,
    running_value: mx.array,
):
    """Merge one key/value tile into an online softmax accumulator."""
    scores = (q32 * float(scale)) @ kv[:, None].swapaxes(-1, -2).astype(mx.float32)
    scores = mx.where(mask, scores, mx.full(scores.shape, -float("inf")))
    tile_max = mx.max(scores, axis=-1)
    next_max = mx.maximum(running_max, tile_max)
    prior_scale = mx.exp(running_max - next_max)
    tile_weights = mx.exp(scores - next_max[..., None])
    next_sum = running_sum * prior_scale + tile_weights.sum(axis=-1)
    next_value = (
        running_value * prior_scale[..., None]
        + tile_weights @ kv[:, None].astype(mx.float32)
    )
    return next_max, next_sum, next_value


def _dsv4_selected_attention_accumulate(
    q32: mx.array,
    selected_kv: mx.array,
    mask: mx.array,
    *,
    scale: float,
    running_max: mx.array,
    running_sum: mx.array,
    running_value: mx.array,
):
    """Merge per-query CSA-selected values into an online softmax state."""
    scores = mx.einsum(
        "bhqd,bqkd->bhqk",
        q32 * float(scale),
        selected_kv.astype(mx.float32),
    )
    scores = mx.where(mask, scores, mx.full(scores.shape, -float("inf")))
    tile_max = mx.max(scores, axis=-1)
    next_max = mx.maximum(running_max, tile_max)
    prior_scale = mx.exp(running_max - next_max)
    tile_weights = mx.exp(scores - next_max[..., None])
    next_sum = running_sum * prior_scale + tile_weights.sum(axis=-1)
    next_value = running_value * prior_scale[..., None] + mx.einsum(
        "bhqk,bqkd->bhqd",
        tile_weights,
        selected_kv.astype(mx.float32),
    )
    return next_max, next_sum, next_value


def _dsv4_selected_query_rows(q: mx.array, topk: mx.array) -> int:
    """Bound selected-value and score temporaries during multi-token prefill."""
    batch, heads, seq_len, head_dim = map(int, q.shape)
    selected_rows = int(topk.shape[-1])
    # Per query: selected BF16 values, fp32 per-head scores/masks, and the
    # fp32 online-softmax output accumulator. Keep the combined estimate under
    # the same 128 MiB budget used by the indexer scan.
    bytes_per_query = batch * (
        selected_rows * (head_dim * 2 + heads * 4 + 1)
        + heads * head_dim * 4
    )
    return max(
        1,
        min(seq_len, _DSV4_POOL_TILE_TARGET_BYTES // max(1, bytes_per_query)),
    )


def _dsv4_selected_pool_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled,
    *,
    offset: int,
    window: int,
    ratio: int,
    scale: float,
    sinks: mx.array,
    topk: mx.array,
) -> mx.array:
    """Attend to CSA top-k by decoding exactly the selected q8 rows.

    The 128-dimensional indexer remains a bounded full-pool scan. Once it has
    selected global row indices, the 512-dimensional compressor pool must not
    be scanned again: each query tile gathers code/scale/min leaves for only
    those row occurrences and performs the local+selected online softmax.
    """
    batch, heads, seq_len, head_dim = map(int, q.shape)
    local_rows = int(local_kv.shape[2])
    local_mask = _dsv4_window_visibility(
        batch,
        seq_len,
        int(offset),
        int(window),
        local_rows,
    )
    query_rows = _dsv4_selected_query_rows(q, topk)
    outputs = []
    for start in range(0, seq_len, query_rows):
        if outputs:
            # Materialize the previous query tile without blocking (memory
            # bound for long prefill; decode has exactly one tile and pays
            # nothing).
            mx.async_eval(outputs[-1])
        end = min(seq_len, start + query_rows)
        q_tile = q[:, :, start:end]
        q32 = q_tile.astype(mx.float32)
        selected_indices = topk[:, start:end].astype(mx.int32)
        selected_kv = pooled.gather_dequantized_rows(selected_indices)
        tile_len = end - start
        running_max = mx.broadcast_to(
            sinks.astype(mx.float32).reshape(1, heads, 1),
            (batch, heads, tile_len),
        )
        running_sum = mx.ones((batch, heads, tile_len), dtype=mx.float32)
        running_value = mx.zeros(
            (batch, heads, tile_len, head_dim),
            dtype=mx.float32,
        )
        if local_rows:
            running_max, running_sum, running_value = _dsv4_attention_accumulate(
                q32,
                local_kv.squeeze(1),
                local_mask[:, :, start:end],
                scale=scale,
                running_max=running_max,
                running_sum=running_sum,
                running_value=running_value,
            )

        q_pos = int(offset) + mx.arange(start, end)
        visible = ((selected_indices + 1) * int(ratio)) <= (
            q_pos[None, :, None] + 1
        )
        running_max, running_sum, running_value = (
            _dsv4_selected_attention_accumulate(
                q32,
                selected_kv,
                visible[:, None],
                scale=scale,
                running_max=running_max,
                running_sum=running_sum,
                running_value=running_value,
            )
        )
        output = (running_value / running_sum[..., None]).astype(q.dtype)
        outputs.append(output)
    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def _dsv4_tiled_pool_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled,
    *,
    offset: int,
    window: int,
    ratio: int,
    scale: float,
    sinks: mx.array,
    topk: mx.array | None = None,
) -> mx.array:
    """DSV4 local plus compressed attention without a full BF16 pool view."""
    if topk is not None:
        return _dsv4_selected_pool_attention(
            q,
            local_kv,
            pooled,
            offset=offset,
            window=window,
            ratio=ratio,
            scale=scale,
            sinks=sinks,
            topk=topk,
        )

    batch, heads, seq_len, head_dim = map(int, q.shape)
    q32 = q.astype(mx.float32)
    running_max = mx.broadcast_to(
        sinks.astype(mx.float32).reshape(1, heads, 1),
        (batch, heads, seq_len),
    )
    # Attention sinks contribute to the softmax denominator with a zero value.
    running_sum = mx.ones((batch, heads, seq_len), dtype=mx.float32)
    running_value = mx.zeros(
        (batch, heads, seq_len, head_dim),
        dtype=mx.float32,
    )

    local_rows = int(local_kv.shape[2])
    if local_rows:
        local_mask = _dsv4_window_visibility(
            batch,
            seq_len,
            int(offset),
            int(window),
            local_rows,
        )
        running_max, running_sum, running_value = _dsv4_attention_accumulate(
            q32,
            local_kv.squeeze(1),
            local_mask,
            scale=scale,
            running_max=running_max,
            running_sum=running_sum,
            running_value=running_value,
        )

    tile_rows = _dsv4_pool_tile_rows(q, int(pooled.shape[-1]))
    q_pos = int(offset) + mx.arange(seq_len)
    first_tile = True
    for start, tile in pooled.iter_dequantized_tiles(max_rows=tile_rows):
        if not first_tile:
            # Non-blocking graph bound between tiles (see _dsv4_tiled_index_topk).
            mx.async_eval(running_max, running_sum, running_value)
        first_tile = False
        rows = int(tile.shape[1])
        k_idx = int(start) + mx.arange(rows)
        visible = ((k_idx[None, :] + 1) * int(ratio)) <= (q_pos[:, None] + 1)
        pool_mask = mx.broadcast_to(
            visible[None, None, :, :],
            (batch, 1, seq_len, rows),
        )
        running_max, running_sum, running_value = _dsv4_attention_accumulate(
            q32,
            tile,
            pool_mask,
            scale=scale,
            running_max=running_max,
            running_sum=running_sum,
            running_value=running_value,
        )
    return (running_value / running_sum[..., None]).astype(q.dtype)


# Cache of unit weight tensors for mx.fast.rms_norm (per-head Q norm uses
# no learned weights; reusing a single-allocated ones tensor avoids
# realloc per call across 43 layers per token).
_Q_NORM_WEIGHT_CACHE = {}

def _get_q_norm_ones(head_dim, dtype):
    key = (head_dim, dtype)
    w = _Q_NORM_WEIGHT_CACHE.get(key)
    if w is None:
        w = mx.ones((head_dim,), dtype=dtype)
        _Q_NORM_WEIGHT_CACHE[key] = w
    return w


class DeepseekV4Attention(nn.Module):
    """MLA with low-rank Q and grouped low-rank O.

    Per-layer RoPE: reference PR #1192 uses different rope configs based on
    `compress_ratio` for the layer. Layers with compress_ratio=0 (first + last)
    use base rope_theta=10000 with NO YaRN. Layers with compress_ratio>0
    (middle 41 layers) use compress_rope_theta=160000 WITH YaRN.

    We don't implement compressor/indexer yet, but we MUST still use the
    correct per-layer rope config or all middle layers drift catastrophically
    (std grows exponentially, hitting bf16 inf by layer 40).
    """
    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads  # typically 1 for DSV4
        self.head_dim = args.head_dim
        self.rope_head_dim = args.qk_rope_head_dim
        self.nope_head_dim = args.head_dim - args.qk_rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.o_groups = args.o_groups

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=args.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        # Grouped low-rank O: wo_a(grouped input) → wo_b(concat to hidden)
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(self.o_groups * self.o_lora_rank, self.hidden_size, bias=False)
        self.attn_sink = mx.zeros((self.n_heads,))

        self.softmax_scale = self.head_dim ** -0.5

        # Resolve per-layer compress_ratio from args.compress_ratios (bundle
        # config.json carries this as an explicit list of 43+1=44 entries).
        compress_ratios = getattr(args, "compress_ratios", None)
        if compress_ratios and layer_id < len(compress_ratios):
            compress_ratio = compress_ratios[layer_id]
        else:
            n = args.num_hidden_layers
            if layer_id == 0 or layer_id == n - 1:
                compress_ratio = 0
            else:
                i = layer_id - 1
                compress_ratio = 4 if i % 2 else 128
        self.compress_ratio = compress_ratio

        # Per-layer RoPE: compress_ratio > 0 uses compress_rope_theta + YaRN.
        if compress_ratio:
            rope_theta = args.compress_rope_theta
            rope_scaling = args.rope_scaling
        else:
            rope_theta = args.rope_theta
            rope_scaling = None
        self.rope = DeepseekV4RoPE(
            args.qk_rope_head_dim, rope_theta, rope_scaling, args.max_position_embeddings,
        )
        self.compress_rope = self.rope

        # Instantiate Compressor + Indexer for layers with compress_ratio > 0.
        if compress_ratio:
            self.compressor = Compressor(args, compress_ratio, self.head_dim)
            if compress_ratio == 4:
                self.indexer = Indexer(args, compress_ratio)

    def _decode_pre_region(self, x, pos):
        """Compiled decode projections: 3 qmm + norms + ropes + fp8 kernel in
        one dispatch. Weights are trace constants (inference-static)."""
        compiled = self.__dict__.get("_decode_pre_compiled")
        if compiled is None:
            def _impl(xin, p):
                B, L = xin.shape[:2]
                q_residual = self.q_norm(self.wq_a(xin))
                q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
                q = mx.fast.rms_norm(
                    q,
                    weight=_get_q_norm_ones(self.head_dim, q.dtype),
                    eps=self.args.rms_norm_eps,
                )
                q = q.transpose(0, 2, 1, 3)
                kv = (
                    self.kv_norm(self.wkv(xin))
                    .reshape(B, L, 1, self.head_dim)
                    .transpose(0, 2, 1, 3)
                )
                q = _dsv4_rope_rotate(q, p, self.rope.inv_freq, inverse=False)
                kv = _dsv4_rope_rotate(kv, p, self.rope.inv_freq, inverse=False)
                kv = _fp8_qat_non_rope(kv, self.rope_head_dim)
                return q_residual, q, kv

            compiled = mx.compile(_impl)
            self.__dict__["_decode_pre_compiled"] = compiled
        return compiled(x, pos)

    def _decode_out_math(self, o, p):
        """Raw decode output math: inverse rope + grouped o-proj + wo_b.
        Traced either standalone (`_decode_out_region`) or inside the layer
        decode tail region."""
        o = _dsv4_rope_rotate(o, p, self.rope.inv_freq, inverse=True)
        B, H, L, D = o.shape
        o = o.transpose(0, 2, 1, 3).reshape(B, L, H * D)
        o = self._grouped_output_projection(o)
        return self.wo_b(o)

    def _decode_out_region(self, out, pos):
        """Compiled decode output path: inverse rope + grouped o-proj + wo_b."""
        compiled = self.__dict__.get("_decode_out_compiled")
        if compiled is None:
            compiled = mx.compile(self._decode_out_math)
            self.__dict__["_decode_out_compiled"] = compiled
        return compiled(out, pos)

    def __call__(self, x, mask=None, cache=None, defer_out=False):
        # Match PR #1192 V4Attention forward. Handles compress_ratio>0 layers
        # via Compressor + Indexer, appending pooled global context to local KV.
        # defer_out=True (decode tail region path) returns (sdpa_out, pos) so
        # the caller can fuse the output projection into its own region.
        B, L, _ = x.shape
        local_cache = cache if isinstance(cache, DeepseekV4Cache) else cache
        offset = local_cache.offset if local_cache is not None else 0

        use_compiled = _DSV4_COMPILE_REGIONS and L == 1
        if use_compiled:
            pos = mx.arange(offset, offset + 1, dtype=mx.float32)
            q_residual, q, kv = self._decode_pre_region(x, pos)
        else:
            q_residual = self.q_norm(self.wq_a(x))
            q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
            # Per-head RMSNorm via mx.fast.rms_norm (1 fused Metal kernel vs 3 ops).
            # Uses unit weight tensor — DSV4 has no learned per-head norm weight.
            q = mx.fast.rms_norm(
                q,
                weight=_get_q_norm_ones(self.head_dim, q.dtype),
                eps=self.args.rms_norm_eps,
            )
            q = q.transpose(0, 2, 1, 3)

            kv = self.kv_norm(self.wkv(x)).reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)

            q = _apply_partial_rope(q, self.rope, offset)
            kv = _apply_partial_rope(kv, self.rope, offset)
            kv = _fp8_qat_non_rope(kv, self.rope_head_dim)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv
        attn_mask = mask
        tiled_pool_out = None

        if self.compress_ratio:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            # FAST PATH: when NOT using DeepseekV4Cache (i.e., plain KVCache),
            # the compressor has no buffer state to accumulate. For L < compress_ratio
            # the pooled output is empty and gets no-op concat. Skip entirely to
            # save ~150 matmuls per token across 41 compress_ratio>0 layers.
            #
            # Only run Compressor/Indexer if:
            # - v4_cache is provided (state carries across calls), OR
            # - L >= compress_ratio (enough tokens to produce non-empty pool in one call)
            if v4_cache is not None or L >= self.compress_ratio:
                _pm = _PHASE_MEM_LOG and L > 1
                if _pm:
                    _phase_mem(self.layer_id, "pre")
                pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
                if _pm:
                    _phase_mem(self.layer_id, "compressor", pooled)
                indexer_pooled = None
                if hasattr(self, "indexer"):
                    # The native DSV4 indexer owns a second compressor and must
                    # advance on every ratio-4 pass.  Deferring this update
                    # until sparse selection is needed creates a shorter
                    # indexer pool, corrupting both top-k positions and native
                    # block-delta export once the main pool crosses index_topk.
                    indexer_pooled = self.indexer.update_pool(
                        x,
                        self.compress_rope,
                        v4_cache,
                        offset,
                    )
                    if _pm:
                        _phase_mem(self.layer_id, "idx-update", indexer_pooled)
                    main_rows = int(pooled.shape[1])
                    indexer_rows = int(indexer_pooled.shape[1])
                    if indexer_rows != main_rows:
                        raise RuntimeError(
                            "DSV4 compressor/indexer pool row misalignment: "
                            f"compressor={main_rows} indexer={indexer_rows} "
                            f"offset={int(offset)} input_tokens={int(L)}"
                        )
                if pooled.shape[1] > 0:
                    topk = None
                    if (
                        indexer_pooled is not None
                        and pooled.shape[1] > self.indexer.index_topk
                    ):
                        topk = self.indexer.select(
                            x,
                            q_residual,
                            self.rope,
                            indexer_pooled,
                            offset,
                        )
                        if _pm:
                            _phase_mem(self.layer_id, "select", topk)

                    if getattr(pooled, "is_dsv4_quantized_pool_view", False):
                        tiled_pool_out = _dsv4_tiled_pool_attention(
                            q,
                            full_kv,
                            pooled,
                            offset=offset,
                            window=self.args.sliding_window,
                            ratio=self.compress_ratio,
                            scale=self.softmax_scale,
                            sinks=self.attn_sink.astype(q.dtype),
                            topk=topk,
                        )
                        if _pm:
                            _phase_mem(self.layer_id, "pool-attn", tiled_pool_out)
                    elif L == 1:
                        # Decode fast path: materialize only the selected rows
                        # for the single query. This is bounded by index_topk
                        # and avoids carrying a full pool mask through SDPA.
                        decode_out = None
                        if (
                            topk is not None
                            and dsv4_indexed_decode_attention is not None
                        ):
                            # Env-gated fused Metal kernel: reads local +
                            # selected pool rows directly through the index
                            # list (no materialized gather) and runs one
                            # split-K softmax(qK^T)V pass for the single
                            # query row.
                            decode_out = dsv4_indexed_decode_attention(
                                q,
                                full_kv,
                                pooled,
                                topk,
                                scale=float(self.softmax_scale),
                                sinks=self.attn_sink,
                            )
                        if decode_out is not None:
                            tiled_pool_out = decode_out
                            attn_mask = None
                        else:
                            if topk is not None:
                                idx = topk[:, None, :, :, None]
                                expanded = mx.broadcast_to(
                                    pooled[:, None, None, :, :],
                                    (B, 1, L, pooled.shape[1], self.head_dim),
                                )
                                pooled_kv = mx.take_along_axis(
                                    expanded,
                                    mx.broadcast_to(
                                        idx,
                                        idx.shape[:-1] + (self.head_dim,),
                                    ),
                                    axis=3,
                                ).reshape(B, 1, -1, self.head_dim)
                            else:
                                pooled_kv = pooled[:, None]
                            full_kv = mx.concatenate(
                                [full_kv, pooled_kv], axis=2
                            )
                            attn_mask = None
                    else:
                        # Prefill path: keep the compressed pool flat and
                        # describe visibility with a compact bool mask. The old
                        # code expanded to (B, 1, L, P, D) and then gathered
                        # L*topk rows, which caused multi-GB/TB allocations and
                        # leaked query i into query j's selected pool slice.
                        heads16_out = None
                        if (
                            topk is not None
                            and dsv4_heads16_prefill_attention is not None
                        ):
                            # Env-gated Metal kernel: attends each query to
                            # window + selected pool rows only, skipping the
                            # (B, 1, L, P) membership mask and full-pool SDPA.
                            heads16_out = dsv4_heads16_prefill_attention(
                                q,
                                full_kv,
                                pooled,
                                topk,
                                offset=int(offset),
                                window=int(self.args.sliding_window),
                                ratio=int(self.compress_ratio),
                                scale=float(self.softmax_scale),
                                sinks=self.attn_sink,
                            )
                        if heads16_out is not None:
                            tiled_pool_out = heads16_out
                            attn_mask = None
                        else:
                            local_mask = _dsv4_window_visibility(
                                B, L, offset, self.args.sliding_window, full_kv.shape[2],
                            )
                            comp_mask = _dsv4_compressed_visibility(
                                B, L, offset, pooled.shape[1], self.compress_ratio,
                            )
                            if topk is not None:
                                k_idx = mx.arange(pooled.shape[1])
                                selected = (
                                    topk[..., None] == k_idx[None, None, None, :]
                                ).any(axis=-2)
                                comp_mask = comp_mask & selected[:, None, :, :]
                            full_kv = mx.concatenate(
                                [full_kv, pooled[:, None]], axis=2
                            )
                            attn_mask = mx.concatenate(
                                [local_mask, comp_mask], axis=-1
                            )

        if attn_mask is not None:
            # DSV4 has heterogeneous attention state: SWA-only layers may use a
            # full KVCache while HSA/CSA layers use DeepseekV4Cache
            # (RotatingKVCache local window + cumulative pool rows). For
            # layers that did not build a DSV4-specific bool mask above, adapt
            # the shared model mask to this layer's actual key length.
            if attn_mask.shape[-1] > full_kv.shape[2]:
                attn_mask = attn_mask[..., -full_kv.shape[2]:]
            elif full_kv.shape[2] > attn_mask.shape[-1]:
                if getattr(attn_mask, "dtype", None) == mx.bool_:
                    pad = mx.ones(
                        attn_mask.shape[:-1] + (full_kv.shape[2] - attn_mask.shape[-1],),
                        dtype=mx.bool_,
                    )
                else:
                    pad = mx.zeros(
                        attn_mask.shape[:-1] + (full_kv.shape[2] - attn_mask.shape[-1],),
                        dtype=attn_mask.dtype,
                    )
                attn_mask = mx.concatenate([attn_mask, pad], axis=-1)

        if tiled_pool_out is None:
            out = scaled_dot_product_attention(
                q, full_kv, full_kv,
                cache=local_cache, scale=self.softmax_scale, mask=attn_mask,
                sinks=self.attn_sink.astype(q.dtype),
            )
        else:
            out = tiled_pool_out
        if use_compiled:
            if defer_out:
                return out, pos
            return self._decode_out_region(out, pos)
        out = _apply_partial_rope(out, self.rope, offset, inverse=True)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        out = self.wo_b(out)
        if _PHASE_MEM_LOG and L > 1 and self.compress_ratio:
            _phase_mem(self.layer_id, "attn-out", out)
        return out

    def _grouped_output_projection(self, out):
        """Match PR #1192 V4Attention._grouped_output_projection — handles
        both QuantizedLinear and plain paths for wo_a."""
        B, L = out.shape[:2]
        group_feat = (self.n_heads * self.head_dim) // self.o_groups
        out = out.reshape(B, L, self.o_groups, group_feat)

        if isinstance(self.wo_a, nn.QuantizedLinear):
            out = out.transpose(2, 0, 1, 3)
            weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            scales = self.wo_a.scales.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            biases = (
                None if self.wo_a.biases is None
                else self.wo_a.biases.reshape(self.o_groups, self.o_lora_rank, -1)[:, None]
            )
            out = mx.quantized_matmul(
                out, weight, scales=scales, biases=biases, transpose=True,
                group_size=self.wo_a.group_size, bits=self.wo_a.bits,
                mode=getattr(self.wo_a, "mode", "affine"),
            )
            out = out.transpose(1, 2, 0, 3).reshape(B, L, self.o_groups * self.o_lora_rank)
            if "bias" in self.wo_a:
                out = out + self.wo_a.bias
            return out

        weight = self.wo_a.weight.reshape(self.o_groups, self.o_lora_rank, group_feat)
        out = mx.einsum("bsgd,grd->bsgr", out, weight)
        out = out.reshape(B, L, self.o_groups * self.o_lora_rank)
        if "bias" in self.wo_a:
            out = out + self.wo_a.bias
        return out


# ---------- MoE ----------

@mx.compile
def _sqrt_softplus_router_scores(gates: mx.array):
    """Stable official DSV4 ``sqrt(softplus(logit))`` router scores."""

    return mx.sqrt(nn.softplus(gates))


@mx.compile
def sqrtsoftplus_select(
    gates: mx.array,
    bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """DSV4 scoring: sqrt(softplus(gates)) + bias → top-k, then renorm.

    `gates` is expected to already be fp32 (caller must cast). Returns
    inds as int32 (required by mlx's gather_qmm path).
    """
    scores = _sqrt_softplus_router_scores(gates)
    orig_scores = scores
    scores = scores + bias
    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    scores = scores * routed_scaling_factor
    return inds, scores


class Gate(nn.Module):
    """DSV4 MoE gate. Supports both hash-routing (first N layers) and
    score-based (sqrtsoftplus + noaux_tc bias) modes."""
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.hash = layer_id < args.num_hash_layers
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        if self.hash:
            self.tid2eid = mx.zeros(
                (args.vocab_size, args.num_experts_per_tok), dtype=mx.int32,
            )
        else:
            self.bias = mx.zeros((args.n_routed_experts,))

    def __call__(self, x, input_ids=None):
        # Reference PR #1192: gate logits matmul in fp32 explicitly to avoid
        # bf16 accumulation error across 256 experts × hidden=4096.
        # The transposed fp32 copy is cached: rebuilding it lazily per call
        # materializes ~4MB × n_layers of cast traffic on every decode token.
        w_t = getattr(self, "_gate_w_t_f32", None)
        if w_t is None or getattr(self, "_gate_w_src", None) is not self.weight:
            w_t = self.weight.T.astype(mx.float32)
            mx.eval(w_t)
            self._gate_w_t_f32 = w_t
            self._gate_w_src = self.weight
        gates = x.astype(mx.float32) @ w_t
        if self.hash:
            # Hash: deterministic per-token lookup (ignoring gates beyond
            # scoring for weights). Use original scores as weights.
            scores = _sqrt_softplus_router_scores(gates)
            assert input_ids is not None, "hash-routed layer requires input_ids"
            inds = self.tid2eid[input_ids].astype(mx.int32)
            weights = mx.take_along_axis(scores, inds, axis=-1)
            if self.args.norm_topk_prob:
                weights = weights / mx.sum(weights, axis=-1, keepdims=True)
            weights = weights * self.args.routed_scaling_factor
            return inds, weights
        else:
            return sqrtsoftplus_select(
                gates, self.bias, self.args.num_experts_per_tok,
                self.args.routed_scaling_factor, self.args.norm_topk_prob,
            )


@mx.compile
def _dsv4_swiglu_fp32(gate, up, swiglu_limit: float):
    """DSV4 SwiGLU with gate/up clamping to ±swiglu_limit (gate is clamped
    to max only; up is clamped symmetrically), retaining the official FP32
    activation domain for optional router weighting before the down projection.

    Official 0731 ``Expert.forward`` casts gate/up to FP32, computes the
    limited SwiGLU, applies the FP32 route weight, and only then casts back to
    the hidden dtype before ``w2``.
    """
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, a_min=-swiglu_limit, a_max=swiglu_limit)
        gate = mx.clip(gate, a_min=None, a_max=swiglu_limit)
    return nn.silu(gate) * up


@mx.compile
def _dsv4_swiglu(gate, up, swiglu_limit: float):
    """Unweighted DSV4 SwiGLU, restored to the projection input dtype."""

    return _dsv4_swiglu_fp32(gate, up, swiglu_limit).astype(gate.dtype)


class _DSV4SwiGLU(nn.Module):
    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def __call__(self, x_up, x_gate):
        return _dsv4_swiglu(x_gate, x_up, self.swiglu_limit)


class MLP(nn.Module):
    """SwiGLU expert / shared expert FFN. Uses mlx_lm naming convention."""
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        d = args.hidden_size
        mi = intermediate_size if intermediate_size is not None else args.moe_intermediate_size
        self.swiglu_limit = getattr(args, "swiglu_limit", 10.0)
        self.gate_proj = nn.Linear(d, mi, bias=False)
        self.down_proj = nn.Linear(mi, d, bias=False)
        self.up_proj = nn.Linear(d, mi, bias=False)

    def __call__(self, x):
        # Match PR #1192 DeepseekV4MLP — no act_quant_sim wrapping.
        return self.down_proj(_dsv4_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


def _dsv4_accumulate_moe(routed, shared, output_shape, output_dtype):
    """Accumulate routed and shared expert outputs exactly as official 0731."""

    combined = routed.astype(mx.float32).sum(axis=-2).reshape(output_shape)
    combined = combined + shared.astype(mx.float32)
    return combined.astype(output_dtype)


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.num_experts_per_tok = args.num_experts_per_tok
        self.gate = Gate(args, layer_id)
        swiglu_limit = getattr(args, "swiglu_limit", 10.0)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
            activation=_DSV4SwiGLU(swiglu_limit),
        )
        self.shared_experts = MLP(args, intermediate_size=args.moe_intermediate_size)

    def _weighted_routed_experts(self, x, inds, scores):
        """Run selected experts in the official 0731 precision/order.

        MLX's generic ``SwitchGLU`` applies ``down_proj`` before returning the
        per-route outputs.  DSV4 instead multiplies the FP32 limited-SwiGLU
        activation by its FP32 route score, casts once to the hidden dtype,
        and then applies the selected expert's bias-free down projection.
        This preserves the generic switch layer's expert-sorted execution
        without changing ``mlx_lm`` globally.
        """

        switch = self.switch_mlp
        dtype = x.dtype
        expanded = mx.expand_dims(x, (-2, -3))
        do_sort = inds.size >= 64
        idx = inds
        route_scores = scores.astype(mx.float32)
        inv_order = None

        if do_sort:
            *_, routes_per_token = inds.shape
            flat_indices = inds.flatten()
            order = mx.argsort(flat_indices)
            inv_order = mx.argsort(order)
            expanded = expanded.flatten(0, -3)[order // routes_per_token]
            idx = flat_indices[order]
            route_scores = route_scores.flatten()[order]

        if switch.training:
            idx = mx.stop_gradient(idx)
        x_up = switch.up_proj(expanded, idx, sorted_indices=do_sort)
        x_gate = switch.gate_proj(expanded, idx, sorted_indices=do_sort)
        activated = _dsv4_swiglu_fp32(
            x_gate,
            x_up,
            self.args.swiglu_limit,
        )
        activated = (activated * route_scores[..., None, None]).astype(dtype)
        routed = switch.down_proj(
            activated,
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            routed = routed[inv_order]
            routed = mx.unflatten(routed, 0, inds.shape)
        return routed.squeeze(-2)

    def _moe_math(self, x, input_ids=None):
        """Raw MoE math: gate (+topk or hash lookup), 3 gather_qmm, swiglu,
        shared expert, fp32 accumulate. Traced standalone or inside a region.

        Official 0731 initializes the MoE accumulator in FP32, adds every
        routed expert and the shared expert there, then casts once to x.dtype.
        The uint32 cast is belt-and-suspenders — mlx gather_qmm in
        QuantizedSwitchLinear strictly requires int32; argpartition return
        dtype varies by mlx version.
        """
        inds, scores = self.gate(x, input_ids=input_ids)
        inds = inds.astype(mx.uint32)
        routed = self._weighted_routed_experts(x, inds, scores)
        return _dsv4_accumulate_moe(
            routed,
            self.shared_experts(x),
            x.shape,
            x.dtype,
        )

    def _decode_moe_region(self, x, input_ids):
        """Whole-MoE compiled decode region — one dispatch."""
        compiled = self.__dict__.get("_decode_moe_compiled")
        if compiled is None:
            if self.gate.hash:
                def _impl(xin, ids):
                    return self._moe_math(xin, ids)
            else:
                def _impl(xin):
                    return self._moe_math(xin, None)

            compiled = mx.compile(_impl)
            self.__dict__["_decode_moe_compiled"] = compiled
        if self.gate.hash:
            return compiled(x, input_ids)
        return compiled(x)

    def __call__(self, x, input_ids=None):
        if (
            _DSV4_COMPILE_REGIONS
            and x.shape[1] == 1
            and not self.training
            and not (self.gate.hash and input_ids is None)
        ):
            return self._decode_moe_region(x, input_ids)
        return self._moe_math(x, input_ids)


# ---------- Block with mHC ----------

def _dsv4_hc_post_ops(x, residual, post, comb):
    """Apply the official DSV4 mHC post-residual contraction.

    Official 0731 computes::

        sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    For ``comb[..., i, j]`` and ``residual[..., i, d]`` this produces
    ``out[..., j, d] = sum_i comb[..., i, j] * residual[..., i, d]``.  The
    contraction is therefore ``comb.T @ residual``, not ``comb @ residual``.
    Reduce the source ``i`` axis explicitly. MLX's fast batched matmul path is
    not numerically neutral even for FP32 inputs, while the source-order
    four-term reduction is bit-identical to the official-shaped broadcast and
    avoids its ``[B, S, hc, hc, D]`` temporary.
    """

    comb_fp32 = comb.astype(mx.float32)
    residual_fp32 = residual.astype(mx.float32)
    residual_mix = (
        comb_fp32[..., 0, :, None] * residual_fp32[..., 0, None, :]
    )
    for source_index in range(1, int(comb.shape[-2])):
        residual_mix = residual_mix + (
            comb_fp32[..., source_index, :, None]
            * residual_fp32[..., source_index, None, :]
        )
    y = post.astype(mx.float32)[..., None] * x.astype(mx.float32)[..., None, :]
    return (y + residual_mix).astype(x.dtype)


def _make_hc_post_decode_kernel():
    """Fuse the source-order four-way mHC decode contraction on Metal.

    The ordinary MLX expression materializes four full hidden-width products
    and three additions for each of the two mHC residuals in every layer.  A
    single-token decode therefore builds 86 copies of that graph.  This kernel
    retains the same FP32 source-axis order and casts only the final result,
    but emits one output element per Metal thread without intermediates.
    """

    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = r"""
        const uint gid = thread_position_in_grid.x;
        const uint d = gid % D;
        const uint residual_row = gid / D;
        const uint target_hc = residual_row % HC;
        const uint batch_row = residual_row / HC;

        float residual_mix =
            static_cast<float>(comb[batch_row * HC * HC + target_hc])
            * static_cast<float>(residual[batch_row * HC * D + d]);
        for (uint source_hc = 1; source_hc < HC; ++source_hc) {
            const float term =
                static_cast<float>(
                    comb[batch_row * HC * HC + source_hc * HC + target_hc])
                * static_cast<float>(
                    residual[batch_row * HC * D + source_hc * D + d]);
            residual_mix = residual_mix + term;
        }

        const float direct =
            static_cast<float>(post[batch_row * HC + target_hc])
            * static_cast<float>(x[batch_row * D + d]);
        y[gid] = static_cast<outT>(direct + residual_mix);
    """
    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_post_decode",
        input_names=["x", "residual", "post", "comb"],
        output_names=["y"],
        source=source,
    )


_hc_post_decode_kernel = _make_hc_post_decode_kernel()


def _dsv4_hc_post_kernel_call(x, residual, post, comb):
    hc_mult = int(comb.shape[-1])
    hidden_size = int(x.shape[-1])
    x_contiguous = mx.contiguous(x)
    residual_contiguous = mx.contiguous(residual)
    post_contiguous = mx.contiguous(post)
    comb_contiguous = mx.contiguous(comb)
    output_shape = (*x.shape[:-1], hc_mult, hidden_size)
    return _hc_post_decode_kernel(
        inputs=[
            x_contiguous,
            residual_contiguous,
            post_contiguous,
            comb_contiguous,
        ],
        template=[
            ("HC", hc_mult),
            ("D", hidden_size),
            ("outT", x_contiguous.dtype),
        ],
        grid=(int(x_contiguous.size) * hc_mult, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[x_contiguous.dtype],
    )[0]


_dsv4_hc_post_fused = None


def _dsv4_hc_post(x, residual, post, comb):
    """Dispatch the exact fused decode contraction with a pure-MLX fallback."""

    try:
        use_metal = mx.default_device() == mx.gpu and mx.metal.is_available()
    except Exception:
        use_metal = False
    hc_mult = int(comb.shape[-1])
    if (
        _hc_post_decode_kernel is None
        or not use_metal
        or len(x.shape) < 3
        or int(x.shape[-2]) != 1
        or hc_mult != int(comb.shape[-2])
        or int(residual.shape[-2]) != hc_mult
        or int(x.size) == 0
    ):
        return _dsv4_hc_post_ops(x, residual, post, comb)

    if _DSV4_COMPILE_REGIONS:
        global _dsv4_hc_post_fused
        if _dsv4_hc_post_fused is None:
            _dsv4_hc_post_fused = mx.compile(_dsv4_hc_post_kernel_call)
        return _dsv4_hc_post_fused(x, residual, post, comb)
    return _dsv4_hc_post_kernel_call(x, residual, post, comb)


def _dsv4_norm_preserve_activation_dtype(norm, x):
    """Apply a folded FP32 norm without promoting the activation stream."""

    activation_dtype = x.dtype
    return norm(x).astype(activation_dtype)


_DSV4_HC_PRE_COMPILED: Dict[tuple, Any] = {}


def _hc_pre_impl(x, fn_w, scale, base, rms_eps, hc_mult, sinkhorn_iters, hc_eps):
    """Raw mHC-pre math shared by the standalone compiled region and the
    per-layer decode tail region (traced inside either)."""
    shape = x.shape
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    rsqrt = mx.rsqrt(
        mx.mean(x_flat.square(), axis=-1, keepdims=True) + rms_eps
    )
    mixes = (x_flat @ fn_w.T) * rsqrt
    pre, post, comb = hc_split_sinkhorn(
        mixes, scale, base, hc_mult, sinkhorn_iters, hc_eps,
    )
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
    return y.astype(x.dtype), post, comb


def _get_hc_pre_compiled(rms_eps, hc_mult, sinkhorn_iters, hc_eps):
    """One compiled mHC-pre region shared by every layer (weights are inputs).

    The sinkhorn Metal kernel traces cleanly inside `mx.compile`; fusing the
    flatten/rsqrt/mix/reduce glue around it collapses ~10 host dispatches per
    call (86 calls/token) into one.
    """
    key = (rms_eps, hc_mult, sinkhorn_iters, hc_eps)
    fn = _DSV4_HC_PRE_COMPILED.get(key)
    if fn is None:
        def _impl(x, fn_w, scale, base):
            return _hc_pre_impl(
                x, fn_w, scale, base, rms_eps, hc_mult, sinkhorn_iters, hc_eps,
            )

        fn = mx.compile(_impl)
        _DSV4_HC_PRE_COMPILED[key] = fn
    return fn


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.self_attn = DeepseekV4Attention(args, layer_id=layer_id)
        self.mlp = MoE(args, layer_id)  # all DSV4 layers are MoE
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.hidden_size
        self.hc_attn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_ffn_fn = mx.zeros((mix_hc, hc_dim))
        self.hc_attn_base = mx.zeros((mix_hc,))
        self.hc_ffn_base = mx.zeros((mix_hc,))
        self.hc_attn_scale = mx.zeros((3,))
        self.hc_ffn_scale = mx.zeros((3,))

    def _hc_pre(self, x, fn, scale, base):
        # x: (B, L, hc_mult, D)
        if _DSV4_COMPILE_REGIONS and x.shape[1] == 1:
            compiled = _get_hc_pre_compiled(
                self.args.rms_norm_eps, self.args.hc_mult,
                self.args.hc_sinkhorn_iters, self.args.hc_eps,
            )
            return compiled(x, fn, scale, base)
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + self.args.rms_norm_eps)
        mixes = (x_flat @ fn.T) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, self.args.hc_mult,
            self.args.hc_sinkhorn_iters, self.args.hc_eps,
        )
        y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y.astype(x.dtype), post, comb

    def _hc_post(self, x, residual, post, comb):
        return _dsv4_hc_post(x, residual, post, comb)

    def _decode_tail_region(self, sdpa_out, residual, post, comb, pos, input_ids):
        """One compiled region for the whole decode tail: attention output
        projection + hc_post(attn) + hc_pre(ffn) + post-LN + MoE + hc_post(ffn).

        Merging these five compiled calls + two raw norms into a single region
        removes ~4 host dispatch floors per layer per token. Weights are
        captured as trace constants (per-layer closure); pos/ids are array
        inputs so value changes never retrace.
        """
        compiled = self.__dict__.get("_decode_tail_compiled")
        if compiled is None:
            args = self.args

            def _tail(o, res, p1, c1, p, ids):
                out = self.self_attn._decode_out_math(o, p)
                h = _dsv4_hc_post_kernel_call(out, res, p1, c1)
                h2, p2, c2 = _hc_pre_impl(
                    h, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                    args.rms_norm_eps, args.hc_mult,
                    args.hc_sinkhorn_iters, args.hc_eps,
                )
                h2 = _dsv4_norm_preserve_activation_dtype(
                    self.post_attention_layernorm, h2,
                )
                m = self.mlp._moe_math(h2, ids)
                return _dsv4_hc_post_kernel_call(m, h, p2, c2)

            if self.mlp.gate.hash:
                def _impl(o, res, p1, c1, p, ids):
                    return _tail(o, res, p1, c1, p, ids)
            else:
                def _impl(o, res, p1, c1, p):
                    return _tail(o, res, p1, c1, p, None)

            compiled = mx.compile(_impl)
            self.__dict__["_decode_tail_compiled"] = compiled
        if self.mlp.gate.hash:
            return compiled(sdpa_out, residual, post, comb, pos, input_ids)
        return compiled(sdpa_out, residual, post, comb, pos)

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        if (
            _DSV4_COMPILE_REGIONS
            and x.shape[1] == 1
            and _hc_post_decode_kernel is not None
            and not self.mlp.training
            and not (self.mlp.gate.hash and input_ids is None)
        ):
            try:
                tail_ok = mx.default_device() == mx.gpu and mx.metal.is_available()
            except Exception:
                tail_ok = False
            if tail_ok:
                residual = x
                xh, post, comb = self._hc_pre(
                    x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
                )
                xh = self.input_layernorm(xh)
                sdpa_out, pos = self.self_attn(
                    xh, mask=mask, cache=cache, defer_out=True,
                )
                return self._decode_tail_region(
                    sdpa_out, residual, post, comb, pos, input_ids,
                )

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.input_layernorm(x)
        sub = _dsv4_attn_subchunk_tokens()
        if cache is not None and sub and x.shape[1] >= 2 * sub:
            # Sub-chunked attention: each slice sees the cache offset advanced
            # by the previous slice's update_and_fetch, so the per-slice mask
            # from the layer's own cache is exact. MoE below still consumes
            # the full concatenated chunk for gather_qmm batch throughput.
            outs = []
            for s in range(0, x.shape[1], sub):
                xs = x[:, s:s + sub]
                sub_mask = create_attention_mask(
                    xs, cache,
                    window_size=self.args.sliding_window,
                    return_array=True,
                )
                outs.append(self.self_attn(xs, mask=sub_mask, cache=cache))
            x = mx.concatenate(outs, axis=1)
        else:
            x = self.self_attn(x, mask=mask, cache=cache)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        # Folded-AWQ bundles keep ``ffn_norm.weight`` in FP32 so the inverse
        # per-channel scale is not rounded in storage.  MLX RMSNorm promotes
        # its result to the weight dtype, which would otherwise make every
        # routed expert qmm (and the rest of the network) run in FP32.  The
        # DSV4 activation stream is FP16/BF16: retain the FP32 norm parameter
        # and computation, then restore the incoming activation dtype before
        # the MoE projections.
        x = _dsv4_norm_preserve_activation_dtype(
            self.post_attention_layernorm,
            x,
        )
        x = self.mlp(x, input_ids=input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x


# ---------- Top-level model ----------

class DeepseekV4Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.embed = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DeepseekV4DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        self.hc_head_fn = mx.zeros((args.hc_mult, args.hc_mult * args.hidden_size))
        self.hc_head_base = mx.zeros((args.hc_mult,))
        self.hc_head_scale = mx.zeros((1,))
        self._layerwise_prefill_logged = False

    def _hc_head_reduce(self, x):
        # x: (B, L, hc_mult, D) → (B, L, D) via sigmoid-weighted sum
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
        rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + self.args.rms_norm_eps)
        mixes = (x_flat @ self.hc_head_fn.T) * rsqrt
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.hc_eps
        y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y.astype(x.dtype)

    def __call__(self, input_ids, cache=None, mask=None):
        if cache is None:
            cache = [None] * len(self.layers)
        try:
            cache_offset = int(getattr(cache[0], "offset", 0) or 0)
        except (IndexError, TypeError, ValueError):
            cache_offset = 0
        try:
            chunk_tokens = int(input_ids.shape[-1])
        except (AttributeError, IndexError, TypeError, ValueError):
            chunk_tokens = 0
        layerwise_prefill = _layerwise_prefill_materialization_enabled(
            input_ids, cache_offset + chunk_tokens
        )
        if layerwise_prefill and not self._layerwise_prefill_logged:
            logger.info(
                "DSV4 layerwise prefill materialization enabled: tokens=%d "
                "final_context=%d layers=%d (bounds lazy CSA/HCA attention "
                "graph lifetime)",
                chunk_tokens,
                cache_offset + chunk_tokens,
                len(self.layers),
            )
            self._layerwise_prefill_logged = True
        h = self.embed(input_ids)
        # Expand to hc_mult copies for mHC. Must be materialized (not a broadcast
        # view) — matches torch reference `h.unsqueeze(2).repeat(1, 1, hc_mult, 1)`.
        # Subsequent `flatten(start_axis=2)` inside `_hc_pre` would see wrong
        # strided data from a broadcast view.
        h = mx.tile(h[..., None, :], (1, 1, self.args.hc_mult, 1))
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            # Match PR #1192 reference: pass an explicit mask array (not
            # "causal" string), with sliding-window semantics. Native SDPA
            # needs an array mask for the `sinks` code path to work.
            first_cache = cache[0]
            mask = create_attention_mask(
                h[:, :, 0, :], first_cache,
                window_size=self.args.sliding_window,
                return_array=True,
            )
        _mem_log = layerwise_prefill and os.environ.get(
            "DSV4_PREFILL_MEM_LOG", ""
        ) == "1"
        for _li, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask=mask, cache=c, input_ids=input_ids)
            if layerwise_prefill:
                mx.eval(h)
                if _mem_log:
                    logger.info(
                        "DSV4 prefill-mem layer=%d active=%.2fGB "
                        "cache=%.2fGB peak=%.2fGB",
                        _li,
                        mx.get_active_memory() / 2**30,
                        mx.get_cache_memory() / 2**30,
                        mx.get_peak_memory() / 2**30,
                    )
        h = self._hc_head_reduce(h)
        return self.norm(h)


class Model(nn.Module):
    """mlx_lm entry-point class — what load_jangtq_model / mlx-lm factory expects."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        # vMLX health reads this shared dictionary directly. The observation
        # fields update when both shared transform dispatch families execute,
        # so health can distinguish an environment request from runtime branch
        # observation without claiming every eventual call site has executed.
        self._vmlx_dsv4_activation_qat_status = _DSV4_ACTIVATION_QAT_STATUS
        self.model = DeepseekV4Model(args)
        # Tied weight option not confirmed for DSV4 — use separate lm_head
        # (config has tie_word_embeddings=false)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids, cache=None, mask=None):
        h = self.model(input_ids, cache=cache, mask=mask)
        if h.shape[1] > 1:
            # Every consumer samples the final position only; projecting a
            # full prefill chunk through the 129k vocab head is ~2 TFLOP of
            # discarded logits per 2048-token chunk.
            h = h[:, -1:, :]
        # CRITICAL: reference does lm_head matmul in FP32
        # (inference/model.py ParallelHead.get_logits: `F.linear(x[:, -1].float(), self.weight)`
        # with self.weight stored as fp32). Accumulating 4096-dim contraction
        # in bf16 can add ~0.5 error per logit — comparable to the margin
        # between correct vs incorrect arithmetic answers.
        w = self.lm_head.weight
        if hasattr(self.lm_head, "scales"):
            # Quantized lm_head — dequantize then fp32 matmul
            w_f = mx.dequantize(
                self.lm_head.weight, self.lm_head.scales,
                getattr(self.lm_head, "biases", None),
                group_size=self.lm_head.group_size,
                bits=self.lm_head.bits,
                mode=getattr(self.lm_head, "mode", "affine"),
            ).astype(mx.float32)
        else:
            w_f = w.astype(mx.float32)
        h_f = h.astype(mx.float32)
        return h_f @ w_f.T

    def make_cache(self):
        """Build the per-layer native DSV4 cache topology.

        With ``DSV4_LONG_CTX=1``, every attention layer owns a bounded local
        SWA ring.  Compressed layers wrap that ring in ``DeepseekV4Cache`` so
        their CSA/HCA compressor and sparse-indexer state survives across
        chunks; zero-compression layers use a plain ``RotatingKVCache``.  The
        latter is important: the reference DSV4 attention path still applies
        the 128-token local window when ``compress_ratio == 0``.  An unbounded
        ``KVCache`` preserves masked logits but retains the entire prompt and
        defeats the architecture's long-context memory contract.

        ``DSV4_LONG_CTX=0`` retains the legacy short-prompt cache for explicit
        diagnostics only.  vMLX production enables native long-context mode.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        import os
        long_ctx = os.environ.get("DSV4_LONG_CTX", "0") == "1"
        pool_quant = os.environ.get("DSV4_POOL_QUANT", "0") == "1"
        pool_cache_cls = DeepseekV4Cache
        if pool_quant:
            # Pool quant is a user-visible native cache contract, not an
            # optional optimization. Silently falling back to BF16 leaves the
            # environment and /health claiming q8 while the model retains a
            # different cache class and memory footprint.
            from .pool_quant_cache import PoolQuantizedV4Cache

            pool_cache_cls = PoolQuantizedV4Cache
        caches = []
        for layer in self.model.layers:
            if not long_ctx:
                caches.append(KVCache())
                continue

            compress_ratio = layer.self_attn.compress_ratio
            if compress_ratio:
                # Pass per-layer `compress_ratio` so `DeepseekV4Cache.trim()`
                # can do proportional pool-row truncation instead of the
                # v2.5.14 full reset (better long-context multi-turn perf:
                # only the latest `n // ratio` pool rows are dropped per
                # trim, the kept-prefix pool survives).
                caches.append(pool_cache_cls(
                    self.args.sliding_window,
                    compress_ratio=compress_ratio,
                ))
            else:
                caches.append(RotatingKVCache(
                    max_size=self.args.sliding_window,
                    keep=0,
                ))
        return caches

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        """Map DSV4 source keys → mlx_lm conventions + stack experts.

        DSV4 ckpt conventions:
          embed.weight                               → model.embed.weight
          head.weight                                → lm_head.weight
          norm.weight                                → model.norm.weight
          layers.N.attn.{wq_a/wq_b/wkv/kv_norm/q_norm/wo_a/wo_b}.weight
                                                     → model.layers.N.self_attn.{...}.weight
          layers.N.attn_norm.weight                  → model.layers.N.input_layernorm.weight
          layers.N.ffn_norm.weight                   → model.layers.N.post_attention_layernorm.weight
          layers.N.ffn.gate.{weight|bias|tid2eid}    → model.layers.N.mlp.gate.{...}
          layers.N.ffn.shared_experts.{w1|w2|w3}.*   → model.layers.N.mlp.shared_experts.{gate/down/up}_proj.*
          layers.N.ffn.experts.E.{w1|w2|w3}.*        → STACK into model.layers.N.mlp.switch_mlp.{gate/down/up}_proj.*
          layers.N.attn.attn_sink                    → model.layers.N.self_attn.attn_sink
          layers.N.hc_{attn/ffn}_{fn/base/scale}     → model.layers.N.hc_{...}
          layers.N.attn.compressor.*                 → model.layers.N.self_attn.compressor.* (unused Phase 7.5B.2)
          layers.N.attn.indexer.*                    → model.layers.N.self_attn.indexer.*   (unused Phase 7.5B.2)
          mtp.0.*                                    → dropped (MTP not run at inference)
          hc_head_{fn/base/scale}                    → model.hc_head_{...}

        W1→gate_proj, W2→down_proj, W3→up_proj (per DSV convention).
        """
        import mlx.core as mx
        import re

        w1w2w3 = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

        out = {}
        for k, v in weights.items():
            # Drop MTP at inference
            if k.startswith("mtp."):
                continue
            # Keep compressor/indexer weights — needed for DSV4-Flash layers
            # with compress_ratio > 0 (most layers) to produce correct attention
            # over compressed global context. Without them, residual stream
            # explodes over 43 layers.
            # Global
            if k == "embed.weight":
                out["model.embed.weight"] = v; continue
            if k == "head.weight" or k == "head.biases" or k == "head.scales":
                # Map quantized head's .weight/.scales/.biases
                out["lm_head." + k[len("head."):]] = v; continue
            if k == "norm.weight":
                out["model.norm.weight"] = v; continue
            if k in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
                out["model." + k] = v; continue

            m = re.match(r"layers\.(\d+)\.(.+)", k)
            if not m:
                out["model." + k] = v  # pass-through
                continue
            L, rest = m.group(1), m.group(2)
            pfx = f"model.layers.{L}"

            # Norms
            if rest == "attn_norm.weight":
                out[f"{pfx}.input_layernorm.weight"] = v; continue
            if rest == "ffn_norm.weight":
                out[f"{pfx}.post_attention_layernorm.weight"] = v; continue

            # mHC
            if rest.startswith("hc_"):
                out[f"{pfx}.{rest}"] = v; continue

            # Attention (including compressor.* and indexer.* sub-modules)
            if rest.startswith("attn."):
                inner = rest[len("attn."):]
                out[f"{pfx}.self_attn.{inner}"] = v; continue

            # FFN
            if rest.startswith("ffn."):
                inner = rest[len("ffn."):]
                # Gate
                if inner.startswith("gate."):
                    out[f"{pfx}.mlp.gate.{inner[len('gate.'):]}"] = v; continue
                # Shared experts
                m2 = re.match(r"shared_experts\.(w[123])\.(weight|scales|biases)$", inner)
                if m2:
                    proj = w1w2w3[m2.group(1)]
                    out[f"{pfx}.mlp.shared_experts.{proj}.{m2.group(2)}"] = v; continue
                # Routed experts — collect for stacking
                m3 = re.match(r"experts\.(\d+)\.(w[123])\.(weight|scales|biases)$", inner)
                if m3:
                    # Temporary marker — will be stacked below
                    out[f"__TEMP__{pfx}.mlp.experts.{m3.group(1)}.{w1w2w3[m3.group(2)]}.{m3.group(3)}"] = v
                    continue
                # Fallback
                out[f"{pfx}.mlp.{inner}"] = v; continue

            out[f"{pfx}.{rest}"] = v

        # Stack routed experts across all layers
        n_experts = self.args.n_routed_experts
        for L in range(self.args.num_hidden_layers):
            pfx = f"model.layers.{L}.mlp"
            for proj in ("gate_proj", "down_proj", "up_proj"):
                for kind in ("weight", "scales", "biases"):
                    keys_e = [f"__TEMP__{pfx}.experts.{e}.{proj}.{kind}" for e in range(n_experts)]
                    if keys_e[0] in out:
                        stacked = mx.stack([out.pop(k) for k in keys_e])
                        out[f"{pfx}.switch_mlp.{proj}.{kind}"] = stacked

        # Final guard: no __TEMP__ keys should remain
        leftovers = [k for k in out if k.startswith("__TEMP__")]
        if leftovers:
            raise RuntimeError(f"sanitize left {len(leftovers)} unstacked TEMP keys, "
                               f"e.g. {leftovers[0]}")
        return out
