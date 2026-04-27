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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Optional, List

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs, create_attention_mask, scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU


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


# ---------- Phase HCMix (2026-04-24): fused rsqrt+matmul kernel ----------
#
# Combines mx.fast.rms_norm + matmul + cast-to-fp32 into ONE Metal dispatch.
# Replaces the 3-dispatch pattern in `_hc_pre`:
#   x_normed = mx.fast.rms_norm(x_flat, ones, eps)
#   mixes = (x_normed @ fn.T).astype(mx.float32)
#
# Algorithm per token (one threadgroup):
#   1. Compute sum(x^2) via threadgroup reduction
#   2. rsqrt = 1/sqrt(sum/HC_DIM + eps)
#   3. For each i in [0, MIX_HC):
#        mixes[i] = sum_j(x[j] * fn[i, j]) * rsqrt
#
# Each token gets one threadgroup of TG_SIZE=256 threads. Each thread
# loads HC_DIM/TG_SIZE = 64 elements of x_flat into threadgroup memory,
# then participates in the dot-product reductions.

def _make_hc_premix_kernel():
    """Single Metal kernel: rsqrt(mean(x^2)+eps) + matmul-with-fn.
    Returns mixes directly (fp32). Skips both intermediate ops."""
    try:
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            return None
    except Exception:
        return None

    source = """
        // Constants from template:
        //   HC_DIM   = hc_mult * hidden  (e.g. 4 * 4096 = 16384)
        //   MIX_HC   = (2 + hc_mult) * hc_mult  (e.g. (2+4)*4 = 24)
        //   TG_SIZE  = 256
        // M3 threadgroup memory limit is 32 KB. We CANNOT store x_norm
        // in threadgroup memory (16384 half = 32 KB, leaves no room for
        // the reduction buffer). Instead: stream x directly from device
        // memory for each pass. Apple's GPU L2 cache should hit on the
        // re-reads since x is small (32 KB) and stays warm.
        constexpr int HC_DIM_C = HC_DIM;
        constexpr int MIX_HC_C = MIX_HC;
        constexpr int TG_SIZE_C = TG_SIZE;
        constexpr int ELEM_PER_THREAD = HC_DIM_C / TG_SIZE_C;

        uint tid = thread_position_in_threadgroup.x;
        uint tg_id = threadgroup_position_in_grid.x;
        threadgroup float partial[TG_SIZE_C];  // 1 KB, fits comfortably

        const device half *x_row = x + tg_id * HC_DIM_C;

        // Step 1: compute partial sum(x^2). Each thread reads its
        // ELEM_PER_THREAD = 64 elements of x. Cache the values in registers
        // for re-use in step 2 (avoids one re-read).
        float my_ssq = 0.0f;
        // We can't store all 64 elements in regs across MIX_HC iterations
        // (would blow register pressure). Just compute sum(x^2) here.
        for (int k = 0; k < ELEM_PER_THREAD; ++k) {
            int j = tid + k * TG_SIZE_C;
            float vf = static_cast<float>(x_row[j]);
            my_ssq += vf * vf;
        }
        partial[tid] = my_ssq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: tree reduction → total sum(x^2) in partial[0]
        for (uint s = TG_SIZE_C / 2; s > 0; s >>= 1) {
            if (tid < s) {
                partial[tid] += partial[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float total_ssq = partial[0];
        float rsqrt_val = metal::rsqrt(total_ssq / static_cast<float>(HC_DIM_C)
                                       + static_cast<float>(eps[0]));

        // Step 3: compute MIX_HC dot products. Each iteration re-reads x.
        // x is only 32 KB → L1/L2 cache hits expected on subsequent passes.
        // fn is 24 × 16384 × 2 = 768 KB; first read may miss but subsequent
        // experts re-use cached lines if they share row tiles.
        device float *mixes_row = mixes + tg_id * MIX_HC_C;
        for (int i = 0; i < MIX_HC_C; ++i) {
            float my_dot = 0.0f;
            for (int k = 0; k < ELEM_PER_THREAD; ++k) {
                int j = tid + k * TG_SIZE_C;
                my_dot += static_cast<float>(x_row[j]) *
                          static_cast<float>(fn[i * HC_DIM_C + j]);
            }
            partial[tid] = my_dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint s = TG_SIZE_C / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    partial[tid] += partial[tid + s];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                mixes_row[i] = partial[0] * rsqrt_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    """

    return mx.fast.metal_kernel(
        name="deepseek_v4_hc_premix",
        input_names=["x", "fn", "eps"],
        output_names=["mixes"],
        source=source,
    )


_hc_premix_kernel = _make_hc_premix_kernel()


def hc_premix(x_flat: mx.array, fn: mx.array, eps_arr: mx.array,
              hc_dim: int, mix_hc: int):
    """Fused rsqrt+matmul. x_flat must be (B*L, hc_dim) fp16. fn is (mix_hc, hc_dim).
    eps_arr is (1,) fp32."""
    if _hc_premix_kernel is None:
        # Fallback: ops path
        ones = mx.ones(hc_dim, dtype=x_flat.dtype)
        x_normed = mx.fast.rms_norm(x_flat, ones, float(eps_arr[0].item()))
        return (x_normed @ fn.T).astype(mx.float32)

    # x_flat shape: (..., hc_dim). We need to flatten leading dims to (N, hc_dim).
    leading = x_flat.shape[:-1]
    n_tokens = 1
    for d in leading:
        n_tokens *= d
    return _hc_premix_kernel(
        inputs=[x_flat, fn, eps_arr],
        template=[("HC_DIM", hc_dim), ("MIX_HC", mix_hc), ("TG_SIZE", 256)],
        grid=(n_tokens, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*leading, mix_hc)],
        output_dtypes=[mx.float32],
    )[0]


def hc_split_sinkhorn(
    mixes: mx.array, hc_scale: mx.array, hc_base: mx.array,
    hc_mult: int, iters: int = 20, eps: float = 1e-6,
):
    """Public API — dispatches to fused Metal kernel when available.
    Same output semantics as `_hc_split_sinkhorn_ops`.
    """
    if _hc_split_sinkhorn_kernel is None:
        return _hc_split_sinkhorn_ops(mixes, scale=hc_scale, base=hc_base,
                                       hc_mult=hc_mult, iters=iters, eps=eps)
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


def act_quant_sim(x: mx.array, block_size: int = 128) -> mx.array:
    """FP8 e4m3 fake-quant on activations (block_size=128 per last dim).

    Reference (`inference/kernel.py act_quant_kernel`) applies this before
    every Linear with FP4/FP8 weights during inference. The model was
    TRAINED expecting this activation noise — skipping it gives weights
    cleaner-than-trained inputs, which paradoxically hurts accuracy.

    e4m3fn has fp8_max=448. Per-block scale = amax/fp8_max rounded to
    power-of-2 (fast_round_scale). Quantize → 8-bit e4m3 levels → dequant.

    Enable with env DSV4_ACT_QUANT=1; default OFF because it adds overhead
    and most runtime queries work without it. For arithmetic-heavy
    reasoning, turn ON.
    """
    import os as _os
    if _os.environ.get("DSV4_ACT_QUANT", "0") != "1":
        return x
    orig_dtype = x.dtype
    shape = x.shape
    if shape[-1] % block_size != 0:
        return x
    x32 = x.astype(mx.float32)
    reshaped = x32.reshape(*shape[:-1], -1, block_size)
    fp8_max = 448.0
    amax = mx.max(mx.abs(reshaped), axis=-1, keepdims=True)
    # fast_round_scale: 2^ceil(log2(amax/fp8_max)), zero-safe
    raw_scale = amax / fp8_max
    # Handle zero blocks — leave them as 1.0 (no-op)
    safe_scale = mx.where(raw_scale > 1e-30, raw_scale, mx.ones_like(raw_scale))
    log2s = mx.ceil(mx.log2(safe_scale))
    scale = mx.power(mx.array(2.0, dtype=mx.float32), log2s)
    # e4m3fn has 3 mantissa bits → ~8 equal-ratio levels per binade.
    # Approximation: uniform 8-bit levels within ±fp8_max. Real e4m3 is
    # log-spaced per binade; this simplification is close enough for
    # the fake-quant effect the model was trained with.
    normalized = reshaped / scale
    # Round-trip through 8-bit signed range [-127, 127] scaled to [-448, 448]
    q = mx.round(normalized * (127.0 / fp8_max)) * (fp8_max / 127.0)
    q = mx.clip(q, -fp8_max, fp8_max)
    result = (q * scale).reshape(shape)
    return result.astype(orig_dtype)


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
        # Phase-Rope (2026-04-24): cache wavelength for mx.fast.rope. Critical
        # convention difference: mx.fast.rope's `freqs=` arg expects WAVELENGTHS
        # (period = 1/frequency = base^(2i/D) * yarn-correction), NOT frequencies.
        # mlx-lm's YarnRoPE in rope_utils.py uses `freq_extra = base ** (...)`.
        # We have inv_freq = 1/wavelength, so wavelength = 1/inv_freq.
        # Verified bit-exact against manual cos/sin path: max_diff 2.4e-7 fp32.
        wavelength = 1.0 / inv_freq
        self._wavelength = (wavelength,)
        self._neg_wavelength = (-wavelength,)

    @property
    def inv_freq(self):
        return self._inv_freq[0]

    def __call__(self, x, offset=0, inverse=False, positions=None):
        # mx.fast.rope fast path (single Metal kernel vs ~8 ops manually).
        # `traditional=True` matches DSV4's interleaved (x[..., 0], x[..., 1])
        # rotation layout. Inverse handled by negating wavelength (cos stays,
        # sin flips → equivalent to original `sin = -sin`).
        if positions is not None:
            # Positions arg path uses manual cos/sin (mx.fast.rope offset is int
            # or 1D array of B-element offsets, NOT arbitrary per-token positions).
            return self._call_manual(x, offset=offset, inverse=inverse, positions=positions)
        wave = self._neg_wavelength[0] if inverse else self._wavelength[0]
        return mx.fast.rope(
            x, self.dims, traditional=True, base=None, scale=1.0,
            offset=offset, freqs=wave,
        )

    def _call_manual(self, x, offset=0, inverse=False, positions=None):
        """Slow path for the rare positions-array case (used by Compressor
        when computing absolute-position RoPE for pooled keys). Kept for
        completeness; not on the decode hot path."""
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


def _apply_partial_rope(x, rope, offset=0, inverse=False, positions=None):
    """Apply RoPE to the LAST `rope.dims` of x (DSV4 partial-rope convention).
    Kept as a wrapper for now because mx.fast.rope's natively partial mode
    rotates the FIRST `dims` features, but DSV4 places the rope dims AT THE
    END (after nope). So we still need split + rope-on-pe + concat — but
    each step is cheap and rope itself is the fast path."""
    rope_dim = rope.dims
    if x.shape[-1] == rope_dim:
        return rope(x, offset=offset, inverse=inverse, positions=positions)
    nope, pe = mx.split(x, [x.shape[-1] - rope_dim], axis=-1)
    pe = rope(pe, offset=offset, inverse=inverse, positions=positions)
    return mx.concatenate([nope, pe], axis=-1)


# ── PR #1195 helpers (ported 2026-04-25) ──────────────────────────────
# These mirror `_build_window_mask`, `_compressed_visibility`, `_indexer_score`
# from `mlx_lm/models/deepseek_v4.py` head 905df9c. They build SDPA-compatible
# 4D bool masks and run the indexer score reduction in a single compile graph.

def _build_window_mask(B: int, S: int, offset, window: int, window_len: int) -> mx.array:
    """Visibility mask for the window-side of the cache (compressed=0 layers
    + the rotating window portion of compressed layers).

    Returns (B, 1, S, window_len) bool, broadcastable to SDPA scores.
    """
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
        end = off + S
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = end[:, None] - window_len + cache_k[None, :]
        win_visible = (
            (raw_pos_at_k[:, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[:, None, :] > q_pos[:, :, None] - window)
        )
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
        cache_k = mx.arange(window_len, dtype=mx.int32)
        raw_pos_at_k = (offset + S) - window_len + cache_k
        win_visible = (
            (raw_pos_at_k[None, None, :] <= q_pos[:, :, None])
            & (raw_pos_at_k[None, None, :] > q_pos[:, :, None] - window)
        )
    return win_visible[:, None, :, :]


def _compressed_visibility(B: int, S: int, offset, compressed_len: int, ratio: int) -> mx.array:
    """Block-causal staircase: pool[k] visible to query at q_pos iff
    (k+1)*ratio <= q_pos+1.

    Returns (B, 1, S, compressed_len) bool, broadcastable to SDPA scores.
    """
    if isinstance(offset, mx.array):
        off = offset.astype(mx.int32).reshape(-1)
        q_pos = off[:, None] + mx.arange(S, dtype=mx.int32)
    else:
        q_pos = mx.broadcast_to(
            offset + mx.arange(S, dtype=mx.int32)[None, :], (B, S)
        )
    k = mx.arange(compressed_len, dtype=mx.int32)
    comp_visible = (k + 1)[None, None, :] * ratio <= (q_pos + 1)[:, :, None]
    return comp_visible[:, None, :, :]


@mx.compile
def _indexer_score_reduction(
    scores_qkT: mx.array, weights: mx.array, scale: float, n_heads_inv_sqrt: float
) -> mx.array:
    """Compile graph: relu(scores) * scale * (weights * n_heads^-0.5).T.sum(axis=heads).

    `scores_qkT`: (B, n_heads, L, P) raw q @ k^T (fp32)
    `weights`:    (B, L, n_heads) head importance per query (fp32)
    Returns:      (B, L, P) reduced scores."""
    s = mx.maximum(scores_qkT, 0) * scale
    w = weights * n_heads_inv_sqrt
    return (s * w.swapaxes(-1, -2)[..., None]).sum(axis=1)


@mx.compile
def _attn_partial_rope_fused(
    x: mx.array, offset, rd: int, freqs: mx.array, inverse_scale: float,
) -> mx.array:
    """PR #1195 port: split nope+pe, rope on pe, concat back — all in one
    compile graph.

    `inverse_scale` = -1.0 for inverse RoPE, 1.0 for forward.
    `freqs` = wavelength tensor (already cached on the rope module).
    """
    nope = x[..., :-rd]
    pe = mx.fast.rope(
        x[..., -rd:], rd, traditional=True, base=None,
        scale=inverse_scale, offset=offset, freqs=freqs,
    )
    return mx.concatenate([nope, pe], axis=-1)


class DeepseekV4Cache:
    """Simplified cache for DSV4: wraps a plain KVCache with compressor/indexer
    state buffers (cross-window pooling). For short prompts (<sliding_window),
    the plain KVCache is equivalent to RotatingKVCache."""
    def __init__(self, sliding_window):
        from mlx_lm.models.cache import RotatingKVCache
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        self.compressor_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}
        self.indexer_state = {"buffer_kv": None, "buffer_gate": None, "pooled": None}

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
        local_state = None if self.local.empty() else self.local.state
        return (
            local_state,
            tuple(self.compressor_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
            tuple(self.indexer_state[k] for k in ("buffer_kv", "buffer_gate", "pooled")),
        )

    @state.setter
    def state(self, value):
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
        return self.local.trim(n)

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
        return total

    def _branch_state(self, key):
        return self.indexer_state if key == "indexer_state" else self.compressor_state

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
    def __init__(self, config, compress_ratio, head_dim):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
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

    def __call__(self, x, rope, cache, start_pos, state_key="compressor_state"):
        B, _, _ = x.shape
        kv = self.wkv(x)
        gate = self.wgate(x)
        if cache is None:
            usable = (kv.shape[1] // self.compress_ratio) * self.compress_ratio
            ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
            pool_base = start_pos
        else:
            ready_kv, ready_gate, pool_base = cache.accumulate_windows(
                kv, gate, state_key, self.compress_ratio, start_pos
            )
        if ready_kv.shape[1] == 0:
            new_pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            W = ready_kv.shape[1] // self.compress_ratio
            kv = ready_kv.reshape(B, W, self.compress_ratio, self.out_dim)
            gate = ready_gate.reshape(B, W, self.compress_ratio, self.out_dim) + self.ape.astype(ready_gate.dtype)
            if self.overlap:
                kv = self._overlap_transform(kv, 0.0)
                gate = self._overlap_transform(gate, -float("inf"))
            weights = mx.softmax(gate.astype(mx.float32), axis=2, precise=True).astype(kv.dtype)
            new_pooled = (kv * weights).sum(axis=2)
            new_pooled = self.norm(new_pooled.astype(x.dtype))
            positions = (
                mx.arange(new_pooled.shape[1], dtype=mx.float32) * self.compress_ratio
                + pool_base
            )
            new_pooled = _apply_partial_rope(new_pooled[:, None], rope, positions=positions).squeeze(1)
        if cache is not None:
            return cache.update_pool(new_pooled, state_key)
        return new_pooled


class Indexer(nn.Module):
    def __init__(self, config, compress_ratio):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False)
        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x, q_residual, rope, position_rope, cache, start_pos):
        B, L, _ = x.shape
        pooled = self.compressor(x, rope, cache, start_pos, state_key="indexer_state")
        if pooled.shape[1] == 0:
            return None
        offset = start_pos
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        q = _apply_partial_rope(q, position_rope, offset)
        # Compile the score reduction (relu + scale + weight * sum-over-heads)
        # into one graph. Was 4 separate ops; @mx.compile fuses them.
        raw_scores = q.astype(mx.float32) @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
        weights = self.weights_proj(x).astype(mx.float32)
        scores = _indexer_score_reduction(
            raw_scores, weights, self.scale, self.n_heads ** -0.5,
        )
        k = min(self.index_topk, pooled.shape[1])
        return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]


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
        self.attn_sink = mx.zeros((self.n_kv_heads, 1, self.head_dim))

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

    def __call__(self, x, mask=None, cache=None):
        # Match PR #1192 V4Attention forward. Handles compress_ratio>0 layers
        # via Compressor + Indexer, appending pooled global context to local KV.
        B, L, _ = x.shape
        local_cache = cache if isinstance(cache, DeepseekV4Cache) else cache
        offset = local_cache.offset if local_cache is not None else 0

        # Phase-MLAFuse (2026-04-24, ralph iter, profiler-driven):
        # `wq_a` and `wkv` both project x (B,L,4096) to small low-rank outputs
        # (1024 and 512). On decode they are TWO quantized matmul dispatches
        # × 43 layers = 86 dispatches/token. Fuse via cached concatenated
        # weight tensor: ONE quantized matmul per layer producing concat
        # output (B,L,1536), then split. Saves 43 dispatches/token.
        # Activates only when an instance-level _wq_a_kv_fused tensor has
        # been built (e.g. by a load_jangtq post-load hook). Skipped if not
        # built — math identical either way.
        fused = getattr(self, "_wq_a_kv_fused", None)
        if fused is not None:
            qa_kv = fused(x)
            q_a_out = qa_kv[..., :self.q_lora_rank]
            kv_out = qa_kv[..., self.q_lora_rank:]
            q_residual = self.q_norm(q_a_out)
            kv_pre_norm = kv_out
        else:
            q_residual = self.q_norm(self.wq_a(x))
            kv_pre_norm = self.wkv(x)
        q = self.wq_b(q_residual).reshape(B, L, self.n_heads, self.head_dim)
        # Per-head RMSNorm via mx.fast.rms_norm. Phase-NoOnesQNorm (2026-04-24,
        # ralph iter, ported from mlx-lm PR #1189 line 875): pass weight=None.
        # mlx-fast supports weightless rms_norm — skips the multiply-by-weight
        # step entirely. Saves the unit-tensor read AND the broadcast multiply.
        q = mx.fast.rms_norm(q, weight=None, eps=self.args.rms_norm_eps)
        q = q.transpose(0, 2, 1, 3)

        kv = self.kv_norm(kv_pre_norm).reshape(B, L, 1, self.head_dim).transpose(0, 2, 1, 3)

        # Phase-RopeFused (2026-04-25, ralph iter): bit-exact verified
        # against _apply_partial_rope at offsets 0/7/31, both forward and
        # inverse (max_abs_diff = 0.000e+00 fp32 in /tmp/rope_fused_verify.py).
        # Replaces 3 ops per call (split + rope + concat) with one
        # @mx.compile graph. Saves ~258 dispatches/token across 3 calls/layer
        # × 43 layers. Convention: _wavelength + scale=+1.0 → forward,
        # scale=-1.0 → inverse.
        # Re-validated end-to-end on this MacBook with chat-template-wrapped
        # prompt: DSV4 produces "Paris" correctly. The earlier "regression"
        # was the raw-text-into-instruct-model trap, not the rope fusion.
        rope_dim = self.rope.dims
        wavelength = self.rope._wavelength[0]
        q = _attn_partial_rope_fused(q, offset, rope_dim, wavelength, 1.0)
        kv = _attn_partial_rope_fused(kv, offset, rope_dim, wavelength, 1.0)

        if local_cache is not None:
            kv, _ = local_cache.update_and_fetch(kv, kv)
        full_kv = kv

        # FAST PATH: for now, only run Compressor/Indexer if explicitly enabled.
        # Pool-accumulation fix 2026-04-25 replaced the previous
        # (B, 1, L_q, top_k, head_dim) materialization with a flat-pool +
        # bool-visibility-mask path (memory: O(L_q×P) for the mask vs the
        # previous O(L_q×top_k×head_dim) for the full expansion). Long-context
        # default still off pending end-to-end verification on a 32K-prompt
        # bench, but the SDPA shape bug that triggered the original revert
        # should no longer fire.
        # Initialize comp_mask_extra at function scope so the post-block
        # `if comp_mask_extra is not None` check never UnboundLocalErrors when
        # use_long_ctx=False or compress_ratio=0. (Bug found 2026-04-25.)
        comp_mask_extra = None
        # Track whether the long-context path modified `full_kv` (window+compressed
        # concatenation). If so, the external mlx-lm mask doesn't match the new
        # K dimension and we must build our own mask (or set mask=None for S=1).
        long_ctx_kv_modified = False
        import os
        use_long_ctx = os.environ.get("DSV4_LONG_CTX", "0") == "1"
        if self.compress_ratio and use_long_ctx:
            v4_cache = cache if isinstance(cache, DeepseekV4Cache) else None
            # FAST PATH: when NOT using DeepseekV4Cache (i.e., plain KVCache),
            # the compressor has no buffer state to accumulate. For L < compress_ratio
            # the pooled output is empty and gets no-op concat. Skip entirely to
            # save ~150 matmuls per token across 41 compress_ratio>0 layers.
            #
            # Only run Compressor/Indexer if:
            # - v4_cache is provided (state carries across calls), OR
            # - L >= compress_ratio (enough tokens to produce non-empty pool in one call)
            # POOL-ACCUMULATION FIX 2026-04-25 — Compressor + Indexer should
            # produce a (B, 1, P, head_dim) pool where each query at position i
            # attends only to: (a) compressed entries causally available at i
            # (j-th pool block represents tokens j*ratio..(j+1)*ratio-1, so
            # visible iff (j+1)*ratio <= i+1) AND (b) the indexer's top-k
            # selection for that query.
            #
            # Previous code materialized (B, 1, L, top_k, head_dim) ≈ 8 GB at
            # L=8K, then flattened to (B, 1, L*top_k, head_dim) and concat'd
            # with the window keys. Default SDPA mask did not segment this
            # concatenation, so query i incorrectly attended to query j's
            # selected pool slice. Fix: keep pool flat, build a (B, 1, L, P)
            # bool visibility mask covering both causal staircase AND topk
            # selection, concatenate it onto the window-side mask before SDPA.
            # Mirrors PR #1195's two-mode selection:
            #   S=1 decode: GATHER top_k pool rows directly (kv_all gets only
            #     top_k extra entries, mask=None — much smaller intermediate).
            #   S>1 prefill: keep pool flat, build (B, 1, L, P) bool mask =
            #     causal_staircase AND indexer_selected, concat to window mask.
            # Both modes avoid the (L_q × top_k × head_dim) materialization
            # the previous code did. See research/JANGTQ-COMPRESSOR-INDEXER-PROPER-FIX-2026-04-25.md.
            if v4_cache is not None or L >= self.compress_ratio:
                pooled = self.compressor(x, self.compress_rope, v4_cache, offset)
                P = pooled.shape[1]
                if P > 0:
                    if hasattr(self, "indexer"):
                        topk = self.indexer(x, q_residual, self.compress_rope, self.rope, v4_cache, offset)
                    else:
                        topk = None

                    if L == 1 and topk is not None:
                        # S=1 GATHER fast path: materialize ONLY top_k rows
                        # for the single query. Result is (B, top_k, head_dim)
                        # not (B, P, head_dim) — saves P/top_k× memory.
                        d = self.head_dim
                        K = topk.shape[-1]
                        # pooled: (B, P, d), topk: (B, 1, K)
                        idx = mx.broadcast_to(topk[:, None, :, :, None], (B, 1, 1, K, d))
                        expanded = mx.broadcast_to(pooled[:, None, None, :, :], (B, 1, 1, P, d))
                        gathered = mx.take_along_axis(expanded, idx, axis=3).reshape(B, K, d)
                        full_kv = mx.concatenate([full_kv, gathered[:, None]], axis=2)
                        long_ctx_kv_modified = True
                        # mask=None for S=1: gather already encodes selection,
                        # window keys are all-visible to single query.
                    else:
                        # S>1 prefill (or S=1 with no indexer): mask path.
                        # Use _compressed_visibility helper (PR #1195 port).
                        pooled_flat = pooled[:, None]  # (B, 1, P, head_dim)
                        comp_visible = _compressed_visibility(
                            B, L, offset, P, self.compress_ratio,
                        )
                        if topk is not None:
                            k_idx = mx.arange(P, dtype=mx.int32)
                            selected = (topk[..., None] == k_idx[None, None, None, :]).any(axis=-2)
                            comp_visible = comp_visible & selected[:, None, :, :]
                        full_kv = mx.concatenate([full_kv, pooled_flat], axis=2)
                        comp_mask_extra = comp_visible
                        long_ctx_kv_modified = True

        # Mask construction for long-context path:
        # When compressed_len > 0 we've reshaped K to (window + compressed),
        # which doesn't match the external mlx-lm-provided mask (sized for
        # the original prompt-length × cache-length). IGNORE the external
        # mask and build our own from scratch (PR #1195 convention,
        # see its V4Attention.__call__ at line 1095-1108). Build:
        #   win_mask  = sliding-window visibility (B, 1, L, L_win)
        #   comp_mask = staircase × indexer-selection (B, 1, L, P)
        # then concat along the K axis.
        # For S==1 decode with no compressed (or when comp_mask_extra
        # already encoded selection via gather), mask=None lets SDPA
        # do its normal causal handling.
        if comp_mask_extra is not None:
            L_win = full_kv.shape[2] - comp_mask_extra.shape[-1]
            win_mask = _build_window_mask(B, L, offset, self.args.sliding_window, L_win)
            sdpa_mask = mx.concatenate([win_mask, comp_mask_extra], axis=-1)
        elif long_ctx_kv_modified:
            # S=1 GATHER path: full_kv = window + top-k gathered rows.
            # External mlx-lm mask doesn't match this new K. The gather
            # already encoded selection, and a single query attends to all
            # of (window + selected) — pass mask=None so SDPA does default
            # behavior (no mask → all keys visible).
            sdpa_mask = None
        else:
            sdpa_mask = mask

        out = scaled_dot_product_attention(
            q, full_kv, full_kv,
            cache=None, scale=self.softmax_scale, mask=sdpa_mask,
            sinks=None,
        )
        # Phase-NoInverseRope FAILED (2026-04-24, ralph iter): mlx-lm PR #1189
        # SKIPS this inverse-rope but our model REQUIRES it — without it
        # decoded output is gibberish ("\n22 { {" tokens). PR #1189 must use
        # a different rope convention (V not rotated?) or has a quality bug.
        # PR #1192 + Apple reference + our impl all REQUIRE inverse-rope.
        # Phase-RopeFused (2026-04-25): inverse rope via fused helper with
        # `_wavelength + scale=-1.0` (cos stays, sin flips). Bit-exact
        # against legacy `_apply_partial_rope(inverse=True)` and validated
        # end-to-end with chat-template prompts. Saves 2 of 3 inverse-rope
        # dispatches per layer.
        out = _attn_partial_rope_fused(out, offset, rope_dim, wavelength, -1.0)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.n_heads * self.head_dim)
        out = self._grouped_output_projection(out)
        return self.wo_b(out)

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

# Phase-ScalarZero (PR #1189 line 471 reference): pre-allocated scalar zero
# for sqrtsoftplus's logaddexp. Avoids the per-call `mx.zeros_like(gates)`
# allocation. Module-level constant since all layers share the same value.
_SCORE_ZERO = mx.array(0.0, dtype=mx.float32)


@mx.compile
def sqrtsoftplus_select(
    gates: mx.array,
    bias: mx.array,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """DSV4 scoring: sqrt(softplus(gates)) + bias → top-k, then renorm.

    `gates` is expected to already be fp32 (caller must cast).
    Phase-Idx (2026-04-24): return inds as uint32 directly so the
    MoE caller no longer needs `.astype(uint32)` (Bug 1.13). Saves
    1 dispatch per layer × 43 = 43 dispatches/token.
    """
    # Phase-Logaddexp (2026-04-24): match PR #1192 reference exactly.
    # Phase-ScalarZero (2026-04-24, ralph iter, from PR #1189 line 471):
    # use module-level _SCORE_ZERO scalar instead of mx.zeros_like(gates).
    # Saves one allocation per call (per layer × 43 = 43 fewer allocs/token).
    scores = mx.sqrt(mx.logaddexp(gates, _SCORE_ZERO))
    orig_scores = scores
    scores = scores + bias
    k = top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.uint32)
    scores = mx.take_along_axis(orig_scores, inds.astype(mx.int32), axis=-1)
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
        gates = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        if self.hash:
            # Hash: deterministic per-token lookup (ignoring gates beyond
            # scoring for weights). Use original scores as weights.
            scores = mx.sqrt(mx.log1p(mx.exp(gates)))
            assert input_ids is not None, "hash-routed layer requires input_ids"
            inds = self.tid2eid[input_ids].astype(mx.uint32)
            weights = mx.take_along_axis(scores, inds.astype(mx.int32), axis=-1)
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
def _dsv4_swiglu(gate, up, swiglu_limit: float):
    """DSV4 SwiGLU with gate/up clamping to ±swiglu_limit (gate is clamped
    to max only; up is clamped symmetrically). Without the clamp, deep MoE
    stacks diverge numerically.

    IMPORTANT: torch reference does `gate.float() * up.float()` — silu and
    multiply in fp32. We match that here to avoid per-layer precision drift.
    """
    out_dtype = gate.dtype
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    if swiglu_limit > 0:
        up = mx.clip(up, a_min=-swiglu_limit, a_max=swiglu_limit)
        gate = mx.clip(gate, a_min=None, a_max=swiglu_limit)
    return (nn.silu(gate) * up).astype(out_dtype)


class _DSV4SwiGLU(nn.Module):
    def __init__(self, swiglu_limit: float):
        super().__init__()
        self.swiglu_limit = swiglu_limit

    def __call__(self, x_up, x_gate):
        return _dsv4_swiglu(x_gate, x_up, self.swiglu_limit)


# ---------- Phase A REVERTED (2026-04-24) ----------
# `@partial(mx.compile, shapeless=True)` on _hc_pre / _hc_post helpers
# regressed JANGTQ from 18.25 → 9.75 tok/s AND broke output coherence
# ("one one one" loop). Two failure modes suspected:
#   1. compile-cache thrashing: 43 layers × 3 helpers = 129 distinct
#      graphs traced per first-call (each layer has its own input array
#      identities; shapeless cache may key on those).
#   2. precision drift through the compiled trace — the fp32 cast that
#      was inline at the call site got hoisted/reordered in a way that
#      changed numerical behavior.
# Plain helpers reverted. See research/DSV4-RUNTIME-ARCHITECTURE.md §21.


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

    def __call__(self, x, input_ids=None):
        inds, scores = self.gate(x, input_ids=input_ids)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype).reshape(x.shape)
        y = y + self.shared_experts(x)
        return y


# ---------- Block with mHC ----------

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
        # Phase-RMSFast (2026-04-24): cache unit-weight tensor for
        # mx.fast.rms_norm in _hc_pre. Shape = (hc_mult * hidden).
        # Phase-FP16-Pre (2026-04-24, ralph iter): allocate as fp16 (default
        # dtype of model weights). Lets mx.fast.rms_norm + matmul run uniformly
        # in fp16 — Apple Metal tensor cores accumulate in fp32 internally so
        # numerical precision is preserved. Skips one fp32 cast of the 16384
        # element x_flat per call.
        self._hc_rms_ones = mx.ones(hc_dim, dtype=mx.float16)
        self._hc_rms_eps = float(args.rms_norm_eps)

    def _hc_pre(self, x, fn, scale, base):
        # x: (B, L, hc_mult, D)
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2)  # (B, L, hc_mult * D = 16384 for DSV4)
        x_flat_f32 = x_flat.astype(mx.float32)
        # ── CRITICAL FP32-CAST FIX (2026-04-25, ralph iter 50) ─────────────────
        # Symptom: DSV4 generated garbage on Mac Studio (M3 Ultra) but coherent
        # output on MacBook (M4) using identical code + bundle. Pattern:
        # "What is 17+28?" → "17 plus plus plus" instead of "45".
        #
        # Root cause: bf16 accumulation overflow in `mean(square(x_flat))` for
        # the 16384-dim vector. M3 Ultra's SIMD lanes implicitly accumulated in
        # bf16 (256-element mantissa adds saturate around 2^7 of relative
        # magnitude); M4 happened to use fp32 lanes. Same code path, different
        # numerical behavior — model produced wrong logits → token loop.
        #
        # Fix: explicit `astype(mx.float32)` BEFORE the reduction, force the
        # square-and-sum to run in fp32 lanes. Cast `rsqrt` back to input dtype
        # for the multiply (which is element-wise and stays well-conditioned).
        # This matches what the pip-installed Metal kernel does internally —
        # restoring the verified-coherent path.
        #
        # Mirrored in Swift `hcCollapse` (DeepseekV4JANGTQ.swift:1338) which
        # ALREADY did this via `.asType(.float32)` at the start. So Swift was
        # never broken on this issue.
        #
        # Verified: math "17+28=45" correct, full memoized Fibonacci function
        # produced. 75 tok/s prefill, 22.5 tok/s decode on M3 Ultra.
        # Memory note: `feedback_hc_pre_fp32_cast.md`.
        x_flat_f32 = x_flat.astype(mx.float32)
        var = mx.mean(mx.square(x_flat_f32), axis=-1, keepdims=True)
        rsqrt = mx.rsqrt(var + self._hc_rms_eps).astype(x_flat.dtype)
        x_normed = x_flat * rsqrt
        x_normed = x_normed * self._hc_rms_ones
        mixes = (x_normed @ fn.T).astype(mx.float32)
        pre, post, comb = hc_split_sinkhorn(
            mixes, scale, base, self.args.hc_mult,
            # Phase-Iters (2026-04-24): kept at config default (20). Bench
            # at iters=10 showed no measurable speedup (sinkhorn loop is
            # entirely inside one Metal kernel; iters affect kernel internal
            # time but not dispatch count, and on M3 Ultra the inner loop
            # is too small to matter). Output is identical so the knob is
            # safe if ever needed; expose via env in case future hardware
            # benefits.
            int(__import__("os").environ.get(
                "DSV4_SINKHORN_ITERS", str(self.args.hc_sinkhorn_iters))),
            self.args.hc_eps,
        )
        # Phase-FP16-Pre: cast pre to x.dtype FIRST (small: B*L*hc_mult elements)
        # then mul-sum stays in fp16. Avoids fp32 promotion of the larger
        # (B, L, hc_mult, D) tensor + final downcast. ~Net same dispatches but
        # smaller tensors moved through fp16/fp32 boundary.
        pre_t = pre.astype(x.dtype)
        y = mx.sum(pre_t[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y, post, comb

    def _hc_post(self, x, residual, post, comb):
        # x: (B, L, D); residual: (B, L, hc_mult, D); return (B, L, hc_mult, D)
        # Reference: y[b,s,i,d] = post[b,s,i] * x[b,s,d]
        #                       + sum_j comb[b,s,i,j] * residual[b,s,j,d]
        # Phase-FP16 (2026-04-24, ralph iter 1): eliminate fp32 casts.
        # Apple Silicon Metal tensor cores accumulate matmul in fp32 internally
        # even with fp16 inputs — explicit upcast was redundant. comb and post
        # come from the sinkhorn fused kernel as fp32; cast them DOWN to x.dtype
        # (one cast each instead of three casts of large tensors).
        # Saves: 4 dispatches per call (x_to_fp32, residual_to_fp32, multiply,
        # add result_to_target) → 1 dispatch (down-cast comb + post). × 2 per
        # layer × 43 = ~344 dispatches/token.
        post_t = post.astype(x.dtype)
        comb_t = comb.astype(x.dtype)
        y = post_t[..., None] * x[..., None, :] + mx.matmul(comb_t, residual)
        return y

    def __call__(self, x, mask=None, cache=None, input_ids=None):
        if x.ndim == 3:
            # Auto-fix for warmup or direct layer calls missing mHC dim
            x = mx.tile(x[..., None, :], (1, 1, self.args.hc_mult, 1))
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.post_attention_layernorm(x)
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
        # Phase-RMSFast + FP16-Pre: unit-weight tensor for HyperHead's RMS norm,
        # allocated as fp16 to keep the whole reduce in fp16 tensor-core path.
        self._hc_head_rms_ones = mx.ones(args.hc_mult * args.hidden_size, dtype=mx.float16)
        self._hc_head_rms_eps = float(args.rms_norm_eps)

    def _hc_head_reduce(self, x):
        # x: (B, L, hc_mult, D) → (B, L, D) via sigmoid-weighted sum
        # Phase-FP16-Pre: keep x_flat at native dtype (fp16). Sigmoid + scale +
        # bias all run fp16; final sum-reduce stays fp16. No fp32 round-trip.
        shape = x.shape
        x_flat = mx.flatten(x, start_axis=2)
        x_normed = mx.fast.rms_norm(x_flat, self._hc_head_rms_ones, self._hc_head_rms_eps)
        mixes = x_normed @ self.hc_head_fn.T
        pre = mx.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.args.hc_eps
        y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
        return y

    def __call__(self, input_ids, cache=None, mask=None):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
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
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c, input_ids=input_ids)
        h = self._hc_head_reduce(h)
        return self.norm(h)


class Model(nn.Module):
    """mlx_lm entry-point class — what load_jangtq_model / mlx-lm factory expects."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = DeepseekV4Model(args)
        # Tied weight option not confirmed for DSV4 — use separate lm_head
        # (config has tie_word_embeddings=false)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids, cache=None, mask=None):
        h = self.model(input_ids, cache=cache, mask=mask)
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
        """Build per-layer cache objects.
        SHORT-PROMPT-SAFE default: use plain KVCache for all layers. Compressor
        + Indexer fast-path is taken in DeepseekV4Attention (cache is None for
        v4-state, so pooled is empty and skipped). This makes prompts up to
        sliding_window=128 tokens behave identically to the pre-make_cache path.
        For >128 tokens, attention falls back to local-only sliding-window context
        (still coherent, but loses pooled-global benefit). To enable full
        long-context behavior with Compressor + Indexer, set the env var
        DSV4_LONG_CTX=1 — then compress_ratio>0 layers get DeepseekV4Cache.
        """
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        import os
        # Default "0" = plain KVCache (sliding-window-only). DSV4_LONG_CTX=1
        # enables DeepseekV4Cache (Compressor+Indexer pooled global). The
        # long-ctx path has a known prefill shape bug on prompts that hit
        # compress_ratio (manifests as `Shapes (L,L) and (1,H,L,big) cannot
        # be broadcast` from SDPA mask). Until that bug is fixed, default
        # off — sliding window is short-prompt-safe and matches verified
        # 67% pass@1 HumanEval result on Mac Studio.
        long_ctx = os.environ.get("DSV4_LONG_CTX", "0") == "1"
        caches = []
        for layer in self.model.layers:
            if long_ctx and layer.self_attn.compress_ratio:
                caches.append(DeepseekV4Cache(self.args.sliding_window))
            else:
                caches.append(KVCache())
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
                m3 = re.match(r"experts\.(\d+)\.(w[123])\.(weight|scales|biases|tq_packed|tq_norms|tq_bits)$", inner)
                if m3:
                    # Temporary marker — will be stacked below
                    out[f"__TEMP__{pfx}.mlp.experts.{m3.group(1)}.{w1w2w3[m3.group(2)]}.{m3.group(3)}"] = v
                    continue
                # Fallback
                out[f"{pfx}.mlp.{inner}"] = v; continue

            out[f"{pfx}.{rest}"] = v

        # Stack routed experts across all layers
        # Phase-Stack-Fast (2026-04-24): previous nested loops with repeated pop/stack
        # triggered OOM/timeouts on DSV4 (33,792 tensors). Group then stack.
        n_experts = self.args.n_routed_experts
        groups = {}
        for k in list(out.keys()):
            if k.startswith("__TEMP__"):
                # __TEMP__model.layers.L.mlp.experts.E.PROJ.KIND
                parts = k.split(".")
                layer_id = parts[2]
                proj = parts[6]
                kind = parts[7]
                group_key = f"model.layers.{layer_id}.mlp.switch_mlp.{proj}.{kind}"
                groups.setdefault(group_key, [None] * n_experts)
                groups[group_key][int(parts[5])] = out.pop(k)

        for group_key, tensors in groups.items():
            if all(t is not None for t in tensors):
                out[group_key] = mx.stack(tensors)
            else:
                missing = [i for i, t in enumerate(tensors) if t is None]
                raise RuntimeError(f"Expert stacking failed for {group_key}: missing experts {missing[:5]}...")

        # Final guard: no __TEMP__ keys should remain
        leftovers = [k for k in out if k.startswith("__TEMP__")]
        if leftovers:
            raise RuntimeError(f"sanitize left {len(leftovers)} unstacked TEMP keys, "
                               f"e.g. {leftovers[0]}")
        return out
