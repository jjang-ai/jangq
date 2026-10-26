"""DeepSeek-V4.1-Flash text model in MLX, for JANG affine bundles.

A batch-1 port of the release's inference/model.py (autoregressive path only):
hyper-connections (hc_mult=4, Sinkhorn mixing), latent attention over a
128-token sliding window plus compressed KV picked by the two-level indexer,
cross-layer KV / index / candidate sharing, Engram n-gram lookups at layers 1
and 14, sqrtsoftplus-routed MoE with one shared expert.

Not ported: the vision tower (bundle tensors are present but unused here),
DSpark speculative heads (not in the bundle), and the release's FP8/FP4
activation round-trips (activations stay bf16/fp32 here). Prefill must be a
single call starting at position 0, then one token per call -- the same
contract as the release's generate.py.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jang_tools.dsv41.engram import EngramLayout, NgramHasher, open_engram_tables

# Forwards of at least ENGRAM_LAZY_MIN tokens read the later Engram layers' rows while the GPU
# runs the layers before them (DSV41._engram_rows(lazy=True)); shorter ones read inline.
ENGRAM_LAZY_MIN = 64

GROUP_SIZE = 64


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "off", "no"}


# Decode / DSpark verify forwards of at most this many tokens submit each layer to the GPU as soon as it
# is built (mx.async_eval per layer) instead of building all 40 layers and then waiting, so the GPU runs
# the early layers while Python builds the later ones (ds4 #1261, oMLX language.py). Bit-identical:
# only the scheduling changes. VMLX_DSV41_LAYER_ASYNC=0 restores the single end-of-graph eval.
LAYER_ASYNC = _env_on("VMLX_DSV41_LAYER_ASYNC")
LAYER_ASYNC_MAX_ROWS = 16
# Router logits for <= 8 rows from the bf16 gate weight with MLX's f32 GEMV order (metal_decode.py):
# no per-call f32 copy of the 384 x 5120 weight. 1 row: bit-identical; 2-8 rows: each row exactly its
# 1-row decode logits. VMLX_DSV41_ROUTER_GEMV=0 restores the f32-cast matmul.
ROUTER_GEMV = _env_on("VMLX_DSV41_ROUTER_GEMV")
# Routed + shared experts for <= 8 rows in two Metal dispatches (metal_decode.moe_decode, mlx-serve
# MOE_GU/MOE_DN) instead of ~25. 1 row: bit-identical; 2-8 rows: each row exactly its 1-row decode.
# VMLX_DSV41_FUSED_MOE=0 restores gather_qmm + qlinear.
FUSED_MOE = _env_on("VMLX_DSV41_FUSED_MOE")
# Hyper-connection stage for <= 8 rows in three Metal dispatches per sublayer (metal_decode.hc_*:
# mixes + collapse + RMSNorm, Sinkhorn split, expand) instead of ~14, each MLX op's order reproduced.
# Bit-identical (checked once per process against the stock ops; a mismatch keeps the stock path).
# VMLX_DSV41_FUSED_HC=0 restores the stock ops.
FUSED_HC = _env_on("VMLX_DSV41_FUSED_HC")


@dataclass
class Args:
    vocab_size: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    swiglu_limit: float
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict
    n_routed_experts: int
    num_experts_per_tok: int
    routed_scaling_factor: float
    norm_topk_prob: bool
    sliding_window: int
    compress_ratios: list
    compress_rope_theta: float
    kv_source_layer_ids: list
    index_source_layer_ids: list
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    engram_layer_ids: list
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, text_cfg: dict) -> "Args":
        names = cls.__dataclass_fields__.keys() - {"raw"}
        return cls(**{k: text_cfg[k] for k in names}, raw=text_cfg)


# ----------------------------------------------------------------------------- rope

def yarn_inv_freq(dim, base, original_seq_len, factor, beta_fast, beta_slow) -> np.ndarray:
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    if original_seq_len > 0:
        def corr(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs.astype(np.float32)


# Positions given as (start, step) -- position of row i = (start + i) * step, every call site except a multi-segment
# batch -- run through mx.fast.rope: one kernel instead of the angle / cos / sin / rotate graph (~8 dispatches per
# call, 3-4 calls per layer per forward). Same fp32 angles (MLX scales the integer position, then divides by the
# wavelength) and the same bf16 store. VMLX_DSV41_FAST_ROPE=0 keeps the compiled cos/sin path.
FAST_ROPE = os.environ.get("VMLX_DSV41_FAST_ROPE", "1").strip().lower() not in {"0", "false", "off", "no"}


class Rope:
    """Interleaved-pair rotary embedding on the last `dim` features, at arbitrary positions."""

    def __init__(self, dim, inv_freq: np.ndarray):
        self.dim = dim
        self.inv_freq = mx.array(inv_freq)
        wl = (1.0 / np.asarray(inv_freq, np.float64)).astype(np.float32)  # mx.fast.rope takes wavelengths
        self.wavelen, self.neg_wavelen = mx.array(wl), mx.array(-wl)

    def __call__(self, x: mx.array, positions, inverse: bool = False) -> mx.array:
        """x [..., L, (heads,) D] where the L axis matches `positions`; rotates x[..., -dim:].
        positions: an int array [L], or (start, step) for positions (start + i) * step."""
        if isinstance(positions, tuple):
            start, step = int(positions[0]), int(positions[1])
            if FAST_ROPE:
                return _fast_rope(x, self.dim, start, step, self.neg_wavelen if inverse else self.wavelen)
            positions = (start + mx.arange(x.shape[1])) * step
        return _rope_fn(self.dim, x.ndim, inverse)(x, positions, self.inv_freq)


def _fast_rope(x, dim, start: int, step: int, wavelen):
    head, tail = x[..., :-dim], x[..., -dim:]
    if x.ndim == 4:  # [B, L, H, D]: mx.fast.rope wants the sequence axis at -2
        t = mx.fast.rope(tail.transpose(0, 2, 1, 3), dim, traditional=True, base=None, scale=float(step),
                         offset=start, freqs=wavelen).transpose(0, 2, 1, 3)
    else:
        t = mx.fast.rope(tail, dim, traditional=True, base=None, scale=float(step), offset=start, freqs=wavelen)
    return mx.concatenate([head, t], axis=-1)


_ROPE = {}


def _rope_fn(dim: int, ndim: int, inverse: bool):
    key = (dim, ndim, inverse)
    if key not in _ROPE:
        def rope(x, positions, inv_freq):
            ang = positions.astype(mx.float32)[:, None] * inv_freq[None]  # [L, dim/2]
            cos, sin = mx.cos(ang), mx.sin(ang)
            if inverse:
                sin = -sin
            if ndim == 4:  # [B, L, H, D]
                cos, sin = cos[:, None, :], sin[:, None, :]
            head, tail = x[..., :-dim], x[..., -dim:]
            t = tail.astype(mx.float32).reshape(*tail.shape[:-1], dim // 2, 2)
            a, b = t[..., 0], t[..., 1]
            rot = mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(tail.shape)
            return mx.concatenate([head, rot.astype(x.dtype)], axis=-1)

        _ROPE[key] = mx.compile(rope)
    return _ROPE[key]


# ----------------------------------------------------------------------------- layers

class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        # fp32 statistics, weight applied before the cast back -- same order as the release
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _fp8_rows_kernel():
    """vMLX's multi-row mxfp8 QMV (each weight read once for 2..8 rows, every row bit-identical to the 1-row
    kernel), or None when vMLX is not importable. VMLX_FP8_ROWS_QMV=0 disables it per call."""
    try:
        from vmlx_engine.metal.fp8_rows_qmv import fp8_rows_qmv
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return fp8_rows_qmv


FP8_ROWS = _fp8_rows_kernel()


class MxLinear(nn.QuantizedLinear):
    """mxfp8 Linear. 2..8 rows (DSpark verify / draft) go through FP8_ROWS: MLX switches mxfp8 kernels above a few rows
    (DeepSeek-V4.1, 40 layers, 6 rows: wo_b 3.05 -> 7.04 ms, wq_b 2.48 -> 4.33 ms vs 1 row) although the rows need no
    extra weight bytes; the multi-row kernel also makes each verify row equal its 1-row decode on these projections."""

    def __call__(self, x):
        rows = x.size // x.shape[-1]
        if FP8_ROWS is not None and 2 <= rows <= 8:
            y = FP8_ROWS(x.reshape(1, rows, x.shape[-1]), self.weight, self.scales)
            if y is not None:
                return y.reshape(*x.shape[:-1], -1)
        return super().__call__(x)


def qlinear(in_dim, out_dim, bits):
    """bits: affine bit width (group GROUP_SIZE), or "mxfp8" (JANGHT bundles: byte-exact repack of the release FP8,
    E4M3 weights with an E8M0 scale per 32 inputs)."""
    if bits == "mxfp8":
        return MxLinear(in_dim, out_dim, bias=False, group_size=32, bits=8, mode="mxfp8")
    return nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=GROUP_SIZE, bits=bits)


def grouped_qmm(x, lin, groups, rows):
    """x [G, ..., K] (or [..., G, 1, K]) through the block-diagonal quantized Linear `lin` viewed as `groups` blocks of
    `rows` outputs: one batched quantized matmul. Affine (biases) and mxfp8 (no biases) storage."""
    w = lin.weight.reshape(groups, rows, -1)
    sc = lin.scales.reshape(groups, rows, -1)
    if FP8_ROWS is not None and getattr(lin, "mode", "affine") == "mxfp8" and x.ndim == 3 and x.shape[0] == groups \
            and 2 <= x.shape[1] <= 8:   # group-major multi-row (verify): each group's weights read once
        y = FP8_ROWS(x, w, sc)
        if y is not None:
            return y
    bi = getattr(lin, "biases", None)
    bi = None if bi is None else bi.reshape(groups, rows, -1)
    return mx.quantized_matmul(x, w, sc, bi, transpose=True, group_size=lin.group_size, bits=lin.bits,
                               mode=getattr(lin, "mode", "affine"))


class DenseLinear(nn.Module):
    """Source-precision weight kept as stored (bf16 / f32)."""

    def __init__(self, in_dim, out_dim, dtype=mx.bfloat16):
        super().__init__()
        self.weight = mx.zeros((out_dim, in_dim), dtype=dtype)

    def __call__(self, x, compute=None):
        w = self.weight if compute is None else self.weight.astype(compute)
        return (x if compute is None else x.astype(compute)) @ w.T


class QuantGrowBuffer:
    """Append-only [1, T, D] buffer whose rows are stored 4-bit (affine, group_size GROUP_SIZE) so the
    DSA compressed/index pools match the release's FP4 cache footprint (~4x smaller than bf16; a 1M-ctx
    cache is ~890 MB, not ~3.2 GB). append() quantizes the incoming bf16 rows; view() dequantizes the
    first n back to bf16 for the attention/indexer read path (the release dequantizes on read too). The
    transient dequantized view is freed each step; only the 4-bit store is resident."""

    STEP = 2048
    BITS = 4

    def __init__(self):
        self.wq = None       # [1, T, D*BITS/32] uint32 packed
        self.scales = None   # [1, T, D/GROUP_SIZE]
        self.biases = None
        self.n = 0
        self.D = None

    def append(self, x: mx.array):
        k = x.shape[1]
        if k == 0:
            return
        self.D = x.shape[2]
        wq, s, b = mx.quantize(x[0].astype(mx.bfloat16), group_size=GROUP_SIZE, bits=self.BITS)
        wq, s, b = wq[None], s[None], b[None]
        if self.wq is None or self.n + k > self.wq.shape[1]:
            cap = ((self.n + k + self.STEP - 1) // self.STEP) * self.STEP
            nwq = mx.zeros((1, cap, wq.shape[2]), dtype=mx.uint32)
            ns = mx.zeros((1, cap, s.shape[2]), dtype=s.dtype)
            nb = mx.zeros((1, cap, b.shape[2]), dtype=b.dtype)
            if self.wq is not None:
                nwq[:, : self.n] = self.wq[:, : self.n]
                ns[:, : self.n] = self.scales[:, : self.n]
                nb[:, : self.n] = self.biases[:, : self.n]
            self.wq, self.scales, self.biases = nwq, ns, nb
        self.wq[:, self.n : self.n + k] = wq
        self.scales[:, self.n : self.n + k] = s
        self.biases[:, self.n : self.n + k] = b
        self.n += k

    def view(self, n=None):
        n = self.n if n is None else n
        if self.wq is None or n == 0:
            return None
        deq = mx.dequantize(self.wq[0, :n], self.scales[0, :n], self.biases[0, :n],
                            group_size=GROUP_SIZE, bits=self.BITS)
        return deq[None]

    def gather(self, idx: mx.array) -> mx.array:
        """Dequantize only the rows ``idx`` ([M] int32, each in [0, n)) -> [M, D] bf16. Bit-identical
        to ``view()[0][idx]`` (dequantization is per row/group), without materializing the pool."""
        wq = mx.take(self.wq[0], idx, axis=0)
        sc = mx.take(self.scales[0], idx, axis=0)
        bi = mx.take(self.biases[0], idx, axis=0)
        return mx.dequantize(wq, sc, bi, group_size=GROUP_SIZE, bits=self.BITS)

    def snapshot(self):
        return (self.wq, self.scales, self.biases, self.n, self.D)

    def load(self, s):
        self.wq, self.scales, self.biases, self.n, self.D = s

    def load_dense(self, x):
        """Replace the buffer with the 4-bit quantization of a dense [1,T,D] bf16 array (used to
        restore from a bf16 prefix-cache state without changing the stored serialization format)."""
        self.wq = self.scales = self.biases = None
        self.n = 0
        self.D = None
        if x is not None and x.shape[1] > 0:
            self.append(x)


class LayerCache:
    def __init__(self):
        self.window = None  # [1, <=win, D] raw post-RoPE KV, oldest first
        self.compress_kv = QuantGrowBuffer()   # 4-bit resident (FP4-parity footprint)
        self.index_k = QuantGrowBuffer()       # 4-bit resident
        self.pend_kv = None  # compressor: tokens of the not yet complete group (ratio > 1), fp32
        self.pend_score = None


class Shared:
    """What attention layers hand down the stack (release: SharedAttentionRuntime)."""

    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None
        self._views = {}
        self.capture = False  # stash per-token projections for DSV41.commit_prefix (DSpark verify)

    def view(self, buf, n):
        """``buf.view(n)`` memoized for this forward: a pool shared by several consumer layers (layer
        20's index pool feeds 5 indexers; each compressed pool up to 6 attention layers) is
        dequantized once per forward instead of once per consumer. Lives only as long as this
        Shared (one forward), so nothing extra stays resident between steps."""
        key = (id(buf), int(n), int(getattr(buf, "n", 0)))
        hit = self._views.get(key)
        if hit is None or hit[0] is not buf:
            hit = (buf, buf.view(n))
            self._views[key] = hit
        return hit[1]


# Queries per segment up to which attention gathers only its selected compressed rows from the 4-bit
# pool instead of dequantizing the whole pool (decode = 1, DSpark verify <= bs + 1).
GATHER_MAX_QUERIES = 8
# Prompt-suffix pruning (Model.__call__ prompt_end): the layers after the last KV-source layer own no
# cache but their 128-token windows, so each one only needs the prompt rows the last prompt token can
# still reach through the windows above it. VMLX_DSV41_SUFFIX_PRUNE=0 runs every row through every layer.
SUFFIX_PRUNE = os.environ.get("VMLX_DSV41_SUFFIX_PRUNE", "1").strip().lower() not in {"0", "false", "off", "no"}


def sparse_attention(q, kv, sink, idxs, scale, chunk=256):
    """q [1,S,H,D], kv [1,N,D] (keys == values), sink [H], idxs [1,S,K] int32 (-1 = none)."""
    q, kv, idxs = q[0], kv[0].astype(mx.float32), idxs[0]
    S, H, D = q.shape
    outs = []
    for c in range(0, S, chunk):
        outs.append(_attn_chunk(q[c : c + chunk], kv, sink, idxs[c : c + chunk], scale))
    return mx.concatenate(outs, axis=0)[None] if len(outs) > 1 else outs[0][None]


@mx.compile
def _attn_core(q, kv, sink, ix, scale):
    valid = ix >= 0
    g = mx.take(kv, mx.where(valid, ix, 0), axis=0)  # [C,K,D]
    s = (q.astype(mx.float32) @ g.transpose(0, 2, 1)) * scale  # [C,H,K]
    s = mx.where(valid[:, None, :], s, -mx.inf)
    m = mx.maximum(mx.max(s, axis=-1, keepdims=True), sink[None, :, None])
    p = mx.exp(s - m)
    denom = mx.sum(p, axis=-1, keepdims=True) + mx.exp(sink[None, :, None] - m)
    return (p @ g) / denom


def _attn_chunk(q, kv, sink, ix, scale):
    return _attn_core(q, kv, sink, ix, mx.array(scale, mx.float32))


# Prefill rows at least this long take the block top-k (topk_indices): MLX's GPU argpartition is a full
# sort of the row, which at long context dominates the indexer (2048 x 100k keys: 32.7 -> 13.5 ms,
# 256 x 1M: 51.9 -> 3.2 ms for the selection). Decode (<= GATHER_MAX_QUERIES rows) keeps the direct
# partition: faster there (170k context: 25.1 vs 23.9 tok/s) and its bf16 index scores tie often.
TOPK_BLOCK_MIN = 65536
TOPK_BLOCK = 64
# Fused index scores take the bf16 q / index keys / weights as is (the reference scores in bf16) on the
# M5 matrix units when vMLX supports it; VMLX_DSV41_INDEX_BF16=0 keeps fp32 operands.
FUSED_INDEX_BF16 = os.environ.get("VMLX_DSV41_INDEX_BF16", "1").strip().lower() not in {"0", "false", "off", "no"}


def topk_indices(scores, k):
    """Indices of the k largest along the last axis (unordered).

    Long rows: the k largest values lie in the k blocks (of TOPK_BLOCK) with the largest maxima, so
    select those blocks first and run the exact selection over their k*TOPK_BLOCK entries. Same
    values selected; only the choice among exactly tied values can differ. Indices past the row end
    (block padding) are -inf entries; callers mask indices past the visible length."""
    T = scores.shape[-1]
    if k >= T:
        return mx.broadcast_to(mx.arange(T, dtype=mx.int32), scores.shape)
    if scores.ndim == 2 and scores.shape[0] > GATHER_MAX_QUERIES and T >= TOPK_BLOCK_MIN and T // TOPK_BLOCK > k:
        C, b = scores.shape[0], TOPK_BLOCK
        pad = -T % b
        s = mx.concatenate([scores, mx.full((C, pad), -mx.inf, dtype=scores.dtype)], axis=-1) if pad else scores
        nb = s.shape[1] // b
        top_b = mx.argpartition(-mx.max(s.reshape(C, nb, b), axis=-1), kth=k - 1, axis=-1)[:, :k]
        cand = (top_b[:, :, None] * b + mx.arange(b)[None, None]).reshape(C, -1)
        sel = mx.argpartition(-mx.take_along_axis(s, cand, axis=-1), kth=k - 1, axis=-1)[:, :k]
        return mx.take_along_axis(cand, sel, axis=-1).astype(mx.int32)
    return mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)


def select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    """Level one of the two-level top-k. logits [C,T] with unreachable at -inf; lens [C,1]."""
    C, T = logits.shape
    pad = -T % block_size
    sc = mx.concatenate([logits, mx.full((C, pad), -mx.inf)], axis=-1) if pad else logits
    sc = mx.max(sc.reshape(C, -1, block_size), axis=-1)
    nb = sc.shape[-1]
    last = (compress_lens - 1) // block_size
    sc = mx.where(mx.arange(nb)[None] == last, mx.inf, sc)
    if topk_blocks >= nb:
        keep = sc > -mx.inf
    else:
        top = mx.argpartition(-sc, kth=topk_blocks - 1, axis=-1)[:, :topk_blocks]
        vals = mx.take_along_axis(sc, top, axis=-1)
        keep = mx.put_along_axis(mx.zeros((C, nb), dtype=mx.bool_), top, vals > -mx.inf, axis=-1)
    return mx.repeat(keep, block_size, axis=-1)[:, :T]


class Compressor(nn.Module):
    def __init__(self, args: Args, ratio: int):
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(args.head_dim, args.rms_norm_eps)
        self.wkv = DenseLinear(args.hidden_size, args.head_dim)
        if ratio > 1:
            self.wgate = DenseLinear(args.hidden_size, args.head_dim)

    def __call__(self, x, start_pos, cache: LayerCache, capture: bool = False):
        """Latents for the groups completed by this chunk, and the index of the first one.

        Group g pools tokens g*r .. g*r+r-1; a trailing partial group waits in the cache, so any
        chunking (single prefill, continuation, one token per step) yields the same groups."""
        r = self.ratio
        if r == 1:  # one token per group: a plain projection, no pooling
            latent = self.norm(self.wkv(x))
            if capture:
                cache.verify_proj = latent
            return latent, start_pos
        kv, score = self.wkv(x, mx.float32), self.wgate(x, mx.float32)
        if capture:
            cache.verify_proj = (kv, score)
        return self._pool(kv, score, start_pos, cache, x.dtype)

    def commit(self, proj, keep: int, start_pos: int, cache: LayerCache):
        """Replay only the first ``keep`` tokens of a captured forward onto the restored cache:
        the same groups / pending tail a forward of exactly those tokens would leave."""
        if self.ratio == 1:
            return proj[:, :keep], start_pos
        kv, score = proj
        return self._pool(kv[:, :keep], score[:, :keep], start_pos, cache, mx.bfloat16)

    def _pool(self, kv, score, start_pos, cache: LayerCache, dtype):
        r = self.ratio
        n_pend = 0
        if cache.pend_kv is not None:
            n_pend = cache.pend_kv.shape[1]
            kv = mx.concatenate([cache.pend_kv, kv], axis=1)
            score = mx.concatenate([cache.pend_score, score], axis=1)
        total = kv.shape[1]
        full = total - total % r
        g0 = (start_pos - n_pend) // r
        cache.pend_kv = kv[:, full:] if full < total else None
        cache.pend_score = score[:, full:] if full < total else None
        if full == 0:
            return None, g0
        B = kv.shape[0]
        kv = kv[:, :full].reshape(B, -1, r, kv.shape[-1])
        score = score[:, :full].reshape(B, -1, r, score.shape[-1])
        out = mx.sum(kv * mx.softmax(score, axis=2), axis=2)
        return self.norm(out.astype(dtype)), g0


def _fused_index_scores():
    """vMLX's fused DSA index-score Metal kernel (`dsa_pool_index_scores`), or None when vMLX is not
    importable or VMLX_DSV41_FUSED_INDEX=0. The kernel returns None for shapes it does not admit, and
    the caller then takes the stock path."""
    if os.environ.get("VMLX_DSV41_FUSED_INDEX", "1").strip().lower() in {"0", "false", "off", "no"}:
        return None
    try:
        from vmlx_engine.metal.dsa_index_scores import dsa_pool_index_scores
    except ImportError:  # vMLX not installed: the stock bf16 indexer path
        return None
    return dsa_pool_index_scores


class Indexer(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        self.owns_k = layer_id in args.kv_source_layer_ids
        self.ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer_id
        self.uses_candidates = 0 <= args.candidate_source_layer_id < layer_id
        self.n_heads, self.head_dim = args.index_n_heads, args.index_head_dim
        self.topk = args.index_topk
        self.cand_blocks, self.cand_size = args.candidate_topk_blocks, args.candidate_block_size
        pfx = f"layers.{layer_id}.attn.indexer"
        self.wq_b = qlinear(args.q_lora_rank, self.n_heads * self.head_dim, qbits[f"{pfx}.wq_b"])
        self.weights_proj = DenseLinear(args.hidden_size, self.n_heads)
        if self.owns_k:
            self.wk = DenseLinear(args.head_dim, self.head_dim)
            self.k_norm = RMSNorm(self.head_dim, args.rms_norm_eps)

    def __call__(self, x, qr, latent, g0, start_pos, rope: Rope, cache: LayerCache, shared: Shared):
        B, S, _ = x.shape
        r, end = self.ratio, start_pos + S
        if self.owns_k and latent is not None:
            k = self.k_norm(self.wk(latent))
            cache.index_k.append(rope(k, (g0, r)))
        if self.owns_k:
            shared.index_k = cache.index_k
        T = end // r
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        # the release scores in bf16 end to end; near-tied positions follow that rounding
        q = rope(q, (start_pos, 1))[0].astype(mx.bfloat16)  # [S,h,d]
        index_k = shared.view(shared.index_k, T)[0].astype(mx.bfloat16)  # [T,d]
        w = (self.weights_proj(x) * (self.head_dim ** -0.5 * self.n_heads ** -0.5))[0]  # [S,h] bf16
        k = min(self.topk, T)
        fused = _fused_index_scores() if S > GATHER_MAX_QUERIES and T >= 32 else None
        if fused is not None:
            # Prefill: one fused kernel per query chunk computes sum_h w * relu(q . k) in fp32 without
            # materializing [C, h, T] (100k keys: 164 -> 97 ms per index layer per 2048-token chunk).
            # The chunk keeps the [C, T] fp32 scores near 1 GB at any context length.
            chunk = max(16, min(S, (1 << 28) // T))
            qf, kf, wf = q.astype(mx.float32), index_k.astype(mx.float32)[None], w.astype(mx.float32)
        else:
            chunk = max(16, min(512, (1 << 26) // max(1, self.n_heads * T)))
        out_idx, out_cand = [], []
        for c in range(0, S, chunk):
            qi = q[c : c + chunk]
            sc = None
            if fused is not None:
                sc = None
                if FUSED_INDEX_BF16:
                    try:  # vMLX >= 391ea34f: bf16 operands on the M5 matrix units (~4x the fp32 kernel)
                        sc = fused(q[None, c : c + chunk], index_k[None], w[None, c : c + chunk], 1.0,
                                   bf16_operands=True)
                    except TypeError:  # older vMLX: fp32 operands only
                        sc = None
                if sc is None:
                    sc = fused(qf[None, c : c + chunk], kf, wf[None, c : c + chunk], 1.0)
                sc = None if sc is None else sc[0]
            if sc is None:
                sc = mx.maximum(qi @ index_k.T, 0)  # [C,h,T]
                sc = mx.sum((sc * w[c : c + chunk, :, None]).astype(mx.float32), axis=1).astype(mx.bfloat16)
                sc = sc.astype(mx.float32)  # [C,T]
            # compressed positions a query can see: a group is visible once the query passed its last token
            lens = ((start_pos + mx.arange(c, c + qi.shape[0]) + 1) // r)[:, None]
            sc = mx.where(mx.arange(T)[None] >= lens, -mx.inf, sc)
            if self.is_candidate_source:
                out_cand.append(select_candidate_blocks(sc, lens, self.cand_blocks, self.cand_size))
            elif self.uses_candidates:
                sc = mx.where(shared.candidates[c : c + chunk], sc, -mx.inf)
            idx = topk_indices(sc, k)
            out_idx.append(mx.where(idx < lens, idx, -1).astype(mx.int32))
        if self.is_candidate_source:
            shared.candidates = mx.concatenate(out_cand, axis=0)
        return mx.concatenate(out_idx, axis=0)[None]


def append_compressed_rows(raw, widx, cidx, pool, T, shared, gather=False):
    """Append the compressed-pool rows the queries attend to after the window rows.

    raw [1,L,D] window KV, widx [1,S,W] window indices into raw, cidx [1,S,K] top-k compressed indices
    in [0,T) (-1 = none). Returns (kv, idxs) for sparse_attention.

    gather=False: the whole pool (dequantized once per forward via shared.view) is appended and cidx
    is offset into it. gather=True (decode / DSpark verify): only the S*K selected rows are dequantized
    from the 4-bit pool, each query's K slots get their own rows in the same order -- bit-identical
    attention, without the full [T,D] dequant + concat + fp32 cast (~0.5-1 GB per layer per token at 1M
    context)."""
    if gather and isinstance(pool, QuantGrowBuffer):
        S, K = cidx.shape[1], cidx.shape[2]
        valid = cidx >= 0
        rows = pool.gather(mx.where(valid, cidx, 0).reshape(-1).astype(mx.int32))  # [S*K, D]
        slot = mx.arange(S * K, dtype=mx.int32).reshape(1, S, K) + raw.shape[1]
        cidx = mx.where(valid, slot, -1)
        kv = mx.concatenate([raw, rows[None].astype(raw.dtype)], axis=1)
    else:
        comp = shared.view(pool, T) if shared is not None else pool.view(T)
        cidx = mx.where(cidx >= 0, cidx + raw.shape[1], -1)
        kv = mx.concatenate([raw, comp], axis=1)
    return kv, mx.concatenate([widx, cidx], axis=-1)


class Seg:
    """One sequence's slice of the token axis in a forward call."""

    __slots__ = ("lo", "n", "start", "caches", "shared")

    def __init__(self, lo, n, start, caches):
        self.lo, self.n, self.start, self.caches = lo, n, start, caches
        self.shared = Shared()


class Attention(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        pfx = f"layers.{layer_id}.attn"
        d, H = args.hidden_size, args.num_attention_heads
        self.layer_id = layer_id
        self.n_heads, self.head_dim, self.rd = H, args.head_dim, args.qk_rope_head_dim
        self.n_groups, self.o_lora_rank = args.o_groups, args.o_lora_rank
        self.window = args.sliding_window
        self.ratio = args.compress_ratios[layer_id]
        self.scale = self.head_dim ** -0.5
        self.attn_sink = mx.zeros((H,), mx.float32)
        self.wq_a = qlinear(d, args.q_lora_rank, qbits[f"{pfx}.wq_a"])
        self.q_norm = RMSNorm(args.q_lora_rank, args.rms_norm_eps)
        self.wq_b = qlinear(args.q_lora_rank, H * self.head_dim, qbits[f"{pfx}.wq_b"])
        self.wkv = qlinear(d, self.head_dim, qbits[f"{pfx}.wkv"])
        self.kv_norm = RMSNorm(self.head_dim, args.rms_norm_eps)
        self.wo_a = qlinear(H * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, qbits[f"{pfx}.wo_a"])
        self.wo_b = qlinear(self.n_groups * self.o_lora_rank, d, qbits[f"{pfx}.wo_b"])
        self.is_kv_source = layer_id in args.kv_source_layer_ids
        self.is_index_source = layer_id in args.index_source_layer_ids
        if self.is_kv_source:
            self.compressor = Compressor(args, self.ratio)
        if self.is_index_source:
            self.indexer = Indexer(args, layer_id, qbits)
        rs = args.rope_scaling
        if self.ratio:
            inv = yarn_inv_freq(self.rd, args.compress_rope_theta, rs["original_max_position_embeddings"],
                                rs["factor"], rs["beta_fast"], rs["beta_slow"])
        else:
            inv = yarn_inv_freq(self.rd, args.rope_theta, 0, 1, 0, 0)
        self._rope = Rope(self.rd, inv)

    def __call__(self, x, segs):
        """x [1,N,D] holds the tokens of every segment back to back; projections run batched,
        window / compression / indexing / attention run per segment against its own cache."""
        B, N, _ = x.shape
        pos = mx.concatenate([mx.arange(s.start, s.start + s.n) for s in segs]) if len(segs) > 1 \
            else (segs[0].start, 1)
        qr = self.q_norm(self.wq_a(x))
        q = self._rope(self.wq_b(qr).reshape(B, N, self.n_heads, self.head_dim), pos)
        kv = self._rope(self.kv_norm(self.wkv(x)), pos)
        outs = []
        for s in segs:
            sl = slice(s.lo, s.lo + s.n)
            outs.append(self._core(x[:, sl], qr[:, sl], q[:, sl], kv[:, sl], s.start, s.caches[self.layer_id], s.shared))
        o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        o = self._rope(o, pos, inverse=True).astype(mx.bfloat16)
        # wo_a is block-diagonal over groups: one batched quantized matmul per group
        if B * N > 1:
            # Every multi-row forward (prefill and DSpark verify): group-major [G, tokens, K] so each
            # group is one matrix multiply reading its weights once. The token-major layout below runs
            # one 1-row product per token per group (2048-token chunk: 18.6 ms -> 2.1 ms per layer;
            # 6-row verify, 40 layers, mxfp8: 10.6 -> 4.5 ms). Verify rows were never bit-identical to
            # 1-row decode anyway (the other projections switch kernels with the row count).
            og = o.reshape(B * N, self.n_groups, -1).transpose(1, 0, 2)
            og = grouped_qmm(og, self.wo_a, self.n_groups, self.o_lora_rank)
            return self.wo_b(og.transpose(1, 0, 2).reshape(B, N, -1))
        # Decode: one row.
        o = grouped_qmm(o.reshape(B * N, self.n_groups, 1, -1), self.wo_a, self.n_groups, self.o_lora_rank)
        return self.wo_b(o.reshape(B, N, -1))

    def _core(self, x, qr, q, kv, start, cache: LayerCache, shared: Shared):
        S = x.shape[1]
        win = self.window
        prev = cache.window
        raw = kv if prev is None else mx.concatenate([prev, kv], axis=1)
        base = start - (0 if prev is None else prev.shape[1])  # absolute position of raw[0]
        K = min(win, start + S)
        wpos = (start + mx.arange(S))[:, None] - (K - 1) + mx.arange(K)[None]  # each query's window
        widx = mx.where(wpos < 0, -1, wpos - base)[None].astype(mx.int32)
        cache.window = raw[:, -win:]
        if shared.capture:
            cache.verify_kv = (start, kv)

        if self.ratio:
            T, cidx = self._compressed(x, qr, start, cache, shared)
            if T:
                raw, widx = append_compressed_rows(raw, widx, cidx, shared.compress_kv, T, shared,
                                                   gather=S <= GATHER_MAX_QUERIES)
        return sparse_attention(q, raw, self.attn_sink, widx, self.scale)

    def _compressed(self, x, qr, start_pos, cache: LayerCache, shared: Shared):
        B, S, _ = x.shape
        r = self.ratio
        T = (start_pos + S) // r
        latent, g0 = None, 0
        if self.is_kv_source:
            latent, g0 = self.compressor(x, start_pos, cache, capture=shared.capture)
            shared.compress_kv = cache.compress_kv
        if self.is_index_source:
            if T == 0:
                idxs = mx.zeros((B, S, 0), mx.int32)
            else:
                idxs = self.indexer(x, qr, latent, g0, start_pos, self._rope, cache, shared)
            shared.topk_idxs = idxs
        else:
            idxs = shared.topk_idxs
        if latent is not None:
            # a latent stands for the first token of its group
            cache.compress_kv.append(self._rope(latent, (g0, r)))
        # T compressed rows are visible to this segment (read from shared.compress_kv by the caller)
        return T, idxs


class SwitchProj(nn.Module):
    def __init__(self, n_experts, in_dim, out_dim, bits):
        super().__init__()
        self.bits = bits
        self.weight = mx.zeros((n_experts, out_dim, in_dim * bits // 32), mx.uint32)
        self.scales = mx.zeros((n_experts, out_dim, in_dim // GROUP_SIZE), mx.bfloat16)
        self.biases = mx.zeros((n_experts, out_dim, in_dim // GROUP_SIZE), mx.bfloat16)

    def __call__(self, x, idx, sorted_indices):
        return mx.gather_qmm(x, self.weight, self.scales, self.biases, rhs_indices=idx, transpose=True,
                             group_size=GROUP_SIZE, bits=self.bits, sorted_indices=sorted_indices)


class SwitchMLP(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        d, f, E = args.hidden_size, args.moe_intermediate_size, args.n_routed_experts
        pfx = f"layers.{layer_id}.ffn.experts"
        self.gate_proj = SwitchProj(E, d, f, qbits[f"{pfx}.gate_proj"])
        self.up_proj = SwitchProj(E, d, f, qbits[f"{pfx}.up_proj"])
        self.down_proj = SwitchProj(E, f, d, qbits[f"{pfx}.down_proj"])


class SharedExpert(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        d, f = args.hidden_size, args.moe_intermediate_size
        pfx = f"layers.{layer_id}.ffn.shared_experts"
        self.w1 = qlinear(d, f, qbits[f"{pfx}.w1"])
        self.w2 = qlinear(f, d, qbits[f"{pfx}.w2"])
        self.w3 = qlinear(d, f, qbits[f"{pfx}.w3"])


class Gate(nn.Module):
    def __init__(self, args: Args):
        super().__init__()
        E = args.n_routed_experts
        self.weight = mx.zeros((E, args.hidden_size), mx.bfloat16)
        self.bias = mx.zeros((E,), mx.float32)
        self.bias_vl = mx.zeros((E,), mx.float32)


def jangt_fused_moe(xf, idx, w, experts, shared, limit):
    """vMLX's fused JANGT routed + mxfp8 shared-expert decode (4 dispatches instead of ~14, every row bit-identical to
    the stock path run alone; self-checked on first use), or None (not importable / not eligible / disabled)."""
    if not FUSED_MOE or xf.shape[0] > GATHER_MAX_QUERIES:
        return None
    try:
        from vmlx_engine.jangt.fused_moe import jt_moe_decode
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    return jt_moe_decode(xf, idx, w, experts, shared, limit)


def jangt_experts(qbits, prefix: str, D: int, I: int, E: int):
    """Routed experts stored JANGT (JANGHT bundles): Hadamard-128 + signs rotation, trellis L=12 at K bits per weight,
    outlier input columns, down column scales folded into its input rotation. Run through the vMLX JANGT kernels
    (vmlx_engine.jangt.switch.JTMixedSwitchGLU): rotate-once fused gate+up+SwiGLU(+clamp), rotated weighted down,
    expert-sorted NAX tiles for prefill. Parameter names match the bundle (<prefix>.<proj>.jt_*)."""
    from vmlx_engine.jangt.switch import JTMixedSwitchGLU

    gu, dn = qbits[f"{prefix}.gate_proj"], qbits[f"{prefix}.down_proj"]
    if qbits[f"{prefix}.up_proj"] != gu:
        raise ValueError(f"jangt: gate/up formats differ for {prefix}")
    n_out = qbits.jangt.get("outlier_columns", {})
    return JTMixedSwitchGLU(D, I, E, gu, dn, int(n_out.get(f"{prefix}.gate_proj", 0)),
                            int(n_out.get(f"{prefix}.down_proj", 0)), qbits.swiglu_limit)


class FFN(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        self.gate = Gate(args)
        pfx = f"layers.{layer_id}.ffn.experts"
        if isinstance(qbits[f"{pfx}.gate_proj"], dict):
            self.experts = jangt_experts(qbits, pfx, args.hidden_size, args.moe_intermediate_size, args.n_routed_experts)
        else:
            self.experts = SwitchMLP(args, layer_id, qbits)
        self.shared_experts = SharedExpert(args, layer_id, qbits)


class Engram(nn.Module):
    def __init__(self, args: Args, layer_id: int, layout: EngramLayout, qbits: dict):
        super().__init__()
        self.hash_index = layout.layer_ids.index(layer_id)
        self.dim, self.hc = args.hidden_size, args.hc_mult
        n_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = qlinear(n_cols * layout.head_dim, args.hidden_size * (args.hc_mult + 1), qbits[f"layers.{layer_id}.engram.wkv"])
        self.q_weight = mx.ones((args.hc_mult, args.hidden_size))
        self.k_weight = mx.ones((args.hc_mult, args.hidden_size))
        self.eps = args.rms_norm_eps

    def __call__(self, h, rows, text_mask=None):
        """h [1,S,hc,D]; rows [1,S,n_cols,head_dim] bf16 fetched from disk.
        text_mask [1,S] (1.0 text, 0.0 image) zeros the engram contribution on image tokens."""
        kv = self.wkv(rows.reshape(*rows.shape[:2], -1))
        key = kv[..., : self.hc * self.dim].astype(mx.float32).reshape(*kv.shape[:2], self.hc, self.dim)
        value = kv[..., self.hc * self.dim :].astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        hf = h.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(hf * hf, -1) + self.eps) * mx.rsqrt(mx.mean(key * key, -1) + self.eps)
        dot = mx.sum(hf * weight * key, -1) * rstd * self.dim ** -0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        if text_mask is not None:
            gate = gate * text_mask[..., None]  # image tokens contribute no n-gram
        return (hf + gate[..., None] * value[:, :, None, :]).astype(h.dtype)


@mx.compile
def _hc_pre(x, pre):
    return mx.sum(pre[..., None] * x.astype(mx.float32), axis=2).astype(x.dtype)


@mx.compile
def _hc_post(x, residual, post, comb):
    y = post[..., None] * x[:, :, None, :].astype(mx.float32)
    y = y + mx.einsum("bsjk,bsjd->bskd", comb, residual.astype(mx.float32))
    return y.astype(x.dtype)


_HC_KERNELS = {}

_HC_SRC = """
    uint t = thread_position_in_grid.x;
    if (t >= N) return;
    const device float* m = mixes + t * 24;
    float s0 = scale[0], s1 = scale[1], s2 = scale[2];
    for (int j = 0; j < 4; j++) {
        pre[t * 4 + j] = 1.0f / (1.0f + metal::precise::exp(-(m[j] * s0 + base[j]))) + EPS;
        post[t * 4 + j] = 2.0f / (1.0f + metal::precise::exp(-(m[4 + j] * s1 + base[4 + j])));
    }
    float c[16];
    for (int i = 0; i < 16; i++) c[i] = m[8 + i] * s2 + base[8 + i];
    for (int r = 0; r < 4; r++) {
        float mx_ = c[r * 4];
        for (int k = 1; k < 4; k++) mx_ = metal::max(mx_, c[r * 4 + k]);
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) { c[r * 4 + k] = metal::precise::exp(c[r * 4 + k] - mx_); sum += c[r * 4 + k]; }
        for (int k = 0; k < 4; k++) c[r * 4 + k] = c[r * 4 + k] / sum + EPS;
    }
    for (int k = 0; k < 4; k++) {
        float cs = c[k] + c[4 + k] + c[8 + k] + c[12 + k];
        for (int r = 0; r < 4; r++) c[r * 4 + k] = c[r * 4 + k] / (cs + EPS);
    }
    for (int it = 0; it < ITERS - 1; it++) {
        for (int r = 0; r < 4; r++) {
            float rs = c[r * 4] + c[r * 4 + 1] + c[r * 4 + 2] + c[r * 4 + 3];
            for (int k = 0; k < 4; k++) c[r * 4 + k] = c[r * 4 + k] / (rs + EPS);
        }
        for (int k = 0; k < 4; k++) {
            float cs = c[k] + c[4 + k] + c[8 + k] + c[12 + k];
            for (int r = 0; r < 4; r++) c[r * 4 + k] = c[r * 4 + k] / (cs + EPS);
        }
    }
    for (int i = 0; i < 16; i++) comb[t * 16 + i] = c[i];
"""


def _hc_split_metal(m, scale, base, iters: int, eps: float):
    """pre/post/comb + Sinkhorn for hc=4 in one thread per token (release kernel order)."""
    key = (iters, eps)
    if key not in _HC_KERNELS:
        src = _HC_SRC.replace("ITERS", str(int(iters))).replace("EPS", f"{float(eps):.9e}f")
        _HC_KERNELS[key] = mx.fast.metal_kernel(
            name=f"dsv41_hc_split_{iters}", input_names=["mixes", "scale", "base", "N_arr"],
            output_names=["pre", "post", "comb"], source=src.replace("N)", "N_arr[0])"))
    lead = m.shape[:-1]
    flat = m.reshape(-1, m.shape[-1]).astype(mx.float32)
    n = flat.shape[0]
    pre, post, comb = _HC_KERNELS[key](
        inputs=[flat, scale.astype(mx.float32), base.astype(mx.float32), mx.array([n], mx.uint32)],
        output_shapes=[(n, 4), (n, 4), (n, 16)], output_dtypes=[mx.float32] * 3,
        grid=(n, 1, 1), threadgroup=(min(n, 256), 1, 1))
    return pre.reshape(*lead, 4), post.reshape(*lead, 4), comb.reshape(*lead, 4, 4)


_HC_SPLIT = {}


def _hc_split(hc: int, iters: int, eps: float):
    """Compiled pre/post/comb split + Sinkhorn (thousands of tiny ops per token otherwise)."""
    key = (hc, iters, eps)
    if key not in _HC_SPLIT:
        def split(m, scale, base):
            pre = mx.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
            post = 2 * mx.sigmoid(m[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
            comb = (m[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(*m.shape[:-1], hc, hc)
            comb = mx.softmax(comb, axis=-1) + eps
            comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
            for _ in range(iters - 1):
                comb = comb / (mx.sum(comb, axis=-1, keepdims=True) + eps)
                comb = comb / (mx.sum(comb, axis=-2, keepdims=True) + eps)
            return pre, post, comb

        _HC_SPLIT[key] = mx.compile(split)
    return _HC_SPLIT[key]


_HC_FUSED_OK: dict = {}


def _hc_fused_ok(blk) -> bool:
    """Once per (D, eps) per process: the fused hc stage / expand reproduce the stock ops bit for bit
    on a random stream with this block's weights (guards against another MLX build). Only where f32
    GEMMs round to TF32 (M5 matrix units, MLX default): elsewhere the expand's 4-term sum follows another
    order, a random-stream check can pass by chance and real decode then diverges, so stock ops."""
    from jang_tools.dsv41.metal_decode import _tf32
    D = blk.hc_attn_fn.shape[1] // blk.hc
    tf32 = _tf32()
    key = (D, blk.eps, blk.attn_norm.eps, blk.hc_eps, blk.iters, tf32)
    ok = _HC_FUSED_OK.get(key)
    if ok is None and not tf32:
        ok = _HC_FUSED_OK[key] = False
    if ok is None:
        ok = False
        try:
            from jang_tools.dsv41.metal_decode import hc_post_decode, hc_stage_decode
            k = mx.random.key(7)
            x = (mx.random.normal((1, 2, blk.hc, D), key=k) * 3).astype(mx.bfloat16)
            pre = mx.softmax(mx.random.normal((1, 2, blk.hc), key=mx.random.key(8)), axis=-1)
            h = (mx.random.normal((1, 2, D), key=mx.random.key(9)) * 2).astype(mx.bfloat16)
            st = hc_stage_decode(x[0], pre[0], blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base,
                                 blk.attn_norm.weight, blk.attn_norm.eps, blk.eps, blk.hc_eps, blk.iters)
            if st is not None:
                ok = True
                for i in range(2):  # stock, one row at a time
                    xi = x[:, i:i + 1]
                    ref = (blk.attn_norm(_hc_pre(xi, pre[:, i:i + 1])),
                           *blk.hc_mixes(xi, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base))
                    for a, b in zip(st, ref):
                        ok = ok and mx.array_equal(a[i], b.reshape(b.shape[2:])).item()
                    post = hc_post_decode(h[0, i:i + 1], x[0, i:i + 1], st[2][i:i + 1], st[3][i:i + 1])
                    ok = ok and post is not None and mx.array_equal(
                        post, _hc_post(h[:, i:i + 1], xi, st[2][None, i:i + 1], st[3][None, i:i + 1])[0]).item()
        except (RuntimeError, ValueError, TypeError):  # kernel build / launch failed: stock path
            ok = False
        _HC_FUSED_OK[key] = ok
    return ok


class Block(nn.Module):
    def __init__(self, args: Args, layer_id: int, layout: EngramLayout, qbits: dict):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(args, layer_id, qbits)
        self.ffn = FFN(args, layer_id, qbits)
        if layer_id in layout.layer_ids:
            self.engram = Engram(args, layer_id, layout, qbits)
        self.attn_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        hc = args.hc_mult
        mix = (2 + hc) * hc
        for p in ("attn", "ffn"):
            setattr(self, f"hc_{p}_fn", mx.zeros((mix, hc * args.hidden_size), mx.float32))
            setattr(self, f"hc_{p}_base", mx.zeros((mix,), mx.float32))
            setattr(self, f"hc_{p}_scale", mx.zeros((3,), mx.float32))
        self.hc, self.iters, self.hc_eps, self.eps = hc, args.hc_sinkhorn_iters, args.hc_eps, args.rms_norm_eps
        self.topk, self.route_scale = args.num_experts_per_tok, args.routed_scaling_factor
        self.norm_topk = args.norm_topk_prob
        self.swiglu_limit = args.swiglu_limit
        self.capture = None  # set to fn(layer_id, e_idx[Nk], gate_up_in[Nk,D], down_in[Nk,F]) to collect stats

    # hyper-connections -------------------------------------------------------
    def hc_mixes(self, x, fn, scale, base):
        B, S = x.shape[:2]
        xf = x.reshape(B, S, -1).astype(mx.float32)
        rs = mx.rsqrt(mx.mean(xf * xf, -1, keepdims=True) + self.eps)
        m = (xf @ fn.T) * rs  # [B,S,mix]
        if self.hc == 4:
            return _hc_split_metal(m, scale, base, self.iters, self.hc_eps)
        return _hc_split(self.hc, self.iters, self.hc_eps)(m, scale, base)

    @staticmethod
    def hc_pre(x, pre):
        return _hc_pre(x, pre)

    @staticmethod
    def hc_post(x, residual, post, comb):
        return _hc_post(x, residual, post, comb)

    # moe ----------------------------------------------------------------------
    def moe(self, x, image_mask=None):
        B, S, D = x.shape
        xf = x.reshape(-1, D)
        g = self.ffn.gate
        logits = None
        if ROUTER_GEMV and xf.shape[0] <= GATHER_MAX_QUERIES:
            from jang_tools.dsv41.metal_decode import router_logits
            logits = router_logits(xf, g.weight)
        if logits is None:
            logits = xf.astype(mx.float32) @ g.weight.astype(mx.float32).T
        scores = mx.sqrt(nn.softplus(logits))
        # VL routing: image tokens use the separately-trained bias_vl (release Gate.forward)
        bias = g.bias if image_mask is None else mx.where(image_mask.reshape(-1, 1), g.bias_vl, g.bias)
        idx = mx.argpartition(-(scores + bias), kth=self.topk - 1, axis=-1)[:, : self.topk]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk and self.topk > 1:
            w = w / (mx.sum(w, -1, keepdims=True) + 1e-20)
        w = w * self.route_scale

        sw = self.ffn.experts
        if getattr(sw, "is_jangt", False):
            if self.capture is None:
                y = jangt_fused_moe(xf, idx, w, sw, self.ffn.shared_experts, self.swiglu_limit)
                if y is not None:
                    return y.reshape(B, S, D)
            return self._shared(xf, sw.routed(xf, idx, w).astype(mx.float32)).astype(x.dtype).reshape(B, S, D)
        n = xf.shape[0]
        if FUSED_MOE and n <= GATHER_MAX_QUERIES and self.capture is None:
            from jang_tools.dsv41.metal_decode import moe_decode
            y = moe_decode(xf, idx, w, sw, self.ffn.shared_experts, self.swiglu_limit)
            if y is not None:
                return y.reshape(B, S, D)
        do_sort = n * self.topk >= 64
        xe = xf[:, None, None, :]  # [N,1,1,D]
        if do_sort:
            flat = idx.reshape(-1)
            order = mx.argsort(flat)
            inv = mx.argsort(order)
            xe = xe.reshape(n, D)[order // self.topk][:, None, :]  # [N*k,1,D]
            e_idx = flat[order]
        else:
            e_idx = idx
        gate = sw.gate_proj(xe, e_idx, do_sort).astype(mx.float32)
        up = sw.up_proj(xe, e_idx, do_sort).astype(mx.float32)
        lim = self.swiglu_limit
        if lim > 0:
            up = mx.clip(up, -lim, lim)
            gate = mx.minimum(gate, lim)
        h = nn.silu(gate) * up
        if do_sort:
            h = h * w.reshape(-1)[order][:, None, None]
        else:
            h = h * w[:, :, None, None]
        if self.capture is not None and do_sort:  # calibration: per-expert projection inputs
            self.capture(self.layer_id, e_idx, xe.reshape(-1, D), h.reshape(-1, h.shape[-1]))
        y = sw.down_proj(h.astype(mx.bfloat16), e_idx, do_sort).astype(mx.float32)
        if do_sort:
            y = y[inv].reshape(n, self.topk, D)
        else:
            y = y.reshape(n, self.topk, D)
        y = mx.sum(y, axis=1)
        return self._shared(xf, y).astype(x.dtype).reshape(B, S, D)

    def _shared(self, xf, y):
        """y (fp32 routed sum) + the shared expert of xf, same clamp."""
        se, lim = self.ffn.shared_experts, self.swiglu_limit
        sg = se.w1(xf).astype(mx.float32)
        su = se.w3(xf).astype(mx.float32)
        if lim > 0:
            su = mx.clip(su, -lim, lim)
            sg = mx.minimum(sg, lim)
        return y + se.w2((nn.silu(sg) * su).astype(mx.bfloat16)).astype(mx.float32)

    def _stage(self, xs, pre, p):
        """Fused hc_mixes + hc_pre + RMSNorm for sublayer p ("attn" / "ffn"), xs [S, 4, D]."""
        from jang_tools.dsv41.metal_decode import hc_stage_decode
        norm = getattr(self, f"{p}_norm")
        return hc_stage_decode(xs, pre, getattr(self, f"hc_{p}_fn"), getattr(self, f"hc_{p}_scale"),
                               getattr(self, f"hc_{p}_base"), norm.weight, norm.eps, self.eps, self.hc_eps, self.iters)

    @staticmethod
    def _post(h, xs, post, comb):
        from jang_tools.dsv41.metal_decode import hc_post_decode
        out = hc_post_decode(h, xs, post, comb)
        return out if out is not None else _hc_post(h[None], xs[None], post[None], comb[None])[0]

    def _call_fused_hc(self, x, segs, pre_mix, image_mask):
        S = x.shape[1]
        xs = x[0]
        st = self._stage(xs, pre_mix.reshape(S, self.hc), "attn")
        if st is None:
            return None
        xin, a_pre, a_post, a_comb = st
        xs = self._post(self.attn(xin[None], segs)[0], xs, a_post, a_comb)
        st = self._stage(xs, a_pre, "ffn")
        xin, f_pre, f_post, f_comb = st
        xs = self._post(self.moe(xin[None], image_mask)[0], xs, f_post, f_comb)
        return xs[None], f_pre[None]

    def __call__(self, x, segs, pre_mix, image_mask=None):
        if (FUSED_HC and x.shape[0] == 1 and x.shape[1] <= GATHER_MAX_QUERIES and self.hc == 4
                and _hc_fused_ok(self)):
            out = self._call_fused_hc(x, segs, pre_mix, image_mask)
            if out is not None:
                return out
        residual = x
        a_pre, a_post, a_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), segs)
        x = self.hc_post(h, residual, a_post, a_comb)
        residual = x
        f_pre, f_post, f_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.moe(self.ffn_norm(self.hc_pre(x, a_pre)), image_mask)
        x = self.hc_post(h, residual, f_post, f_comb)
        return x, f_pre


class Model(nn.Module):
    def __init__(self, args: Args, qbits: dict, layout: EngramLayout, vision_args=None):
        super().__init__()
        self.args = args
        self.embed = nn.QuantizedEmbedding(args.vocab_size, args.hidden_size, group_size=GROUP_SIZE, bits=qbits["embed"])
        self.layers = [Block(args, i, layout, qbits) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.head = qlinear(args.hidden_size, args.vocab_size, qbits["head"])
        self.layout = layout
        self.vision_args = vision_args
        # dSpark/MTP reads the mean-over-hc hidden at the INPUT of these layers (release Transformer.forward)
        self.target_layer_ids = tuple(args.raw.get("dspark_target_layer_ids", ()))
        # First layer prompt-suffix pruning may shorten: past the last KV-source (its compressor and index
        # keys need every row) and past every Engram layer. Layers from here on keep only a window cache.
        self.prune_from = max([*args.kv_source_layer_ids, *layout.layer_ids, args.candidate_source_layer_id]) + 1
        if vision_args is not None:
            from jang_tools.dsv41.vision import ViT, Aligner, IMAGE, IMAGE_START, IMAGE_END, IMAGE_NEW_LINE
            self.vision = ViT(vision_args)
            self.aligner = Aligner(vision_args)
            self.image_start = mx.zeros((args.hidden_size,), mx.bfloat16)
            self.image_end = mx.zeros((args.hidden_size,), mx.bfloat16)
            self.image_newline = mx.zeros((args.hidden_size,), mx.bfloat16)
            self._img_types = (IMAGE, IMAGE_START, IMAGE_END, IMAGE_NEW_LINE)

    def encode_image(self, patches, n_vit_h, n_vit_w):
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, h, images):
        """Overwrite each image's token span in h[0] with ViT/aligner features + learned delimiters."""
        IMAGE, IMAGE_START, IMAGE_END, IMAGE_NEW_LINE = self._img_types
        for img in images:
            types = np.asarray(img.types)
            s = img.start
            feats = self.encode_image(img.patches, img.n_vit_h, img.n_vit_w).astype(h.dtype)  # [n_image_slots, D]
            fi = 0
            rows = []
            for t in types:
                if t == IMAGE:
                    rows.append(feats[fi]); fi += 1
                elif t == IMAGE_START:
                    rows.append(self.image_start.astype(h.dtype))
                elif t == IMAGE_END:
                    rows.append(self.image_end.astype(h.dtype))
                else:  # IMAGE_NEW_LINE
                    rows.append(self.image_newline.astype(h.dtype))
            span = mx.stack(rows, axis=0)  # [len(types), D]
            h[0, s : s + len(types)] = span
        return h

    def prune_rows(self, layer_id: int, seg_start: int, n: int, prompt_end: int) -> int:
        """Trailing rows of an n-row forward starting at seg_start that layer ``layer_id`` must run so the
        logits at ``prompt_end - 1`` and every layer's window cache at ``prompt_end`` are exact.

        A pruned layer's 128-token window reaches 127 rows back, so its first 127 kept rows see a partial
        window and only its last keep - 127 outputs are exact; the layer above must not read further
        back than that. The last layer needs its window (128 rows), hence keep = 1 + (L - i) * 127 for L
        layers (layer 39: 128, layer 21: 2414). At least one row always runs, so a forward that lies
        entirely before the kept range still leaves a window row at the right position."""
        win = self.args.sliding_window
        keep = 1 + (len(self.layers) - layer_id) * (win - 1)
        lo = max(seg_start, prompt_end - keep)
        return max(1, min(n, seg_start + n - lo))

    def __call__(self, tokens: mx.array, segs: list, engram_rows: dict, all_logits: bool = False,
                 images=None, image_mask: mx.array | None = None, return_hidden: bool = False,
                 prompt_end: int | None = None):
        """tokens [1,N]: the segments' tokens back to back. engram_rows: {layer_id: [1,N,cols,head_dim]}.
        images: list of ImageInput to splice into the prefill (start_pos 0). image_mask [1,N] bool,
        True inside image spans. Returns fp32 logits at each segment's last token (or every token).
        With return_hidden, also returns main_hidden [1,N,dim*len(target_layer_ids)] (bf16): the
        mean-over-hc hidden at the INPUT of each dSpark target layer, concatenated -- the MTP head's
        input (release Transformer.forward collects h.mean(dim=2) at target_layer_ids).
        prompt_end: absolute position where the prompt ends (this forward ends at or before it). The
        layers from prune_from run only the trailing rows the last prompt token still depends on
        (prune_rows); the returned logits are exact only when this forward ends at prompt_end, and
        main_hidden only for the last 128 rows before prompt_end (rows before are zero)."""
        h = self.embed(tokens)
        if images:
            h = self.merge_image_embeddings(h, images)
        hc = self.args.hc_mult
        h = mx.repeat(h[:, :, None, :], hc, axis=2)
        pre_mix = mx.zeros((*tokens.shape, hc), mx.float32)
        pre_mix[..., 0] = 1.0
        text_mask = None if image_mask is None else (~image_mask).astype(mx.float32)
        main_hiddens = []
        N = tokens.shape[1]
        prune = (SUFFIX_PRUNE and prompt_end is not None and len(segs) == 1 and not all_logits and N > 1
                 and prompt_end >= segs[0].start + N)
        seg0, cur = segs[0], N
        per_layer_async = LAYER_ASYNC and tokens.shape[-1] <= LAYER_ASYNC_MAX_ROWS
        for i, layer in enumerate(self.layers):
            if prune and i >= self.prune_from:
                n = min(cur, self.prune_rows(i, seg0.start, N, prompt_end))
                if n < N:
                    # Rows before the kept suffix never reach this layer: its window restarts at the
                    # suffix (the stale rows would sit at the wrong positions; only rows whose outputs
                    # nothing reads see the shortened window).
                    seg0.caches[i].window = None
                if n < cur:
                    h, pre_mix = h[:, -n:], pre_mix[:, -n:]
                    if image_mask is not None:
                        image_mask = image_mask[:, -n:]
                    sh = seg0.shared
                    if sh.topk_idxs is not None:
                        sh.topk_idxs = sh.topk_idxs[:, -n:]
                    if sh.candidates is not None:
                        sh.candidates = sh.candidates[-n:]
                    seg = Seg(0, n, seg0.start + N - n, seg0.caches)
                    seg.shared = sh
                    segs, cur = [seg], n
            if i in engram_rows:
                rows = engram_rows[i]
                if callable(rows):  # still being read: start the layers so far on the GPU meanwhile
                    mx.async_eval(h, pre_mix)
                    rows = rows()
                h = layer.engram(h, rows, text_mask)
            if return_hidden and i in self.target_layer_ids:
                mh = mx.mean(h, axis=2)  # [1,cur,dim] at this layer's input
                if cur < N:  # pruned: rows before the suffix are not computed (zeros, never read)
                    mh = mx.pad(mh, [(0, 0), (N - cur, 0), (0, 0)])
                main_hiddens.append(mh)
            h, pre_mix = layer(h, segs, pre_mix, image_mask)
            if per_layer_async:
                mx.async_eval(h, pre_mix)
        if not all_logits:
            last = mx.array([sg.lo + sg.n - 1 for sg in segs])
            h, pre_mix = h[:, last], pre_mix[:, last]
        h = Block.hc_pre(h, pre_mix)
        # fp32 head input -> fp32 logits, as the reference (F.linear(x.float(), weight)). In bf16 the
        # top-2 logits tie exactly every ~50-100 tokens, and a batched verify forward breaks those
        # ties differently from single-token decode, so DSpark drifted from greedy AR.
        logits = self.head(self.norm(h).astype(mx.float32))[0]
        if return_hidden:
            main_hidden = mx.concatenate(main_hiddens, axis=-1) if main_hiddens else None
            return logits, main_hidden
        return logits


# ----------------------------------------------------------------------------- loading

def _infer_bits(bundle: Path) -> dict[str, int]:
    from jang_tools.dsv41.st_io import read_header

    index = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
    shapes = {}
    for fn in sorted(set(index.values())):
        header, _ = read_header(bundle / fn)
        for k, v in header.items():
            if k != "__metadata__":
                shapes[k] = v["shape"]
    bits = {}
    for k, shp in shapes.items():
        if k.endswith(".scales"):
            base = k[: -len(".scales")]
            wshape = shapes[base + ".weight"]
            bits[base] = wshape[-1] * 32 // (shp[-1] * GROUP_SIZE)
    return bits


class BundleSpecs(dict):
    """Per-module formats read from config.json "quantization" (JANGHT bundles), in the qbits shape the modules take:
    affine -> bit width (group GROUP_SIZE), mxfp8 -> "mxfp8", jangt -> {"mode": "jangt", "kbits": K}. A module the
    config does not list raises (fail closed, no guessed format). Also carries the bundle's "jangt" block."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.q = cfg["quantization"]
        self.jangt = cfg.get("jangt") or {}
        if self.jangt and (int(self.jangt.get("version", 0)) != 1 or self.jangt.get("code") != "v2_halfbits"
                           or int(self.jangt.get("state_bits", 0)) != 12):
            raise ValueError(f"unsupported jangt block {self.jangt}")
        text = cfg.get("text_config", cfg)
        self.swiglu_limit = float(self.jangt.get("swiglu_limit", text.get("swiglu_limit", 0.0)) or 0.0)

    @staticmethod
    def wants(cfg: dict) -> bool:
        q = cfg.get("quantization")
        return isinstance(q, dict) and any(isinstance(v, dict) and v.get("mode") in ("mxfp8", "jangt") for v in q.values())

    def __missing__(self, key):
        e = self.q.get(key)
        if not isinstance(e, dict):
            raise KeyError(f"bundle config.json quantization has no entry for {key}")
        mode = e.get("mode", "affine")
        if mode == "mxfp8":
            v = "mxfp8"
        elif mode == "jangt":
            v = {"mode": "jangt", "kbits": e["kbits"]}
        elif mode == "affine" and int(e.get("group_size", GROUP_SIZE)) == GROUP_SIZE:
            v = int(e["bits"])
        else:
            raise ValueError(f"unsupported quantization {e} for {key}")
        self[key] = v
        return v


def quantize_listed_linears(model: nn.Module, cfg: dict, prefixes=("vision.", "aligner.")) -> int:
    """Quantize the plain Linears under `prefixes` that config.json "quantization" lists (JANGHT bundles store the
    vision tower / aligner Linears affine 8-bit); the rest of those modules stay as stored. Returns the count."""
    q = cfg.get("quantization") or {}
    listed = {k: v for k, v in q.items() if isinstance(v, dict) and k.startswith(prefixes)}
    if not listed:
        return 0
    n = [0]

    def pred(path, m):
        e = listed.get(path)
        if e is None or not isinstance(m, nn.Linear):
            return False
        n[0] += 1
        return {"group_size": int(e["group_size"]), "bits": int(e["bits"]), "mode": e.get("mode", "affine")}

    nn.quantize(model, class_predicate=pred)
    if n[0] != len(listed):
        raise RuntimeError(f"config lists {len(listed)} quantized vision/aligner Linears, the model has {n[0]}")
    return n[0]


class DSV41:
    """Loaded bundle + tokenizer + Engram hasher/tables, with a batch-1 session."""

    def __init__(self, bundle: str | Path, engram_dir: str | Path | None = None, wired: bool = True):
        from tokenizers import Tokenizer

        bundle = Path(bundle).expanduser()
        cfg = json.loads((bundle / "config.json").read_text())
        text = cfg["text_config"]
        self.args = Args.from_config(text)
        self.eos_id = cfg.get("eos_token_id", 1)
        layout = EngramLayout.from_config(text)
        # JANGHT bundles declare every module's format (mxfp8 / affine / jangt); affine bundles are read from shapes
        qbits = BundleSpecs(cfg) if BundleSpecs.wants(cfg) else _infer_bits(bundle)
        self.cfg = cfg
        # vision tower (VL): weights live in the bundle as BF16 passthrough; enabled when present
        self.vision_args = None
        if "vision_config" in cfg and cfg["vision_config"].get("num_hidden_layers", 0) > 0:
            from jang_tools.dsv41.vision import VisionArgs
            self.vision_args = VisionArgs.from_config(cfg)
        self.model = Model(self.args, qbits, layout, self.vision_args)
        if self.vision_args is not None and isinstance(qbits, BundleSpecs):
            quantize_listed_linears(self.model, cfg)

        index = json.loads((bundle / "model.safetensors.index.json").read_text())["weight_map"]
        skip = () if self.vision_args is not None else ("vision.", "aligner.", "image_")
        if wired:
            info = mx.device_info()
            mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
        from mlx.utils import tree_flatten

        shapes = {k: v.shape for k, v in tree_flatten(self.model.parameters())}
        for fn in sorted(set(index.values())):
            w = mx.load(str(bundle / fn))
            w = [(k, v) for k, v in w.items() if not k.startswith(skip)]
            bad = [(k, v.shape, shapes.get(k)) for k, v in w if k in shapes and tuple(v.shape) != tuple(shapes[k])]
            if bad:
                raise RuntimeError(f"shape mismatch in {fn}: {bad[:3]}")
            self.model.load_weights(w, strict=False)
            mx.eval(self.model.parameters())
        self._check_loaded(index, skip)
        if isinstance(qbits, BundleSpecs):
            self._fp32_compute_weights()

        self.tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        self.hasher = NgramHasher(text, layout, bundle / "tokenizer.json")
        edir = Path(engram_dir).expanduser() if engram_dir else bundle / "engram"
        self.tables = open_engram_tables(edir, layout.layer_ids)
        self.layout = layout
        self.reset()

    def _fp32_compute_weights(self):
        """fp32 copies, once, of the pooled compressors' wkv / wgate (only ever called with compute=fp32).
        Value-identical; removes a cast kernel and its fp32 write per layer per forward. (The router gate stays bf16:
        metal_decode.router_logits reads it directly with MLX's f32 GEMV order.)"""
        for blk in self.model.layers:
            comp = getattr(blk.attn, "compressor", None)
            if comp is not None and comp.ratio > 1:
                comp.wkv.weight = comp.wkv.weight.astype(mx.float32)
                comp.wgate.weight = comp.wgate.weight.astype(mx.float32)
        mx.eval(self.model.parameters())

    def _check_loaded(self, index, skip):
        from mlx.utils import tree_flatten

        params = dict(tree_flatten(self.model.parameters()))
        expected = {k for k in index if not k.startswith(skip)}
        missing = expected - params.keys()
        unset = params.keys() - expected
        if missing or unset:
            raise RuntimeError(f"weight mismatch: {len(missing)} unused bundle tensors, {len(unset)} unloaded params "
                               f"(e.g. {sorted(missing)[:3]} / {sorted(unset)[:3]})")

    def reset(self):
        self.session = self.new_session()

    # sessions ---------------------------------------------------------------
    def new_session(self) -> "Session":
        return Session(self.args.num_hidden_layers)

    @property
    def pos(self) -> int:
        return self.session.pos

    @property
    def caches(self):
        return self.session.caches

    def _engram_rows(self, ids: np.ndarray, start_pos: int, sess: "Session | None" = None,
                     keep: np.ndarray | None = None, lazy: bool = False) -> dict:
        """{layer_id: rows [1,S,cols,head_dim]}. With lazy, the reads run in the background (layer
        by layer, first layer first) and only the first is waited for here; later layers get a
        callable the forward resolves when it reaches them, after queueing the earlier layers on
        the GPU, so their cold SSD reads overlap compute (one forward, no extra weight passes).
        Rows already requested by a matching `engram_lookahead` are used instead of reading."""
        sess = sess or self.session
        ahead = getattr(sess, "engram_ahead", None)
        sess.engram_ahead = None
        if ahead is not None and keep is None and ahead[0] == start_pos and np.array_equal(ahead[1], ids):
            _, _, pending, sess.history = ahead
        else:
            hashes, sess.history = self.hasher.hash(ids, start_pos, sess.history, keep)  # [S, n_layers, cols]
            if not lazy:
                return {lid: self.tables[lid].lookup(hashes[:, j])[None] for j, lid in enumerate(self.layout.layer_ids)}
            pending = self._read_ahead(hashes)
        rows = {}
        for k, (lid, (t, f, shape)) in enumerate(zip(self.layout.layer_ids, pending)):
            if k == 0:
                rows[lid] = t.finish(shape, f.result())[None]
            else:
                rows[lid] = lambda t=t, f=f, shape=shape: t.finish(shape, f.result())[None]
        return rows

    def _read_ahead(self, hashes: np.ndarray) -> list:
        """Start background reads of every Engram layer's rows for hashes [S, n_layers, cols], one
        layer after another so the first layer's rows (needed first) get every I/O lane."""
        out, prev = [], None
        for j, lid in enumerate(self.layout.layer_ids):
            t = self.tables[lid]
            f = t.prefetch(hashes[:, j], after=prev)
            out.append((t, f, hashes[:, j].shape))
            prev = f
        return out

    def _start_lookahead(self, sess, ids: list[int]):
        """Request the rows for `ids`, the tokens the next forward on `sess` will append, while
        this forward runs on the GPU. The hashes need only token ids and the n-gram history;
        the history they produce is applied when that forward consumes the record."""
        arr = np.asarray(ids, np.int64)
        hashes, history = self.hasher.hash(arr, sess.pos, sess.history, None)
        sess.engram_ahead = (sess.pos, arr, self._read_ahead(hashes), history)

    def forward(self, ids: list[int], sess: "Session | None" = None, all_logits: bool = False,
                images=None, token_types=None, return_hidden: bool = False,
                engram_lookahead: list[int] | None = None, capture_verify: bool = False,
                prompt_end: int | None = None) -> mx.array:
        """Append tokens to a session (prefill, continuation or one decode step).
        images/token_types carry VL inputs (image spans must be in a single start_pos-0 prefill).
        engram_lookahead: the next chunk of a chunked prefill; its Engram rows are read while this
        forward computes (text tokens only).
        capture_verify: keep references to this forward's per-token cache inputs so commit_prefix
        can later keep only a prefix of these tokens (DSpark partial accept, no second forward).
        prompt_end: absolute position where the prompt being prefilled ends (Model.__call__): the late
        layers then skip prompt rows the end of the prompt no longer depends on. Pass it only for
        prefill chunks whose logits are discarded or that end exactly at prompt_end.
        Returns fp32 logits [vocab] at the last token ([len(ids), vocab] with all_logits).
        With return_hidden also returns main_hidden [len(ids), dim*n_targets] for the dSpark head."""
        sess = sess or self.session
        arr = np.asarray(ids, np.int64)
        image_mask = None
        if token_types is not None:
            tt = np.asarray(token_types)
            image_mask = mx.array((tt >= 0)[None])  # [1,N] bool, True inside image spans
        # image tokens take no part in an n-gram (release engram_mask = ~image_mask)
        engram_mask = None if token_types is None else (np.asarray(token_types) < 0)
        rows = self._engram_rows(arr, sess.pos, sess, engram_mask, lazy=len(arr) >= ENGRAM_LAZY_MIN)
        seg = Seg(0, len(ids), sess.pos, sess.caches)
        seg.shared.capture = capture_verify
        out = self.model(mx.array(arr[None].astype(np.int32)), [seg], rows, all_logits=all_logits,
                         images=images, image_mask=image_mask, return_hidden=return_hidden,
                         prompt_end=None if capture_verify else prompt_end)
        sess.pos += len(ids)
        if engram_lookahead:
            self._start_lookahead(sess, engram_lookahead)
        if return_hidden:
            logits, main_hidden = out
            mh = None if main_hidden is None else main_hidden[0]  # [N, dim*n_targets]
            return (logits if all_logits else logits[0]), mh
        return out if all_logits else out[0]

    def commit_prefix(self, sess, ck, keep: int):
        """Keep only the first ``keep`` tokens of the capture_verify forward that followed checkpoint
        ``ck`` -- the cache state a forward of exactly those tokens would have left, with no second
        forward. Per layer: restore, then re-append the kept tokens' window rows and replay the
        compressor on their captured projections (same groups, pending tail, pool/index rows and
        RoPE positions as in Attention._compressed). History and position are truncated."""
        history = sess.history                 # includes the verify's tokens (absolute positions)
        sess.restore(ck)
        start = sess.pos
        for layer, cache in zip(self.model.layers, sess.caches):
            attn = layer.attn
            cap = getattr(cache, "verify_kv", None)
            if cap is None:
                raise RuntimeError("commit_prefix needs a capture_verify forward after the checkpoint")
            v_start, kv = cap
            assert v_start == start, (v_start, start)
            prev = cache.window
            raw = kv[:, :keep] if prev is None else mx.concatenate([prev, kv[:, :keep]], axis=1)
            cache.window = raw[:, -attn.window:]
            if attn.ratio and attn.is_kv_source:
                r = attn.ratio
                latent, g0 = attn.compressor.commit(cache.verify_proj, keep, start, cache)
                if latent is not None:
                    pos = (g0, r)
                    if attn.is_index_source and attn.indexer.owns_k:
                        ix = attn.indexer
                        cache.index_k.append(attn._rope(ix.k_norm(ix.wk(latent)), pos))
                    cache.compress_kv.append(attn._rope(latent, pos))
            cache.verify_kv = None
            cache.verify_proj = None
        sess.history = history[: start + keep]
        sess.pos = start + keep

    def decode_batch(self, sessions: list["Session"], tokens: list[int]) -> mx.array:
        """One token for each of several sessions in a single forward. Returns logits [B, vocab]."""
        rows_parts = [self._engram_rows(np.asarray([t], np.int64), s.pos, s) for s, t in zip(sessions, tokens)]
        rows = {lid: mx.concatenate([r[lid] for r in rows_parts], axis=1) for lid in self.layout.layer_ids}
        segs = [Seg(i, 1, s.pos, s.caches) for i, s in enumerate(sessions)]
        logits = self.model(mx.array([tokens], mx.int32), segs, rows)
        for s in sessions:
            s.pos += 1
        return logits


class Session:
    """Per-sequence state: layer caches, Engram compressed-token history, position."""

    def __init__(self, n_layers: int):
        self.caches = [LayerCache() for _ in range(n_layers)]
        self.history = np.zeros(0, np.int64)
        self.pos = 0

    def checkpoint(self):
        """O(1) snapshot (refs only) for speculative verification. QuantGrowBuffer.append either
        grows a fresh buffer or writes past `.n` in place, so saving each buffer's snapshot() and the
        window/pend refs is enough: bytes below `.n` are never mutated and view only reads [:n]."""
        caches = []
        for c in self.caches:
            caches.append((
                c.window,
                c.compress_kv.snapshot(),
                c.index_k.snapshot(),
                c.pend_kv, c.pend_score,
            ))
        return (caches, self.history, self.pos)

    def restore(self, ck):
        caches, history, pos = ck
        for c, s in zip(self.caches, caches):
            c.window = s[0]
            c.compress_kv.load(s[1])
            c.index_k.load(s[2])
            c.pend_kv, c.pend_score = s[3], s[4]
        self.history = history
        self.pos = pos
