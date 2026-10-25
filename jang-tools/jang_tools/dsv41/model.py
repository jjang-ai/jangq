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
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jang_tools.dsv41.engram import EngramLayout, NgramHasher, open_engram_tables

GROUP_SIZE = 64


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


class Rope:
    """Interleaved-pair rotary embedding on the last `dim` features, at arbitrary positions."""

    def __init__(self, dim, inv_freq: np.ndarray):
        self.dim = dim
        self.inv_freq = mx.array(inv_freq)

    def __call__(self, x: mx.array, positions: mx.array, inverse: bool = False) -> mx.array:
        """x [..., L, (heads,) D] where the L axis matches `positions`; rotates x[..., -dim:]."""
        return _rope_fn(self.dim, x.ndim, inverse)(x, positions, self.inv_freq)


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


def qlinear(in_dim, out_dim, bits):
    return nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=GROUP_SIZE, bits=bits)


class DenseLinear(nn.Module):
    """Source-precision weight kept as stored (bf16 / f32)."""

    def __init__(self, in_dim, out_dim, dtype=mx.bfloat16):
        super().__init__()
        self.weight = mx.zeros((out_dim, in_dim), dtype=dtype)

    def __call__(self, x, compute=None):
        w = self.weight if compute is None else self.weight.astype(compute)
        return (x if compute is None else x.astype(compute)) @ w.T


class GrowBuffer:
    """Append-only [1, T, D] buffer with amortized growth."""

    STEP = 2048

    def __init__(self):
        self.buf = None
        self.n = 0

    def append(self, x: mx.array):
        k = x.shape[1]
        if k == 0:
            return
        if self.buf is None or self.n + k > self.buf.shape[1]:
            cap = ((self.n + k + self.STEP - 1) // self.STEP) * self.STEP
            new = mx.zeros((x.shape[0], cap, x.shape[2]), dtype=x.dtype)
            if self.buf is not None:
                new[:, : self.n] = self.buf[:, : self.n]
            self.buf = new
        self.buf[:, self.n : self.n + k] = x
        self.n += k

    def view(self, n=None):
        n = self.n if n is None else n
        if self.buf is None:
            return None
        return self.buf[:, :n]


class LayerCache:
    def __init__(self):
        self.window = None  # [1, <=win, D] raw post-RoPE KV, oldest first
        self.compress_kv = GrowBuffer()
        self.index_k = GrowBuffer()
        self.pend_kv = None  # compressor: tokens of the not yet complete group (ratio > 1), fp32
        self.pend_score = None


class Shared:
    """What attention layers hand down the stack (release: SharedAttentionRuntime)."""

    def __init__(self):
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None


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


def topk_indices(scores, k):
    """Indices of the k largest along the last axis (unordered)."""
    if k >= scores.shape[-1]:
        return mx.broadcast_to(mx.arange(scores.shape[-1], dtype=mx.int32), scores.shape)
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

    def __call__(self, x, start_pos, cache: LayerCache):
        """Latents for the groups completed by this chunk, and the index of the first one.

        Group g pools tokens g*r .. g*r+r-1; a trailing partial group waits in the cache, so any
        chunking (single prefill, continuation, one token per step) yields the same groups."""
        r = self.ratio
        if r == 1:  # one token per group: a plain projection, no pooling
            return self.norm(self.wkv(x)), start_pos
        kv, score = self.wkv(x, mx.float32), self.wgate(x, mx.float32)
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
        return self.norm(out.astype(x.dtype)), g0


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
            cache.index_k.append(rope(k, (g0 + mx.arange(latent.shape[1])) * r))
        if self.owns_k:
            shared.index_k = cache.index_k
        T = end // r
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        # the release scores in bf16 end to end; near-tied positions follow that rounding
        q = rope(q, mx.arange(start_pos, end))[0].astype(mx.bfloat16)  # [S,h,d]
        index_k = shared.index_k.view(T)[0].astype(mx.bfloat16)  # [T,d]
        w = (self.weights_proj(x) * (self.head_dim ** -0.5 * self.n_heads ** -0.5))[0]  # [S,h] bf16
        k = min(self.topk, T)
        chunk = max(16, min(512, (1 << 26) // max(1, self.n_heads * T)))
        out_idx, out_cand = [], []
        for c in range(0, S, chunk):
            qi = q[c : c + chunk]
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
            else mx.arange(segs[0].start, segs[0].start + segs[0].n)
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
        o = o.reshape(B * N, self.n_groups, 1, -1)
        wa = self.wo_a
        w = wa.weight.reshape(self.n_groups, self.o_lora_rank, -1)
        sc = wa.scales.reshape(self.n_groups, self.o_lora_rank, -1)
        bi = wa.biases.reshape(self.n_groups, self.o_lora_rank, -1)
        o = mx.quantized_matmul(o, w, sc, bi, transpose=True, group_size=wa.group_size, bits=wa.bits)
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

        if self.ratio:
            comp, cidx = self._compressed(x, qr, start, cache, shared)
            if comp is not None:
                cidx = mx.where(cidx >= 0, cidx + raw.shape[1], -1)
                raw = mx.concatenate([raw, comp], axis=1)
                widx = mx.concatenate([widx, cidx], axis=-1)
        return sparse_attention(q, raw, self.attn_sink, widx, self.scale)

    def _compressed(self, x, qr, start_pos, cache: LayerCache, shared: Shared):
        B, S, _ = x.shape
        r = self.ratio
        T = (start_pos + S) // r
        latent, g0 = None, 0
        if self.is_kv_source:
            latent, g0 = self.compressor(x, start_pos, cache)
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
            cache.compress_kv.append(self._rope(latent, (g0 + mx.arange(latent.shape[1])) * r))
        if T == 0:
            return None, idxs
        return shared.compress_kv.view(T), idxs


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


class FFN(nn.Module):
    def __init__(self, args: Args, layer_id: int, qbits: dict):
        super().__init__()
        self.gate = Gate(args)
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


HC_METAL = True
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
        if self.hc == 4 and HC_METAL:
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
        scores = mx.sqrt(nn.softplus(xf.astype(mx.float32) @ g.weight.astype(mx.float32).T))
        # VL routing: image tokens use the separately-trained bias_vl (release Gate.forward)
        bias = g.bias if image_mask is None else mx.where(image_mask.reshape(-1, 1), g.bias_vl, g.bias)
        idx = mx.argpartition(-(scores + bias), kth=self.topk - 1, axis=-1)[:, : self.topk]
        w = mx.take_along_axis(scores, idx, axis=-1)
        if self.norm_topk and self.topk > 1:
            w = w / (mx.sum(w, -1, keepdims=True) + 1e-20)
        w = w * self.route_scale

        sw = self.ffn.experts
        n = xf.shape[0]
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

        se = self.ffn.shared_experts
        sg = se.w1(xf).astype(mx.float32)
        su = se.w3(xf).astype(mx.float32)
        if lim > 0:
            su = mx.clip(su, -lim, lim)
            sg = mx.minimum(sg, lim)
        y = y + se.w2((nn.silu(sg) * su).astype(mx.bfloat16)).astype(mx.float32)
        return y.astype(x.dtype).reshape(B, S, D)

    def __call__(self, x, segs, pre_mix, image_mask=None):
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

    def __call__(self, tokens: mx.array, segs: list, engram_rows: dict, all_logits: bool = False,
                 images=None, image_mask: mx.array | None = None, return_hidden: bool = False):
        """tokens [1,N]: the segments' tokens back to back. engram_rows: {layer_id: [1,N,cols,head_dim]}.
        images: list of ImageInput to splice into the prefill (start_pos 0). image_mask [1,N] bool,
        True inside image spans. Returns fp32 logits at each segment's last token (or every token).
        With return_hidden, also returns main_hidden [1,N,dim*len(target_layer_ids)] (bf16): the
        mean-over-hc hidden at the INPUT of each dSpark target layer, concatenated -- the MTP head's
        input (release Transformer.forward collects h.mean(dim=2) at target_layer_ids)."""
        h = self.embed(tokens)
        if images:
            h = self.merge_image_embeddings(h, images)
        hc = self.args.hc_mult
        h = mx.repeat(h[:, :, None, :], hc, axis=2)
        pre_mix = mx.zeros((*tokens.shape, hc), mx.float32)
        pre_mix[..., 0] = 1.0
        text_mask = None if image_mask is None else (~image_mask).astype(mx.float32)
        main_hiddens = []
        for i, layer in enumerate(self.layers):
            if i in engram_rows:
                h = layer.engram(h, engram_rows[i], text_mask)
            if return_hidden and i in self.target_layer_ids:
                main_hiddens.append(mx.mean(h, axis=2))  # [1,N,dim] at this layer's input
            h, pre_mix = layer(h, segs, pre_mix, image_mask)
        if not all_logits:
            last = mx.array([sg.lo + sg.n - 1 for sg in segs])
            h, pre_mix = h[:, last], pre_mix[:, last]
        h = Block.hc_pre(h, pre_mix)
        logits = self.head(self.norm(h)).astype(mx.float32)[0]
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
        qbits = _infer_bits(bundle)
        # vision tower (VL): weights live in the bundle as BF16 passthrough; enabled when present
        self.vision_args = None
        if "vision_config" in cfg and cfg["vision_config"].get("num_hidden_layers", 0) > 0:
            from jang_tools.dsv41.vision import VisionArgs
            self.vision_args = VisionArgs.from_config(cfg)
        self.model = Model(self.args, qbits, layout, self.vision_args)

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

        self.tokenizer = Tokenizer.from_file(str(bundle / "tokenizer.json"))
        self.hasher = NgramHasher(text, layout, bundle / "tokenizer.json")
        edir = Path(engram_dir).expanduser() if engram_dir else bundle / "engram"
        self.tables = open_engram_tables(edir, layout.layer_ids)
        self.layout = layout
        self.reset()

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
                     keep: np.ndarray | None = None) -> dict:
        sess = sess or self.session
        hashes, sess.history = self.hasher.hash(ids, start_pos, sess.history, keep)  # [S, n_layers, cols]
        return {lid: self.tables[lid].lookup(hashes[:, j])[None] for j, lid in enumerate(self.layout.layer_ids)}

    def forward(self, ids: list[int], sess: "Session | None" = None, all_logits: bool = False,
                images=None, token_types=None, return_hidden: bool = False) -> mx.array:
        """Append tokens to a session (prefill, continuation or one decode step).
        images/token_types carry VL inputs (image spans must be in a single start_pos-0 prefill).
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
        rows = self._engram_rows(arr, sess.pos, sess, engram_mask)
        seg = Seg(0, len(ids), sess.pos, sess.caches)
        out = self.model(mx.array(arr[None].astype(np.int32)), [seg], rows, all_logits=all_logits,
                         images=images, image_mask=image_mask, return_hidden=return_hidden)
        sess.pos += len(ids)
        if return_hidden:
            logits, main_hidden = out
            mh = None if main_hidden is None else main_hidden[0]  # [N, dim*n_targets]
            return (logits if all_logits else logits[0]), mh
        return out if all_logits else out[0]

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
        """O(1) snapshot (refs only) for speculative verification. GrowBuffer.append either grows a
        fresh buffer or writes past `.n` in place, so saving (buf, n) and restoring both is enough:
        the bytes below `.n` are never mutated, and the view only ever reads [:n]."""
        caches = []
        for c in self.caches:
            caches.append((
                c.window,
                c.compress_kv.buf, c.compress_kv.n,
                c.index_k.buf, c.index_k.n,
                c.pend_kv, c.pend_score,
            ))
        return (caches, self.history, self.pos)

    def restore(self, ck):
        caches, history, pos = ck
        for c, s in zip(self.caches, caches):
            c.window = s[0]
            c.compress_kv.buf, c.compress_kv.n = s[1], s[2]
            c.index_k.buf, c.index_k.n = s[3], s[4]
            c.pend_kv, c.pend_score = s[5], s[6]
        self.history = history
        self.pos = pos
