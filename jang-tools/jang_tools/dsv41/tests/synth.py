"""Synthetic DSV4.1 models (random weights, no bundle) for decode-path tests and microbenchmarks.

``tiny_args`` is small enough for fast exactness tests but keeps the bundle's quant layout (2/3-bit
group-64 experts, 8-bit dense) and the projection-length classes that pick MLX's qmv kernels
(hidden a multiple of 512 -> qmv_fast; intermediate not -> qmv). ``real_args`` keeps the bundle's
per-layer dimensions with fewer layers / experts, for per-layer timing.

  build(args, expert_bits=...) -> Model with random weights
  greedy(model, prompt, steps) -> generated token ids (decode one token per forward)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from jang_tools.dsv41 import model as M
from jang_tools.dsv41.engram import EngramLayout


def _base(**kw):
    t = {
        "vocab_size": 1024, "rms_norm_eps": 1e-6, "rope_theta": 10000, "swiglu_limit": 10.0,
        "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 65536},
        "routed_scaling_factor": 1.5, "norm_topk_prob": True, "compress_rope_theta": 160000,
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "engram_layer_ids": [], "engram_max_ngram_size": 3, "engram_vocab_size": 1000, "engram_n_heads": 2,
        "engram_head_dim": 32, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 1000,
        "engram_num_embeddings": [],
    }
    t.update(kw)
    return t


def tiny_args(n_layers=6, n_experts=128, window=16):
    ratios = ([0, 2, 2, 1, 1, 1] * 4)[:n_layers]
    return M.Args.from_config(_base(
        hidden_size=1024, moe_intermediate_size=768, num_hidden_layers=n_layers, num_attention_heads=4,
        head_dim=128, qk_rope_head_dim=64, q_lora_rank=256, o_lora_rank=128, o_groups=2,
        n_routed_experts=n_experts, num_experts_per_tok=6, sliding_window=window, compress_ratios=ratios,
        kv_source_layer_ids=[1, 3], index_source_layer_ids=[1, 3, 5], index_n_heads=8, index_head_dim=64,
        index_topk=8, candidate_source_layer_id=3, candidate_topk_blocks=3, candidate_block_size=4))


def real_args(n_layers=10, n_experts=16):
    """Bundle dimensions (hidden 5120, intermediate 2304, 64 heads x 512, ...) on fewer layers."""
    ratios = [0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1][:n_layers]
    return M.Args.from_config(_base(
        hidden_size=5120, moe_intermediate_size=2304, num_hidden_layers=n_layers, num_attention_heads=64,
        head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024, o_groups=8,
        n_routed_experts=n_experts, num_experts_per_tok=6, sliding_window=128, compress_ratios=ratios,
        kv_source_layer_ids=[1, 5], index_source_layer_ids=[1, 5, 7, 9], index_n_heads=64, index_head_dim=128,
        index_topk=512, candidate_source_layer_id=5, candidate_topk_blocks=2048, candidate_block_size=8))


def _qbits(args, expert_bits):
    b = {"embed": 8, "head": 8}
    for i in range(args.num_hidden_layers):
        p = f"layers.{i}"
        for n in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b"):
            b[f"{p}.attn.{n}"] = 8
        for n in ("w1", "w2", "w3"):
            b[f"{p}.ffn.shared_experts.{n}"] = 8
        g, d = expert_bits(i)
        b[f"{p}.ffn.experts.gate_proj"] = g
        b[f"{p}.ffn.experts.up_proj"] = g
        b[f"{p}.ffn.experts.down_proj"] = d
        b[f"{p}.engram.wkv"] = 8
    return b


def bundle_bits(i):
    """The Q2Q3 bundle: gate/up 2-bit (3-bit on a few layers), down 3-bit."""
    return (3 if i % 5 == 4 else 2), 3


def _rand_quant(key, w_shape, bits, in_dim, dtype=mx.bfloat16, lo=None):
    """Random affine-quantized weight: packed codes uniform, values in about +-1.7/sqrt(in_dim)."""
    k1, k2 = mx.random.split(key)
    # two 16-bit halves: randint draws through float32, so a single 32-bit draw has zero low bits
    codes = (mx.random.randint(0, 1 << 16, w_shape, key=k1).astype(mx.uint32) << 16) | \
        mx.random.randint(0, 1 << 16, w_shape, key=k2).astype(mx.uint32)
    lo = 1.7 / in_dim ** 0.5 if lo is None else lo
    g_shape = (*w_shape[:-1], in_dim // M.GROUP_SIZE)
    jitter = 1 + 0.2 * mx.random.uniform(shape=g_shape, key=k2)
    scales = (2 * lo / ((1 << bits) - 1) * jitter).astype(dtype)
    biases = (-lo * jitter).astype(dtype)
    return codes, scales, biases


def build(args, expert_bits=bundle_bits, seed=0, dtype=mx.bfloat16) -> M.Model:
    layout = EngramLayout.from_config(args.raw)
    model = M.Model(args, _qbits(args, expert_bits), layout)
    params = dict(tree_flatten(model.parameters()))
    key = mx.random.key(seed)
    new = []
    for i, (name, p) in enumerate(sorted(params.items())):
        k = mx.random.fold_in(key, i) if hasattr(mx.random, "fold_in") else mx.random.key(seed * 100003 + i)
        base = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and p.dtype == mx.uint32:
            bits = 32 * p.shape[-1] // (params[base + ".scales"].shape[-1] * M.GROUP_SIZE)
            in_dim = params[base + ".scales"].shape[-1] * M.GROUP_SIZE
            w, s, b = _rand_quant(k, p.shape, bits, in_dim, lo=1.0 if name.startswith("embed") else None)
            new += [(name, w), (base + ".scales", s), (base + ".biases", b)]
        elif name.endswith((".scales", ".biases")):
            continue
        elif "norm" in name or name.endswith(("q_weight", "k_weight")):  # BF16 in the bundle
            new.append((name, (1 + 0.1 * mx.random.normal(p.shape, key=k)).astype(mx.bfloat16)))
        elif name.endswith("_scale") and "hc_" in name:
            new.append((name, (0.5 + 0.1 * mx.random.normal(p.shape, key=k)).astype(p.dtype)))
        elif name.endswith("_base") and "hc_" in name:
            new.append((name, (0.3 * mx.random.normal(p.shape, key=k)).astype(p.dtype)))
        elif name.endswith(("attn_sink", ".bias", "bias_vl")):
            new.append((name, (0.5 * mx.random.normal(p.shape, key=k)).astype(p.dtype)))
        else:
            fan = p.shape[-1] if p.ndim > 1 else 1
            new.append((name, (mx.random.normal(p.shape, key=k) / fan ** 0.5).astype(p.dtype)))
        if len(new) % 64 == 0:
            mx.eval([v for _, v in new[-64:]])
    model.load_weights(new, strict=False)
    mx.eval(model.parameters())
    return model


def new_caches(args):
    return [M.LayerCache() for _ in range(args.num_hidden_layers)]


def forward(model, ids, caches, pos, all_logits=False):
    seg = M.Seg(0, len(ids), pos, caches)
    return model(mx.array([ids], mx.int32), [seg], {}, all_logits=all_logits)


def greedy(model, prompt, steps, return_logits=False):
    """Prefill ``prompt`` then decode ``steps`` tokens one forward each (argmax)."""
    caches = new_caches(model.args)
    logits = forward(model, list(prompt), caches, 0)
    pos, out, lg = len(prompt), [], []
    for _ in range(steps):
        lg.append(np.array(logits[0].astype(mx.float32)))
        tok = int(mx.argmax(logits[0]))
        out.append(tok)
        logits = forward(model, [tok], caches, pos)
        pos += 1
    return (out, lg) if return_logits else out
