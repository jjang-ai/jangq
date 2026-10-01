"""Is the BF16 source an upcast of an FP8 grid? For each tensor test (a) FP8-e4m3 128x128 block grid (block amax/448
scale) and (b) MXFP8 grid (per-32 power-of-two scale): fraction of elements exactly representable."""
import os
import json, sys
from pathlib import Path
import mlx.core as mx
SRC = Path(os.environ["JANGH_SOURCE"])
H = json.loads(Path(os.environ["JANGH_HEADERS"]).read_text())

def mx8_frac(W):
    o, i = W.shape; G = W.reshape(o, i // 32, 32)
    a = mx.maximum(mx.abs(G).max(-1, keepdims=True), 1e-30)
    best = mx.zeros(G.shape[:2] + (1,))
    for de in (0, -1):
        sc = 2.0 ** (mx.ceil(mx.log2(a / 448.0)) + de)
        rt = mx.from_fp8(mx.to_fp8(mx.clip(G / sc, -448, 448)), dtype=mx.float32) * sc
        best = mx.maximum(best, mx.mean((rt == G).astype(mx.float32), -1, keepdims=True))
    return float(mx.mean(best))

def blk_frac(W):
    o, i = W.shape
    if o % 128 or i % 128: return float("nan")
    B = W.reshape(o // 128, 128, i // 128, 128)
    s = mx.maximum(mx.abs(B).max((1, 3), keepdims=True), 1e-30) / 448.0
    rt = mx.from_fp8(mx.to_fp8(mx.clip(B / s, -448, 448)), dtype=mx.float32) * s
    return float(mx.mean((mx.abs(rt - B) <= 1e-6 * mx.abs(B)).astype(mx.float32)))

names = ["model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight", "model.layers.1.self_attn.o_proj.weight",
         "model.layers.0.self_attn.indexer.wq.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.experts.7.gate_proj.weight",
         "model.layers.1.mlp.experts.7.down_proj.weight", "lm_head.weight", "model.embed_tokens.weight", "model.layers.20.self_attn.q_proj.weight"]
cache = {}
for n in names:
    f = H[n][0]
    if f not in cache: cache.clear(); cache[f] = mx.load(str(SRC / f))
    W = cache[f][n].astype(mx.float32)[:4096]
    print(f"{n:55s} mxfp8-exact {mx8_frac(W):.4f}  fp8-128blk-exact {blk_frac(W):.4f}", flush=True)
