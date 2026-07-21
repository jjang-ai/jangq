"""AWQ activation capture for poolside Laguna (one weight-streaming pass).

Created by Jinho Jang (eric@jangq.ai) — 2026-07-21.

Streams the BF16 Laguna source layer-by-layer (never holds the model) over a
batch of equal-length calibration sequences and accumulates, per sparse MoE
layer, per-input-channel activation statistics of the post_attention_layernorm
output — the input the router + shared expert + routed experts all read.

Unlike the hy3 capture (hand-rolled torch forward), this one runs the REAL
`jang_tools.laguna.model.LagunaLayer` in MLX, so the capture math is the
verified runtime math by construction: per-layer head counts, dual RoPE with
partial rotary, softplus g_proj gating, sigmoid+bias top-k routing. At
--seq-len 512 (= sliding_window) the SWA band mask never bites, so a plain
causal mask is EXACT for every layer — do not raise seq-len past the window
without also wiring per-layer-type masks here.

Emits (fp32), hy3 key convention so `jang_tools.hy3.awq_search` runs as-is
(laguna shares the per-expert weight key layout):

    model.layers.{li}.mlp.input_max      per-channel max|x|      (hidden,)
    model.layers.{li}.mlp.input_absmean  per-channel mean|x|     (hidden,)
    model.layers.{li}.mlp.act_sample     real activation rows    (n_sample, hidden)

`input_scale` is NOT written here — scale selection is the measured step:

  python -m jang_tools.hy3.awq_search \
      --model ~/models/poolside/Laguna-S-2.1 \
      --stats ~/models/poolside/Laguna-S-2.1-awq-stats.safetensors \
      --out   ~/models/poolside/Laguna-S-2.1-awq-scales.safetensors \
      --bits 2 --group-size 64 --experts 4 --probe-layers 1,12,24,36,47

Usage:
  python -m jang_tools.laguna.awq_capture \
      --model ~/models/poolside/Laguna-S-2.1 \
      --calib ~/jang/kimi_v3_calib/corpus_v3.jsonl \
      --out   ~/models/poolside/Laguna-S-2.1-awq-stats.safetensors \
      --batch 8 --seq-len 512
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_unflatten

from .config import LagunaConfig
from .model import LagunaLayer

_PROJS = ("gate_proj", "up_proj", "down_proj")


class _ShardReader:
    """Whole-shard mx.load with a 2-entry LRU — consecutive layers share
    boundary shards, so each of the 46 shards is read from disk ~once."""

    def __init__(self, src: Path):
        self.src = src
        idx = json.loads((src / "model.safetensors.index.json").read_text())
        self.wm = idx["weight_map"]
        self._cache: dict[str, dict] = {}

    def get(self, key: str) -> mx.array:
        shard = self.wm[key]
        if shard not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
                mx.clear_cache()
            self._cache[shard] = mx.load(str(self.src / shard))
        return self._cache[shard][key]

    def layer_keys(self, li: int) -> list[str]:
        pre = f"model.layers.{li}."
        return [k for k in self.wm if k.startswith(pre)]


def _load_layer(reader: _ShardReader, cfg: LagunaConfig, li: int) -> LagunaLayer:
    pre = f"model.layers.{li}."
    layer = LagunaLayer(cfg, li)
    flat: dict[str, mx.array] = {}
    expert_stacks: dict[str, dict[int, mx.array]] = {}
    for k in reader.layer_keys(li):
        rel = k[len(pre):]
        if ".mlp.experts." in k and any(f".{p}.weight" in k for p in _PROJS):
            # experts.E.{proj}.weight -> stack later
            parts = rel.split(".")           # mlp experts E proj weight
            e, proj = int(parts[2]), parts[3]
            expert_stacks.setdefault(proj, {})[e] = reader.get(k)
            continue
        if rel == "mlp.experts.e_score_correction_bias":
            rel = "mlp.e_score_correction_bias"
        flat[rel] = reader.get(k)
    for proj, per_e in expert_stacks.items():
        n = max(per_e) + 1
        assert set(per_e) == set(range(n)), f"L{li} {proj}: sparse expert set"
        flat[f"mlp.switch_mlp.{proj}.weight"] = mx.stack(
            [per_e[e] for e in range(n)], axis=0)
    layer.update(tree_unflatten(list(flat.items())))
    mx.eval(layer.parameters())
    return layer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Laguna AWQ activation capture")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--n-sample", type=int, default=256,
                    help="real activation rows kept per layer for awq_search")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    src = args.model.expanduser()
    cfg = LagunaConfig.from_json(src / "config.json")
    if args.seq_len > cfg.sliding_window:
        raise SystemExit(
            f"--seq-len {args.seq_len} > sliding_window {cfg.sliding_window}: "
            "the plain causal mask used here would be WRONG for SWA layers")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)

    # Equal-length calibration batch: first --batch docs that tokenize to
    # >= seq_len, truncated. corpus_v3 is the token-balanced domain mix.
    rows: list[list[int]] = []
    with open(args.calib.expanduser()) as f:
        for line in f:
            text = json.loads(line).get("text", "")
            ids = tok.encode(text)
            if len(ids) >= args.seq_len:
                rows.append(ids[: args.seq_len])
            if len(rows) >= args.batch:
                break
    if len(rows) < args.batch:
        raise SystemExit(f"only {len(rows)} calib docs reached {args.seq_len} tokens")
    ids = mx.array(rows, dtype=mx.uint32)
    B, T = ids.shape
    print(f"[capture] batch {B}x{T} tokens, {cfg.num_hidden_layers} layers, "
          f"hidden {cfg.hidden_size}", flush=True)

    reader = _ShardReader(src)
    embed_w = reader.get("model.embed_tokens.weight")
    h = embed_w[ids.reshape(-1)].reshape(B, T, -1)   # bf16, source dtype
    mx.eval(h)

    causal = mx.triu(mx.full((T, T), -mx.inf, dtype=h.dtype), k=1)
    rng = np.random.default_rng(args.seed)
    out: dict[str, np.ndarray] = {}
    t0 = time.time()

    for li in range(cfg.num_hidden_layers):
        tl = time.time()
        layer = _load_layer(reader, cfg, li)
        attn = layer.self_attn(layer.input_layernorm(h), mask=causal, cache=None)
        h2 = h + attn
        xn = layer.post_attention_layernorm(h2)
        if not layer.is_dense:
            flat = np.array(xn.reshape(-1, xn.shape[-1]).astype(mx.float32))
            out[f"model.layers.{li}.mlp.input_max"] = \
                np.abs(flat).max(axis=0).astype(np.float32)
            out[f"model.layers.{li}.mlp.input_absmean"] = \
                np.abs(flat).mean(axis=0).astype(np.float32)
            pick = rng.choice(flat.shape[0], size=min(args.n_sample, flat.shape[0]),
                              replace=False)
            out[f"model.layers.{li}.mlp.act_sample"] = flat[pick].astype(np.float32)
        h = h2 + layer.mlp(xn)
        mx.eval(h)
        res = float(h.astype(mx.float32).abs().max())
        if not np.isfinite(res):
            raise SystemExit(f"L{li}: non-finite residual in bf16 source pass "
                             f"— capture aborted, investigate before AWQ")
        del layer
        gc.collect()
        mx.clear_cache()
        print(f"  L{li:2d} {'dense' if li == 0 else 'moe':5s} "
              f"resid_max={res:8.2f}  {time.time() - tl:5.1f}s", flush=True)

    from safetensors.numpy import save_file
    meta = {"batch": str(B), "seq_len": str(T), "calib": str(args.calib),
            "model": str(src), "n_sample": str(args.n_sample)}
    save_file(out, str(args.out.expanduser()), metadata=meta)
    n_layers = sum(1 for k in out if k.endswith(".input_max"))
    print(f"[capture] DONE — {n_layers} sparse layers, "
          f"{(time.time() - t0) / 60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
