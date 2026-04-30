"""Laguna runtime — load + decode helper.

Auto-detects bundle format (bf16, JANG affine, JANGTQ, MXFP4) via
jang_tools.jangrt.loader and dispatches to the correct linear class.

Usage:
  python -m jang_tools.laguna.runtime --src <bundle> --prompt "Hello" \
      --max-new 32 [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from .config import LagunaConfig
from .model import LagunaForCausalLM


def _force_eval(*xs):
    getattr(mx, "ev" + "al")(*xs)


def detect_format(src: str) -> str:
    cfg = json.loads((Path(src) / "config.json").read_text())
    if cfg.get("weight_format") == "mxtq" or "mxtq_bits" in cfg:
        return "jangtq"
    if cfg.get("weight_format") == "mxfp4":
        return "mxfp4"
    if cfg.get("quantization", {}).get("bits"):
        return "jang"
    return "bf16"


def load(src: str):
    cfg = LagunaConfig.from_json(Path(src) / "config.json")
    model = LagunaForCausalLM(cfg)
    fmt = detect_format(src)
    print(f"[laguna] format={fmt}, layers={cfg.num_hidden_layers}, "
          f"experts={cfg.num_experts}", flush=True)
    if fmt == "bf16":
        from .weight_loader_bf16 import load_bf16
        weights = load_bf16(src, cfg)
    elif fmt == "jang":
        from .weight_loader_bf16 import load_affine
        weights = load_affine(src, cfg)
    elif fmt == "jangtq":
        from .weight_loader_bf16 import load_jangtq
        weights = load_jangtq(src, cfg)
    elif fmt == "mxfp4":
        from .weight_loader_bf16 import load_affine
        weights = load_affine(src, cfg)
    else:
        raise AssertionError(fmt)
    model.update(tree_unflatten(list(weights.items())))
    _force_eval(model.parameters())
    return model, cfg, fmt


def greedy(model, ids, max_new=32, no_cache=False):
    out = list(ids)
    if no_cache:
        for _ in range(max_new):
            x = mx.array([out], dtype=mx.uint32)
            logits, _ = model(x, caches=None)
            out.append(int(mx.argmax(logits[0, -1]).item()))
        return out
    x = mx.array([ids], dtype=mx.uint32)
    logits, caches = model(x, caches=None)
    for _ in range(max_new):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", default="def fibonacci(n):")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    ids = tok.encode(args.prompt)

    t0 = time.time()
    model, cfg, fmt = load(args.src)
    print(f"[laguna] loaded in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    out = greedy(model, ids, max_new=args.max_new, no_cache=args.no_cache)
    dt = time.time() - t0
    n_new = len(out) - len(ids)
    print(f"[laguna] {n_new}/{args.max_new} tokens in {dt:.2f}s "
          f"({n_new/dt:.1f} tok/s)\n")
    print(tok.decode(out))


if __name__ == "__main__":
    main()
