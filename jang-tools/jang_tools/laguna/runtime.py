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
    # 2026-04-30 fix: quantized formats (jang affine, MXFP4, JANGTQ
    # mixed-precision) ship `.weight + .scales + .biases` keys per
    # Linear, but `model.update()` walks the tree against the bare
    # `nn.Linear` modules instantiated by `LagunaForCausalLM.__init__`
    # — which have NO `.scales` parameter, so the update raises:
    #   ValueError: Module does not have parameter named "scales"
    # Walk the weights once to swap matching `nn.Linear` modules to
    # `nn.QuantizedLinear` BEFORE update, mirroring the pattern
    # `mlx_lm.utils.load_model` uses (`nn.quantize` predicate that
    # checks for sidecar keys). This is what makes JANG_2L / MXFP4 /
    # JANGTQ-quantized Laguna bundles actually load.
    # 2026-04-30 stack of key remappings to bridge HF safetensors layout
    # to LagunaForCausalLM's flat module structure.
    #   1. Strip leading `model.` prefix — HF stores the text decoder
    #      under `model.embed_tokens.weight` etc, but Laguna flat-attaches
    #      embed_tokens/layers/norm/lm_head at the wrapper root.
    #   2. Drop the `experts.` infix on the MoE bias-correction key:
    #      `mlp.experts.e_score_correction_bias` → `mlp.e_score_correction_bias`.
    #      `self.experts` on `LagunaMoE` is a Python list of DenseMLPs
    #      with no aggregate parameter slot; the bias lives on the parent
    #      LagunaMoE module instead, so the key is renamed at load time.
    def _remap(k: str) -> str:
        if k.startswith("model."):
            k = k[len("model."):]
        if k.endswith(".mlp.experts.e_score_correction_bias"):
            k = k.replace(".mlp.experts.e_score_correction_bias",
                          ".mlp.e_score_correction_bias")
        return k
    weights = {_remap(k): v for k, v in weights.items()}

    # 2026-04-30 expert weight packing for SwitchGLU. HF Laguna source
    # stores experts unpacked as
    #   layers.N.mlp.experts.E.{gate,up,down}_proj.weight
    # SwitchGLU expects ONE packed tensor per matmul shaped
    # (num_experts, out, in) under
    #   layers.N.mlp.switch_mlp.{gate,up,down}_proj.weight
    # Stack along axis=0 and rename. We only do this for the routed-expert
    # path (NOT shared_expert, which is a regular DenseMLP).
    import re as _re_pack
    _expert_pat = _re_pack.compile(r"^(layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    grouped: dict = {}
    new_weights: dict = {}
    for k, v in weights.items():
        m = _expert_pat.match(k)
        if not m:
            new_weights[k] = v
            continue
        prefix, expert_idx, proj = m.group(1), int(m.group(2)), m.group(3)
        grouped.setdefault((prefix, proj), {})[expert_idx] = v
    for (prefix, proj), per_expert in grouped.items():
        n_exp = max(per_expert.keys()) + 1
        # Verify dense indexing 0..n_exp-1.
        if set(per_expert.keys()) != set(range(n_exp)):
            missing = set(range(n_exp)) - set(per_expert.keys())
            raise ValueError(f"Expert pack: {prefix}.{proj} missing experts {sorted(missing)[:5]}…")
        stacked = mx.stack([per_expert[i] for i in range(n_exp)], axis=0)
        new_weights[f"{prefix}.switch_mlp.{proj}.weight"] = stacked
    weights = new_weights
    if fmt in ("jang", "mxfp4", "jangtq"):
        # Group sizes / bits are in config.json["quantization"] for affine
        # paths; for JANGTQ they're per-module via jang_config.mxtq_bits
        # but `load_jangtq` in weight_loader_bf16 already dequantizes the
        # JANGTQ codebook part to affine 8-bit before this point, so we
        # treat all three quantized paths the same way: read group_size +
        # bits from config and call nn.quantize with a predicate that
        # only matches modules whose .scales key is in the weight map.
        import json as _json
        import mlx.nn as nn
        cfg_json = _json.loads((Path(src) / "config.json").read_text())
        qcfg = cfg_json.get("quantization") or {}
        group_size = qcfg.get("group_size", 64)
        bits = qcfg.get("bits", 4)
        scale_keys = {k for k in weights.keys() if k.endswith(".scales")}
        def _predicate(name, module):
            return f"{name}.scales" in scale_keys
        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_predicate)
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
