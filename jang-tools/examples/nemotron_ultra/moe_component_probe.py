"""Profile real Nemotron Ultra MoE decode components.

This loads the JANGTQ_1L bundle and times the pieces inside selected
NemotronHMoE layers for a single decode-shaped hidden state. The goal is to
localize token/s bottlenecks without perturbing the full generator loop.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx

from jang_tools.load_jangtq import load_jangtq_model


DEFAULT_BUNDLE = Path(
    "/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L"
)


def _time(label: str, fn, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return {
        "label": label,
        "samples_s": samples,
        "median_ms": ordered[len(ordered) // 2] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _first_moe_layers(model, limit: int) -> list[tuple[int, object]]:
    found = []
    for i, layer in enumerate(model.backbone.layers):
        if getattr(layer, "block_type", None) == "E":
            found.append((i, layer))
            if len(found) >= limit:
                break
    return found


def _profile_layer(layer, hidden, repeats: int, warmup: int) -> dict:
    moe = layer.mixer
    normed = layer.norm(hidden)
    mx.eval(normed)

    inds, scores = moe.gate(normed)
    x_latent = moe.fc1_latent_proj(normed) if moe.moe_latent_size is not None else normed
    routed = moe.switch_mlp(x_latent, inds)
    combined = (routed * scores[..., None]).sum(axis=-2).astype(routed.dtype)
    weighted_decode = None
    weighted_decode_fn = getattr(moe.switch_mlp, "_jangtq_weighted_decode", None)
    if weighted_decode_fn is not None:
        weighted_decode = weighted_decode_fn(x_latent, inds, scores)
    projected = moe.fc2_latent_proj(combined) if moe.moe_latent_size is not None else combined
    shared = moe.shared_experts(normed) if getattr(moe.config, "n_shared_experts", None) is not None else 0
    eval_items = [inds, scores, x_latent, routed, combined, projected, shared]
    if weighted_decode is not None:
        eval_items.append(weighted_decode)
    mx.eval(*eval_items)

    timings = [
        _time("norm", lambda: layer.norm(hidden), repeats, warmup),
        _time("gate", lambda: moe.gate(normed)[0], repeats, warmup),
    ]
    if moe.moe_latent_size is not None:
        timings.append(_time("fc1_latent_proj", lambda: moe.fc1_latent_proj(normed), repeats, warmup))
    timings.extend(
        [
            _time("switch_mlp", lambda: moe.switch_mlp(x_latent, inds), repeats, warmup),
            _time(
                "score_weighted_sum",
                lambda: (routed * scores[..., None]).sum(axis=-2).astype(routed.dtype),
                repeats,
                warmup,
            ),
        ]
    )
    if weighted_decode_fn is not None:
        timings.append(_time("weighted_decode", lambda: weighted_decode_fn(x_latent, inds, scores), repeats, warmup))
    if moe.moe_latent_size is not None:
        timings.append(_time("fc2_latent_proj", lambda: moe.fc2_latent_proj(combined), repeats, warmup))
    if getattr(moe.config, "n_shared_experts", None) is not None:
        timings.append(_time("shared_experts", lambda: moe.shared_experts(normed), repeats, warmup))
    timings.append(_time("full_moe", lambda: moe(normed), repeats, warmup))

    return {
        "block_type": getattr(layer, "block_type", None),
        "hidden_shape": list(hidden.shape),
        "indices_shape": list(inds.shape),
        "scores_shape": list(scores.shape),
        "latent_shape": list(x_latent.shape),
        "routed_shape": list(routed.shape),
        "timings": timings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--wired-limit-gb", type=int, default=105)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    os.environ.setdefault("JANGTQ_WIRED_LIMIT_GB", str(args.wired_limit_gb))
    model, _ = load_jangtq_model(args.bundle, skip_params_eval=True)
    hidden_size = int(model.args.hidden_size)
    hidden = mx.ones((1, 1, hidden_size), dtype=mx.bfloat16)
    mx.eval(hidden)

    layers = []
    for idx, layer in _first_moe_layers(model, args.layers):
        layers.append(
            {
                "layer_index": idx,
                **_profile_layer(layer, hidden, args.repeats, args.warmup),
            }
        )

    result = {
        "bundle": str(args.bundle),
        "wired_limit_gb": args.wired_limit_gb,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "layers": layers,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
