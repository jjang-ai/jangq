"""Profile real Nemotron Ultra Mamba decode components.

This loads the JANGTQ_1L bundle and times the pieces inside selected
NemotronHMamba2Mixer layers for a single decode-shaped hidden state. It is a
diagnostic probe only; it does not modify runtime code or the model bundle.
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


def _first_mamba_layers(model, limit: int) -> list[tuple[int, int, object]]:
    found = []
    cache_ordinal = 0
    for layer_index, layer in enumerate(model.backbone.layers):
        if layer.block_type in ("M", "*"):
            this_cache = cache_ordinal
            cache_ordinal += 1
        else:
            this_cache = -1
        if getattr(layer, "block_type", None) == "M":
            found.append((layer_index, this_cache, layer))
            if len(found) >= limit:
                break
    return found


def _split_projected(mamba, projected):
    gate, conv_input, dt = mx.split(
        projected,
        [mamba.intermediate_size, mamba.intermediate_size + mamba.conv_dim],
        axis=-1,
    )
    return gate, conv_input, dt


def _split_conv(mamba, conv_output):
    return mx.split(
        conv_output,
        [
            mamba.intermediate_size,
            mamba.intermediate_size + mamba.n_groups * mamba.ssm_state_size,
        ],
        axis=-1,
    )


def _profile_layer(layer, cache_entry, hidden, repeats: int, warmup: int) -> dict:
    mamba = layer.mixer
    normed = layer.norm(hidden)
    projected = mamba.in_proj(normed)
    gate, conv_input, dt = _split_projected(mamba, projected)
    conv_output = mamba._conv(conv_input, cache_entry, None)
    hidden_states_ssm, b_state, c_state = _split_conv(mamba, conv_output)
    ssm_out = mamba._ssm(hidden_states_ssm, b_state, c_state, dt, cache_entry, None)
    gated = mamba.norm(ssm_out, gate)
    out = mamba.out_proj(gated)
    mx.eval(normed, projected, gate, conv_input, dt, conv_output, ssm_out, gated, out)

    timings = [
        _time("outer_norm", lambda: layer.norm(hidden), repeats, warmup),
        _time("in_proj", lambda: mamba.in_proj(normed), repeats, warmup),
        _time("conv", lambda: mamba._conv(conv_input, cache_entry, None), repeats, warmup),
        _time(
            "ssm_update",
            lambda: mamba._ssm(hidden_states_ssm, b_state, c_state, dt, cache_entry, None),
            repeats,
            warmup,
        ),
        _time("mamba_norm_gated", lambda: mamba.norm(ssm_out, gate), repeats, warmup),
        _time("out_proj", lambda: mamba.out_proj(gated), repeats, warmup),
        _time("full_mamba_mixer", lambda: mamba(normed, mask=None, cache=cache_entry), repeats, warmup),
    ]

    return {
        "block_type": getattr(layer, "block_type", None),
        "hidden_shape": list(hidden.shape),
        "normed_shape": list(normed.shape),
        "projected_shape": list(projected.shape),
        "gate_shape": list(gate.shape),
        "conv_input_shape": list(conv_input.shape),
        "conv_output_shape": list(conv_output.shape),
        "ssm_out_shape": list(ssm_out.shape),
        "conv_dim": int(mamba.conv_dim),
        "intermediate_size": int(mamba.intermediate_size),
        "num_heads": int(mamba.num_heads),
        "n_groups": int(mamba.n_groups),
        "ssm_state_size": int(mamba.ssm_state_size),
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
    cache = model.make_cache()
    mx.eval(hidden)

    layers = []
    for layer_index, cache_ordinal, layer in _first_mamba_layers(model, args.layers):
        layers.append(
            {
                "layer_index": layer_index,
                "cache_ordinal": cache_ordinal,
                **_profile_layer(layer, cache[cache_ordinal], hidden, args.repeats, args.warmup),
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
