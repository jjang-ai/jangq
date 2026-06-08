"""Live MiMo JANGTQ MoE prefill component probe.

Runs one layer's MoE path on real activations from a local bundle and times the
router, sort, gate/up, down, scatter, and weighting pieces. It is intentionally
single-process and single-layer to avoid the memory pressure from full smoke
loops.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401


def _sync_time(fn, repeat: int) -> float:
    mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repeat):
        mx.eval(fn())
    return (time.perf_counter() - start) * 1000.0 / repeat


def _render_prompt(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    model, tokenizer = load(
        str(args.bundle),
        lazy=True,
        tokenizer_config={"trust_remote_code": True},
    )
    ids = _render_prompt(tokenizer, args.prompt)
    input_ids = mx.array([ids], dtype=mx.int32)
    h = model.model.embed_tokens(input_ids)
    for layer_idx in range(args.layer):
        layer = model.model.layers[layer_idx]
        h = layer(h, mask=None, cache=None)
        mx.eval(h)

    layer = model.model.layers[args.layer]
    moe = layer.mlp
    x = layer.post_attention_layernorm(
        h + layer.self_attn(layer.input_layernorm(h), mask=None, cache=None)
    )
    mx.eval(x)
    bsz, seq_len, hidden = x.shape
    x_flat = x.reshape(-1, hidden)
    gate = moe.gate
    switch = moe.switch_mlp

    router_ms = _sync_time(lambda: gate(x_flat), args.repeat)
    topk_idx, topk_w = gate(x_flat)
    mx.eval(topk_idx, topk_w)

    x_exp = mx.expand_dims(x_flat, (-2, -3))
    sort_ms = _sync_time(lambda: _gather_sort(x_exp, topk_idx), args.repeat)
    sorted_x, sorted_idx, inv_order = _gather_sort(x_exp, topk_idx)
    mx.eval(sorted_x, sorted_idx, inv_order)

    gate_up_ms = _sync_time(
        lambda: switch.activation(
            switch.up_proj(sorted_x, sorted_idx, sorted_indices=True),
            switch.gate_proj(sorted_x, sorted_idx, sorted_indices=True),
        ),
        args.repeat,
    )
    x_act = switch.activation(
        switch.up_proj(sorted_x, sorted_idx, sorted_indices=True),
        switch.gate_proj(sorted_x, sorted_idx, sorted_indices=True),
    )
    mx.eval(x_act)

    down_ms = _sync_time(
        lambda: switch.down_proj(x_act, sorted_idx, sorted_indices=True),
        args.repeat,
    )
    x_out = switch.down_proj(x_act, sorted_idx, sorted_indices=True)
    mx.eval(x_out)

    scatter_ms = _sync_time(
        lambda: _scatter_unsort(x_out, inv_order, topk_idx.shape),
        args.repeat,
    )
    unsorted = _scatter_unsort(x_out, inv_order, topk_idx.shape)
    mx.eval(unsorted)

    weight_sum_ms = _sync_time(
        lambda: (unsorted.squeeze(-2) * topk_w[..., None]).sum(axis=1),
        args.repeat,
    )

    result = {
        "bundle": str(args.bundle),
        "layer": args.layer,
        "prompt_tokens": len(ids),
        "seq_len": seq_len,
        "dispatches": int(seq_len) * int(gate.top_k),
        "repeat": args.repeat,
        "component_ms": {
            "router": router_ms,
            "sort": sort_ms,
            "gate_up": gate_up_ms,
            "down": down_ms,
            "scatter": scatter_ms,
            "weight_sum": weight_sum_ms,
            "sum_known": router_ms + sort_ms + gate_up_ms + down_ms + scatter_ms + weight_sum_ms,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
