"""Distributed inference runtime for MiMo-V2.5-Pro.

Modes
-----
    sanity    — load weights on each node, do one forward, all_sum a digest
    decode    — interactive / batch decode using EP-sharded experts
    eval      — run pass@1 / MMLU harness through the distributed model

Sharding (default plan, see distributed/sharding.py):
    EP : experts split by RAM ratio (Studio 256 / MacBook 128 -> 2:1)
    PP : every layer replicated; attention runs locally on every node
    TP : off

MoE forward is the only place that needs comms. Per token, each node
computes its locally-owned experts and contributes zero for others, then
all_sum produces the merged routed output. Dense layer-0 runs the same
on every node (replicated weights).

Loaders
-------
    --src points at one of:
        FP8 source       (auto-detect via config.quantization_config.fmt == "e4m3")
        JANG_2L bundle   (auto-detect via mx.quantize affine sidecar)
        JANGTQ2 bundle   (auto-detect via mxtq_bits / routed_expert_bits)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from ..distributed.jaccl_init import init_world, print_zero, realize
from ..distributed.sharding import default_mimo_v2_plan
from .config import MiMoV2Config
from .model import MiMoV2ForCausalLM


def detect_format(src: str) -> str:
    """Return one of: 'fp8' | 'jang' | 'jangtq'."""
    cfg = json.loads((Path(src) / "config.json").read_text())
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") == "fp8":
        return "fp8"
    if "mxtq_bits" in cfg or "routed_expert_bits" in cfg:
        return "jangtq"
    if qc.get("group_size") and qc.get("bits"):
        return "jang"
    raise ValueError(f"cannot detect bundle format at {src}")


def build_distributed_model(src: str, world):
    cfg = MiMoV2Config.from_json(f"{src}/config.json")
    plan = default_mimo_v2_plan(world.rank, world.size,
                                n_experts=cfg.n_routed_experts,
                                num_layers=cfg.num_hidden_layers,
                                ram_weights=[1.0, 0.5][:world.size])
    print_zero(world, f"plan: rank {world.rank} owns "
                      f"{len(plan.my_experts)} of {cfg.n_routed_experts} experts")
    fmt = detect_format(src)
    print_zero(world, f"format: {fmt}")

    model = MiMoV2ForCausalLM(cfg)
    if fmt == "fp8":
        from .weight_loader import load_fp8_to_bf16
        flat = load_fp8_to_bf16(src, cfg)
    elif fmt == "jang":
        from .jang_loader import load_jang
        flat = load_jang(src, cfg, plan=plan)
    elif fmt == "jangtq":
        from .jangtq_loader import load_jangtq
        flat = load_jangtq(src, cfg, plan=plan)
    else:
        raise AssertionError(fmt)

    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(flat.items())))
    realize(model.parameters())
    return model, cfg, plan


def sanity(model, world, cfg):
    x = mx.array([[1, 2, 3, 4, 5]], dtype=mx.uint32)
    t0 = time.perf_counter()
    logits, _ = model(x, caches=None)
    realize(logits)
    dt = time.perf_counter() - t0

    # cross-rank digest must match if sharding is correct
    digest = mx.sum(logits.astype(mx.float32))
    realize(digest)
    summed = mx.distributed.all_sum(digest)
    realize(summed)
    print_zero(world, f"sanity: forward {dt*1e3:.1f} ms  "
                      f"digest sum-of-ranks={float(summed.item()):.4f}")


def decode_loop(model, world, cfg, src: str, prompt: str, max_new: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    ids = tok.encode(prompt)
    x = mx.array([ids], dtype=mx.uint32)
    logits, caches = model(x, caches=None)
    realize(logits)
    out = list(ids)
    for _ in range(max_new):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
        realize(logits)
    print_zero(world, "\n" + tok.decode(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("sanity", "decode"), default="sanity")
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=16)
    args = ap.parse_args()

    world = init_world()
    print_zero(world, f"jang dist: rank {world.rank}/{world.size} backend={world.backend}")

    model, cfg, plan = build_distributed_model(args.src, world)

    if args.mode == "sanity":
        sanity(model, world, cfg)
    else:
        decode_loop(model, world, cfg, args.src, args.prompt, args.max_new)


if __name__ == "__main__":
    main()
