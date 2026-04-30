"""Convert MiMo-V2.5-Pro FP8 source -> JANG_2L (mx.quantize affine).

JANGTQ2 conversion lives in `convert_jangtq.py` (calls into
jang_tools.turboquant.* the same way convert_dsv4_jangtq.py does).

Per-module overrides (matches upstream ignored_layers):
    - All `o_proj` layers stay bf16
    - lm_head stays bf16 (small, end-of-pipe)
    - embed_tokens stays bf16
    - MTP head stays bf16 (separate file)
    - Routed experts (model.layers.*.mlp.experts.*.{gate,up,down}_proj)
      use bits=2 with gate=4 / down=3 floor (project_mlp_asymmetry)
    - All other linears use bits=2

Bake the 2026-04-25 invariants into config.json:
    - mxtq_bits = 2 (so vmlx-swift loaders pick up the right codebook)
    - routed_expert_bits = 2
    - rope_parameters {rope_theta:.., rope_type:.., partial_rotary_factor:..}
      (transformers >= 4.50 contract)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import MiMoV2Config


def per_module_quant_overrides(cfg: MiMoV2Config) -> dict:
    overrides: dict[str, dict] = {}
    for i, ig in enumerate(cfg.fp8_ignored_layers):
        # ig already names full upstream path, e.g. model.layers.5.self_attn.o_proj
        overrides[ig] = {"bits": None}
    overrides["lm_head"] = {"bits": None}
    overrides["model.embed_tokens"] = {"bits": None}

    for L in range(cfg.num_hidden_layers):
        if not cfg.moe_layer_freq[L]:
            continue
        for e in range(cfg.n_routed_experts):
            base = f"model.layers.{L}.mlp.experts.{e}"
            overrides[f"{base}.gate_proj"] = {"bits": 4}
            overrides[f"{base}.down_proj"] = {"bits": 3}
            overrides[f"{base}.up_proj"]   = {"bits": 2}
    return overrides


def write_jang_2l_config(src: str, dst: str, cfg: MiMoV2Config):
    src_cfg = json.loads((Path(src) / "config.json").read_text())
    out = dict(src_cfg)

    out["quantization"] = {
        "group_size": 64,
        "bits": 2,
        **per_module_quant_overrides(cfg),
    }
    out["mxtq_bits"] = 2          # 2026-04-25 invariant
    out["routed_expert_bits"] = 2 # 2026-04-25 invariant
    out["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": float(cfg.rope_theta),
        "partial_rotary_factor": float(cfg.partial_rotary_factor),
    }
    Path(dst).mkdir(parents=True, exist_ok=True)
    (Path(dst) / "config.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}/config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="FP8 source bundle")
    ap.add_argument("--dst", required=True, help="JANG_2L output bundle")
    ap.add_argument("--profile", choices=("JANG_2L",), default="JANG_2L")
    args = ap.parse_args()

    cfg = MiMoV2Config.from_json(f"{args.src}/config.json")
    write_jang_2l_config(args.src, args.dst, cfg)

    # Weight conversion is the heavy step; delegated to a streaming routine
    # that mirrors convert_dsv4_jang.py. Implemented in _stream_quantize.
    _stream_quantize(args.src, args.dst, cfg)


def _stream_quantize(src: str, dst: str, cfg: MiMoV2Config):
    """Stream FP8 -> bf16 -> mx.quantize(affine) shard by shard.

    Writing the quantized output as new safetensors shards, plus a fresh
    model.safetensors.index.json. Matches jang_tools.dsv4.convert_dsv4_jang
    (read it for the exact streaming pattern; this is the MiMoV2 mirror).
    """
    raise NotImplementedError(
        "Hook up to jang_tools.dsv4.convert_dsv4_jang style streamer once "
        "FP8 source download completes. Files in place to plug in."
    )


if __name__ == "__main__":
    main()
