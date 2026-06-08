"""Probe MiMo routed-expert affine quantization error on a real hidden state.

This is a narrow diagnostic for the current JANG_2L incoherence issue. It
compares source FP8-dequant expert outputs against local affine requantized
variants for the first routed layer, using the already-converted bundle only to
produce a realistic layer-1 MoE input and selected expert ids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401
from jang_tools.mimo_v2.weight_loader import MiMoShardIndex


PROFILES = {
    "222g64": (2, 2, 2, 64),
    "222g32": (2, 2, 2, 32),
    "224g64": (2, 2, 4, 64),
    "422g64": (4, 2, 2, 64),
    "242g64": (2, 4, 2, 64),
    "333g64": (3, 3, 3, 64),
    "444g64": (4, 4, 4, 64),
}


def _qdq(weight: torch.Tensor, bits: int, group_size: int) -> mx.array:
    arr = mx.array(weight.numpy())
    qw, qs, qb = mx.quantize(arr, group_size=group_size, bits=bits)
    return mx.dequantize(qw, qs, qb, group_size=group_size, bits=bits, mode="affine", dtype=mx.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=3)
    args = parser.parse_args()

    model, _ = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    h = model.model.embed_tokens(ids)

    layer0 = model.model.layers[0]
    h = h + layer0.self_attn(layer0.input_layernorm(h), mask="causal", cache=None)
    h = h + layer0.mlp(layer0.post_attention_layernorm(h))
    mx.eval(h)

    layer1 = model.model.layers[1]
    h = h + layer1.self_attn(layer1.input_layernorm(h), mask="causal", cache=None)
    x = layer1.post_attention_layernorm(h).reshape(-1, h.shape[-1])
    topk_idx, _ = layer1.mlp.gate(x)
    mx.eval(x, topk_idx)

    selected = np.array(topk_idx)[0, : args.experts].tolist()
    loader = MiMoShardIndex(args.src)
    x_mx = x[0].astype(mx.float32)
    x_torch = torch.from_numpy(np.array(x_mx))
    print(f"selected_experts={selected}")

    for expert_idx in selected:
        gate_t = loader.read_tensor(f"model.layers.1.mlp.experts.{expert_idx}.gate_proj.weight")
        up_t = loader.read_tensor(f"model.layers.1.mlp.experts.{expert_idx}.up_proj.weight")
        down_t = loader.read_tensor(f"model.layers.1.mlp.experts.{expert_idx}.down_proj.weight")
        source = F.linear(F.silu(F.linear(x_torch, gate_t)) * F.linear(x_torch, up_t), down_t)
        source_rms = torch.sqrt(torch.mean(source * source)) + 1e-12
        print(f"expert={expert_idx} source_rms={float(source_rms):.6f}")

        for name, (gate_bits, up_bits, down_bits, group_size) in PROFILES.items():
            gate = _qdq(gate_t, gate_bits, group_size)
            up = _qdq(up_t, up_bits, group_size)
            down = _qdq(down_t, down_bits, group_size)
            actual = (nn.silu(x_mx @ gate.T) * (x_mx @ up.T)) @ down.T
            mx.eval(actual)
            actual_t = torch.from_numpy(np.array(actual))
            diff = source - actual_t
            rel = torch.sqrt(torch.mean(diff * diff)) / source_rms
            print(f"  {name}: rel_rmse={float(rel):.6f} maxerr={float(diff.abs().max()):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
