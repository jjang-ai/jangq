"""Stream MiMo-V2.6 SOURCE weights through the fresh MLX runtime, layer by layer.

The full model (166 GiB) does not fit in 128 GB, so every consumer that needs
the exact source function (golden perplexity checks, calibration capture,
reference logits for KL) builds ONE decoder layer at a time from the
checkpoint, runs all tokens through it, and frees it.

Exactness: routed experts are loaded as the raw MXFP4 bytes (bit-exact MLX
mxfp4, see mxfp4_codec.py); FP8 tensors are dequantized with their 128x128
block scales to bf16; bf16 tensors pass through. This is the source model up to
bf16 rounding of the FP8 dequant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from .mxfp4_codec import mxfp4_raw_to_mlx
from .v26_model import DecoderLayer, ModelArgs
from .weight_loader import MiMoShardIndex, deinterleave_tp_qkv_rows


def load_args(src: Path) -> ModelArgs:
    cfg = json.loads((Path(src) / "config.json").read_text())
    fields = ModelArgs.__dataclass_fields__
    return ModelArgs(**{k: v for k, v in cfg.items() if k in fields})


def _mx(t: torch.Tensor, dtype=mx.bfloat16) -> mx.array:
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16).astype(dtype)
    return mx.array(t.float().numpy()).astype(dtype)


class SourceStream:
    def __init__(self, src: str | Path, *, qkv_layout: str = "tp4"):
        self.src = Path(src)
        self.idx = MiMoShardIndex(self.src)
        self.args = load_args(self.src)
        if qkv_layout not in {"tp4", "contiguous"}:
            raise ValueError(qkv_layout)
        self.qkv_layout = qkv_layout

    # ---------------------------------------------------------------- tensors
    def dense(self, name: str, dtype=mx.bfloat16) -> mx.array:
        """Read a non-expert tensor (FP8 dequantized) with NO qkv reordering."""
        if ".self_attn.qkv_proj." in name and self.idx.is_fp8_weight(name):
            raise ValueError("read fused qkv through SourceStream.qkv()")
        if self.idx.is_fp8_weight(name):
            from .fp8_block_codec import dequant_fp8_e4m3_scale_inv
            w = self.idx._open(self.idx.weight_map[name]).get_tensor(name)
            sname = name[: -len(".weight")] + ".weight_scale_inv"
            s = self.idx._open(self.idx.weight_map[sname]).get_tensor(sname)
            return _mx(dequant_fp8_e4m3_scale_inv(w, s, out_dtype=torch.float32), dtype)
        t = self.idx._open(self.idx.weight_map[name]).get_tensor(name)
        return _mx(t, dtype)

    def qkv(self, layer: int, layout: str | None = None) -> mx.array:
        name = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        if (layout or self.qkv_layout) == "tp4":
            return _mx(self.idx.read_tensor(name, out_dtype=torch.float32))  # rank-blocked dequant + de-interleave
        return self.dense(name)

    def expert_stack(self, layer: int, proj: str) -> tuple[mx.array, mx.array]:
        ws, ss = [], []
        for e in range(self.args.n_routed_experts):
            w, s = self.idx.read_mxfp4_raw(f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight")
            w, s = mxfp4_raw_to_mlx(w, s)
            ws.append(w)
            ss.append(s)
        return mx.array(np.stack(ws)), mx.array(np.stack(ss))

    # ----------------------------------------------------------------- layers
    def build_layer(self, layer: int, *, qkv_layout: str | None = None) -> DecoderLayer:
        a = self.args
        mod = DecoderLayer(a, layer)
        p = f"model.layers.{layer}"
        mod.input_layernorm.weight = self.dense(f"{p}.input_layernorm.weight")
        mod.post_attention_layernorm.weight = self.dense(f"{p}.post_attention_layernorm.weight")
        at = mod.self_attn
        at.qkv_proj.weight = self.qkv(layer, qkv_layout)
        at.o_proj.weight = self.dense(f"{p}.self_attn.o_proj.weight")
        if at.has_sink:
            at.attention_sink_bias = self.dense(f"{p}.self_attn.attention_sink_bias", mx.float32)
        if a.moe_layer_freq[layer]:
            mlp = mod.mlp
            mlp.gate.weight = self.dense(f"{p}.mlp.gate.weight", mx.float32)
            mlp.gate.e_score_correction_bias = self.dense(f"{p}.mlp.gate.e_score_correction_bias", mx.float32)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(mlp.switch_mlp, proj)
                q = lin.to_quantized(group_size=32, bits=4, mode="mxfp4")
                q.weight, q.scales = self.expert_stack(layer, proj)
                if getattr(q, "biases", None) is not None:
                    q.biases = None
                setattr(mlp.switch_mlp, proj, q)
        else:
            for proj in ("gate_proj", "up_proj", "down_proj"):
                getattr(mod.mlp, proj).weight = self.dense(f"{p}.mlp.{proj}.weight")
        mx.eval(mod.parameters())
        return mod

    def embed(self, ids: mx.array) -> mx.array:
        if not hasattr(self, "_embed"):
            self._embed = self.dense("model.embed_tokens.weight")
        return self._embed[ids]

    def head(self, h: mx.array) -> mx.array:
        if not hasattr(self, "_head"):
            self._final_norm = self.dense("model.norm.weight")
            self._lm_head = self.dense("lm_head.weight")
        h = mx.fast.rms_norm(h, self._final_norm, self.args.layernorm_epsilon)
        return h @ self._lm_head.T


def layer_masks(args: ModelArgs, T: int):
    from mlx_lm.models.base import create_causal_mask
    return create_causal_mask(T, 0), create_causal_mask(T, 0, window_size=args.sliding_window)
