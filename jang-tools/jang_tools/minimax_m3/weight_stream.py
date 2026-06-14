"""Streamed weight reader for a MiniMax-M3 safetensors checkpoint.

Indexes tensor keys across all shards and reads them one at a time (zero-copy
memmap via safetensors framework="pt"). Optionally simulates the JANG_2L
quantization inline (mx.quantize -> mx.dequantize) so a coherence probe tests
EXACTLY the weights the converter will ship, without building the bundle.

Tensor naming (text backbone prefix `language_model.model.`):
  layers.{L}.self_attn.{q,k,v,o}_proj.weight
  layers.{L}.self_attn.{q,k}_norm.weight
  layers.{L}.mlp.{gate,up,down}_proj.weight                       (dense 0-2)
  layers.{L}.block_sparse_moe.gate.weight / .e_score_correction_bias
  layers.{L}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight       (w1=gate w3=up w2=down)
  layers.{L}.block_sparse_moe.shared_experts.{gate,up,down}_proj.weight
  embed_tokens.weight  /  norm.weight  ;  language_model.lm_head.weight

Created by Jinho Jang (eric@jangq.ai).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# JANG_2L policy: (bits, group_size) per tensor class.
JANG2L = {
    "expert": (2, 128),    # routed expert w1/w2/w3 (gs128 to fit sub-110 GB post-prune)
    "shared": (6, 64),     # always-on shared expert (protected)
    "dense":  (6, 64),     # dense MLP layers 0-2 (always-on, only 3 layers)
    "attn":   (8, 64),     # q/k/v/o
    "embed":  (6, 64),     # embed_tokens
    "lmhead": (8, 64),     # lm_head
    # norms, router gate, indexer -> passthrough (no quant)
}


@dataclass
class ShardIndex:
    key_to_shard: dict
    text_prefix: str       # "language_model.model." or "model."
    head_prefix: str       # "language_model." or ""


def build_index(model_dir: Path) -> ShardIndex:
    shards = sorted(Path(model_dir).glob("model-*.safetensors"))
    k2s = {}
    for sp in shards:
        with safe_open(str(sp), framework="pt") as f:
            for k in f.keys():
                k2s[k] = sp
    # detect prefix
    if any(k.startswith("language_model.model.layers.0.") for k in k2s):
        tp, hp = "language_model.model.", "language_model."
    else:
        tp, hp = "model.", ""
    return ShardIndex(key_to_shard=k2s, text_prefix=tp, head_prefix=hp)


def _raw(idx: ShardIndex, key: str) -> torch.Tensor:
    sp = idx.key_to_shard[key]
    with safe_open(str(sp), framework="pt") as f:
        return f.get_tensor(key)


def has(idx: ShardIndex, key: str) -> bool:
    return key in idx.key_to_shard


# ----- inline JANG_2L quant simulation (mlx, bit-exact w/ converter) -----

_MX = None


def _mx():
    global _MX
    if _MX is None:
        import mlx.core as mx
        _MX = mx
    return _MX


def quant_sim(w: torch.Tensor, kind: str, awq_scale=None) -> torch.Tensor:
    """Quantize->dequantize `w` per the JANG_2L policy for `kind`.

    If `awq_scale` (per-input-channel, last axis) is given, reproduce the AWQ
    inference math: quantize (w * s) then divide the dequantized result by s, so
    using it with the ORIGINAL activation equals the folded-norm production path.
    No-op for kinds not in JANG2L (norms/router/indexer passthrough).
    """
    if kind not in JANG2L:
        return w
    bits, gs = JANG2L[kind]
    mx = _mx()
    wf = w.float()
    if awq_scale is not None:
        s = torch.as_tensor(awq_scale, dtype=torch.float32)
        wf = wf * s                      # scale input columns (last axis)
    a = mx.array(wf.numpy())
    qw, qs, qb = mx.quantize(a.astype(mx.float16), group_size=gs, bits=bits)
    dq = np.array(mx.dequantize(qw, qs, qb, group_size=gs, bits=bits).astype(mx.float32))
    out = torch.from_numpy(dq)
    if awq_scale is not None:
        out = out / s
    return out.to(w.dtype)


class WeightStreamer:
    """Reads M3 tensors as torch bf16, optionally simulating JANG_2L quant."""

    supports_stacked = False  # we feed experts per-id (memory-frugal)

    def __init__(self, idx: ShardIndex, quant: str = "none", device="cpu",
                 compute_dtype=torch.bfloat16, awq: dict | None = None):
        self.idx = idx
        self.quant = quant            # "none" | "2L"
        self.device = device
        self.dtype = compute_dtype
        self.awq = awq or {}          # {layer_idx: per-input-channel scale np array}

    def _get(self, key: str, kind: str, awq_scale=None) -> torch.Tensor:
        w = _raw(self.idx, key)
        if self.quant == "2L":
            w = quant_sim(w, kind, awq_scale=awq_scale)
        return w.to(device=self.device, dtype=self.dtype)

    def _passthrough(self, key: str) -> torch.Tensor:
        return _raw(self.idx, key).to(device=self.device, dtype=self.dtype)

    # -- top level --
    def embed(self):
        return self._get(self.idx.text_prefix + "embed_tokens.weight", "embed")

    def final_norm(self):
        return _raw(self.idx, self.idx.text_prefix + "norm.weight").to(self.device, torch.float32)

    def lm_head(self):
        k = self.idx.head_prefix + "lm_head.weight"
        if has(self.idx, k):
            return self._get(k, "lmhead")
        return self.embed()  # tie fallback (M3 is untied, so this rarely fires)

    # -- per layer --
    def attn(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}.self_attn."
        w = {proj: self._get(f"{p}{proj}_proj.weight", "attn")
             for proj in ("q", "k", "v", "o")}
        qn = self._passthrough(f"{p}q_norm.weight") if has(self.idx, f"{p}q_norm.weight") else None
        kn = self._passthrough(f"{p}k_norm.weight") if has(self.idx, f"{p}k_norm.weight") else None
        return w, qn, kn

    def norms(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}."
        return (self._passthrough(p + "input_layernorm.weight"),
                self._passthrough(p + "post_attention_layernorm.weight"))

    def dense_mlp(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}.mlp."
        return {f"{n}_proj": self._get(f"{p}{n}_proj.weight", "dense")
                for n in ("gate", "up", "down")}

    def router(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}.block_sparse_moe."
        rw = self._passthrough(p + "gate.weight")
        bk = p + "e_score_correction_bias"
        rb = self._passthrough(bk) if has(self.idx, bk) else None
        return rw, rb

    def shared_expert(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}.block_sparse_moe.shared_experts."
        if not has(self.idx, p + "gate_proj.weight"):
            return None
        s = self.awq.get(li)
        return {"gate_proj": self._get(f"{p}gate_proj.weight", "shared", awq_scale=s),
                "up_proj": self._get(f"{p}up_proj.weight", "shared", awq_scale=s),
                "down_proj": self._get(f"{p}down_proj.weight", "shared")}

    def make_expert_loader(self, li: int):
        p = f"{self.idx.text_prefix}layers.{li}.block_sparse_moe.experts."
        s = self.awq.get(li)

        def loader(e, allow_none=False):
            if e == "__stacked__":
                return None
            gw = self._get(f"{p}{e}.w1.weight", "expert", awq_scale=s)   # gate
            uw = self._get(f"{p}{e}.w3.weight", "expert", awq_scale=s)   # up
            dw = self._get(f"{p}{e}.w2.weight", "expert")               # down
            return gw, uw, dw

        loader.supports_stacked = False
        return loader
