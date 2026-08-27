"""Layer-streaming executor over the 299 GB fp8 source (128 GB machine).

One layer's weights live in RAM at a time (~12.5 GB bf16 for a MoE layer);
hidden states for the whole corpus stay resident (~0.5 GB / 50k tokens).
Used for: (1) the no-cache greedy coherence gate on the SOURCE,
(2) calibration captures (X1 samples, second moments, router stats),
(3) the source-logit reference pass for the KL acceptance gate.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import Dots3Config
from .fp8 import ShardIndex
from .ops import decoder_layer, rms_norm

LAYER_PREFIX = "model.layers.{i}."

ATTN_KEYS = [
    "input_layernorm", "post_attention_layernorm",
    "self_attn.q_a_proj", "self_attn.q_a_layernorm", "self_attn.q_b_proj",
    "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm",
    "self_attn.kv_b_proj", "self_attn.k_rope_only_layernorm",
    "self_attn.o_proj", "self_attn.g_proj",
]


def _short(k: str) -> str:
    return k.split(".")[-1] if "." in k else k


def load_layer(idx: ShardIndex, cfg: Dots3Config, i: int,
               dtype=mx.bfloat16) -> dict:
    """Load + dequant one decoder layer into an ops.py weight dict."""
    p = LAYER_PREFIX.format(i=i)
    lw: dict[str, mx.array] = {}
    for k in ATTN_KEYS:
        name = p + k + ".weight"
        arr = idx.read_dequant(name)
        key = _short(k)
        lw[key] = mx.array(arr).astype(
            mx.float32 if "layernorm" in key else dtype)
    if cfg.is_moe(i):
        from concurrent.futures import ThreadPoolExecutor
        lw["gate_w"] = mx.array(idx.read_dequant(p + "mlp.gate.weight")).astype(dtype)
        lw["gate_bias"] = mx.array(
            idx.read_dequant(p + "mlp.gate.e_score_correction_bias")).astype(mx.float32)
        for proj, key in (("gate_proj", "experts_gate"), ("up_proj", "experts_up"),
                          ("down_proj", "experts_down")):
            stack = np.empty(
                (cfg.n_routed_experts,) +
                tuple(idx.info(p + f"mlp.experts.0.{proj}.weight")[2]),
                dtype=np.float32)

            def fill(e, _p=p, _proj=proj, _stack=stack):
                _stack[e] = idx.read_dequant(
                    _p + f"mlp.experts.{e}.{_proj}.weight")

            with ThreadPoolExecutor(max_workers=8) as ex:
                list(ex.map(fill, range(cfg.n_routed_experts)))
            lw[key] = mx.array(stack).astype(dtype)
            del stack
        for proj, key in (("gate_proj", "shared_gate"), ("up_proj", "shared_up"),
                          ("down_proj", "shared_down")):
            lw[key] = mx.array(
                idx.read_dequant(p + f"mlp.shared_experts.{proj}.weight")).astype(dtype)
    else:
        for proj, key in (("gate_proj", "mlp_gate"), ("up_proj", "mlp_up"),
                          ("down_proj", "mlp_down")):
            lw[key] = mx.array(idx.read_dequant(p + f"mlp.{proj}.weight")).astype(dtype)
    mx.eval(list(lw.values()))
    return lw


def free_layer(lw: dict) -> None:
    lw.clear()
    gc.collect()
    mx.clear_cache()


class StreamModel:
    def __init__(self, model_dir: str | Path, memory_limit_gb: float = 48.0):
        self.dir = Path(model_dir)
        self.cfg = Dots3Config.load(self.dir)
        self.idx = ShardIndex(self.dir)
        mx.set_memory_limit(int(memory_limit_gb * 1024**3))
        self._embed = None

    # -- bookends ---------------------------------------------------------
    def embed(self, token_ids: list[list[int]]) -> list[mx.array]:
        if self._embed is None:
            self._embed = mx.array(
                self.idx.read_dequant("model.embed_tokens.weight")).astype(mx.bfloat16)
            mx.eval(self._embed)
        return [self._embed[mx.array(t)][None] for t in token_ids]

    def head_logits(self, hidden: mx.array) -> mx.array:
        """hidden [B,S,H] -> logits f32 (applies final norm + lm_head)."""
        norm_w = mx.array(self.idx.read_dequant("model.norm.weight")).astype(mx.float32)
        h = rms_norm(hidden, norm_w, self.cfg.rms_norm_eps)
        head = mx.array(self.idx.read_dequant("lm_head.weight")).astype(mx.bfloat16)
        logits = (h.astype(mx.bfloat16) @ head.T).astype(mx.float32)
        mx.eval(logits)
        del head
        mx.clear_cache()
        return logits

    # -- streaming forward --------------------------------------------------
    def forward_all(self, states: list[mx.array], layer_cb=None,
                    n_layers: int | None = None,
                    progress: bool = True) -> list[mx.array]:
        """Run every sequence through all layers, one layer resident at a time.
        states: list of [1, S_i, H] embeddings, modified through the stack.
        layer_cb(i, seq_j, capture_dict) receives per-sequence captures."""
        n = n_layers if n_layers is not None else self.cfg.num_hidden_layers
        for i in range(n):
            t0 = time.time()
            lw = load_layer(self.idx, self.cfg, i)
            t_load = time.time() - t0
            t0 = time.time()
            for j, h in enumerate(states):
                cap = {} if layer_cb is not None else None
                out = decoder_layer(h, lw, self.cfg, i, capture=cap)
                mx.eval(out)
                states[j] = out
                if layer_cb is not None:
                    layer_cb(i, j, cap)
                    del cap
            free_layer(lw)
            if progress:
                print(f"  layer {i:2d} load {t_load:5.1f}s "
                      f"compute {time.time()-t0:6.1f}s "
                      f"mem {mx.get_active_memory()/1e9:5.1f}GB", flush=True)
        return states

    # -- no-cache greedy gate ------------------------------------------------
    def greedy_generate(self, token_ids: list[int], max_new: int = 12,
                        eos: tuple[int, ...] = (151643, 151668)) -> list[int]:
        """Full re-stream per token: THE manual no-cache greedy loop.
        ~1 full 299 GB weight read per generated token — this is a gate,
        not a chat loop."""
        ids = list(token_ids)
        for step in range(max_new):
            t0 = time.time()
            states = self.embed([ids])
            states = self.forward_all(states, progress=False)
            logits = self.head_logits(states[0])
            nxt = int(mx.argmax(logits[0, -1]).item())
            ids.append(nxt)
            print(f"[greedy {step+1}/{max_new}] +{nxt} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if nxt in eos:
                break
        return ids
