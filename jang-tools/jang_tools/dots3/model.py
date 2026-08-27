"""dots3 JANG bundle runtime (text path) — fast in-RAM inference with KV cache.

Loads a bundle produced by convert_dots3_jang.py (MLX-native affine/mxfp4
storage), builds per-layer weight dicts with QW wrappers, and generates with
the shared ops.py numerics. This is the verification/caption runtime, not a
serving stack.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx

from .config import Dots3Config
from .ops import QW, decoder_layer, linear, rms_norm

EOS_DEFAULT = (151643, 151668)


def _load_all_shards(bundle: Path) -> dict[str, mx.array]:
    idx = json.loads((bundle / "model.safetensors.index.json").read_text())
    tensors: dict[str, mx.array] = {}
    for shard in sorted(set(idx["weight_map"].values())):
        tensors.update(mx.load(str(bundle / shard)))
    return tensors


class BundleModel:
    def __init__(self, bundle: str | Path, verbose: bool = True):
        self.dir = Path(bundle)
        self.cfg = Dots3Config.load(self.dir)
        cfg_json = json.loads((self.dir / "config.json").read_text())
        self.overrides: dict = {
            k: v for k, v in cfg_json.get("quantization", {}).items()
            if isinstance(v, dict)}
        t0 = time.time()
        raw = _load_all_shards(self.dir)
        self.layers: list[dict] = []
        for i in range(self.cfg.num_hidden_layers):
            self.layers.append(self._layer_dict(raw, i))
        self.embed_q = self._qw_or_plain(raw, "model.embed_tokens.weight")
        self.head = self._qw_or_plain(raw, "lm_head.weight")
        self.final_norm = raw["model.norm.weight"].astype(mx.float32)
        # embeddings: dequantize once for row lookup
        if isinstance(self.embed_q, QW):
            e = self.embed_q
            self.embed = mx.dequantize(e.wq, e.scales, e.biases,
                                       group_size=e.group_size, bits=e.bits,
                                       mode=e.mode).astype(mx.bfloat16)
        else:
            self.embed = self.embed_q.astype(mx.bfloat16)
        mx.eval(self.embed)
        if verbose:
            print(f"bundle mapped in {time.time()-t0:.1f}s "
                  f"({len(raw)} tensors)", flush=True)
        self._raw = raw

    # -- weight plumbing ---------------------------------------------------
    def _qw_or_plain(self, raw: dict, name: str):
        stem = name[:-len(".weight")] if name.endswith(".weight") else name
        ov = self.overrides.get(stem)
        if ov and (stem + ".scales") in raw:
            return QW(raw[name], raw[stem + ".scales"],
                      raw.get(stem + ".biases"), ov["group_size"], ov["bits"],
                      ov.get("mode", "affine"))
        return raw[name]

    def _layer_dict(self, raw: dict, i: int) -> dict:
        p = f"model.layers.{i}."
        lw: dict = {}
        short = {
            "input_layernorm": p + "input_layernorm.weight",
            "post_attention_layernorm": p + "post_attention_layernorm.weight",
            "q_a_proj": p + "self_attn.q_a_proj.weight",
            "q_a_layernorm": p + "self_attn.q_a_layernorm.weight",
            "q_b_proj": p + "self_attn.q_b_proj.weight",
            "kv_a_proj_with_mqa": p + "self_attn.kv_a_proj_with_mqa.weight",
            "kv_a_layernorm": p + "self_attn.kv_a_layernorm.weight",
            "kv_b_proj": p + "self_attn.kv_b_proj.weight",
            "k_rope_only_layernorm": p + "self_attn.k_rope_only_layernorm.weight",
            "o_proj": p + "self_attn.o_proj.weight",
            "g_proj": p + "self_attn.g_proj.weight",
        }
        for k, name in short.items():
            v = self._qw_or_plain(raw, name)
            if "layernorm" in k and not isinstance(v, QW):
                v = v.astype(mx.float32)
            lw[k] = v
        if self.cfg.is_moe(i):
            lw["gate_w"] = raw[p + "mlp.gate.weight"]
            lw["gate_bias"] = raw[p + "mlp.gate.e_score_correction_bias"].astype(mx.float32)
            for proj, key in (("gate_proj", "experts_gate"),
                              ("up_proj", "experts_up"),
                              ("down_proj", "experts_down")):
                lw[key] = self._qw_or_plain(raw, p + f"mlp.switch_mlp.{proj}.weight")
            for proj, key in (("gate_proj", "shared_gate"),
                              ("up_proj", "shared_up"),
                              ("down_proj", "shared_down")):
                lw[key] = self._qw_or_plain(raw, p + f"mlp.shared_experts.{proj}.weight")
        else:
            for proj, key in (("gate_proj", "mlp_gate"), ("up_proj", "mlp_up"),
                              ("down_proj", "mlp_down")):
                lw[key] = self._qw_or_plain(raw, p + f"mlp.{proj}.weight")
        return lw

    # -- forward -------------------------------------------------------------
    def new_cache(self) -> list[dict]:
        return [{} for _ in range(self.cfg.num_hidden_layers)]

    def forward_embeds(self, x: mx.array, caches: list[dict] | None) -> mx.array:
        """x: [1, S, H] pre-built embeddings (multimodal path)."""
        from .ops import mla_attention, moe_forward, mlp_dense
        for i, lw in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            g = self.cfg.geom(i)
            h = rms_norm(x, lw["input_layernorm"], self.cfg.rms_norm_eps)
            x = x + mla_attention(h, lw, g, self.cfg.rms_norm_eps,
                                  self.cfg.apply_lora_rescale,
                                  self.cfg.k_rope_only_layernorm, cache=cache)
            h = rms_norm(x, lw["post_attention_layernorm"], self.cfg.rms_norm_eps)
            if self.cfg.is_moe(i):
                x = x + moe_forward(h, lw, self.cfg)
            else:
                B, S, H = h.shape
                x = x + mlp_dense(h.reshape(-1, H), lw["mlp_gate"],
                                  lw["mlp_up"], lw["mlp_down"]
                                  ).reshape(B, S, H).astype(x.dtype)
        h = rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        return linear(h[:, -1:], self.head).astype(mx.float32)

    def generate_from_embeds(self, x: mx.array, max_new: int = 96,
                             temperature: float = 0.0,
                             eos: tuple[int, ...] = EOS_DEFAULT,
                             verbose: bool = True) -> list[int]:
        """Prefill from prepared embeddings, then decode by token id."""
        import time
        caches = self.new_cache()
        t0 = time.time()
        mx.eval(self.forward_embeds(x, caches))
        out, cur = [], None
        for _ in range(max_new):
            logits = (self.forward_embeds(
                self.embed[mx.array([cur])][None], caches)
                if cur is not None else self.forward_embeds(x[:, -1:], caches))
            nxt = int(mx.argmax(logits[0, -1]).item())
            out.append(nxt)
            if nxt in eos:
                break
            cur = nxt
        if verbose:
            print(f"  prefill+decode {len(out)} tok in {time.time()-t0:.1f}s",
                  flush=True)
        return out

    def forward(self, ids: list[int], caches: list[dict] | None) -> mx.array:
        x = self.embed[mx.array(ids)][None]
        for i, lw in enumerate(self.layers):
            cache = caches[i] if caches is not None else None
            g = self.cfg.geom(i)
            h = rms_norm(x, lw["input_layernorm"], self.cfg.rms_norm_eps)
            from .ops import mla_attention, moe_forward, mlp_dense
            x = x + mla_attention(h, lw, g, self.cfg.rms_norm_eps,
                                  self.cfg.apply_lora_rescale,
                                  self.cfg.k_rope_only_layernorm, cache=cache)
            h = rms_norm(x, lw["post_attention_layernorm"], self.cfg.rms_norm_eps)
            if self.cfg.is_moe(i):
                x = x + moe_forward(h, lw, self.cfg)
            else:
                B, S, H = h.shape
                x = x + mlp_dense(h.reshape(-1, H), lw["mlp_gate"],
                                  lw["mlp_up"], lw["mlp_down"]
                                  ).reshape(B, S, H).astype(x.dtype)
        h = rms_norm(x, self.final_norm, self.cfg.rms_norm_eps)
        return linear(h[:, -1:], self.head).astype(mx.float32)

    # -- generation ------------------------------------------------------------
    def generate(self, ids: list[int], max_new: int = 128,
                 temperature: float = 0.0, top_p: float = 0.95,
                 eos: tuple[int, ...] = EOS_DEFAULT,
                 prefill_chunk: int = 512, verbose: bool = True) -> list[int]:
        caches = self.new_cache()
        t0 = time.time()
        for s in range(0, len(ids) - 1, prefill_chunk):
            chunk = ids[s: min(s + prefill_chunk, len(ids) - 1)]
            mx.eval(self.forward(chunk, caches))
        t_prefill = time.time() - t0
        out = []
        cur = [ids[-1]]
        t0 = time.time()
        for _ in range(max_new):
            logits = self.forward(cur, caches)[0, -1]
            if temperature <= 0:
                nxt = int(mx.argmax(logits).item())
            else:
                logits = logits / temperature
                probs = mx.softmax(logits, axis=-1)
                sp = mx.sort(probs, axis=-1)[::-1]
                cum = mx.cumsum(sp, axis=-1)
                cutoff = sp[int(mx.argmax(cum >= top_p).item())]
                probs = mx.where(probs >= cutoff, probs, 0.0)
                probs = probs / probs.sum()
                nxt = int(mx.random.categorical(mx.log(probs + 1e-20)).item())
            out.append(nxt)
            if nxt in eos:
                break
            cur = [nxt]
        dt = time.time() - t0
        if verbose:
            print(f"prefill {len(ids)-1} tok in {t_prefill:.1f}s | "
                  f"decode {len(out)} tok in {dt:.1f}s "
                  f"({len(out)/max(dt,1e-9):.1f} tok/s) | "
                  f"peak {mx.get_peak_memory()/1e9:.1f} GB", flush=True)
        return out
