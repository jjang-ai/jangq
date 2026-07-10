"""Hy3-preview MLX model — wrapper over mlx_lm.models.dots1 with Hy3-specific
MoE layout (`mlp.switch_mlp.*`) so JANGTQ-quantized routed experts
hydrate cleanly through the TurboQuant kernel path.

Architecture is otherwise identical to dots1: GQA + qk_norm + RoPE,
sigmoid router with `e_score_correction_bias` (DSV3-style aux-free
balancing), shared expert, dense layer 0 (`first_k_dense_replace=1`).

Key differences vs dots1:
- Hy3 ships routed experts at `mlp.experts.{e}.{proj}.weight`; dots1
  also uses per-expert source tensors but stacks them into
  `mlp.experts.{proj}.weight`. JANGTQ's loader stacks instead into
  `mlp.switch_mlp.{proj}.tq_packed`. Hy3 follows JANGTQ here.
- Field names in `config.json`: `num_experts`, `num_shared_experts`,
  `route_norm`, `router_scaling_factor`, `rope_parameters` — remapped
  to dots1's expected names in `ModelArgs.from_dict`.
- Layer N (MTP) is dropped at sanitize for the first runtime pass
  (`mtp_mode='preserved_disabled'`).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import dots1
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import SwitchGLU


_HY3_TO_DOTS1_FIELD_MAP = {
    "num_experts": "n_routed_experts",
    "num_shared_experts": "n_shared_experts",
    "route_norm": "norm_topk_prob",
    "router_scaling_factor": "routed_scaling_factor",
}


@dataclass
class ModelArgs(dots1.ModelArgs):
    # 1 for Hy3 final (model.layers.80 = DSV3-style MTP layer). Kept on args so
    # the vmlx native-MTP patch can attach the head; 0 disables cleanly.
    num_nextn_predict_layers: int = 0

    @classmethod
    def from_dict(cls, params):
        remapped = {}
        for k, v in params.items():
            remapped[_HY3_TO_DOTS1_FIELD_MAP.get(k, k)] = v
        if "rope_parameters" in remapped and isinstance(remapped["rope_parameters"], dict):
            rope = remapped.pop("rope_parameters")
            remapped.setdefault("rope_theta", rope.get("rope_theta", 10000.0))
        remapped.setdefault("n_group", 1)
        remapped.setdefault("topk_group", 1)
        return super().from_dict(remapped)


def build_args_from_hy3_config(cfg: dict) -> ModelArgs:
    return ModelArgs.from_dict(cfg)


class Hy3HeadRMSNorm(nn.Module):
    """Per-head RMSNorm that accepts either flat or pre-reshaped queries/keys.

    The original `Dots1Attention.__call__` reshapes q/k into per-head shape
    `[..., n_heads, head_dim]` before applying `nn.RMSNorm(head_dim)`. The
    JANGTQ P18 QKV-fusion patch replaces `__call__` and applies `q_norm`/
    `k_norm` BEFORE the reshape — feeding `[..., n_heads * head_dim]` into
    a 128-dim RMSNorm would silently normalize only the last 128 of 8192
    elements.

    This norm reshapes flat input to `[..., n_heads, head_dim]`, applies
    RMSNorm over `head_dim`, and reshapes back. Pre-reshaped input passes
    through directly. Stored under the same `weight` parameter name as
    `nn.RMSNorm` so the bundle's `q_norm.weight`/`k_norm.weight` tensors
    load without remapping.
    """

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((head_dim,))
        self.eps = eps
        self.head_dim = head_dim

    def __call__(self, x: mx.array) -> mx.array:
        head_dim = self.head_dim
        if x.shape[-1] != head_dim:
            n_heads = x.shape[-1] // head_dim
            x = x.reshape(*x.shape[:-1], n_heads, head_dim)
            x = mx.fast.rms_norm(x, self.weight, self.eps)
            return x.reshape(*x.shape[:-2], n_heads * head_dim)
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Hy3Attention(dots1.Dots1Attention):
    """`Dots1Attention` with the `use_qk_norm=True` flag that JANGTQ's P18
    QKV-fusion patch checks for, and per-head RMSNorms that auto-reshape
    flat-input tensors so the fused path produces the same output as the
    original reshape-then-normalize path."""

    def __init__(self, args: ModelArgs):
        super().__init__(args)
        head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        # Replace the parent's `nn.RMSNorm(head_dim)` instances with norms
        # that handle both flat and reshaped inputs.
        self.q_norm = Hy3HeadRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = Hy3HeadRMSNorm(head_dim, eps=args.rms_norm_eps)
        self.use_qk_norm = True


class Hy3MoE(nn.Module):
    """Sigmoid router + expert_bias + 1 shared expert MoE.

    The SwitchGLU sits at `mlp.switch_mlp.{gate_proj,up_proj,down_proj}` —
    matches the path JANGTQ's `_hydrate_jangtq_model` expects when it
    stacks per-expert quantized tensors and calls back to replace the
    target with a TurboQuantLinear-backed SwitchGLU.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.n_shared_experts = args.n_shared_experts

        self.gate = dots1.Dots1TopkRouter(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
        )
        self.shared_experts = dots1.Dots1MLP(
            args=args,
            intermediate_size=args.moe_intermediate_size * args.n_shared_experts,
        )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class Hy3DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Hy3Attention(args)
        if layer_idx >= args.first_k_dense_replace:
            self.mlp = Hy3MoE(args)
        else:
            self.mlp = dots1.Dots1MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Hy3MTPLayer(nn.Module):
    """Hy3 native MTP head (model.layers.80 re-namespaced to mtp.{i}.* by the
    JANG converter). Mirrors vLLM's HYV3MultiTokenPredictorLayer exactly:

        fused = eh_proj(concat([enorm(embed(next_ids)), hnorm(hidden)], -1))
        out   = final_layernorm(decoder_block(fused))

    where hidden is the backbone's PRE-final-norm last-layer output and
    decoder_block is a full Hy3 layer (GQA + qk_norm + sigmoid-router MoE +
    shared expert). Output feeds the SHARED base lm_head (fp32 contract).
    Recursive drafting feeds the post-final-norm output back as `hidden`
    (same as vLLM's spec_step recursion — it gets hnorm'ed again)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
        # layer_idx >= first_k_dense_replace -> MoE mlp (the MTP layer is MoE)
        self.block = Hy3DecoderLayer(args, layer_idx=args.num_hidden_layers)
        self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        next_token_ids: mx.array,
        embed_tokens: nn.Module,
        cache: Optional[Any] = None,
    ) -> mx.array:
        e = self.enorm(embed_tokens(next_token_ids))
        hn = self.hnorm(hidden)
        fused = self.eh_proj(mx.concatenate([e, hn], axis=-1))
        mask = create_attention_mask(fused, cache)
        out = self.block(fused, mask, cache)
        return self.final_layernorm(out)


class Hy3InnerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Hy3DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        """Returns the PRE-final-norm hidden state; ``Model.__call__`` applies
        ``self.norm`` before the lm_head so the MTP head can fuse the raw
        backbone hidden (DSV3/vLLM contract)."""
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Hy3InnerModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def _head_logits(self, normed: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(normed)
        # `enable_lm_head_fp32=True` in Hy3 config — accumulate the 4096-dim
        # contraction in fp32. Mirror DSV4's pattern: dequantize the
        # quantized lm_head weight, then matmul in fp32. Without this the
        # bf16 accumulation drifts logits by ~0.5/elem on the dim=4096
        # contraction, enough to flip plausible top-k token picks toward
        # high-baseline-energy junk tokens (doubled letters, etc).
        if hasattr(self.lm_head, "scales"):
            w_f = mx.dequantize(
                self.lm_head.weight,
                self.lm_head.scales,
                getattr(self.lm_head, "biases", None),
                group_size=self.lm_head.group_size,
                bits=self.lm_head.bits,
                mode=getattr(self.lm_head, "mode", "affine"),
            ).astype(mx.float32)
        else:
            w_f = self.lm_head.weight.astype(mx.float32)
        return normed.astype(mx.float32) @ w_f.T

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        return_hidden: bool = False,
        return_logits: bool = True,
        n_confirmed: int = 0,  # accepted for driver parity; Hy3 is pure-KV
    ):
        hidden = self.model(inputs, cache)
        if not return_logits:
            return hidden
        logits = self._head_logits(self.model.norm(hidden))
        if return_hidden:
            return logits, hidden
        return logits

    # ── native MTP surface (vmlx batch_generator contract) ──
    # `self.mtp` is only attached by vmlx's mlx_lm_mtp patch (gated on
    # is_mtp_active()); with it absent this model is indistinguishable from a
    # stock no-MTP build and sanitize strips mtp.* weights.

    def attach_mtp(self) -> None:
        n = int(getattr(self.args, "num_nextn_predict_layers", 0) or 0)
        if n <= 0:
            raise ValueError("num_nextn_predict_layers=0 — no MTP head to attach")
        self.mtp = [Hy3MTPLayer(self.args) for _ in range(n)]

    def mtp_forward(self, hidden_states, next_token_ids, mtp_cache,
                    return_hidden: bool = False):
        h = hidden_states
        if mtp_cache is None:
            mtp_cache = [None] * len(self.mtp)
        for blk, c in zip(self.mtp, mtp_cache):
            h = blk(h, next_token_ids, self.model.embed_tokens, c)
        logits = self._head_logits(h)
        if return_hidden:
            return logits, h
        return logits

    def make_mtp_cache(self):
        if hasattr(self, "mtp"):
            from mlx_lm.models.cache import KVCache

            return [KVCache() for _ in self.mtp]
        return []

    @property
    def layers(self):
        return self.model.layers

    def sanitize(self, weights):
        N = self.args.num_hidden_layers
        n_experts = self.args.n_routed_experts

        # 1) MTP weights. New JANG bundles ship the head re-namespaced to
        #    mtp.{i}.* with FINAL param names (they load directly onto
        #    Hy3MTPLayer). Keep them only when the head is attached (vmlx
        #    native-MTP patch); otherwise strip. Legacy source-style
        #    model.layers.{N}.* MTP tensors (preview bundles, raw bf16
        #    source) are always dropped — the head is never loaded from them.
        weights = {
            k: v for k, v in weights.items()
            if not k.startswith(f"model.layers.{N}.")
        }
        if not hasattr(self, "mtp"):
            weights = {
                k: v for k, v in weights.items() if not k.startswith("mtp.")
            }

        # 2) Rename hy_v3-specific tensor keys onto our model's expected paths.
        for layer in range(self.args.first_k_dense_replace, N):
            prefix = f"model.layers.{layer}.mlp"

            # router.gate.weight -> gate.weight (dots1.Dots1TopkRouter expects `weight`)
            for suf in ("weight", "scales", "biases"):
                src = f"{prefix}.router.gate.{suf}"
                dst = f"{prefix}.gate.{suf}"
                if src in weights:
                    weights[dst] = weights.pop(src)

            # expert_bias -> gate.e_score_correction_bias
            src = f"{prefix}.expert_bias"
            dst = f"{prefix}.gate.e_score_correction_bias"
            if src in weights:
                weights[dst] = weights.pop(src)

            # shared_mlp -> shared_experts (per-projection, all suffixes)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for suf in ("weight", "scales", "biases"):
                    src = f"{prefix}.shared_mlp.{proj}.{suf}"
                    dst = f"{prefix}.shared_experts.{proj}.{suf}"
                    if src in weights:
                        weights[dst] = weights.pop(src)

            # 3) Stack per-expert routed weights into SwitchGLU under
            #    `switch_mlp.{proj}`. Drop quant metadata that
            #    SwitchGLU doesn't take directly — the JANGTQ load path
            #    re-attaches `tq_packed/tq_norms/tq_bits` through
            #    TurboQuantLinear after this sanitize.
            for proj in ("gate_proj", "up_proj", "down_proj"):
                target_w = f"{prefix}.switch_mlp.{proj}.weight"
                if target_w not in weights:
                    parts_w = []
                    parts_s = []
                    parts_b = []
                    ok = True
                    for e in range(n_experts):
                        wkey = f"{prefix}.experts.{e}.{proj}.weight"
                        if wkey not in weights:
                            ok = False
                            break
                        parts_w.append(weights.pop(wkey))
                        skey = f"{prefix}.experts.{e}.{proj}.scales"
                        bkey = f"{prefix}.experts.{e}.{proj}.biases"
                        if skey in weights:
                            parts_s.append(weights.pop(skey))
                        if bkey in weights:
                            parts_b.append(weights.pop(bkey))
                    if ok:
                        weights[target_w] = mx.stack(parts_w)
                        if parts_s and len(parts_s) == n_experts:
                            weights[f"{prefix}.switch_mlp.{proj}.scales"] = mx.stack(parts_s)
                        if parts_b and len(parts_b) == n_experts:
                            weights[f"{prefix}.switch_mlp.{proj}.biases"] = mx.stack(parts_b)
                # Drop tq_* — re-attached by load_jangtq's TurboQuant replace pass.
                for e in range(n_experts):
                    for suf in ("tq_packed", "tq_norms", "tq_bits"):
                        weights.pop(f"{prefix}.experts.{e}.{proj}.{suf}", None)

        return {k: v for k, v in weights.items() if "rotary_emb.inv_freq" not in k}


def register_mlx_lm_hy3() -> None:
    """Alias this module under `mlx_lm.models.hy_v3` so mlx_lm.utils.load_model
    can resolve `model_type='hy_v3'` to this Model class."""
    sys.modules.setdefault("mlx_lm.models.hy_v3", sys.modules[__name__])
