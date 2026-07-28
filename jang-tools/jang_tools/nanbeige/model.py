"""MLX implementation of Nanbeige 4.x — a **Looped Transformer**.

Reference: `modeling_nanbeige.py` from `Nanbeige/Nanbeige4.2-3B` (rev fab06df).

The per-layer math is byte-for-byte Llama: GQA attention (no bias, no
qk-layernorm), SwiGLU MLP, pre-norm RMSNorm residual blocks, NeoX-style
half-rotation RoPE.  The *only* architectural difference — and the one that
breaks every stock loader — is the loop:

    h = embed(x)
    for loop in range(num_loops):            # 2 for Nanbeige4.2-3B
        for i in range(num_hidden_layers):   # 22, SHARED weights every loop
            h = layers[i](h, cache=cache[i + loop * num_hidden_layers])
        if not skip_loop_final_norm:
            h = model.norm(h)                # final norm runs EVERY loop
    if skip_loop_final_norm:
        h = model.norm(h)
    logits = lm_head(h)

Two consequences a port MUST honour:

1. **The KV cache has `num_hidden_layers * num_loops` slots**, not
   `num_hidden_layers`.  HF derives the slot as
   `layer_idx + loop_idx * num_hidden_layers`
   (`_get_loop_cache_layer_idx`).  Each loop pass attends only to the keys
   *its own pass* wrote.  Sharing one cache across loops silently doubles the
   sequence each layer sees and destroys long-context behaviour.
2. **`position_ids` / RoPE offsets are identical in every loop.**  The loop
   adds depth, not positions.  The offset comes from the cache slot of the
   loop being executed, and every slot advances in lockstep, so one mask
   computed once is valid for all loops.

Config keys that would change the above are all disabled on the shipped
4.2-3B checkpoint and are rejected loudly here rather than silently ignored:
`enable_double_loop_split` (LoopSplit), `enable_hyper_connection` / `enable_mhc`,
`enable_depth_attention`, `emb_neighbor_num` (n-gram embeddings), `loop_share_kv`,
`qk_layernorm`.  Nanbeige4.5 is announced to use them — fail, don't guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False

    # ── Looped Transformer ────────────────────────────────────────────
    num_loops: int = 1
    loop_loss_weights: Optional[List[float]] = None
    skip_loop_final_norm: bool = False

    # ── Features present in modeling_nanbeige.py but OFF in 4.2-3B ────
    # Kept so `from_dict` doesn't drop them and the guard below can fire.
    enable_double_loop_split: bool = False
    loop_middle_layers: Optional[int] = None
    loop_share_kv: bool = False
    enable_hyper_connection: bool = False
    enable_mhc: bool = False
    enable_depth_attention: bool = False
    emb_neighbor_num: Optional[int] = None
    qk_layernorm: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        unsupported = [
            name
            for name, on in (
                ("enable_double_loop_split", self.enable_double_loop_split),
                ("loop_share_kv", self.loop_share_kv),
                ("enable_hyper_connection", self.enable_hyper_connection),
                ("enable_mhc", self.enable_mhc),
                ("enable_depth_attention", self.enable_depth_attention),
                ("qk_layernorm", self.qk_layernorm),
                ("emb_neighbor_num", self.emb_neighbor_num is not None),
            )
            if on
        ]
        if unsupported:
            raise ValueError(
                "nanbeige: this checkpoint enables architecture features this "
                f"runtime does not implement: {', '.join(unsupported)}. "
                "Port them before loading — do not ignore."
            )

    @property
    def total_loops(self) -> int:
        """Loops actually executed. Mirrors `NanbeigeModel._get_num_loops`."""
        if self.loop_loss_weights:
            return len(self.loop_loss_weights) + 1
        return max(1, int(self.num_loops))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        # NOTE: n_heads * head_dim (48*128 = 6144) != hidden_size (3072).
        # q_proj/o_proj are "wide"; never derive head_dim from hidden_size.
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=args.mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=args.mlp_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class NanbeigeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.num_loops = args.total_loops
        self.skip_loop_final_norm = args.skip_loop_final_norm
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        n_slots = self.num_hidden_layers * self.num_loops
        if cache is None:
            cache = [None] * n_slots
        elif len(cache) != n_slots:
            raise ValueError(
                f"nanbeige: cache must have num_hidden_layers * num_loops = "
                f"{n_slots} slots (got {len(cache)}). Slot index is "
                f"layer_idx + loop_idx * {self.num_hidden_layers}."
            )

        # All loops share one mask: every slot carries the same offset because
        # each loop appends exactly the same token span at the same positions.
        mask = create_attention_mask(h, cache[0])

        for loop_idx in range(self.num_loops):
            base = loop_idx * self.num_hidden_layers
            for i, layer in enumerate(self.layers):
                h = layer(h, mask, cache=cache[base + i])
            if not self.skip_loop_final_norm:
                # Final norm applied at the END OF EVERY LOOP; the normed
                # output re-enters layer 0 of the next loop.
                h = self.norm(h)

        if self.skip_loop_final_norm:
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NanbeigeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def make_cache(self, max_kv_size: Optional[int] = None):
        """One cache slot per (layer, loop) — `num_hidden_layers * num_loops`."""
        n_slots = self.args.num_hidden_layers * self.args.total_loops
        if max_kv_size is not None:
            return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(n_slots)]
        return [KVCache() for _ in range(n_slots)]

    @property
    def layers(self):
        return self.model.layers

    @property
    def cache_slots(self) -> int:
        return self.args.num_hidden_layers * self.args.total_loops

    @property
    def quant_predicate(self):
        return None
