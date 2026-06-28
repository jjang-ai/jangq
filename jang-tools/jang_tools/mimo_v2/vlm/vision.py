"""MiMo-V2.5 vision tower in MLX.

Faithful port of ``MiMoVisionTransformer`` from the source
``modeling_mimo_v2.py``. Checkpoint-truth notes (these differ from the source
class declarations and WILL break weight loading if "fixed"):

- ``merger.ln_q`` is LayerNorm with NO bias; ``merger.mlp.{0,2}`` have NO bias.
- block MLP gate/up/down and attention qkv/proj DO have bias.
- ``attn.sinks`` exists only on the 24 non-full-attention blocks; the sink is
  an additive logit bias on key position 0 (NOT an appended softmax column —
  that is the *text* model's sink convention, not the vision tower's).
- window attention is an additive |i-j| > window mask per image chunk, with
  column-major token reorder runs driven by ``vit_window_attn_types``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import VisionConfig


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    # q/k: [L, heads, head_dim]; cos/sin: [L, head_dim] -> [L, 1, head_dim]
    orig_dtype = q.dtype
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    cos = mx.expand_dims(cos.astype(mx.float32), -2)
    sin = mx.expand_dims(sin.astype(mx.float32), -2)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.astype(orig_dtype), k.astype(orig_dtype)


class MiMoVisionPatchEmbed(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        kernel = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nn.Conv3d(
            cfg.in_chans, cfg.hidden_size, kernel_size=kernel, stride=kernel, bias=False
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        cfg = self.cfg
        # input: [L, in_chans * temporal * patch * patch] flattened patches
        x = hidden_states.reshape(
            -1, cfg.in_chans, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size
        )
        # torch NCDHW -> mlx NDHWC
        x = x.transpose(0, 2, 3, 4, 1).astype(self.proj.weight.dtype)
        return self.proj(x).reshape(-1, cfg.hidden_size)


class MiMoVisionSwiGLUMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MiMoVisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig, use_sinks: bool):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.qk_channels
        self.scale = self.head_dim**-0.5
        self.window_size = cfg.visual_token_window_size
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, qkv_dim, bias=True)
        self.proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=True)
        self.use_sinks = use_sinks
        if use_sinks:
            self.sinks = mx.zeros((self.num_heads,))

    def _chunk_mask(self, length: int, full_attn: bool) -> mx.array | None:
        mask = None
        if not full_attn and self.window_size > 0:
            idx = mx.arange(length)
            dist = mx.abs(idx[:, None] - idx[None, :])
            mask = mx.where(dist > self.window_size, float("-inf"), 0.0)
            mask = mx.broadcast_to(mask[None, None], (1, self.num_heads, length, length))
        if self.use_sinks:
            # additive bias on key position 0 only
            col0 = mx.broadcast_to(self.sinks.reshape(1, self.num_heads, 1, 1), (1, self.num_heads, length, 1))
            rest = mx.zeros((1, self.num_heads, length, length - 1))
            sink = mx.concatenate([col0, rest], axis=-1)
            mask = sink if mask is None else mask + sink
        return mask

    def __call__(
        self,
        x: mx.array,
        chunk_lengths: list[int],
        position_embeddings: tuple[mx.array, mx.array],
        full_attn: bool,
    ) -> mx.array:
        seq_len = x.shape[0]
        qkv = self.qkv(x)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].reshape(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim : q_dim + kv_dim].reshape(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim :].reshape(seq_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)

        outputs = []
        start = 0
        for length in chunk_lengths:
            q_c = q[start : start + length].transpose(1, 0, 2)[None]  # [1, H, L, D]
            k_c = k[start : start + length].transpose(1, 0, 2)[None]
            v_c = v[start : start + length].transpose(1, 0, 2)[None]
            mask = self._chunk_mask(length, full_attn)
            if mask is not None:
                mask = mask.astype(q_c.dtype)
            out = mx.fast.scaled_dot_product_attention(q_c, k_c, v_c, scale=self.scale, mask=mask)
            outputs.append(out[0].transpose(1, 0, 2).reshape(length, -1))
            start += length
        return self.proj(mx.concatenate(outputs, axis=0))


class MiMoVisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig, use_sinks: bool):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.attn = MiMoVisionAttention(cfg, use_sinks=use_sinks)
        self.mlp = MiMoVisionSwiGLUMLP(cfg.hidden_size, cfg.intermediate_size)

    def __call__(self, x, chunk_lengths, position_embeddings, full_attn):
        x = x + self.attn(self.norm1(x), chunk_lengths, position_embeddings, full_attn)
        x = x + self.mlp(self.norm2(x))
        return x


class MiMoVisionPatchMerger(nn.Module):
    """Checkpoint truth: ln_q and both mlp linears carry NO bias."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.unit_size = cfg.hidden_size * (cfg.spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(cfg.hidden_size, eps=1e-6, affine=True, bias=False)
        self.mlp = [
            nn.Linear(self.unit_size, self.unit_size, bias=False),
            nn.GELU(),
            nn.Linear(self.unit_size, cfg.out_hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.ln_q(x).reshape(-1, self.unit_size)
        x = self.mlp[0](x)
        x = self.mlp[1](x)
        return self.mlp[2](x)


class VisionModel(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.spatial_merge_unit = cfg.spatial_merge_size**2
        self.fullatt_block_indexes = list(cfg.fullatt_block_indexes)
        self.window_attn_types = cfg.resolved_window_attn_types()
        self.patch_embed = MiMoVisionPatchEmbed(cfg)
        self.blocks = [
            MiMoVisionBlock(cfg, use_sinks=cfg.use_sink and (i not in self.fullatt_block_indexes))
            for i in range(cfg.depth)
        ]
        self.merger = MiMoVisionPatchMerger(cfg)
        half = cfg.qk_channels // 2
        self._inv_freq = 1.0 / (10000.0 ** (mx.arange(0, half, 2, dtype=mx.float32) / half))

    # --- position / index helpers -----------------------------------------

    def _rot_pos_ids(self, grid_thw: list[tuple[int, int, int]]) -> mx.array:
        pos = []
        m = self.spatial_merge_size
        for t, h, w in grid_thw:
            hpos = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
            hpos = hpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            wpos = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
            wpos = wpos.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            ids = mx.stack([hpos, wpos], axis=-1)
            pos.append(mx.tile(ids, (t, 1)))
        return mx.concatenate(pos, axis=0)

    def _rot_pos_emb(self, grid_thw: list[tuple[int, int, int]]) -> mx.array:
        pos_ids = self._rot_pos_ids(grid_thw)
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        freqs = mx.outer(mx.arange(max_grid, dtype=mx.float32), self._inv_freq)
        emb = freqs[pos_ids].reshape(pos_ids.shape[0], -1)  # [L, head_dim//2]
        return mx.concatenate([emb, emb], axis=-1)  # [L, head_dim]

    def _window_index_1d(self, grid_thw: list[tuple[int, int, int]], col: bool = True) -> mx.array:
        out = []
        offset = 0
        m = self.spatial_merge_size
        for t, h, w in grid_thw:
            lh, lw = h // m, w // m
            index = mx.arange(t * lh * lw).reshape(t, lh, lw)
            index_new = index.transpose(0, 2, 1).reshape(-1) if col else index.reshape(-1)
            out.append(index_new + offset)
            offset += t * lh * lw
        return mx.concatenate(out, axis=0)

    def _apply_index(self, tensor: mx.array, index: mx.array) -> mx.array:
        t = tensor.reshape(-1, self.spatial_merge_unit, *tensor.shape[1:])
        return t[index].reshape(-1, *tensor.shape[1:])

    # --- forward ------------------------------------------------------------

    def __call__(self, pixel_values: mx.array, grid_thw) -> mx.array:
        grid = [tuple(int(v) for v in row) for row in grid_thw]
        x = self.patch_embed(pixel_values)

        emb = self._rot_pos_emb(grid)
        window_index = self._window_index_1d(grid, col=True)
        reverse_index = mx.argsort(window_index)

        row_pe = (mx.cos(emb), mx.sin(emb))
        col_emb = self._apply_index(emb, window_index)
        col_pe = (mx.cos(col_emb), mx.sin(col_emb))

        chunk_lengths = []
        for t, h, w in grid:
            chunk_lengths.extend([h * w] * t)

        types = self.window_attn_types
        for i, blk in enumerate(self.blocks):
            wt = types[i]
            if wt == 1 and (i == 0 or types[i - 1] != 1):
                x = self._apply_index(x, window_index)
            if i > 0 and wt != 1 and types[i - 1] == 1:
                x = self._apply_index(x, reverse_index)
            pe = col_pe if wt == 1 else row_pe
            x = blk(x, chunk_lengths, pe, full_attn=i in self.fullatt_block_indexes)

        return self.merger(x)

    # --- weights ------------------------------------------------------------

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        out = {}
        for k, v in weights.items():
            if (
                k.endswith("patch_embed.proj.weight")
                and v.ndim == 5
                and v.shape[-1] != self.cfg.in_chans
            ):
                # torch OIDHW -> mlx ODHWI (idempotent: skip if already ODHWI)
                v = v.transpose(0, 2, 3, 4, 1)
            out[k] = v
        return out
