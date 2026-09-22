"""MiMo-V2.6 vision tower (MLX) + image/video preprocessing (numpy).

Written from Xiaomi's reference ``modeling_mimo_v2.py`` (``MiMoVisionTransformer``)
shipped with ``XiaomiMiMo/MiMo-V2.6-Flash-RL`` and cross-checked against the
serving stacks that the model card points to:

* vLLM   ``vllm/model_executor/models/mimo_v2_omni.py`` +
         ``vllm/transformers_utils/processors/mimo_v2_omni.py``
* SGLang ``python/sglang/srt/models/mimo_vl.py`` +
         ``python/sglang/srt/multimodal/processors/mimo_v2.py``

Nothing here is derived from the older MiMo-V2.5 code in this package.
See ``docs/runtime/mimo-v26-flash-2026-09-22/VISION.md`` for the measured
parity against the torch reference and the list of reference/serving
disagreements (merger norm, sink semantics, preprocessing constants).

Tower summary (config.json ``vision_config``)::

    patches [N, 3*2*16*16] --Conv3d(k=stride=(2,16,16), no bias)--> [N, 1280]
    28 x block: x += attn(RMSNorm(x)); x += SwiGLU(RMSNorm(x))
        attn: GQA 32 q / 8 kv heads, head_dim 64, qkv+proj biased,
              2-D rotary (h, w) on all 64 dims (neox rotate-half),
              per-frame sequences (cu_seqlens), and per layer
              vit_window_attn_types[i]:
                -1 : full attention (blocks 0, 9, 18, 27), no sink
                 0 : sliding window |i-j| <= 64 in ROW-major merge-unit order
                 1 : sliding window |i-j| <= 64 in COLUMN-major merge-unit order
              window layers carry ``sinks[h]`` added to the logit of key 0
              of every sequence (reference + vLLM ``sinks_bias_key0``).
    merger: norm(1280) -> view(-1, 5120) -> Linear(5120,5120) -> GELU(erf)
            -> Linear(5120, 4096); no biases in the checkpoint.

Placeholder tokens per image: ``t*h*w // 4`` with t = 1 (a still image is
duplicated to 2 frames and occupies one temporal patch).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # MLX is only needed for the tower; preprocessing is pure numpy.
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover
    mx = None
    nn = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MiMoV26VisionConfig:
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    num_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64  # reference: getattr(config, "qk_channels", 64)
    out_hidden_size: int = 4096
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    in_channels: int = 3
    rms_norm_eps: float = 1e-6
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [0, 9, 18, 27])
    vit_window_attn_types: list[int] = field(default_factory=list)
    visual_token_window_size: int = 64
    use_sink: bool = True
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, vc: dict) -> "MiMoV26VisionConfig":
        depth = int(vc.get("depth", 28))
        return cls(
            depth=depth,
            hidden_size=int(vc["hidden_size"]),
            intermediate_size=int(vc["intermediate_size"]),
            num_heads=int(vc["num_heads"]),
            num_key_value_heads=int(vc.get("num_key_value_heads", vc["num_heads"])),
            head_dim=int(vc.get("qk_channels", 64)),
            out_hidden_size=int(vc["out_hidden_size"]),
            patch_size=int(vc["patch_size"]),
            temporal_patch_size=int(vc["temporal_patch_size"]),
            spatial_merge_size=int(vc.get("spatial_merge_size", 2)),
            in_channels=int(vc.get("in_channels") or vc.get("in_chans", 3)),
            rms_norm_eps=float(vc.get("rms_norm_eps", 1e-6)),
            fullatt_block_indexes=list(vc.get("fullatt_block_indexes", [])),
            vit_window_attn_types=list(vc.get("vit_window_attn_types") or [-1] * depth),
            visual_token_window_size=int(vc.get("visual_token_window_size", -1)),
            use_sink=bool(vc.get("use_sink", False)),
        )

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "MiMoV26VisionConfig":
        cfg = json.loads((Path(path) / "config.json").read_text())
        return cls.from_dict(cfg["vision_config"])


# ---------------------------------------------------------------------------
# Index / rotary helpers (numpy, shared by tower and tests)
# ---------------------------------------------------------------------------


def _grid_list(grid_thw) -> list[tuple[int, int, int]]:
    arr = np.asarray(grid_thw, dtype=np.int64).reshape(-1, 3)
    return [(int(t), int(h), int(w)) for t, h, w in arr]


def window_index_col(grid_thw, merge: int = 2) -> np.ndarray:
    """Merge-unit permutation row-major -> column-major (reference get_window_index_1d(col=True))."""
    out, base = [], 0
    for t, h, w in _grid_list(grid_thw):
        lh, lw = h // merge, w // merge
        idx = np.arange(t * lh * lw).reshape(t, lh, lw).transpose(0, 2, 1).reshape(-1)
        out.append(idx + base)
        base += t * lh * lw
    return np.concatenate(out)


def rotary_cos_sin(grid_thw, head_dim: int = 64, merge: int = 2, theta: float = 10000.0):
    """float32 (cos, sin) of shape [N, head_dim] in row-major merge-unit order."""
    rot_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
    grids = _grid_list(grid_thw)
    max_grid = max(max(h, w) for _, h, w in grids)
    table = np.outer(np.arange(max_grid, dtype=np.float32), inv_freq).astype(np.float32)
    pos = []
    for t, h, w in grids:
        hp = np.broadcast_to(np.arange(h)[:, None], (h, w))
        wp = np.broadcast_to(np.arange(w)[None, :], (h, w))

        def _m(a):
            return a.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)

        pos.append(np.tile(np.stack([_m(hp), _m(wp)], axis=-1), (t, 1)))
    pos = np.concatenate(pos)
    freqs = table[pos].reshape(pos.shape[0], -1)  # [N, rot_dim]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [N, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def frame_seqlens(grid_thw) -> list[int]:
    """Attention sequences: one per temporal patch of every item (reference cu_seqlens)."""
    out = []
    for t, h, w in _grid_list(grid_thw):
        out += [h * w] * t
    return out


def merged_token_count(grid_thw, merge: int = 2) -> int:
    return sum(t * h * w for t, h, w in _grid_list(grid_thw)) // (merge * merge)


# ---------------------------------------------------------------------------
# MLX tower
# ---------------------------------------------------------------------------

if nn is not None:

    def _rotate_half(x):
        h = x.shape[-1] // 2
        return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    class MiMoV26VisionAttention(nn.Module):
        def __init__(self, cfg: MiMoV26VisionConfig, use_sink: bool):
            super().__init__()
            self.num_heads = cfg.num_heads
            self.num_kv_heads = cfg.num_key_value_heads
            self.head_dim = cfg.head_dim
            self.scale = self.head_dim ** -0.5
            self.window = cfg.visual_token_window_size
            qkv_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
            self.qkv = nn.Linear(cfg.hidden_size, qkv_dim, bias=True)
            self.proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=True)
            if use_sink:
                self.sinks = mx.zeros((self.num_heads,))
            # "key0": reference/vLLM semantics (sinks[h] added to key-0 logit).
            # "virtual": SGLang FA3 semantics (extra null logit in the denominator).
            # "off": ignore sinks (diagnostic only; what vLLM did before PR #49815).
            self.sink_mode = "key0"
            self.q_block = 512

        def _attend(self, q, k, v, full_attn: bool):
            """q [1,H,L,D], k/v [1,Hkv,L,D] -> [1,H,L,D] for ONE sequence."""
            L = q.shape[2]
            sinks = self.get("sinks") if self.sink_mode != "off" else None  # Module is a dict
            windowed = (not full_attn) and self.window > 0 and L > self.window + 1
            if sinks is not None and self.sink_mode == "virtual":
                s = sinks.astype(mx.float32)
                if not windowed:
                    return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, sinks=s)
            if not windowed and sinks is None:
                return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

            outs = []
            w = self.window if windowed else L
            B = self.q_block if windowed else L
            for s0 in range(0, L, B):
                s1 = min(L, s0 + B)
                k0, k1 = max(0, s0 - w), min(L, s1 + w)
                qi = np.arange(s0, s1)[:, None]
                kj = np.arange(k0, k1)[None, :]
                bias = np.where(np.abs(qi - kj) > w, -np.inf, 0.0).astype(np.float32)
                bias = mx.array(bias)[None, None]  # [1,1,Lq,Lk]
                if sinks is not None and self.sink_mode == "key0" and k0 == 0:
                    col0 = mx.zeros((1, 1, 1, k1 - k0)).at[..., 0].add(1.0)
                    bias = bias + col0 * sinks.astype(mx.float32).reshape(1, -1, 1, 1)
                bias = bias.astype(q.dtype)
                kw = {}
                if sinks is not None and self.sink_mode == "virtual":
                    kw["sinks"] = sinks.astype(mx.float32)
                outs.append(
                    mx.fast.scaled_dot_product_attention(
                        q[:, :, s0:s1], k[:, :, k0:k1], v[:, :, k0:k1], scale=self.scale, mask=bias, **kw
                    )
                )
            return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)

        def __call__(self, x, seqlens: Sequence[int], cos, sin, full_attn: bool):
            N = x.shape[0]
            qkv = self.qkv(x)
            qd, kd = self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim
            q = qkv[:, :qd].reshape(N, self.num_heads, self.head_dim)
            k = qkv[:, qd : qd + kd].reshape(N, self.num_kv_heads, self.head_dim)
            v = qkv[:, qd + kd :].reshape(N, self.num_kv_heads, self.head_dim)
            # rotary in fp32, cast back (reference _apply_rotary_pos_emb_vision)
            c, s = cos[:, None, :], sin[:, None, :]
            qf, kf = q.astype(mx.float32), k.astype(mx.float32)
            q = (qf * c + _rotate_half(qf) * s).astype(x.dtype)
            k = (kf * c + _rotate_half(kf) * s).astype(x.dtype)

            outs, off = [], 0
            for L in seqlens:
                qc = q[off : off + L].transpose(1, 0, 2)[None]
                kc = k[off : off + L].transpose(1, 0, 2)[None]
                vc = v[off : off + L].transpose(1, 0, 2)[None]
                o = self._attend(qc, kc, vc, full_attn)
                outs.append(o[0].transpose(1, 0, 2).reshape(L, -1))
                off += L
            o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
            return self.proj(o)

    class MiMoV26VisionMLP(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.gate_proj = nn.Linear(dim, hidden, bias=True)
            self.up_proj = nn.Linear(dim, hidden, bias=True)
            self.down_proj = nn.Linear(hidden, dim, bias=True)

        def __call__(self, x):
            return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

    class MiMoV26VisionBlock(nn.Module):
        def __init__(self, cfg: MiMoV26VisionConfig, use_sink: bool):
            super().__init__()
            self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.norm2 = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
            self.attn = MiMoV26VisionAttention(cfg, use_sink)
            self.mlp = MiMoV26VisionMLP(cfg.hidden_size, cfg.intermediate_size)

        def __call__(self, x, seqlens, cos, sin, full_attn):
            x = x + self.attn(self.norm1(x), seqlens, cos, sin, full_attn)
            return x + self.mlp(self.norm2(x))

    class MiMoV26PatchEmbed(nn.Module):
        """Conv3d with kernel == stride == (2,16,16) is a Linear over (C,T,P,P)."""

        def __init__(self, cfg: MiMoV26VisionConfig):
            super().__init__()
            k = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
            self.proj = nn.Linear(k, cfg.hidden_size, bias=False)

        def __call__(self, x):
            return self.proj(x)

    class MiMoV26PatchMerger(nn.Module):
        def __init__(self, cfg: MiMoV26VisionConfig, norm: str = "rms"):
            super().__init__()
            self.hidden = cfg.hidden_size * cfg.spatial_merge_size ** 2
            if norm == "rms":
                self.ln_q = nn.RMSNorm(cfg.hidden_size, eps=1e-6)
            elif norm == "layernorm":  # HF reference file (nn.LayerNorm, bias absent -> 0)
                self.ln_q = nn.LayerNorm(cfg.hidden_size, eps=1e-6, affine=True, bias=False)
            else:
                raise ValueError(f"merger norm must be 'rms' or 'layernorm', got {norm!r}")
            self.mlp = [
                nn.Linear(self.hidden, self.hidden, bias=False),
                nn.GELU(),
                nn.Linear(self.hidden, cfg.out_hidden_size, bias=False),
            ]

        def __call__(self, x):
            x = self.ln_q(x).reshape(-1, self.hidden)
            for layer in self.mlp:
                x = layer(x)
            return x

    class MiMoVisionTower(nn.Module):
        """MLX port of ``MiMoVisionTransformer.forward(pixel_values, grid_thw)``.

        ``merger_norm``: "rms" (vLLM + SGLang, default) or "layernorm" (the HF
        reference file). The checkpoint only has ``merger.ln_q.weight``; see VISION.md.
        """

        def __init__(self, cfg: MiMoV26VisionConfig, merger_norm: str = "rms"):
            super().__init__()
            self.cfg = cfg
            self.patch_embed = MiMoV26PatchEmbed(cfg)
            self.blocks = [
                MiMoV26VisionBlock(cfg, use_sink=cfg.use_sink and i not in cfg.fullatt_block_indexes)
                for i in range(cfg.depth)
            ]
            self.merger = MiMoV26PatchMerger(cfg, norm=merger_norm)

        # -- runtime knobs (not weights) --
        def set_sink_mode(self, mode: str):
            assert mode in ("key0", "virtual", "off")
            for b in self.blocks:
                b.attn.sink_mode = mode

        def set_q_block(self, n: int):
            for b in self.blocks:
                b.attn.q_block = int(n)

        @staticmethod
        def sanitize(weights: dict) -> dict:
            """Source ``visual.*`` tensors -> this module's parameter names."""
            out = {}
            for k, v in weights.items():
                if k.startswith("visual."):
                    k = k[len("visual.") :]
                if k == "patch_embed.proj.weight" and v.ndim == 5:
                    v = v.reshape(v.shape[0], -1)
                out[k] = v
            return out

        def load_source_weights(self, weights: dict, dtype=None):
            w = self.sanitize(weights)
            if dtype is not None:
                w = {k: v.astype(dtype) for k, v in w.items()}
            self.load_weights(list(w.items()), strict=True)
            return self

        def __call__(self, pixel_values, grid_thw):
            """pixel_values [N, 1536], grid_thw [n_items, 3] -> [N // 4, out_hidden_size]."""
            return self.merger(self.encode_blocks(pixel_values, grid_thw))

        def encode_blocks(self, pixel_values, grid_thw):
            """Patch embed + 28 blocks, row-major merge-unit order, pre-merger [N, 1280]."""
            cfg = self.cfg
            unit = cfg.spatial_merge_size ** 2
            dtype = self.patch_embed.proj.weight.dtype
            x = self.patch_embed(pixel_values.astype(dtype))
            D = x.shape[-1]

            cos_np, sin_np = rotary_cos_sin(grid_thw, cfg.head_dim, cfg.spatial_merge_size, cfg.rope_theta)
            col = window_index_col(grid_thw, cfg.spatial_merge_size)
            rev = np.argsort(col)
            col_mx, rev_mx = mx.array(col), mx.array(rev)

            def apply_index(t, idx):
                return t.reshape(-1, unit, t.shape[-1])[idx].reshape(-1, t.shape[-1])

            row_cs = (mx.array(cos_np), mx.array(sin_np))
            col_cs = (apply_index(row_cs[0], col_mx), apply_index(row_cs[1], col_mx))
            seqlens = frame_seqlens(grid_thw)
            types = cfg.vit_window_attn_types

            for i, blk in enumerate(self.blocks):
                wt = types[i]
                if wt == 1 and (i == 0 or types[i - 1] != 1):
                    x = apply_index(x, col_mx)
                if i > 0 and wt != 1 and types[i - 1] == 1:
                    x = apply_index(x, rev_mx)
                cos, sin = col_cs if wt == 1 else row_cs
                x = blk(x, seqlens, cos, sin, full_attn=i in cfg.fullatt_block_indexes)
            # vLLM restores row order when the LAST block is column-SWA; the HF
            # reference does not. No-op for V2.6 (last type is -1).
            if types[-1] == 1:
                x = apply_index(x, rev_mx)
            assert x.shape == (pixel_values.shape[0], D)
            return x


def load_visual_weights_from_source(model_dir: str | Path) -> dict:
    """Read only the ``visual.*`` tensors from the source safetensors (as MLX arrays)."""
    model_dir = Path(model_dir)
    wm = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files: dict[str, list[str]] = {}
    for k, f in wm.items():
        if k.startswith("visual."):
            files.setdefault(f, []).append(k)
    from safetensors import safe_open

    out = {}
    for f, keys in files.items():
        with safe_open(str(model_dir / f), framework="pt") as fh:
            for k in keys:
                t = fh.get_tensor(k)
                # numpy has no bf16: widen to fp32 (exact) and narrow back in MLX (exact)
                a = mx.array(t.float().numpy())
                out[k] = a.astype(mx.bfloat16) if str(t.dtype) == "torch.bfloat16" else a
    return out


# ---------------------------------------------------------------------------
# Preprocessing (numpy port of the SGLang/vLLM MiMo processor)
# ---------------------------------------------------------------------------

# Both serving stacks hard-code these (0-255 scale, i.e. ImageNet mean/std * 255).
# NOTE: preprocessor_config.json lists CLIP mean/std (Qwen2VLImageProcessor
# defaults) which NEITHER serving stack uses.
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# Token ids (config.json processor_config / tokenizer.json)
VISION_START_ID = 151652  # <|vision_start|>
VISION_END_ID = 151653  # <|vision_end|>
IMAGE_PAD_ID = 151655  # <|image_pad|>
VIDEO_PAD_ID = 151656  # <|video_pad|>
VIDEO_START_ID = 151670  # <|mimo_video_start|>
VIDEO_END_ID = 151671  # <|mimo_video_end|>


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int,
                 variant: str = "sglang") -> tuple[int, int]:
    """(h_bar, w_bar), both multiples of ``factor`` (=patch 16 * merge 2 = 32).

    ``variant`` only matters when min(h, w) < factor: SGLang rounds the upscaled
    edges, vLLM truncates the long edge with int(). Identical otherwise.
    """
    if min(height, width) < factor:
        if variant == "sglang":
            scale = factor / min(height, width)
            height, width = int(round(height * scale)), int(round(width * scale))
        else:
            if height < width:
                height, width = factor, int(width * factor / height)
            else:
                width, height = factor, int(height * factor / width)
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {height}x{width}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def _bilinear_weights(in_size: int, out_size: int):
    """torch upsample_bilinear2d(align_corners=False, no antialias) source taps."""
    scale = np.float32(in_size) / np.float32(out_size)
    dst = np.arange(out_size, dtype=np.float32)
    src = scale * (dst + np.float32(0.5)) - np.float32(0.5)
    src = np.maximum(src, np.float32(0.0))
    i0 = np.floor(src).astype(np.int64)
    i0 = np.minimum(i0, in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    l1 = (src - i0.astype(np.float32)).astype(np.float32)
    l0 = (np.float32(1.0) - l1).astype(np.float32)
    return i0, i1, l0, l1


def resize_bilinear(frames: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """frames [..., C, H, W] float32 -> [..., C, out_h, out_w] (torch F.interpolate bilinear)."""
    frames = np.asarray(frames, dtype=np.float32)
    H, W = frames.shape[-2:]
    y0, y1, ly0, ly1 = _bilinear_weights(H, out_h)
    x0, x1, lx0, lx1 = _bilinear_weights(W, out_w)
    rows = frames[..., y0, :] * ly0[:, None] + frames[..., y1, :] * ly1[:, None]
    return (rows[..., x0] * lx0 + rows[..., x1] * lx1).astype(np.float32)


def _to_chw_float(img) -> np.ndarray:
    try:
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.asarray(img.convert("RGB"))
    except ImportError:  # pragma: no cover
        pass
    a = np.asarray(img)
    if a.ndim != 3:
        raise ValueError(f"expected HWC or CHW image, got shape {a.shape}")
    if a.shape[-1] in (3, 4) and a.shape[0] not in (3, 4):
        a = a[..., :3].transpose(2, 0, 1)
    return a.astype(np.float32)


def flatten_patches(visual: np.ndarray, patch: int = 16, merge: int = 2, tps: int = 2):
    """[T, C, H, W] (T % tps == 0) -> patches [t*h*w, C*tps*patch*patch], grid (t, h, w).

    Row order: t, h-block, w-block, h-in-unit, w-in-unit (merge units of 2x2 are
    contiguous); column order: C, T, P, P (matches Conv3d weight layout).
    """
    T, C, H, W = visual.shape
    gt, gh, gw = T // tps, H // patch, W // patch
    p = visual.reshape(gt, tps, C, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(p).reshape(gt * gh * gw, C * tps * patch * patch), (gt, gh, gw)


def format_timestamp(ts: float) -> str:
    return f"{int(ts // 60):02d}:{int(ts % 60):02d}"


def sample_video_frames(total_frames: int, video_fps: float, fps: float = 1.0,
                        min_frames: int = 8, max_frames: int = 3600) -> tuple[np.ndarray, np.ndarray]:
    """SGLang ``_decode_frames_and_timestamps`` + qwen_vl ``smart_nframes`` (FRAME_FACTOR 2)."""

    def _ceil2(x):
        return int(math.ceil(x / 2) * 2)

    def _floor2(x):
        return int(math.floor(x / 2) * 2)

    lo = _ceil2(min_frames)
    hi = _floor2(min(max_frames, total_frames) if max_frames is not None else min(768, total_frames))
    n = total_frames / video_fps * fps
    n = _floor2(min(min(max(n, lo), hi), total_frames))
    if not (2 <= n <= total_frames):
        raise ValueError(f"nframes should be in [2, {total_frames}], got {n}")
    idx = np.unique(np.linspace(0, total_frames - 1, num=n, dtype=np.int64))
    return idx, idx.astype(np.float32) / np.float32(video_fps)


@dataclass
class VisualItem:
    kind: str  # "image" | "video"
    pixel_values: np.ndarray  # [t*h*w, 1536] float32
    grid_thw: tuple[int, int, int]
    num_tokens: int  # number of <|image_pad|>/<|video_pad|> tokens
    resized_hw: tuple[int, int]
    timestamps: np.ndarray | None = None  # aligned per-frame timestamps (video)
    timestamp_texts: list[str] | None = None  # one per temporal grid (video)


class MiMoV26VisionProcessor:
    """numpy port of the MiMo-V2 image/video processor (SGLang MiMoProcessor, vLLM MiMoVLProcessor).

    Pixel limits come from config.json ``processor_config`` (what both serving
    stacks read), NOT preprocessor_config.json.
    """

    def __init__(self, patch_size=16, merge_size=2, temporal_patch_size=2,
                 image_min_pixels=8192, image_max_pixels=8388608,
                 video_min_pixels=8192, video_max_pixels=8388608,
                 video_total_max_pixels=268435456, fps=1.0, max_frames=3600,
                 min_frames=8, num_frames=None, resize_variant="sglang"):
        self.patch_size, self.merge_size, self.tps = patch_size, merge_size, temporal_patch_size
        self.factor = patch_size * merge_size
        self.image_min_pixels, self.image_max_pixels = image_min_pixels, image_max_pixels
        self.video_min_pixels, self.video_max_pixels = video_min_pixels, video_max_pixels
        self.video_total_max_pixels = video_total_max_pixels
        self.fps, self.max_frames, self.min_frames, self.num_frames = fps, max_frames, min_frames, num_frames
        self.resize_variant = resize_variant

    @classmethod
    def from_config(cls, config: dict) -> "MiMoV26VisionProcessor":
        vc, pc = config.get("vision_config", {}), config.get("processor_config", {}) or {}
        patch = int(vc.get("patch_size", 14))
        merge = int(vc.get("spatial_merge_size", 2))
        unit = patch * merge
        return cls(
            patch_size=patch, merge_size=merge,
            temporal_patch_size=int(vc.get("temporal_patch_size", 2)),
            # same `or` fallbacks as SGLang/vLLM from_hf_config
            image_min_pixels=pc.get("image_min_pixels") or 4 * unit * unit,
            image_max_pixels=pc.get("image_max_pixels") or 4096 * unit * unit,
            video_min_pixels=pc.get("video_min_pixels") or 4 * unit * unit,
            video_max_pixels=pc.get("video_max_pixels") or 4096 * unit * unit,
            video_total_max_pixels=pc.get("video_total_max_pixels") or 16384 * unit * unit,
            fps=pc.get("fps") or 2.0,
            max_frames=pc.get("max_frames") or 256,
            min_frames=pc.get("min_frames") or 8,
            num_frames=pc.get("num_frames"),
        )

    @classmethod
    def from_model_dir(cls, path: str | Path) -> "MiMoV26VisionProcessor":
        return cls.from_config(json.loads((Path(path) / "config.json").read_text()))

    # -- images --
    def image_grid(self, height: int, width: int, min_pixels=None, max_pixels=None):
        h, w = smart_resize(height, width, self.factor,
                            min_pixels or self.image_min_pixels, max_pixels or self.image_max_pixels,
                            self.resize_variant)
        return (1, h // self.patch_size, w // self.patch_size), (h, w)

    def num_image_tokens(self, height: int, width: int, **kw) -> int:
        (t, gh, gw), _ = self.image_grid(height, width, **kw)
        return t * gh * gw // self.merge_size ** 2

    def preprocess_image(self, image, min_pixels=None, max_pixels=None) -> VisualItem:
        chw = _to_chw_float(image)
        _, H, W = chw.shape
        _, (h, w) = self.image_grid(H, W, min_pixels, max_pixels)
        x = resize_bilinear(chw, h, w)
        x = (x - PIXEL_MEAN[:, None, None]) / PIXEL_STD[:, None, None]
        vis = np.repeat(x[None], self.tps, axis=0)  # duplicate still frame to T=2
        patches, grid = flatten_patches(vis, self.patch_size, self.merge_size, self.tps)
        return VisualItem("image", patches.astype(np.float32), grid,
                          grid[0] * grid[1] * grid[2] // self.merge_size ** 2, (h, w))

    # -- videos --
    def preprocess_video(self, frames, timestamps) -> VisualItem:
        """frames: [T, H, W, C] or [T, C, H, W] already-sampled frames; timestamps [T] seconds."""
        f = np.asarray(frames)
        if f.ndim != 4:
            raise ValueError(f"expected 4-D frames, got {f.shape}")
        if f.shape[-1] in (1, 3, 4) and f.shape[1] not in (1, 3, 4):
            f = f[..., :3].transpose(0, 3, 1, 2)
        f = f.astype(np.float32)
        ts = np.asarray(timestamps, dtype=np.float32)
        n = f.shape[0]
        max_px = max(self.video_min_pixels,
                     min(self.video_total_max_pixels * self.tps // n, self.video_max_pixels))
        if n % self.tps:
            pad = self.tps - n % self.tps
            f = np.concatenate([f, np.repeat(f[-1:], pad, axis=0)], axis=0)
            ts = np.concatenate([ts, np.repeat(ts[-1:], pad)], axis=0)
        H, W = f.shape[-2:]
        h, w = smart_resize(H, W, self.factor, self.video_min_pixels, max_px, self.resize_variant)
        x = resize_bilinear(f, h, w)
        x = (x - PIXEL_MEAN[None, :, None, None]) / PIXEL_STD[None, :, None, None]
        patches, grid = flatten_patches(x, self.patch_size, self.merge_size, self.tps)
        ts_texts = [format_timestamp(float(t)) for t in ts[:: self.tps]]
        return VisualItem("video", patches.astype(np.float32), grid,
                          grid[0] * grid[1] * grid[2] // self.merge_size ** 2, (h, w), ts, ts_texts)

    # -- prompt expansion --
    def expand_tokens(self, item: VisualItem, encode: Callable[[str], list[int]] | None = None) -> list[int]:
        """Token ids replacing the chat-template placeholder
        ``<|vision_start|><|image_pad|><|vision_end|>`` (or the video one), SGLang layout.

        image: [vision_start] + [image_pad]*n + [vision_end]
        video: [mimo_video_start] + sum_t(encode("MM:SS") + [vision_start] + [video_pad]*(h*w/4) + [vision_end])
               + [mimo_video_end]
        """
        t, gh, gw = item.grid_thw
        per_grid = gh * gw // self.merge_size ** 2
        if item.kind == "image":
            return [VISION_START_ID] + [IMAGE_PAD_ID] * item.num_tokens + [VISION_END_ID]
        if encode is None:
            raise ValueError("video expansion needs a tokenizer encode() for the MM:SS timestamps")
        ids = [VIDEO_START_ID]
        for txt in item.timestamp_texts:
            ids += list(encode(txt)) + [VISION_START_ID] + [VIDEO_PAD_ID] * per_grid + [VISION_END_ID]
        return ids + [VIDEO_END_ID]


def batch_items(items: Iterable[VisualItem]):
    """Concatenate items for one tower call: (pixel_values [N,1536], grid_thw [n,3])."""
    items = list(items)
    return (np.concatenate([it.pixel_values for it in items], axis=0),
            np.array([it.grid_thw for it in items], dtype=np.int64))
