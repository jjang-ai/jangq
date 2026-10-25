"""MLX port of the DeepSeek-V4.1 vision tower (ViT + Aligner) and image preprocessing.

A faithful port of release `inference/vision.py` + `inference/image_processor.py`. The ViT runs
full bidirectional attention over one image's patch grid with 2D RoPE; the Aligner does a 3x3
spatial merge (9x token reduction) and projects to the text hidden size. Weights are BF16
passthrough in the JANG bundle (names `vision.*`, `aligner.*`, `image_{start,end,newline}`), so the
modules here mirror those names and load via nn.Module.load_weights.

Preprocessing is pure PIL/NumPy. `prepare_vl_inputs` expands each image placeholder id in a token
list into its image span and returns the parallel token-type array the backbone uses to build the
image mask (VL routing bias) and to splice aligner rows into the embedding stream.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from functools import lru_cache
from urllib.request import urlopen

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# token types (parallel to input_ids over an image span); TEXT elsewhere
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class VisionArgs:
    n_layers: int
    dim: int
    n_heads: int
    inter_dim: int
    patch_size: int
    rope_theta: float
    downsample_ratio: int
    max_n_token: int
    min_pixels: int
    max_wh_ratio: float | None
    image_token_id: int
    text_dim: int
    eps: float = 1e-6

    @property
    def rope_dim(self) -> int:
        return self.dim // self.n_heads // 2

    @classmethod
    def from_config(cls, cfg: dict) -> "VisionArgs":
        v = cfg["vision_config"]
        return cls(
            n_layers=v["num_hidden_layers"], dim=v["hidden_size"], n_heads=v["num_attention_heads"],
            inter_dim=v["intermediate_size"], patch_size=v["patch_size"],
            rope_theta=v.get("rope_theta", 10000.0), downsample_ratio=v["downsample_ratio"],
            max_n_token=v.get("max_image_tokens", 1024), min_pixels=v.get("min_pixels", 544 * 544),
            max_wh_ratio=v.get("max_wh_ratio"), image_token_id=cfg["image_token_id"],
            text_dim=cfg["text_config"]["hidden_size"],
        )


# --------------------------------------------------------------------------- ViT

@lru_cache(8)
def _vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """2D RoPE tables, matching release get_vision_cos_sin. Returns cos,sin [n_h*n_w, 1, dim]."""
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))  # [dim/2]
    hpos = np.broadcast_to(np.arange(n_h)[:, None], (n_h, n_w))
    wpos = np.broadcast_to(np.arange(n_w)[None, :], (n_h, n_w))
    pos = np.stack([hpos, wpos], axis=-1).reshape(-1, 2, 1).astype(np.float32)  # [n,2,1]
    freqs = (pos * inv).reshape(n_h * n_w, -1)  # [n, dim]
    cos = mx.array(np.cos(freqs))[:, None, :]
    sin = mx.array(np.sin(freqs))[:, None, :]
    return cos, sin


def _apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x [n, heads, head_dim]; cos/sin [n,1,head_dim/2]. fp32 rotate, cast back."""
    dt = x.dtype
    xf = x.astype(mx.float32)
    x1, x2 = mx.split(xf, 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(dt)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dt = x.dtype
        xf = x.astype(mx.float32)
        xf = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (self.weight * xf).astype(dt)


class PatchEmbed(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.proj = nn.Linear(3 * a.patch_size**2, a.dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x.reshape(x.shape[0], -1))


class VisionAttention(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.n_heads = a.n_heads
        self.head_dim = a.dim // a.n_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(a.dim, 3 * a.dim)
        self.wo = nn.Linear(a.dim, a.dim)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        n = x.shape[0]
        qkv = self.wqkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = _apply_rotary(q.reshape(n, self.n_heads, self.head_dim), cos, sin)
        k = _apply_rotary(k.reshape(n, self.n_heads, self.head_dim), cos, sin)
        v = v.reshape(n, self.n_heads, self.head_dim)
        # [1, heads, n, hd] for SDPA (full bidirectional, no mask)
        q = q.transpose(1, 0, 2)[None]
        k = k.transpose(1, 0, 2)[None]
        v = v.transpose(1, 0, 2)[None]
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o[0].transpose(1, 0, 2).reshape(n, -1)
        return self.wo(o)


class VisionMLP(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.w1 = nn.Linear(a.dim, 2 * a.inter_dim, bias=False)
        self.w2 = nn.Linear(a.inter_dim, a.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.norm1 = _RMSNorm(a.dim, a.eps)
        self.attn = VisionAttention(a)
        self.norm2 = _RMSNorm(a.dim, a.eps)
        self.mlp = VisionMLP(a)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    def __init__(self, a: VisionArgs):
        super().__init__()
        self.rope_dim = a.rope_dim
        self.rope_theta = a.rope_theta
        self.patch_embed = PatchEmbed(a)
        self.blocks = [VisionBlock(a) for _ in range(a.n_layers)]
        self.norm = _RMSNorm(a.dim, a.eps)

    def __call__(self, patches: mx.array, n_h: int, n_w: int) -> mx.array:
        x = self.patch_embed(patches)
        cos, sin = _vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


def _gelu_erf(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class Aligner(nn.Module):
    """3x3 spatial merge (release uses F.unfold) + two-layer projector to the text hidden size."""

    def __init__(self, a: VisionArgs):
        super().__init__()
        self.r = a.downsample_ratio
        self.w1 = nn.Linear(a.dim * self.r**2, a.text_dim)
        self.w2 = nn.Linear(a.text_dim, a.text_dim)

    def __call__(self, x: mx.array, n_h: int, n_w: int) -> mx.array:
        r = self.r
        C = x.shape[-1]
        x = x.reshape(n_h, n_w, C)
        ph, pw = (-n_h) % r, (-n_w) % r  # pad to a multiple of r (bottom/right), matching F.pad
        if ph or pw:
            x = mx.pad(x, [(0, ph), (0, pw), (0, 0)])
        H, W = n_h + ph, n_w + pw
        # F.unfold(kernel=r, stride=r) on [C,H,W] then transpose -> rows=blocks (row-major),
        # cols = (c, kh, kw) flattened. Replicate with reshape/transpose.
        x = x.reshape(H // r, r, W // r, r, C)          # [bh, kh, bw, kw, C]
        x = x.transpose(0, 2, 4, 1, 3)                  # [bh, bw, C, kh, kw]
        x = x.reshape((H // r) * (W // r), C * r * r)    # [n_llm, C*r*r] with col = c*r*r + kh*r + kw
        return self.w2(_gelu_erf(self.w1(x)))


# --------------------------------------------------------------------------- preprocessing

@dataclass
class ImageInput:
    start: int
    patches: mx.array
    n_vit_h: int
    n_vit_w: int
    types: list[int]


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_h: int, best_w: int, patch: int, r: int):
    return math.ceil((best_h // patch) / r), math.ceil((best_w // patch) / r)


def solve_resize_ratio(height, width, patch, r, max_n_token):
    ratio = height / width
    max_w = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h = max_w * ratio
    cell = patch * r
    if max_w < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / patch) * patch, math.floor(width * beta / patch) * patch


def safe_resize(height, width, best_h, best_w, patch, r, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_h, best_w, patch, r)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_h, best_w = solve_resize_ratio(height, width, patch, r, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_h, best_w, patch, r)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_h, best_w


def load_image_bytes(record) -> bytes:
    if isinstance(record, str):  # the chat encoder hands back bare url/path strings
        record = {"url": record}
    data = record.get("data")
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    source = record.get("source")
    if isinstance(source, dict):
        if source.get("data") is not None:
            return base64.b64decode(source["data"])
        if source.get("url"):
            return load_image_bytes({"url": source["url"]})
    url = record.get("url")
    if isinstance(url, str) and url:
        if url.startswith("data:"):
            header, _, payload = url.partition(",")
            if ";base64" not in header:
                raise ValueError(f"Unsupported data URL encoding: {header}")
            return base64.b64decode(payload)
        if url.startswith(("http://", "https://")):
            with urlopen(url, timeout=30) as response:
                return response.read()
        with open(url, "rb") as file:
            return file.read()
    raise ValueError(f"Cannot load image from record: {list(record.keys())}")


def plan_image_grid(width: int, height: int, a: VisionArgs):
    p = a.patch_size
    if a.max_wh_ratio is not None and width > height * a.max_wh_ratio:
        width = height * a.max_wh_ratio
    if 0 < width * height < a.min_pixels:
        ratio = (a.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_w = math.ceil(width / p) * p
    best_h = math.ceil(height / p) * p
    return safe_resize(height, width, best_h, best_w, p, a.downsample_ratio, a.max_n_token)


def load_image(record, a: VisionArgs):
    """Load+transform one image record into ViT patches [n_vit_h*n_vit_w, 3, p, p] (bf16)."""
    from PIL import Image, ImageOps

    p = a.patch_size
    with Image.open(io.BytesIO(load_image_bytes(record))) as source:
        image = source.convert("RGB")
    n_llm_h, n_llm_w, best_h, best_w = plan_image_grid(image.width, image.height, a)
    n_vit_h, n_vit_w = best_h // p, best_w // p
    if a.max_wh_ratio is not None and image.width >= a.max_wh_ratio * image.height:
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0  # [3,H,W]
    x = (x - 0.5) / 0.5
    x = x.reshape(3, n_vit_h, p, n_vit_w, p).transpose(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return mx.array(x.astype(np.float32)).astype(mx.bfloat16), n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> list[int]:
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return types


def prepare_vl_inputs(prompt_tokens: list[int], images: list, a: VisionArgs):
    """Expand each image_token_id in a token list into its image span.

    Returns (tokens, token_types, image_inputs|None). Image-span positions carry a.image_token_id in
    `tokens`; `token_types` is TEXT(-1) for text and the 0..3 enum inside spans."""
    num_ph = sum(t == a.image_token_id for t in prompt_tokens)
    if num_ph != len(images):
        raise ValueError(f"Found {num_ph} image tokens but got {len(images)} images")
    tokens, token_types, image_inputs = [], [], []
    it = iter(images)
    for tok in prompt_tokens:
        if tok != a.image_token_id:
            tokens.append(tok)
            token_types.append(TEXT)
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(next(it), a)
        types = image_token_types(n_llm_h, n_llm_w)
        image_inputs.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
        tokens += [a.image_token_id] * len(types)
        token_types += types
    return tokens, token_types, image_inputs or None
