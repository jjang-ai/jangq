"""FP8 e4m3fn block-128 dequant + shard index for the dots3-note-prev-fp8 source.

The fp8 repo quantizes ONLY language-model 2-D Linears (weight + weight_scale_inv
[ceil(out/128), ceil(in/128)] f32). Vision/audio towers, norms, routers,
embeddings, and e_score biases are bf16/f32 passthrough.

Dequant is dependency-light: raw uint8 → 256-entry f32 LUT (verified against
torch.float8_e4m3fn in tests) → * scale_inv (repeat 128×128, cropped).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def e4m3fn_lut() -> np.ndarray:
    """256-entry float32 lookup table for float8_e4m3fn byte values."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0x0F
        m = b & 0x07
        if e == 0x0F and m == 0x07:
            lut[b] = np.nan          # e4m3fn: only NaN, no inf
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0 ** (-6)
        else:
            lut[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


def dequant_fp8_block(weight_u8: np.ndarray, scale_inv: np.ndarray,
                      block: int = 128) -> np.ndarray:
    """weight_u8: (out, in) uint8 raw e4m3fn bytes; scale_inv: f32 block scales.
    Returns float32 (out, in)."""
    w = e4m3fn_lut()[weight_u8]
    rows, cols = w.shape
    s = np.repeat(np.repeat(scale_inv.astype(np.float32), block, axis=0),
                  block, axis=1)[:rows, :cols]
    return w * s


class ShardIndex:
    """name -> (shard path, dtype, shape, offsets) via safetensors headers."""

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        idx = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = idx["weight_map"]
        self._headers: dict[str, dict] = {}

    def _header(self, shard: str) -> dict:
        h = self._headers.get(shard)
        if h is None:
            p = self.dir / shard
            with open(p, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                h = json.loads(f.read(n))
            h["__data_start__"] = 8 + n
            self._headers[shard] = h
        return h

    def names(self):
        return self.weight_map.keys()

    def info(self, name: str) -> tuple[str, str, list[int]]:
        shard = self.weight_map[name]
        meta = self._header(shard)[name]
        return shard, meta["dtype"], meta["shape"]

    def read_raw(self, name: str) -> tuple[np.ndarray, str, list[int]]:
        """Read tensor bytes via memmap slice. Returns (flat uint8 view copy,
        dtype string, shape)."""
        shard = self.weight_map[name]
        h = self._header(shard)
        meta = h[name]
        start = h["__data_start__"] + meta["data_offsets"][0]
        length = meta["data_offsets"][1] - meta["data_offsets"][0]
        mm = np.memmap(self.dir / shard, dtype=np.uint8, mode="r",
                       offset=start, shape=(length,))
        buf = np.asarray(mm).copy()
        del mm
        return buf, meta["dtype"], meta["shape"]

    def read(self, name: str) -> np.ndarray:
        """Read a tensor as numpy. BF16 -> float32, F8_E4M3 -> raw uint8
        (callers pair with weight_scale_inv via read_dequant)."""
        buf, dtype, shape = self.read_raw(name)
        if dtype == "BF16":
            u16 = buf.view(np.uint16).astype(np.uint32) << 16
            return u16.view(np.float32).reshape(shape)
        if dtype == "F32":
            return buf.view(np.float32).reshape(shape)
        if dtype == "F16":
            return buf.view(np.float16).reshape(shape).astype(np.float32)
        if dtype == "F8_E4M3":
            return buf.reshape(shape)          # raw bytes, caller dequants
        raise ValueError(f"unhandled dtype {dtype} for {name}")

    def read_dequant(self, name: str) -> np.ndarray:
        """Full-precision f32 view of any weight (auto fp8-block dequant)."""
        shard, dtype, shape = self.info(name)
        if dtype == "F8_E4M3":
            scale = self.read(name + "_scale_inv")
            return dequant_fp8_block(self.read(name), scale)
        return self.read(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map
