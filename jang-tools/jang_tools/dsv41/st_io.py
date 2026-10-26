"""Torch-free safetensors access and DSV4.1 source codecs.

The DSV4.1 release stores:
  - non-routed matmuls as F8_E4M3 with one F8_E8M0 scale per 32x32 block
  - routed experts as I8 (two FP4 e2m1 nibbles per byte) with one F8_E8M0
    scale per 32 input columns (i.e. MXFP4)
  - Engram tables as F8_E4M3 rows with one F8_E8M0 scale per 32 columns
  - control tensors as BF16 / F32

Headers are parsed directly so tensors can be read with a single pread and
decoded on the GPU with MLX. Nothing here imports torch.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

_NP_STORAGE = {
    "F8_E4M3": np.uint8,
    "F8_E8M0": np.uint8,
    "I8": np.uint8,
    "U8": np.uint8,
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "I32": np.int32,
    "I64": np.int64,
}

FP4_TABLE = mx.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=mx.float32,
)


@dataclass(frozen=True)
class TensorMeta:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int  # absolute byte offset in the file
    nbytes: int


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


class Shards:
    """Tensor name -> location map for a directory of safetensors shards."""

    def __init__(self, root: str | Path, index_name: str = "model.safetensors.index.json", allow_missing: bool = False):
        self.root = Path(root)
        index = self.root / index_name
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.root.glob("*.safetensors"))
        self.meta: dict[str, TensorMeta] = {}
        self.missing: list[str] = []
        for fn in files:
            path = self.root / fn
            if allow_missing and not path.exists():
                self.missing.append(fn)
                continue
            header, base = read_header(path)
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                self.meta[name] = TensorMeta(path, info["dtype"], tuple(info["shape"]), base + start, end - start)

    def __contains__(self, name: str) -> bool:
        return name in self.meta

    def keys(self):
        return self.meta.keys()

    def raw(self, name: str) -> np.ndarray:
        m = self.meta[name]
        buf = np.empty(m.nbytes, dtype=np.uint8)
        with open(m.path, "rb", buffering=0) as f:
            fd = f.fileno()
            view = memoryview(buf)
            done = 0
            while done < m.nbytes:
                got = os.preadv(fd, [view[done:]], m.offset + done)
                if got <= 0:
                    raise IOError(f"short read for {name}")
                done += got
        return buf.view(_NP_STORAGE[m.dtype]).reshape(m.shape)

    def to_mx(self, name: str) -> mx.array:
        """Raw tensor as an MLX array; BF16 comes back as bfloat16, FP8 / I8 as uint8."""
        m = self.meta[name]
        arr = mx.array(self.raw(name))
        if m.dtype == "BF16":
            arr = arr.view(mx.bfloat16)
        return arr

    def dequant(self, name: str, dtype=mx.bfloat16) -> mx.array:
        """Dequantized weight. Dispatches on the stored dtype and the sibling `.scale`."""
        m = self.meta[name]
        if m.dtype in ("BF16", "F16", "F32"):
            return self.to_mx(name).astype(dtype)
        scale_name = name[: -len(".weight")] + ".scale" if name.endswith(".weight") else name + ".scale"
        w = self.to_mx(name)
        s = self.to_mx(scale_name)
        if m.dtype == "F8_E4M3":
            return dequant_fp8_block(w, s, dtype)
        if m.dtype == "I8":
            return dequant_fp4(w, s, dtype)
        raise ValueError(f"cannot dequantize {name} ({m.dtype})")


def e8m0_to_float(s: mx.array) -> mx.array:
    """UE8M0 byte -> 2 ** (byte - 127), built exactly from the float32 exponent bits.

    Byte 0 (2**-127) is a float32 subnormal: mantissa bit 22. 0xFF is NaN in the spec and never
    stored."""
    b = s.astype(mx.uint32)
    bits = mx.where(b == 0, mx.array(0x00400000, mx.uint32), mx.left_shift(b, 23))
    return bits.view(mx.float32)


def dequant_fp8_block(w: mx.array, s: mx.array, dtype=mx.bfloat16) -> mx.array:
    """F8_E4M3 [out, in] with E8M0 scales [ceil(out/bo), ceil(in/bi)]; block size inferred."""
    out_dim, in_dim = w.shape
    bo = -(-out_dim // s.shape[0])
    bi = -(-in_dim // s.shape[1])
    scale = e8m0_to_float(s)
    scale = mx.repeat(mx.repeat(scale, bo, axis=0)[:out_dim], bi, axis=1)[:, :in_dim]
    return (mx.from_fp8(w, dtype=mx.float32) * scale).astype(dtype)


def dequant_fp4(w: mx.array, s: mx.array, dtype=mx.bfloat16) -> mx.array:
    """Packed FP4 (low nibble = even column) [..., out, in/2] with E8M0 scales per 32 columns.

    FP4 values times a power of two are exact in bfloat16, so the bf16 result is lossless.
    """
    lo = mx.bitwise_and(w, 0x0F)
    hi = mx.right_shift(w, 4)
    vals = mx.stack([FP4_TABLE[lo], FP4_TABLE[hi]], axis=-1).reshape(*w.shape[:-1], w.shape[-1] * 2)
    in_dim = vals.shape[-1]
    block = in_dim // s.shape[-1]
    vals = vals.reshape(*vals.shape[:-1], s.shape[-1], block) * e8m0_to_float(s)[..., None]
    return vals.reshape(*w.shape[:-1], in_dim).astype(dtype)
