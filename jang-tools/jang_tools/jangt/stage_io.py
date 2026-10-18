"""Read routed-expert weights back from JANGT / JANGH stage files or bundles (evaluation only).

A source is a directory holding either per-layer stage files `L##.safetensors` (moe_encode.py / jang_tools.jangh.n05.convert
--keep-stage) or a bundle with `model.safetensors.index.json`. For one layer and projection, `load_proj` returns
(E, N, K) bf16 in the original basis, decoding:
  JANGT ragged  (.jt_packed flat + .jt_off + .jt_kbits [+ .jt_cs])  per-expert widths, down column scales
  JANGT fixed   (.jt_packed (E, N, W) [+ .jt_kbits])
  JANGH         (.tq2_packed + .tq2_scales), bits MEASURED from the packed width (never from a plan)
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx


class Source:
    def __init__(self, d: str | Path):
        self.d = Path(d); self._cache = {}
        idx = self.d / "model.safetensors.index.json"
        self.wmap = json.loads(idx.read_text())["weight_map"] if idx.exists() else None

    def _file_for(self, L: int, key: str):
        if self.wmap is not None:
            f = self.wmap.get(key)
            return None if f is None else self.d / f
        f = self.d / f"L{L:02d}.safetensors"
        return f if f.exists() else None

    def tensors(self, L: int, m: str, suffixes) -> dict | None:
        """{suffix: array} for module m if the source holds it, else None."""
        out = {}
        for sfx in suffixes:
            f = self._file_for(L, m + sfx)
            if f is None:
                return None if not out else out
            if f not in self._cache:
                if len(self._cache) > 1:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[f] = mx.load(str(f))
            if m + sfx not in self._cache[f]:
                return None if not out else out
            out[sfx] = self._cache[f][m + sfx]
        return out

    def has(self, L: int, m: str) -> str | None:
        for kind, sfx in (("jt", ".jt_packed"), ("jh", ".tq2_packed")):
            if self.tensors(L, m, (sfx,)):
                return kind
        return None


def load_proj(src: Source, L: int, proj: str, K: int, E: int = 256) -> mx.array:
    m = f"model.layers.{L}.mlp.switch_mlp.{proj}_proj"
    kind = src.has(L, m)
    if kind == "jt":
        from jang_tools.jangt.moe_encode import dequant_mx, dequant_ragged
        t = src.tensors(L, m, (".jt_packed", ".jt_scale", ".jt_su", ".jt_ocol", ".jt_wout"))
        ex = {s: v for s, v in ((s, src.tensors(L, m, (s,))) for s in (".jt_off", ".jt_kbits", ".jt_cs")) if v}
        sf = {m + k: v for k, v in t.items()}
        for s, v in ex.items():
            sf[m + s] = v[s]
        parts = []
        for e0 in range(0, E, 16):
            e1 = min(E, e0 + 16)
            if m + ".jt_off" in sf:
                W = dequant_ragged(sf, m, e0, e1, K)
            else:
                kb = float(sf[m + ".jt_kbits"].reshape(-1)[0].item()) if m + ".jt_kbits" in sf else 2
                kb = int(kb) if float(kb).is_integer() else kb
                W = dequant_mx(sf[m + ".jt_packed"][e0:e1], sf[m + ".jt_scale"][e0:e1], sf[m + ".jt_su"],
                               sf[m + ".jt_ocol"][e0:e1], sf[m + ".jt_wout"][e0:e1], K, kb)
            parts.append(W.astype(mx.bfloat16)); mx.eval(parts[-1])
        return mx.concatenate(parts, axis=0)
    if kind == "jh":
        import sys
        from jang_tools.jangh.encode import dequant as tq_dequant
        from jang_tools.jangh.format import h32, unpack_bitstream
        t = src.tensors(L, m, (".tq2_packed", ".tq2_scales"))
        pk, sc = t[".tq2_packed"], t[".tq2_scales"]
        b = int(pk.shape[-1]) * 32 // K
        assert b * K == int(pk.shape[-1]) * 32, (m, pk.shape, K)
        parts = []
        for e0 in range(0, E, 16):
            q = unpack_bitstream(pk[e0:e0 + 16], b, K)
            parts.append(h32(tq_dequant(q, sc[e0:e0 + 16], b)).astype(mx.bfloat16)); mx.eval(parts[-1])
        return mx.concatenate(parts, axis=0)
    raise KeyError(f"{m}: not in source {src.d}")
