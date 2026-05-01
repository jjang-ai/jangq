"""A1 — Pool-side KV cache quantization for HSA + CSA layers.

Pool entries from Compressor are READ-ONLY after produce: no further
mutation across decode steps. Currently stored as bf16. Quantizing them
to 4-bit affine in-cache reduces pool memory 4×.

Memory savings at 1M context (DSV4-Flash, head_dim=512):
  - HSA pool (compress_ratio=128): ~7.8K entries × 21 layers ≈ 168 MB → 42 MB
  - CSA pool (compress_ratio=4 overlap): ~250K entries × 20 layers ≈ 5 GB → 1.25 GB
  - Savings: ~3.9 GB on a 1M context

Pool entries are heavily averaged inputs. 4-bit affine bench precedent on
Mistral 3.5 MXFP4 was cos=0.996 round-trip; pool entries are similarly
well-behaved.

Usage:
    DSV4_POOL_QUANT=1 python -m jang_tools.dsv4.runtime ...

When the env var is set, `make_cache()` (in mlx_model.py) returns
`PoolQuantizedV4Cache` instead of `DeepseekV4Cache`. The new cache
intercepts `compressor_state["pooled"]` writes and quantizes in-place.

Implementation: wraps the existing DeepseekV4Cache. On `pooled` write,
applies `mx.quantize(bits=4, group_size=32)` and stores
(weight, scales, biases) as a tuple. On read, applies `mx.dequantize`
and returns bf16. Drop-in compatible — the consumer (Indexer GATHER /
mask path) sees normal bf16 tensors.
"""
from __future__ import annotations
from typing import Optional

import mlx.core as mx


def _quant_pool(p: mx.array) -> tuple:
    """Quantize a pool tensor (B, P, head_dim) to (qw, scales, biases)
    at 4-bit affine, group_size=32 along the head_dim axis."""
    qw, sc, bi = mx.quantize(p, bits=4, group_size=32)
    return (qw, sc.astype(mx.bfloat16), bi.astype(mx.bfloat16))


def _dequant_pool(triple: tuple) -> mx.array:
    qw, sc, bi = triple
    return mx.dequantize(qw, sc.astype(mx.float32), bi.astype(mx.float32),
                         bits=4, group_size=32).astype(mx.bfloat16)


class PoolQuantizedV4Cache:
    """Drop-in replacement for DeepseekV4Cache with quantized pool.

    Mirrors the original API (same `local`, `compressor_state`,
    `indexer_state` dict shapes, same `state` get/set, same `is_trimmable`).
    Only the `pooled` entry of compressor_state and indexer_state is
    transparently quantized when written.
    """

    def __init__(self, sliding_window):
        from mlx_lm.models.cache import RotatingKVCache
        self.local = RotatingKVCache(max_size=sliding_window, keep=0)
        # Internal storage: pooled is a triple if quantized, else mx.array or None
        self._compressor: dict = {"buffer_kv": None, "buffer_gate": None,
                                  "_pooled_q": None}
        self._indexer:    dict = {"buffer_kv": None, "buffer_gate": None,
                                  "_pooled_q": None}

    @property
    def offset(self): return self.local.offset

    @property
    def keys(self): return self.local.keys

    @property
    def values(self): return self.local.values

    @property
    def is_trimmable(self): return self.local.is_trimmable

    @property
    def compressor_state(self) -> dict:
        # Synthesize the original interface: pooled comes back as bf16
        d = {k: self._compressor[k]
             for k in ("buffer_kv", "buffer_gate")}
        q = self._compressor["_pooled_q"]
        d["pooled"] = _dequant_pool(q) if q is not None else None
        return _StateProxy(d, self._compressor)

    @compressor_state.setter
    def compressor_state(self, value):
        for k in ("buffer_kv", "buffer_gate"):
            self._compressor[k] = value.get(k)
        p = value.get("pooled")
        self._compressor["_pooled_q"] = _quant_pool(p) if p is not None else None

    @property
    def indexer_state(self) -> dict:
        d = {k: self._indexer[k] for k in ("buffer_kv", "buffer_gate")}
        q = self._indexer["_pooled_q"]
        d["pooled"] = _dequant_pool(q) if q is not None else None
        return _StateProxy(d, self._indexer)

    @indexer_state.setter
    def indexer_state(self, value):
        for k in ("buffer_kv", "buffer_gate"):
            self._indexer[k] = value.get(k)
        p = value.get("pooled")
        self._indexer["_pooled_q"] = _quant_pool(p) if p is not None else None

    @property
    def state(self):
        local_state = self.local.state
        return (
            local_state,
            (self._compressor["buffer_kv"],
             self._compressor["buffer_gate"],
             self._dequant_or_none(self._compressor["_pooled_q"])),
            (self._indexer["buffer_kv"],
             self._indexer["buffer_gate"],
             self._dequant_or_none(self._indexer["_pooled_q"])),
        )

    @staticmethod
    def _dequant_or_none(q):
        return _dequant_pool(q) if q is not None else None

    @state.setter
    def state(self, value):
        local_state, comp_state, idx_state = value
        self.local.state = local_state
        for k, v in zip(("buffer_kv", "buffer_gate", "pooled"), comp_state):
            if k == "pooled":
                self._compressor["_pooled_q"] = _quant_pool(v) if v is not None else None
            else:
                self._compressor[k] = v
        for k, v in zip(("buffer_kv", "buffer_gate", "pooled"), idx_state):
            if k == "pooled":
                self._indexer["_pooled_q"] = _quant_pool(v) if v is not None else None
            else:
                self._indexer[k] = v

    def update_and_fetch(self, k, v):
        return self.local.update_and_fetch(k, v)


class _StateProxy(dict):
    """Dict that writes-back into a backing storage on assignment so
    `cache.compressor_state["pooled"] = new_pool` round-trips through
    the quantizer."""
    def __init__(self, view: dict, backing: dict):
        super().__init__(view)
        self._backing = backing

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "pooled":
            self._backing["_pooled_q"] = _quant_pool(value) if value is not None else None
        else:
            self._backing[key] = value
