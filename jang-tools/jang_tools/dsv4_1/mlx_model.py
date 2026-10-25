"""DeepSeek-V4.1-Flash (``model_type: deepseek_v41``) as a vMLX engine model.

V4.1 is the V4 family plus Engram n-gram tap layers, a vision tower, dSpark MTP heads, and cross-layer
KV/index/candidate sharing in the latent-attention stack. Rather than re-port that validated compute
into the engine's per-layer-cache idiom, this module WRAPS the proven batch-1 runtime in
``jang_tools.dsv41`` (``DSV41`` / ``Model`` / ``Session``) behind the small contract the engine drives:

    Model(args)                       # lightweight; real weights attached by the loader
    model.layers                      # list, one per decoder layer (engine cache bookkeeping)
    model.make_cache() -> [cache]     # one DeepseekV41Cache per layer, sharing one Session
    model(input_ids, cache, mask)     # append the new tokens, return last-row logits (1,1,V)

The whole sequence state (per-layer KV windows + compressed/index buffers, the Engram compressed-token
history, the position) lives in one ``jang_tools.dsv41.Session``; the per-layer ``DeepseekV41Cache``
objects are thin views onto it so the engine can treat the model like any other (len(cache) == layers,
each reports the shared ``offset``). Weights, tokenizer and the disk-resident Engram tables are loaded by
``load_deepseek_v41`` through the validated ``DSV41`` path (q2/q3 per-tensor affine), so the engine's
generic variable-bit hydration is bypassed for this family.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

try:
    from mlx_lm.models.base import BaseModelArgs
except Exception:  # pragma: no cover - mlx_lm always present in the engine env
    @dataclass
    class BaseModelArgs:  # minimal shim
        @classmethod
        def from_dict(cls, params):
            import inspect
            fields = set(inspect.signature(cls).parameters)
            return cls(**{k: v for k, v in params.items() if k in fields})


@dataclass
class ModelArgs(BaseModelArgs):
    """Only what the engine/registry reads off the class; the full architecture config is parsed by
    ``jang_tools.dsv41.Args`` from ``config.json['text_config']`` inside ``DSV41``."""

    model_type: str = "deepseek_v41"
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    vocab_size: int = 129280
    num_attention_heads: int = 64
    num_key_value_heads: int = 1
    head_dim: int = 512
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # config.json keeps the real fields under text_config; surface the few we expose.
        tc = params.get("text_config", params) if isinstance(params, dict) else params
        merged = {**(tc if isinstance(tc, dict) else {}), **({"model_type": params.get("model_type")} if isinstance(params, dict) else {})}
        return super().from_dict(merged)


class _SeqState:
    """One decoding sequence's shared state, Session-compatible (jang_tools.dsv41.Session duck type):
    the per-layer cache list, the Engram compressed-token history, and the position. The per-layer
    DeepseekV41Cache objects ARE the entries of ``caches`` (they duck-type LayerCache), so
    ``DSV41.forward(ids, seq)`` drives them directly."""

    def __init__(self):
        self.caches: list = []
        self.history = np.zeros(0, np.int64)
        self.pos = 0

    def checkpoint(self):
        """O(1) ref snapshot for DSpark speculative verification (mirrors runtime Session.checkpoint,
        but over DeepseekV41Cache fields). QuantGrowBuffer.append writes past .n in place / regrows, so
        snapshotting (window, compress_kv.snapshot(), index_k.snapshot(), pend) + history + pos and
        restoring them is a safe rollback."""
        snaps = []
        for c in self.caches:
            snaps.append((c.window, c.compress_kv.snapshot(), c.index_k.snapshot(), c.pend_kv, c.pend_score))
        return (snaps, self.history, self.pos)

    def restore(self, ck):
        snaps, history, pos = ck
        for c, s in zip(self.caches, snaps):
            c.window = s[0]
            c.compress_kv.load(s[1])
            c.index_k.load(s[2])
            c.pend_kv, c.pend_score = s[3], s[4]
        self.history = history
        self.pos = pos


def _relink(cache: list) -> "_SeqState":
    """Unify a per-layer DeepseekV41Cache list onto one _SeqState. make_cache() already shares one;
    the engine's per-layer reconstruct builds each cache with its OWN _SeqState (pos on every layer,
    Engram history only on the idx==0 layer), so pull those onto a single canonical seq and point all
    layers at it. Idempotent."""
    canon = cache[0]._seq
    for c in cache:
        s = c._seq
        if s is not canon:
            if s.pos:
                canon.pos = s.pos
            if s.history.size:
                canon.history = s.history
            c._seq = canon
    canon.caches = list(cache)
    return canon


class DeepseekV41Cache:
    """One decoder layer's KV cache for the engine. Duck-types jang_tools.dsv41.LayerCache (the runtime
    reads/writes .window / .compress_kv / .index_k / .pend_kv / .pend_score on it) and adds the engine
    cache contract (offset / state / meta_state / trim / is_trimmable).

    The sliding window is the raw last-``win`` KV; compress_kv / index_k are append-only GrowBuffers of
    the compressed-global pool (one row per ``ratio`` tokens); pend_* is the compressor's partial group.
    Sequence-level position + Engram history live on the shared _SeqState and are serialized on layer 0
    so a restored prefix resumes exactly (offset, n-gram continuation)."""

    def __init__(self, seq: "_SeqState | None" = None, idx: int = 0, ratio: int = 0,
                 sliding_window=None, compress_ratio=None):
        from jang_tools.dsv41.model import QuantGrowBuffer

        # sliding_window/compress_ratio accepted so the engine's DSV4 reconstruct ctor
        # (`cache_cls(sliding_window=.., compress_ratio=..)`) can build us unchanged.
        self._seq = seq if seq is not None else _SeqState()
        self._idx = int(idx)
        self._ratio = int(ratio or compress_ratio or 0)
        # LayerCache fields (names match jang_tools.dsv41.model.LayerCache exactly). compress_kv /
        # index_k are 4-bit RESIDENT (FP4-parity) so the live 1M-ctx cache is ~890 MB, not ~3.2 GB
        # bf16 (the release stores these FP4 too). state() still emits the DEQUANTIZED bf16 (one array
        # per pool), so the prefix/disk-cache serialization + reconstruction format is UNCHANGED;
        # restore re-quantizes via load_dense. See jang_tools.dsv41.model.QuantGrowBuffer.
        self.window = None
        self.compress_kv = QuantGrowBuffer()
        self.index_k = QuantGrowBuffer()
        self.pend_kv = None
        self.pend_score = None
        self._meta = None            # pending meta_state (restore)
        self._pending_state = None   # pending state tuple (restore); finalize when both present

    @property
    def offset(self) -> int:
        return int(self._seq.pos)

    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        """Resident bytes of this layer's KV (window + compressed pools + pend). The memory-aware
        prefix cache sizes/evicts entries by this."""
        total = 0
        for a in (self.window, self.pend_kv, self.pend_score):
            if a is not None:
                total += a.nbytes
        for q in (self.compress_kv, self.index_k):  # 4-bit resident bytes (wq + scales + biases)
            for a in (q.wq, q.scales, q.biases):
                if a is not None and a.shape[1]:
                    total += int(int(a.size) * a.dtype.size * q.n / a.shape[1])
        return int(total)

    # --- serialization (prefix cache in RAM; disk cache on SSD) ---
    @property
    def state(self):
        """Tuple of this layer's live KV arrays (and, on layer 0, the Engram history), in the order
        declared by meta_state. mlx_lm.generate_step evals this each step; the prefix/disk caches store
        and round-trip it through the setter."""
        arrs = []
        if self.window is not None:
            arrs.append(self.window)
        ck = self.compress_kv.view()
        if ck is not None:
            arrs.append(ck)
        ik = self.index_k.view()
        if ik is not None:
            arrs.append(ik)
        if self.pend_kv is not None:
            arrs.append(self.pend_kv)
        if self.pend_score is not None:
            arrs.append(self.pend_score)
        if self._idx == 0 and self._seq.history.size:
            arrs.append(mx.array(self._seq.history.astype(np.int32)))
        return tuple(arrs)

    @state.setter
    def state(self, value):
        # Order-independent restore: the engine may set state before or after meta_state, so stash
        # and finalize once both are present (the array layout is declared in meta.order).
        self._pending_state = tuple(value)
        self._try_finalize()

    @property
    def meta_state(self):
        order = []
        if self.window is not None:
            order.append("window")
        if self.compress_kv.view() is not None:
            order.append("ck")
        if self.index_k.view() is not None:
            order.append("ik")
        if self.pend_kv is not None:
            order.append("pend_kv")
        if self.pend_score is not None:
            order.append("pend_score")
        if self._idx == 0 and self._seq.history.size:
            order.append("hist")
        return json.dumps({
            "order": order, "idx": int(self._idx),
            "ck_n": int(self.compress_kv.n), "ik_n": int(self.index_k.n),
            "pos": int(self._seq.pos),
        })

    @meta_state.setter
    def meta_state(self, value):
        self._meta = json.loads(value) if isinstance(value, str) else dict(value)
        self._try_finalize()

    def _try_finalize(self):
        """Apply a restored (state, meta_state) pair once both have arrived."""
        meta, value = self._meta, self._pending_state
        if meta is None or value is None:
            return
        self._pending_state = None
        self._idx = int(meta.get("idx", self._idx))
        order = meta["order"]
        it = iter(value)
        self.window = next(it) if "window" in order else None
        # state stores the DEQUANTIZED bf16 (one array per pool); re-quantize into the 4-bit resident
        # buffer. meta ck_n/ik_n matches the dense row count, so load_dense restores the same n.
        if "ck" in order:
            self.compress_kv.load_dense(next(it))
        else:
            self.compress_kv.load_dense(None)
        if "ik" in order:
            self.index_k.load_dense(next(it))
        else:
            self.index_k.load_dense(None)
        self.pend_kv = next(it) if "pend_kv" in order else None
        self.pend_score = next(it) if "pend_score" in order else None
        self._seq.pos = int(meta.get("pos", 0))
        if self._idx == 0:
            hist = next(it) if "hist" in order else None
            self._seq.history = (np.asarray(hist).astype(np.int64) if hist is not None
                                 else np.zeros(0, np.int64))

    def trim(self, n: int) -> int:
        """Drop the last ``n`` committed tokens (speculative reject / prefix-length fit). Mirrors
        DeepseekV4Cache.trim: trim the window, drop the proportional tail of the compressed pools,
        clear the compressor's partial group; layer 0 also rolls back pos + Engram history."""
        n = int(n)
        if n <= 0:
            return 0
        if self.window is not None:
            keep = self.window.shape[1] - n
            self.window = self.window[:, :keep] if keep > 0 else None
        self.pend_kv = None
        self.pend_score = None
        r = self._ratio
        if r > 0:
            rows = max(1, n // r)
            for gb in (self.compress_kv, self.index_k):
                gb.n = max(0, gb.n - rows)
                if gb.n == 0:
                    gb.wq = gb.scales = gb.biases = None
        else:
            # ratio 0 layers keep no compressed pool
            for gb in (self.compress_kv, self.index_k):
                gb.wq = gb.scales = gb.biases = None
                gb.n = 0
        if self._idx == 0:
            self._seq.pos = max(0, self._seq.pos - n)
            self._seq.history = self._seq.history[: self._seq.pos]
        return n


class Model(nn.Module):
    """Engine-facing wrapper. Real compute + weights come from an attached ``DSV41`` runtime."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self._rt = None  # DSV41, attached by load_deepseek_v41

    def attach(self, rt) -> "Model":
        self._rt = rt
        return self

    @property
    def runtime(self):
        if self._rt is None:
            raise RuntimeError("DeepseekV41 Model has no runtime attached (use load_deepseek_v41)")
        return self._rt

    @property
    def layers(self):
        return self.runtime.model.layers

    def make_cache(self):
        rt = self.runtime
        ratios = getattr(rt.args, "compress_ratios", None) or [0] * rt.args.num_hidden_layers
        seq = _SeqState()
        seq.caches = [DeepseekV41Cache(seq, i, ratios[i] if i < len(ratios) else 0)
                      for i in range(rt.args.num_hidden_layers)]
        return seq.caches

    def __call__(self, input_ids: mx.array, cache=None, mask=None, all_logits: bool = False,
                 images=None, token_types=None, **kwargs):
        """Append ``input_ids`` (the new chunk) to the sequence; return last-row logits (1,1,V)
        (or every row with ``all_logits``). The engine's per-layer cache list is _SeqState.caches,
        which duck-types the runtime's Session, so DSV41.forward drives it directly. VL inputs
        (``images`` = jang_tools.dsv41.vision.ImageInput list, ``token_types``) splice vision features
        into the start_pos-0 prefill chunk; the runtime handles bias_vl routing + engram masking."""
        rt = self.runtime
        ids = [int(t) for t in mx.array(input_ids).reshape(-1).tolist()]
        if cache and isinstance(cache[0], DeepseekV41Cache):
            seq = _relink(cache)  # unify per-layer caches onto one Session (handles reconstructed lists)
        else:
            seq = rt.session  # fall back to the runtime's default session
        logits = rt.forward(ids, seq, all_logits=all_logits, images=images, token_types=token_types)
        if all_logits:
            return logits[None]            # (1, len, V)
        return logits[None, None, :]       # (1, 1, V)

    def sanitize(self, weights):  # weights load through the DSV41 path, not the generic loader
        return weights


def load_deepseek_v41(model_path, *, engram_dir=None, skip_params_eval: bool = True, wired: bool = True):
    """Load a DeepSeek-V4.1-Flash JANG bundle through the validated runtime and wrap it for the engine.

    Returns ``(model, tokenizer)``. The tokenizer is the bundle's ``tokenizers.Tokenizer``; the caller
    (engine loader) wraps it as needed."""
    from jang_tools.dsv41.model import DSV41

    rt = DSV41(model_path, engram_dir=engram_dir, wired=wired)
    model = Model(ModelArgs(model_type="deepseek_v41", num_hidden_layers=rt.args.num_hidden_layers,
                            hidden_size=rt.args.hidden_size, vocab_size=rt.args.vocab_size,
                            num_attention_heads=rt.args.num_attention_heads, head_dim=rt.args.head_dim))
    model.attach(rt)
    return model, rt.tokenizer
