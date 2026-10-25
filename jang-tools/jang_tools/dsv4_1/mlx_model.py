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
    cache contract (offset / state / meta_state / from_state / is_trimmable / compress_ratio).

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
        # bf16 (the release stores these FP4 too). state() serializes the 4-bit pools as stored (format
        # v2 below), so stored prefix/SSD entries are the same size and bit-exact. See
        # jang_tools.dsv41.model.QuantGrowBuffer.
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
    #
    # Format v2 (``meta_state["fmt"] == 2``): the 4-bit compressed/index pools are emitted AS STORED
    # (packed wq uint32 + scales + biases, first n rows only), never dequantized. The old format emitted
    # the dequantized bf16 view (~3.6x larger: ~3.3 GB per 1M-token entry, ~6.4 GB on SSD after the
    # numpy float32 widening) and re-quantized on restore (not bit-exact). bf16/f16 arrays travel as
    # their exact uint16 bit pattern (the numpy-based block store widens bf16 to float32 otherwise);
    # meta "dtypes" records the original dtype so restore views them back bit-exactly.
    #
    # Size: per token ~0.9 KB at 1M context (compressed pools of the 4 KV-source layers, ratio 2/2/2/1,
    # 512-dim rows at 4 bit = 256 B + 8 groups x 2 x 2 B scales/biases; plus 128-dim index rows), i.e.
    # a stored 1M-token entry is ~0.9 GB, like the resident cache. Fixed per entry: the 40 sliding
    # windows (128 x 512 bf16 each, ~5.2 MB) + the Engram history (int32, 4 B/token on layer 0).
    #
    # Pre-v2 entries are REJECTED (restore raises), never migrated: they were bf16 round-trips and were
    # stored N tokens under an N-1 key (the prompt-boundary off-by-one), so they cannot be trusted.
    FORMAT = 2

    @staticmethod
    def _to_bits(a):
        """(array, dtype-name) with bf16/f16 carried as their exact uint16 bit pattern."""
        name = str(a.dtype).split(".")[-1]
        if a.dtype in (mx.bfloat16, mx.float16):
            return a.view(mx.uint16), name
        return a, name

    @staticmethod
    def _from_bits(a, name):
        if name in ("bfloat16", "float16"):
            want = getattr(mx, name)
            if a.dtype == mx.uint16:
                return a.view(want)
            return a.astype(want)  # a store that widened it (value-exact for bf16/f16 sources)
        want = getattr(mx, name, None)
        if want is not None and a.dtype != want:
            return a.astype(want)
        return a

    def _named_arrays(self):
        """[(name, array)] of this layer's live state, 4-bit pools sliced to their n rows."""
        out = []
        if self.window is not None:
            out.append(("window", self.window))
        for tag, q in (("ck", self.compress_kv), ("ik", self.index_k)):
            if q.wq is not None and q.n > 0:
                out.append((f"{tag}_wq", q.wq[:, : q.n]))
                out.append((f"{tag}_s", q.scales[:, : q.n]))
                out.append((f"{tag}_b", q.biases[:, : q.n]))
        if self.pend_kv is not None:
            out.append(("pend_kv", self.pend_kv))
        if self.pend_score is not None:
            out.append(("pend_score", self.pend_score))
        if self._idx == 0 and self._seq.history.size:
            out.append(("hist", mx.array(self._seq.history.astype(np.int32))))
        return out

    @property
    def state(self):
        """Tuple of this layer's live arrays (format v2, see above), in the order declared by
        meta_state["order"]. Layer 0 also carries the Engram history."""
        return tuple(self._to_bits(a)[0] for _, a in self._named_arrays())

    @state.setter
    def state(self, value):
        # Order-independent restore: the engine may set state before or after meta_state, so stash
        # and finalize once both are present (the array layout is declared in meta.order).
        self._pending_state = tuple(value)
        self._try_finalize()

    @property
    def meta_state(self):
        named = self._named_arrays()
        return json.dumps({
            "fmt": self.FORMAT,
            "order": [n for n, _ in named],
            "dtypes": {n: str(a.dtype).split(".")[-1] for n, a in named},
            "idx": int(self._idx), "ratio": int(self._ratio),
            "ck_n": int(self.compress_kv.n), "ik_n": int(self.index_k.n),
            "ck_D": self.compress_kv.D, "ik_D": self.index_k.D,
            "pos": int(self._seq.pos),
        })

    @meta_state.setter
    def meta_state(self, value):
        if isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        self._meta = json.loads(value) if isinstance(value, str) else dict(value)
        self._try_finalize()

    def _try_finalize(self):
        """Apply a restored (state, meta_state) pair once both have arrived."""
        meta, value = self._meta, self._pending_state
        if meta is None or value is None:
            return
        self._pending_state = None
        self._meta = None
        if int(meta.get("fmt", 0) or 0) != self.FORMAT:
            raise ValueError(
                f"stale DeepseekV41Cache entry (format {meta.get('fmt', 1)}, need {self.FORMAT}): "
                "pre-v2 entries were bf16 round-trips stored at the wrong prompt boundary; rejecting")
        order = list(meta["order"])
        if len(order) != len(value):
            raise ValueError(f"DeepseekV41Cache state has {len(value)} arrays, meta declares {len(order)}")
        dtypes = meta.get("dtypes") or {}
        arrs = {n: self._from_bits(a, dtypes.get(n, "")) for n, a in zip(order, value)}
        self._idx = int(meta.get("idx", self._idx))
        if meta.get("ratio") is not None:
            self._ratio = int(meta["ratio"])
        self.window = arrs.get("window")
        for tag, q in (("ck", self.compress_kv), ("ik", self.index_k)):
            n = int(meta.get(f"{tag}_n", 0) or 0)
            if f"{tag}_wq" in arrs:
                wq, sc, bi = arrs[f"{tag}_wq"], arrs[f"{tag}_s"], arrs[f"{tag}_b"]
                if not (wq.shape[1] == sc.shape[1] == bi.shape[1] == n):
                    raise ValueError(f"DeepseekV41Cache {tag} pool rows {wq.shape[1]} != meta n {n}")
                q.wq, q.scales, q.biases, q.n = wq, sc, bi, n
                q.D = meta.get(f"{tag}_D") or int(wq.shape[2]) * 32 // q.BITS
            else:
                if n:
                    raise ValueError(f"DeepseekV41Cache {tag} pool missing ({n} rows declared)")
                q.wq = q.scales = q.biases = None
                q.n, q.D = 0, None
        self.pend_kv = arrs.get("pend_kv")
        self.pend_score = arrs.get("pend_score")
        self._seq.pos = int(meta.get("pos", 0))
        if self._idx == 0:
            hist = arrs.get("hist")
            self._seq.history = (np.asarray(hist).astype(np.int64) if hist is not None
                                 else np.zeros(0, np.int64))

    @classmethod
    def from_state(cls, state, meta_state):
        """mlx_lm-style constructor (disk prompt cache restore)."""
        c = cls()
        c.meta_state = meta_state
        c.state = state
        return c

    @property
    def compress_ratio(self):
        """The layer's compression ratio, None for a sliding-window-only (ratio 0) layer -- the
        engine's DSV4 record validator requires a ratio in [1, 4096] when one is given. The restore
        does not depend on it (meta_state carries "ratio")."""
        return int(self._ratio) if self._ratio else None

    @property
    def sliding_window(self):
        return None if self.window is None else int(self.window.shape[1])

    def is_trimmable(self) -> bool:
        # Not exactly trimmable: the sliding window keeps only the last `win` rows (older rows cannot be
        # rebuilt), dropping compressed rows needs the ratio-aligned group boundary, and the compressor's
        # partial group (pend_*) is path state. Rollback is checkpoint()/restore() on the _SeqState.
        return False

    def trim(self, n: int) -> int:
        """No-op (returns 0 trimmed): see is_trimmable. The previous lossy trim (window slice, pool
        rows max(1, n // r), cleared pend) produced a state no forward could have produced."""
        return 0


def clone_cache_list(cache: list) -> list:
    """Deep copy of an engine DeepseekV41Cache list (a prompt-boundary snapshot), bit-exact and compact:
    the 4-bit pools are copied as stored (first n rows of wq/scales/biases), never dequantized, and all
    layers share ONE new _SeqState (pos + one copy of the Engram history, not one per layer)."""
    def cp(a):
        return None if a is None else mx.array(a)  # mx.array(a) copies

    src_seq = cache[0]._seq
    seq = _SeqState()
    seq.pos = int(src_seq.pos)
    seq.history = src_seq.history.copy()
    out = []
    for c in cache:
        cl = DeepseekV41Cache(seq, c._idx, c._ratio)
        cl.window = cp(c.window)
        for src, dst in ((c.compress_kv, cl.compress_kv), (c.index_k, cl.index_k)):
            if src.wq is not None and src.n > 0:
                dst.wq = cp(src.wq[:, : src.n])
                dst.scales = cp(src.scales[:, : src.n])
                dst.biases = cp(src.biases[:, : src.n])
                dst.n, dst.D = int(src.n), src.D
        cl.pend_kv = cp(c.pend_kv)
        cl.pend_score = cp(c.pend_score)
        out.append(cl)
    seq.caches = list(out)
    arrays = [a for c in out for _, a in c._named_arrays()]
    if arrays:
        mx.eval(arrays)
    return out


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
        into the start_pos-0 prefill chunk; the runtime handles bias_vl routing + engram masking.
        ``engram_lookahead`` (list of token ids) is the next prefill chunk: its Engram rows are read
        from SSD while this chunk computes."""
        rt = self.runtime
        ids = [int(t) for t in mx.array(input_ids).reshape(-1).tolist()]
        if cache and all(isinstance(c, DeepseekV41Cache) for c in cache):
            seq = _relink(cache)  # unify per-layer caches onto one Session (handles reconstructed lists)
        elif cache:
            # A cache list that is not (entirely) DeepseekV41Cache -- e.g. a generic disk-cache restore
            # that rebuilt unknown classes as empty KVCache. Falling back to the runtime's shared
            # session would silently continue a stale sequence; refuse instead.
            raise TypeError("DeepseekV41 Model needs a DeepseekV41Cache list (got "
                            + ", ".join(sorted({type(c).__name__ for c in cache})) + ")")
        else:
            seq = rt.session  # no engine cache: the runtime's default session (standalone use)
        logits = rt.forward(ids, seq, all_logits=all_logits, images=images, token_types=token_types,
                            engram_lookahead=kwargs.get("engram_lookahead"))
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
