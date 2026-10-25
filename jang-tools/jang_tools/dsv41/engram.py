"""Engram for DSV4.1: n-gram hashing (CPU) + disk-resident FP8 tables.

Port of the release's inference/engram.py. The two tables are ~98 GB each
(384M rows x 256 FP8 bytes + 8 E8M0 scale bytes per row). Each position needs
only (max_ngram_size - 1) * n_heads = 24 rows per Engram layer, so the tables
stay in their original safetensors shards and are read in place with concurrent
pread calls: the OS page cache holds whatever is hot, and none of it is wired
GPU memory.
"""

from __future__ import annotations

import hashlib
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.dsv41.st_io import Shards, e8m0_to_float


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: np.ndarray  # [n_layers, max_ngram_size - 1, n_heads] int64
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, cfg: dict) -> "EngramLayout":
        layer_ids = tuple(cfg["engram_layer_ids"])
        n, heads = cfg["engram_max_ngram_size"], cfg["engram_n_heads"]
        seen: set[int] = set()
        primes = []
        for _ in layer_ids:
            per_ngram = []
            for _ in range(n - 1):
                cur, sizes = cfg["engram_vocab_size"] - 1, []
                for _ in range(heads):
                    cur = _next_prime(cur, seen)
                    seen.add(cur)
                    sizes.append(cur)
                per_ngram.append(sizes)
            primes.append(per_ngram)
        layout = cls(n, layer_ids, tuple(cfg["engram_num_embeddings"]), np.array(primes, np.int64), heads, cfg["engram_head_dim"])
        # bucket ranges tile each table exactly
        assert tuple(int(p.sum()) for p in layout.primes) == layout.num_embeddings, "engram prime layout mismatch"
        return layout


def build_compressed_token_map(tokenizer_json: Path, cache_dir: Path | None = None) -> tuple[np.ndarray, int]:
    """Same normalization as the release: NFKC/NFD/strip accents/lowercase/collapse whitespace."""
    digest = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()[:16]
    cache = (cache_dir or Path.home() / ".cache" / "jang_dsv41") / f"engram_token_map_{digest}.npy"
    if cache.exists():
        lookup = np.load(cache)
        return lookup, int(lookup.max()) + 1

    from tokenizers import Regex, Tokenizer, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    tok = Tokenizer.from_file(str(tokenizer_json))
    n = tok.get_vocab_size(with_added_tokens=True)
    key_to_new: dict[str, int] = {}
    lookup = np.zeros(n, np.int64)
    for tid in range(n):
        text = tok.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = tok.id_to_token(tid)
        else:
            norm = normalizer.normalize_str(text)
            key = norm if norm else text
        lookup[tid] = key_to_new.setdefault(key, len(key_to_new))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, lookup)
    return lookup, len(key_to_new)


def hash_multipliers(layer_ids, max_ngram_size: int, vocab: int) -> np.ndarray:
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    rows = []
    for lid in layer_ids:
        g = np.random.default_rng(10007 * lid)
        rows.append(g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


class NgramHasher:
    """Per-sequence n-gram hash ids. Batch size 1; holds the compressed history across calls."""

    DEAD = -1

    def __init__(self, cfg: dict, layout: EngramLayout, tokenizer_json: Path):
        self.layout = layout
        self.token_map, vocab = build_compressed_token_map(tokenizer_json)
        assert vocab == cfg["engram_compressed_vocab_size"], (vocab, cfg["engram_compressed_vocab_size"])
        self.pad_id = int(self.token_map[cfg["engram_pad_token_id"]])
        flat = layout.primes.reshape(len(layout.layer_ids), -1)
        self.offsets = np.concatenate([np.zeros((flat.shape[0], 1), np.int64), np.cumsum(flat, 1)[:, :-1]], 1)
        self.multipliers = hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab)
        self.history = np.zeros(0, np.int64)

    def reset(self):
        self.history = np.zeros(0, np.int64)

    def __call__(self, input_ids: np.ndarray, start_pos: int) -> np.ndarray:
        """input_ids [L] -> hash ids [L, n_engram_layers, (max_ngram-1)*n_heads] (int64)."""
        out, self.history = self.hash(input_ids, start_pos, self.history)
        return out

    def hash(self, input_ids: np.ndarray, start_pos: int, history: np.ndarray, keep: np.ndarray | None = None):
        """Stateless form: returns (hash ids, updated compressed-token history) for one sequence.
        keep [L] bool (True = real text): image tokens (keep=False) become DEAD so they take part in
        no n-gram and block n-grams that would span them (matches the release engram_mask)."""
        comp = self.token_map[input_ids]
        if keep is not None:
            comp = np.where(keep, comp, self.DEAD)
        history = np.concatenate([history[:start_pos], comp])
        L, n = len(input_ids), self.layout.max_ngram_size
        pos = np.arange(start_pos, start_pos + L)
        toks, blocked = [], np.zeros(L, bool)
        for shift in range(n):
            src = history[np.clip(pos - shift, 0, None)]
            blocked = blocked | (pos < shift) | (src == self.DEAD)
            toks.append(np.where(blocked, self.pad_id, src))
        toks = np.stack(toks, -1)  # [L, n]
        prod = toks[:, None, :] * self.multipliers[None]  # [L, layers, n]
        rolling, out = prod[..., 0], []
        for i in range(1, n):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.layout.primes[None, :, i - 1, :])
        return np.concatenate(out, -1) + self.offsets[None], history


# Cold rows are random 256-byte reads spread over ~98 GB per table, so throughput is set by how
# many reads are in flight. Measured on cold rows: a single-thread memmap gather (one page fault
# at a time) reads ~8-11k rows/s; 16 threads each gathering a slice of the memmap read ~110k
# rows/s (more lanes contend on the VM map and slow down); 64 pread lanes read ~130k rows/s but
# hold the GIL per row, which slows graph building on the main thread during a pipelined
# prefill. The memmap gather is the default; DSV41_ENGRAM_IO=pread selects pread.
_IO_MODE = os.environ.get("DSV41_ENGRAM_IO", "take")
_TAKE_LANES = 16
_TAKE_POOL = ThreadPoolExecutor(_TAKE_LANES, thread_name_prefix="engram-take")
_IO_POOL = ThreadPoolExecutor(64, thread_name_prefix="engram-io")
_IO_TASK = 256
# Prefetch orchestration runs on its own pool so a waiting prefetch never holds an I/O lane.
_PREFETCH_POOL = ThreadPoolExecutor(4, thread_name_prefix="engram-prefetch")


class DiskEngramTable:
    """One layer's FP8 table, read from its safetensors shard in place.

    Rows are read concurrently for every batch size, decode's few rows and prefill's tens of
    thousands alike: the SSD serves random small reads in parallel, while a single gather through
    the memory map takes its page faults one at a time. The OS page cache keeps hot n-grams;
    nothing is wired.
    """

    def __init__(self, shards: Shards, layer_id: int):
        w = shards.meta[f"layers.{layer_id}.engram.embed.weight"]
        s = shards.meta[f"layers.{layer_id}.engram.embed.scale"]
        assert w.dtype == "F8_E4M3" and s.dtype == "F8_E8M0", (w.dtype, s.dtype)
        assert w.path == s.path
        self.rows = np.memmap(w.path, np.uint8, "r", offset=w.offset, shape=w.shape)
        self.scales = np.memmap(s.path, np.uint8, "r", offset=s.offset, shape=s.shape)
        # Random rows: read-ahead around each fault only wastes SSD bandwidth (~1.5x more cold
        # rows/s with MADV_RANDOM).
        for mm in (self.rows, self.scales):
            try:
                mm._mmap.madvise(mmap.MADV_RANDOM)
            except (AttributeError, OSError, ValueError):
                pass
        self.fd = os.open(w.path, os.O_RDONLY)
        self.w_off, self.s_off = w.offset, s.offset
        self.dim = w.shape[1]
        self.sdim = s.shape[1]
        self.block = w.shape[1] // s.shape[1]

    def _pread(self, uniq: np.ndarray):
        """Rows and scales for sorted unique row ids, read concurrently into preallocated
        buffers: _TAKE_LANES slices gathered from the memmap, or pread tasks of _IO_TASK rows."""
        n = len(uniq)
        rows = np.empty((n, self.dim), np.uint8)
        scales = np.empty((n, self.sdim), np.uint8)
        fd, dim, sdim, w_off, s_off = self.fd, self.dim, self.sdim, self.w_off, self.s_off
        ids = uniq.tolist()

        def task(lo):
            for i in range(lo, min(n, lo + _IO_TASK)):
                r = ids[i]
                rows[i] = np.frombuffer(os.pread(fd, dim, w_off + r * dim), np.uint8)
                scales[i] = np.frombuffer(os.pread(fd, sdim, s_off + r * sdim), np.uint8)

        if _IO_MODE == "take":
            step = max(1, -(-n // _TAKE_LANES))

            def take(lo):
                u = uniq[lo : lo + step]
                np.take(self.rows, u, axis=0, out=rows[lo : lo + step])
                np.take(self.scales, u, axis=0, out=scales[lo : lo + step])

            list(_TAKE_POOL.map(take, range(0, n, step)))
            return rows, scales
        if n <= _IO_TASK:
            task(0)
        else:
            list(_IO_POOL.map(task, range(0, n, _IO_TASK)))
        return rows, scales

    def read(self, ids: np.ndarray):
        """Host-side half of lookup (thread-safe, no MLX calls): (inverse map, rows, scales)."""
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        rr, ss = self._pread(uniq)
        return inv, rr, ss

    def prefetch(self, ids: np.ndarray, after=None):
        """Start reading ids' rows in the background (once the future `after`, if given, is done);
        pass the returned future's result to `finish`."""
        def run():
            if after is not None:
                after.result()
            return self.read(ids)
        return _PREFETCH_POOL.submit(run)

    def lookup(self, ids: np.ndarray) -> mx.array:
        """ids [...] -> bf16 rows [..., dim]. Duplicate ids are read once."""
        return self.finish(ids.shape, self.read(ids))

    def finish(self, shape, got) -> mx.array:
        """Dequantize what `read` returned into bf16 rows of `shape + (dim,)`."""
        inv, rr, ss = got
        r = mx.array(rr)
        s = mx.array(ss)
        n = rr.shape[0]
        vals = mx.from_fp8(r, dtype=mx.float32).reshape(n, -1, self.block) * e8m0_to_float(s)[..., None]
        vals = vals.reshape(n, self.dim).astype(mx.bfloat16)
        return vals[mx.array(inv.astype(np.int32))].reshape(*shape, self.dim)


def open_engram_tables(engram_dir: Path, layer_ids) -> dict[int, DiskEngramTable]:
    engram_dir = Path(engram_dir)
    index = "engram.safetensors.index.json" if (engram_dir / "engram.safetensors.index.json").exists() else "none"
    shards = Shards(engram_dir, index_name=index)
    missing = [l for l in layer_ids if f"layers.{l}.engram.embed.weight" not in shards]
    if missing:
        raise FileNotFoundError(f"engram tables for layers {missing} not found under {engram_dir}")
    return {l: DiskEngramTable(shards, l) for l in layer_ids}
