# DeepSeek-V4.1-Flash — runtime and serving contract (vmlx-py)

Companion to `DECODE_DSPARK_METHODS.md` (decode and DSpark methods). This page covers the
pieces between the runtime and the vMLX engine: residency, prefill and Engram I/O, prefix
caching, multimodal, and how to measure all of it. Measured numbers come from an M5 Ultra
(256 GB) serving the `DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX` bundle.

## Components

| piece | file |
|---|---|
| runtime (backbone, caches, Engram, VL splice) | `jang_tools/dsv41/model.py` (`DSV41`, `Session`) |
| Engram hashing + on-disk tables | `jang_tools/dsv41/engram.py` (`NgramHasher`, `DiskEngramTable`) |
| DSpark 3-stage MTP draft + adaptive loop | `jang_tools/dsv41/dspark.py` (`DSparkRunner`) |
| vision tower | `jang_tools/dsv41/vision.py` |
| engine model wrapper + per-layer cache | `jang_tools/dsv4_1/mlx_model.py` (`Model`, `DeepseekV41Cache`, `_SeqState`) |
| engine loader / generator / scheduler | vMLX `vmlx_engine/loaders/load_jangtq_dsv41.py`, `utils/dsv41_dspark_generator.py`, `utils/single_batch_generator.py`, `scheduler.py` |

## Residency

- Backbone weights (main bundle dir, ~197 GB) are loaded into owned, wired memory. No mmap
  compute paths, no streaming, no dequantized copies.
- The two Engram tables (`engram/`, ~98 GB each, layers 1 and 14) are the only weights left
  on SSD. They are read in place from their safetensors shards; the OS page cache keeps hot
  rows (not wired, not counted as owned memory).
- Expected footprint after load: ~197 GB phys_footprint. If a bundle's `engram/` is missing,
  loading fails (`open_engram_tables` raises) instead of running without Engram.

## Prefill and Engram I/O

Each position reads `(max_ngram_size - 1) * n_heads = 24` random 256-byte rows (plus 8 scale
bytes) per Engram layer. On unique text those rows are cold, so prefill is bound by how many
random SSD reads are in flight, not by compute.

Measured (unique text, 2048-token engine chunks, standalone runtime):

| tokens | before (1-thread memmap gather) | cold now | warm pages |
|---|---|---|---|
| 1024 | 68 pp/s | 598 | 741 |
| 2048 | — | 713 | 801 |
| 4096 | 459 | 756 | 794 |
| 8192 | 382 | 768 | 784 |
| 16384 | — | 746 | 762 |

Design, and the rules it encodes:

1. **Concurrent reads.** `DiskEngramTable._pread` gathers from the memmap on 16 threads
   (`np.take` releases the GIL), with `MADV_RANDOM` on the mapping: ~110–146k cold rows/s,
   against 8–11k for a single-thread gather. More than ~16 gather threads collapse on VM-map
   contention. `DSV41_ENGRAM_IO=pread` selects 64-lane `os.pread` (~130k rows/s, but it holds
   the GIL per row).
2. **One forward per chunk.** Every forward streams the routed experts' weights, about 0.14 s
   of fixed cost at prefill batch sizes, so splitting a chunk into sub-chunks to hide reads
   costs more than it saves. Don't add forwards to overlap I/O.
3. **Per-layer lazy rows.** `_engram_rows(lazy=True)`: layer 1's rows are read up front;
   later Engram layers get a callable that `Model.__call__` resolves when it reaches that
   layer, after `mx.async_eval` of the layers before it, so those reads overlap GPU work.
   Layer reads are chained first-layer-first so the first gets every I/O lane.
4. **Next-chunk lookahead.** `forward(ids, sess, engram_lookahead=next_ids)` starts the next
   chunk's hashes and reads while this chunk computes. The record on the session is used
   only if the next forward's start position and token ids match exactly; otherwise it is
   discarded. The engine's chunked prefill loops pass the next chunk (`_model_call(...,
   lookahead=...)`, gated on `model_type == "deepseek_v41"`). Any other caller that prefills
   in chunks should pass the lookahead too; an unchunked prefill gets only (3).
5. Output is bit-identical with and without (3) and (4). Chunk boundaries themselves change
   bf16 numerics slightly (cosine ~0.999 between single-pass and chunked logits), the same as
   the engine's existing chunked prefill.

The 1k case stays near 600 pp/s cold because layer 1's reads (~0.2 s) come before any compute
and the first chunk has nothing to overlap with.

## Prefix cache (engine)

- RAM paged prefix cache plus block-disk L2 (`--use-paged-cache --enable-block-disk-cache`).
  Entries are keyed on N-1 prompt tokens: the snapshot is taken at the prompt boundary,
  before the last prompt token, which is re-fed on a hit.
- Per-tool-call persistence: the terminal snapshot is written when a request finishes;
  the next request waits for it (`Terminal persistence settled ... persist_ms`,
  `Admission waited ... wait_ms` in the server log). Measured: `persist_ms` 7–121 ms, admission
  wait 14–116 ms.
- Stored-size target: ~0.9 GB for a 1M-token entry, the same as the resident 4-bit
  compressed-KV pools (the resident pools measured 905 MB at 1M). Entries must store the 4-bit
  pools as they are, without round-tripping through bf16.

## Multimodal

- VL requests run through the DSpark generator subclass of `SingleBatchGenerator`. The
  scheduler must detect the generator with `isinstance`, not a class-name comparison, or
  images are silently dropped.
- Image spans must be in a single start_pos-0 prefill forward. Image tokens take no part in
  n-grams (`engram_mask = ~image_mask`), and image prefills don't use the lookahead path.

## Measuring

`tests/serve_acceptance.py` drives a real `vmlx serve` process (one model load per run). It
reports cold prefill, decode tok/s by content type, TTFT, RAM/SSD prefix reuse, per-tool-call
reuse, multiturn recall, multimodal answers, reasoning-parser leaks, and the server process
phys_footprint. Run with `VMLX_DSV41_DSPARK=0` for the AR baseline. Load only one copy of the
model at a time: a second ~200 GB load on a 256 GB machine will be killed.

Serving results (DSpark default on, 2048-token chunks, after the Engram I/O change):

| metric | result |
|---|---|
| cold prefill 1.3k | 614 pp/s |
| cold prefill 5.4k | 294 pp/s (DSpark prefill was one unchunked forward; chunked + lookahead pending) |
| decode code / code2 / structured / prose | 53.8 / 42.8 / 83.0 / 29.2 tok/s (AR ~30) |
| TTFT, short prompt | 0.25–0.36 s |
| RAM prefix hit, 3.4k tokens | 11.5 s → 0.58 s |
| tool turn 2 | 280/354 tokens cached, 0.81 s |
| multimodal (red image) | Red / Strawberry / Red |

## Known open items

- DSpark prefill must chunk at the engine step size and pass `engram_lookahead`.
- DSpark must stay at or above AR on low-acceptance text (prose measured 29.2 vs AR ~30).
- Multimodal turns don't reuse the prefix cache yet.
- Stored prefix-cache entries: measure single-entry bytes per token at 8k/32k and confirm the
  ~0.9 GB at 1M target.
