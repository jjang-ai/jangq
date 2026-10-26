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

### Grouped output projection

`wo_a` is block-diagonal over 8 groups. For prefill (more than `GATHER_MAX_QUERIES` rows) it runs
group-major, one matrix multiply per group; token-major ran one 1-row product per token per group
and took a third of each layer (2048-token chunk: 18.6 → 2.1 ms per layer; cold prefill with the draft
resident 708 → 909 pp/s standalone). Decode and DSpark verify keep the per-row products so each verify
row is computed like a 1-row decode. Numerics: prefill logits vs token-major mean cos 0.9954, top-1
98.4%, which is inside the model's own bf16 sensitivity (moving chunk boundaries 1024 → 512: mean cos
0.9943, min 0.73). Measured per-layer costs at a 2048-token chunk (layer 21): MoE 21.5 ms, attention core
12.8 ms, `wo_a` 2.1 ms (was 18.6), `wq_b` 2.9 ms, hyper-connection mixes 2.6 ms, `wq_a` 1.7 ms.

### Serving memory settings that prefill depends on

The standalone numbers above only hold in `vmlx serve` with two settings, both now defaults:

1. **Allocator-cache floor.** The engine bounds MLX's allocator cache at 5% of resident, capped
   at 4 GB. A 2048-token DSV4.1 chunk needs more scratch than that, so Metal buffers were freed
   and re-created inside every chunk. AR serving, 21k-token cold prompt: 4 GB 475 pp/s, 16 GB
   769, MLX default 772. `vmlx_engine/mlx_memory.py` adds a per-family floor (`deepseek_v41`
   16 GB) that raises the computed default only; `VMLX_MLX_CACHE_LIMIT_MB` still overrides
   exactly. The cache is cleared after every chunk, so idle residency is unchanged.
2. **Prefill chunk with the DSpark draft resident.** Metal's recommended working set on a
   256 GB machine is 239.1 GB (222.7 GiB, `iogpu.wired_limit_mb=0`). The backbone is 197 GiB
   resident, and with the 8.3 GB draft 205.4 GiB. During prefill, active memory peaks at
   ~209 GiB; with 2048-token chunks, active plus allocator cache crosses the working-set limit
   and the GPU slows ~30%. The slowdown is GPU-side, not I/O: warm pages are equally slow, and
   an 8.7 GB dummy MLX allocation without the draft reproduces it (760 → 548–620 pp/s).
   Draft resident, 12k-token cold prompt: chunk 2048 545 pp/s, 1536 643, 1024 708, 768 684;
   allocator cap 8–24 GB makes no difference at 1024. `DSV41DSparkGenerator` therefore caps
   the prefill step at 1024 (`VMLX_DSV41_DSPARK_PREFILL_STEP` overrides). Without the draft,
   2048 stays best. The cliff tracks resident memory, not the allocator cap: with no draft and
   2048-token chunks, +0/2/4/6/8 GB of extra resident memory gives 764/772/773/753/588 pp/s, and with
   +8 GB the cap (6–16 GB) changes nothing. Raising `iogpu.wired_limit_mb` is untested.

## Prefix cache (engine)

- RAM paged prefix cache plus block-disk L2 (`--use-paged-cache --enable-block-disk-cache`).
  Entries are keyed on N-1 prompt tokens: the snapshot is taken at the prompt boundary,
  before the last prompt token, which is re-fed on a hit.
- The block-disk directory is namespaced by a fingerprint of the serving configuration and engine
  code, so entries written by a different engine build are not reused (a new namespace directory
  appears instead). Benchmarks of SSD reuse must restart the same build.
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

Serving results (all fixes merged incl. fp32 head, in-place partial-accept commit and group-major `wo_a`;
DSpark default on; unique-text cold prompts; M5 Ultra 256 GB; *the old 1.6k figure was disturbed by a
vmmap sample taken right before it):

| metric | DSpark (default) | AR (`VMLX_DSV41_DSPARK=0`) |
|---|---|---|
| cold prefill 1.6k / 6.5k / 26.5k tokens | 822 / 933 / 872 pp/s (was 341* / 710 / 676 before the group-major `wo_a`) | 21k: 769 pp/s (before) |
| decode code / code2 / structured / prose | 60.8–66.0 / 46.4–51.1 / 82.4–86.6 / 29.5–31.0 tok/s | 30.8 / 29.6 / 31.2 / 29.3 |
| TTFT, short prompt | 0.24–0.52 s | 0.19–0.59 s |
| RAM prefix hit, 3.4k tokens | 5.18 s → 0.38 s | 6.45 s → 0.33 s |
| SSD hit after restart, 3.4k tokens | 3394 cached, 0.38–0.42 s (same as a RAM hit) | |
| tool turn 2 | 280/354 cached, 0.85 s | 280/354, 0.87 s |
| per-tool-call persist / next-request wait | 7–108 ms / ≤ 93 ms | |
| multimodal | Red / Apple / Red | same |
| multimodal reuse (925-token image turn, text follow-up) | 924/941 cached, 1.48 s → 0.26 s; other image misses ("Blue") | |
| KV at 8k / 32k / 128k (resident; stored format v2 within 0.5%) | 12.0 / 33.1 / 117.5 MiB; marginal 900 B/token → **0.88 GiB at 1M** (stored 904 B/token → 0.89 GiB) | |
| 128k cold prefill (2048 chunks, no draft, average) | 527 pp/s | |
| phys_footprint after load / peak | 206.6 / 213.1 GB | 198.1 / 205.8 GB |

Decode cost by verify width (2k context, warm): 1 row 33 ms, 2 rows 44, 4 rows 51, 6 rows 60,
8 rows 67. The 1→2 step is MLX switching from its 1-row qmv kernel; beyond that each row adds ~4 ms of
extra routed experts. vMLX's row-exact multi-row QMV (`vmlx_engine/metal/row_exact_qmv.py`, bit-equal
to 1-row decode, weight fetched once) would flatten the dense part and make verify rows identical to
decode, but it is disabled on the installed MLX ("qmv traversal changed") and needs its kernel
template regenerated for that MLX.

## Known open items

- Prose decode ≈ AR (acceptance ~0.36; the governor keeps it at AR, not above).
- Regenerate `row_exact_qmv` for the installed MLX and use it for DSpark verify rows (speed +
  exact greedy identity).
- `tests/test_chunking.py` assumes 32-dim index heads, which don't fit the 64-wide groups of the
  4-bit pools; it needs real-geometry heads.
- Logits top-2 are often exactly tied (bf16 head output), so chunk boundaries flip first
  tokens; compare head precision with the reference.
- `iogpu.wired_limit_mb` raise not tested.
