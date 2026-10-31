# DeepSeek-V4.1-Flash — runtime and serving contract (vmlx-py)

Companion to `DECODE_DSPARK_METHODS.md` (decode and DSpark methods). This page covers the
pieces between the runtime and the vMLX engine: residency, prefill and Engram I/O, prefix
caching, multimodal, and how to measure all of it. Measured numbers come from an M5 Ultra
(256 GB) serving the `DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX` bundle.

## Speed ledger: why it was slow, and what each piece does

Every row was measured on the real model (M5 Ultra 256 GB) and is exact or bounded as noted. Read this first
before touching a hot path: each fix removes one specific cause, and reverting any one of them brings that
cause back. Details are in the linked sections.

**Prefill** (unique-text cold prompts; the first version ran 68 pp/s at 1k tokens):

| was slow because | fix (where) | measured |
|---|---|---|
| Engram rows (24 random 256-byte reads per position per Engram layer, ~98 GB tables) were gathered by one thread: one page fault in flight | 16 concurrent memmap gathers with `MADV_RANDOM` (`engram.DiskEngramTable._gather`); `np.take` releases the GIL so graph building continues | 8–11k → 110–146k cold rows/s; 1k prefill 68 → 598 pp/s |
| every Engram read waited before any compute | layer 1's rows first, layer 14's rows resolved lazily after `mx.async_eval` of layers 0–13 (`DSV41._engram_rows(lazy=True)`) | reads overlap compute (bit-identical) |
| the next chunk's reads started only when it ran | `engram_lookahead`: the next chunk's hashes and reads start while this one computes (engine passes it) | 4k 459 → 756 pp/s with the above |
| (rule) each forward streams the routed experts once, ~0.14 s at prefill widths | one forward per chunk; never split a chunk to hide I/O | — |
| the engine capped MLX's allocator cache at 4 GB, so every 2048-token chunk freed and re-created its Metal buffers | 16 GB per-family floor (vMLX `mlx_memory.model_cache_limit_floor_bytes`) | 21k 475 → 769 pp/s |
| with the 8.3 GB draft resident, 2048-token chunks crossed Metal's recommended working set and the GPU slowed ~30% | 1024-token chunks while the draft is resident (`DSV41DSparkGenerator`) | 545 → 708 pp/s |
| `wo_a` (block-diagonal over 8 groups) ran token-major: one 1-row product per token per group, a third of each layer | group-major for prefill, one matmul per group (`Attention.__call__`) | 18.6 → 2.1 ms per layer; 708 → 909 pp/s |
| a pool shared by several layers was dequantized once per consumer | `Shared.view` memoizes the dequantized pool for one forward | one dequant per pool per forward |
| index scores materialized `[C, heads, T]` and reduced it | fused `sum_h w * relu(q . k)` kernel (vMLX `dsa_pool_index_scores`) | 100k keys: 164 → 97 ms per index layer |
| MLX's GPU `argpartition` sorts the whole row | block top-k for prefill rows >= 65536 keys (`topk_indices`), exact values | 2.4x at 100k, 16x at 1M |
| the fused scorer ran on simdgroup fp32 (~27 TFLOPS ceiling) | same kernel on the M5 matrix units with bf16 operands (vMLX `dsa_index_scores_mpp`) | 4.3x; 100k avg 831 → 884 pp/s |
| layers 21–39 computed every prompt row, though only the last rows reach the prompt end through their 128-token windows | prompt-suffix pruning, `1 + (40 - i) * 127` trailing rows per layer (`Model.prune_rows`), exact at the prompt end | 32k 938 → 1641, 100k 875 → 1584 pp/s |

**Decode** (AR started at 29–31 tok/s; DSpark at 26.8 in the first live build):

| was slow because | fix (where) | measured |
|---|---|---|
| bf16 logits tied every 50–100 tokens, so batched verify and 1-row decode broke ties differently and DSpark drifted | fp32 LM-head input, as the reference (`Model.__call__`) | correctness (enables the rows below) |
| each DSpark step ran a verify forward and then a second forward to commit the accepted tokens | full accepts keep the verify's state; partial accepts replay only the kept tokens' cache updates (`DSV41.commit_prefix`) | code 44.9 → 57.5 tok/s |
| speculation could run slower than AR on low-acceptance text | wall-clock governor: speculate only while faster than a measured AR step, re-measured as context grows (`dspark._Governor`) | never below AR |
| decode/verify attention dequantized the whole 4-bit pool every step (~0.5–1 GB per layer per token at 1M) | gather only the selected rows from the 4-bit pool (`append_compressed_rows(gather=True)`) | bit-identical; flat in context |
| a decode step was ~150 dispatches per layer, built for all 40 layers before the GPU started | per-layer `mx.async_eval`, bf16 router GEMV, 2-dispatch MoE, fused hyper-connection stage (`metal_decode.py`, M5 in DECODE_DSPARK_METHODS) | AR 29.0 → 39.2 tok/s (2k), 26.0 → 34.3 (100k); DSpark code 65.7 → 90.2 |

**Prefill and decode touch different parameter counts.** A decode token reads 6 of 384 routed experts per
layer plus the shared expert (~10B parameters, a few GB at 2-3 bits): decode is bound by bytes and dispatches per
token, hence the low-bit experts, the fused decode kernels, and DSpark, whose 6-row verify reads attention and
shared weights once for 6 tokens at the cost of up to 36 experts (the governor speculates only while that is
faster). A 2048-token prefill chunk routes ~12k token-expert pairs, so every expert is read once per chunk (~0.14 s)
and shared by ~32 tokens: prefill amortizes the whole expert set per chunk, hence one forward per chunk and
expert-sorted grouped matmuls; a larger chunk halves that per-token cost but it is ~10 % of a chunk's time, and with
the draft resident the chunk is 1024 for memory.

**Memory**: compressed-KV and index pools are stored 4-bit (`QuantGrowBuffer`, the reference's FP4 footprint) and
the window keeps 128 rows: 905 MiB resident at 1M tokens (vs ~3.3 KB/token before).

Measured and removed (no gain): standalone decode attention kernel, MoE expert dedup across verify rows
(DECODE_DSPARK_METHODS M4), 64-lane `os.pread` Engram reads, a scalar fused prefill sparse-attention kernel
(14.4 vs 13.7 ms for the gather path at 2048 queries), bf16 gather in prefill attention (no gain at 256k).

## Components

| piece | file |
|---|---|
| runtime (backbone, caches, Engram, VL splice) | `jang_tools/dsv41/model.py` (`DSV41`, `Session`) |
| Engram hashing + on-disk tables | `jang_tools/dsv41/engram.py` (`NgramHasher`, `DiskEngramTable`) |
| DSpark 3-stage MTP draft + adaptive loop | `jang_tools/dsv41/dspark.py` (`DSparkRunner`) |
| vision tower | `jang_tools/dsv41/vision.py` |
| engine model wrapper + per-layer cache | `jang_tools/dsv4_1/mlx_model.py` (`Model`, `DeepseekV41Cache`, `_SeqState`) |
| engine loader / generator / scheduler | vMLX `vmlx_engine/loaders/load_jangtq_dsv41.py`, `utils/dsv41_dspark_generator.py`, `utils/single_batch_generator.py`, `scheduler.py` |

## Environment flags

Read by this package. Every `VMLX_DSV41_*` flag defaults to on and is turned off by `0`, `false`, `off` or
`no` (any case, `model._env_on`); all but `VMLX_DSV41_FUSED_INDEX` are read once at import. Turning one off
restores the slower path it replaced; numerics are described in the sections above (bit-identical except
the index-score kernels and suffix pruning, which is exact at the prompt end).

| flag | default | effect when on | where |
|---|---|---|---|
| `VMLX_DSV41_LAYER_ASYNC` | on | forwards of <= 16 rows submit each layer to the GPU as it is built (`mx.async_eval`); bit-identical | `model.py:43` |
| `VMLX_DSV41_ROUTER_GEMV` | on | router logits for <= 8 rows from the bf16 gate weight in MLX's f32 GEMV order (`metal_decode.router_logits`) | `model.py:48` |
| `VMLX_DSV41_FUSED_MOE` | on | routed + shared experts for <= 8 rows in two Metal dispatches (`metal_decode.moe_decode`); 1-row JANGT decode through vMLX `jt_moe_decode` | `model.py:52` |
| `VMLX_DSV41_FUSED_HC` | on | hyper-connection stage for <= 8 rows in three dispatches per sublayer; self-checked once per process, only where f32 GEMMs round to TF32 | `model.py:57` |
| `VMLX_DSV41_HC_BOUNDARY` | on | with `FUSED_HC`: each boundary in two dispatches (`metal_decode.hc_mix_decode` + `hc_boundary_decode`), the attention branch's expand folded into the FFN boundary. Bit-equal to the stage except pre / post / comb (the 24 mix dot products and their sum of squares sum in another order); checked once per process. JANGHT1.4 AR: prose 33.05 -> 35.54, code 33.39 -> 36.65 tok/s with `ATTN_SPLIT` | `model.py:62` |
| `VMLX_DSV41_ATTN_SPLIT` | on | decode / verify rows' sparse attention split by keys (`metal_decode.attend_decode`): window rows and 4-bit pool picks read by index, full fp32 sums (the gather + compiled core's matmuls round to TF32 on M5); checked once per process against `sparse_attention` | `model.py:67` |
| `VMLX_DSV41_FAST_ROPE` | on | `(start, step)` positions through `mx.fast.rope` instead of the compiled cos/sin graph | `model.py:123` |
| `VMLX_DSV41_SUFFIX_PRUNE` | on | prompt-suffix pruning of layers 21–39 when `prompt_end` is passed (Prompt-suffix pruning above) | `model.py:356` |
| `VMLX_DSV41_KV_SOURCE_TRIM` | on | with suffix pruning: a forward that ends before layer 21's kept rows runs layer 20's cache side only (window kv, compressor, index keys) and stops (Prompt-suffix pruning below); bit-identical at the prompt end | `model.py:379` |
| `VMLX_DSV41_INDEX_BF16` | on | fused index scores with bf16 operands on the M5 matrix units; off or older vMLX: fp32 operands | `model.py:393` |
| `VMLX_DSV41_INDEX_CHUNK_EVAL` | on | long-context prefill evaluates each index query chunk's top-k before the next chunk (lower transient memory) | `model.py:399` |
| `VMLX_DSV41_FUSED_INDEX` | on | prefill index scores through vMLX `dsa_pool_index_scores` (read per call); off or no vMLX: stock bf16 path | `model.py:500` |
| `MLX_ENABLE_TF32` | on (MLX's flag) | read only to decide whether f32 GEMMs round to TF32 (`metal_decode._tf32`), which gates `VMLX_DSV41_FUSED_HC`; `0` / `false` / `False` keeps the stock hc ops | `metal_decode.py:750` |

Read by vMLX, not here: `VMLX_FP8_ROWS_QMV` (the multi-row mxfp8 QMV behind `model.FP8_ROWS`, per call),
`VMLX_DSV41_DSPARK` (load the draft), `VMLX_DSV41_DSPARK_ADAPTIVE` (governor), `VMLX_DSV41_DSPARK_SAMPLED`
(sampled requests through `SpecSampler`), `VMLX_DSV41_DSPARK_PREFILL_STEP` (prefill chunk with the draft
resident).

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

1. **Concurrent reads.** `DiskEngramTable._gather` gathers from the memmap on 16 threads
   (`np.take` releases the GIL), with `MADV_RANDOM` on the mapping: ~110–146k cold rows/s,
   against 8–11k for a single-thread gather. More than ~16 gather threads collapse on VM-map
   contention. 64-lane `os.pread` reached ~130k rows/s but held the GIL per row and slowed graph
   building, so it was removed.
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

### Long-context indexer

Each of the 8 index layers scores every compressed row for every query (32 heads x 128), so the
indexer is the cost that grows with context. Two changes, both exact up to ties:
- **Fused scores:** prefill (more than 8 rows) uses vMLX's `dsa_pool_index_scores` Metal kernel,
  `sum_h w * relu(q . k)` in fp32 in one pass, instead of materializing `[C, heads, T]` and reducing
  it (100k keys: 164 -> 97 ms per index layer per 2048-token chunk). Query chunks keep the `[C, T]`
  fp32 block near 1 GB. Optional import; `VMLX_DSV41_FUSED_INDEX=0` or no vMLX keeps the bf16 path.
  Numerics vs bf16: prefill logits mean cos 0.9962, top-1 agreement 100%.
- **Block top-k (prefill only):** MLX's GPU `argpartition` sorts the whole row. For prefill rows
  (more than 8 queries) of 65536+ keys, `topk_indices` picks the 512 blocks of 64 with the largest
  maxima (they hold the top-512 values) and partitions only those entries. Identical selected values;
  2.4x faster at 2048 x 100k, 16x at 256 x 1M. Decode keeps the direct partition: for one row it is
  cheaper (170k context: 25.1 vs 23.9 tok/s), and decode index scores take the bf16 path where exact
  ties are common, so a different pick among tied values would change decode output.
- **Matrix units:** the simdgroup fp32 kernel tops out near 27 TFLOPS (fp16 fragments are no faster:
  simdgroup MMA has one rate for half and float). vMLX's `dsa_index_scores_mpp` runs the same fused
  function on the M5 matrix units (Metal 4 tensor ops) with bf16 operands at ~115-120 TFLOPS: 4.3x the
  fp32 kernel at 2048 x 100k keys. The indexer passes its bf16 tensors (`bf16_operands=True`; falls back
  to the fp32 kernel on older vMLX / non-M5 / `VMLX_DSV41_INDEX_BF16=0`). Cold prefill to 100k: average
  831 -> 884 pp/s, 40-50k 913 -> 963, 88-100k 818 -> 910. Numerics vs the fp32 kernel: mean cos 0.9964,
  top-1 100%, greedy continuation identical. The per-layer attention input (window + whole pool, concatenated and cast to fp32) also grows
with context and pushes peak active memory to ~214 GiB at 1M, past the resident cliff below; gathering
in bf16 instead is output-identical but measured no faster at 256k.

### Prompt-suffix pruning

Layers 21–39 own no cache but their 128-token window: the compressed pool, index keys and candidates
they read all come from layer 20 (the last KV source; the Engram layers are 1 and 14). For a prefill that
ends at `prompt_end`, layer i only needs the trailing `1 + (40 - i) * 127` rows (layer 39: 128,
layer 21: 2414): each window reaches 127 rows back, so a layer's first 127 kept rows see a partial
window and the layer above starts 127 rows later. The logits at `prompt_end - 1` and every layer's
window at `prompt_end` are those of a full prefill (`Model.prune_rows`, ported from ds4 #1272).

- `DSV41.forward(..., prompt_end=)`; the engine's `SingleBatchGenerator._prefill` passes the end of each
  prefill call (the N-1 prompt boundary), `DSparkRunner.prefill` the end of the prompt body. Decode,
  DSpark verify (`capture_verify`), `all_logits` and multi-segment forwards never prune.
- A layer that drops rows restarts its window at the kept suffix; a forward entirely before a layer's
  kept range still runs its last row there. DSpark `return_hidden` rows are exact for the last 128
  positions (all the draft reads), zero before.
- Prefix-cache snapshots are only valid at `prompt_end`. Every engine snapshot is taken there.
- A forward that ends before layer 21's kept range (every prefill chunk but those holding the last ~2.4k
  prompt rows) feeds nothing exact from layer 20's output on. Layer 20 then runs only its cache side on
  every row -- `attn_norm(hc_pre(h))`, the window kv, the compressor and the index keys, the same N-row
  calls as the full layer -- and the forward stops: no query side / index scoring / attention / wo / MoE
  at layer 20, no one-row runs of layers 21–39, their windows restart (`Model.trim_layer`,
  `Block.source_rows`). A forward that reaches the kept rows runs the whole layer, so every row the
  prompt end reads is computed by the same calls as before: logits, `return_hidden` rows and every lane
  at the prompt end are bit-identical (`tests/dsv41_kv_source_trim_test.py`). Trimming only part of a
  chunk's rows would not be: MLX's projections round differently at other row counts.
  `VMLX_DSV41_KV_SOURCE_TRIM=0` runs the whole stack on every forward.
- `VMLX_DSV41_SUFFIX_PRUNE=0` disables. CPU test: `tests/test_suffix_prune.py` (tiny random model,
  three pruned layers, chunk sizes 17–256).

Measured (real model, cold unique text, 2048-token chunks, no draft): 8k 1019 → 1452 pp/s, 32k 938 →
1641, 100k 875 → 1584. Exactness at 8k / 32k: last-token logits cos 0.99996 / 0.99994, same top-5,
24 of 24 greedy tokens identical; window caches bit-identical through layer 27, bf16-level differences
above.

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
- Images may sit anywhere in the prompt. The engine's chunked prefill (`SingleBatchGenerator._model_call`)
  passes each image with the chunk that holds its span, `ImageInput.start` made chunk-relative, and
  `token_types` sliced to the chunk; `_dsv41_vl_chunk_len` shortens or extends a chunk so no image is
  split. `DSV41.forward` / `Model.__call__` take any chunk this way. Image tokens take no part in n-grams
  (`engram_mask = ~image_mask`), and chunks with image tokens don't use a lookahead record.

## Measuring

`tests/serve_acceptance.py` drives a real `vmlx serve` process (one model load per run). It
reports cold prefill, decode tok/s by content type, TTFT, RAM/SSD prefix reuse, per-tool-call
reuse, multiturn recall, multimodal answers, reasoning-parser leaks, and the server process
phys_footprint. Run with `VMLX_DSV41_DSPARK=0` for the AR baseline. Load only one copy of the
model at a time: a second ~200 GB load on a 256 GB machine will be killed.

Serving results (2026-10-26: prompt-suffix pruning + fused decode kernels (DECODE_DSPARK_METHODS M5) on top of
the fp32 head, in-place partial-accept commit, group-major `wo_a` and matrix-unit index scores; DSpark
default on; unique-text cold prompts; M5 Ultra 256 GB; previous round in parentheses):

| metric | DSpark (default) | AR (`VMLX_DSV41_DSPARK=0`) |
|---|---|---|
| cold prefill 1.6k / 6.5k / 26.5k tokens (1024-token chunks, draft resident) | 845 / 1331 / 1554 pp/s (822 / 955 / 940) | 21k: 769 pp/s (before) |
| decode code / code2 / structured / prose | 90.2 / 62.3 / 111.9 / 38.7 tok/s (60.8–66.0 / 46.1–51.1 / 82.4–86.7 / 29.5–31.0) | AR greedy 39.1–39.2 at 2k, 34.3 at 100k (29.0 / 26.0 before M5) |
| decode after a 170k-token context (AR) | 25.0 tok/s; tokens identical with/without the block top-k | |
| TTFT, short prompt | 0.21–0.46 s | 0.19–0.59 s |
| RAM prefix hit, 3.4k tokens | 3.38 s → 0.36 s | 6.45 s → 0.33 s |
| SSD hit after restart, 3.4k tokens | 3394 cached, 0.34 s (same as a RAM hit) | |
| tool turn 2 | 280/354 cached, 0.75 s | 280/354, 0.87 s |
| per-tool-call persist / next-request wait | 2–118 ms / ≤ 102 ms (22 writes) | |
| multimodal | Red / Apple / Red | same |
| multimodal reuse (925-token image turn, text follow-up) | 924/941 cached, 1.20 s → 0.22 s; other image misses ("Blue") | |
| KV measured at 256k / 512k / **1M** tokens (resident; stored format v2) | 230.0 / 455.0 / **905.0 MiB** (920 / 910 / 905 B/token); stored 231.0 / 457.0 / **909.0 MiB** — 1M = 949 MB | |
| long-context cold prefill (2048 chunks, no draft) | with suffix pruning 8k / 32k / 100k: 1452 / 1641 / 1584 pp/s (875 at 100k without); matrix-unit index scores alone 100k: 884 pp/s avg (40–50k 963, 88–100k 910); before the matrix-unit kernel 1M: 332 avg (windows 0–128k 792, 128–256k 618, 256–384k 431, 384–512k 366, 512–640k 306, 640–768k 270, 768–896k 234, 896k–1M 203) | |
| phys_footprint after load / end / peak | 206.8 / 206.9 / 213.2 GB | 198.1 / 205.8 GB |

Decode cost by verify width before M5 (2k context, warm): 1 row 33 ms, 2 rows 44, 4 rows 51, 6 rows 60,
8 rows 67. The 1→2 step is MLX switching from its 1-row qmv kernel; beyond that each row adds ~4 ms of
extra routed experts. vMLX's row-exact multi-row QMV (`vmlx_engine/metal/row_exact_qmv.py`, bit-equal
to 1-row decode, weight fetched once) would flatten the dense part and make verify rows identical to
decode, but it is disabled on the installed MLX ("qmv traversal changed") and needs its kernel
template regenerated for that MLX.

### Long-context retrieval (2026-10-27)

Greedy, thinking off, question about a line at ~54 % depth, JANG_2K, `jang_tools` runtime (same answers through
`vmlx serve`; vmlx-swift misses the same hard variant):

| prompt | 300k | 461k | 844k–925k |
|---|---|---|---|
| one distinctive sentence in natural text (package sources) | | correct, p 0.994 | correct at 844k, p 0.986 |
| `Doc line N: value V`, newline-joined, random 3-digit values | | correct, p 0.90 | correct at 925k, p 0.99 |
| hard: `Doc line r1m-N: value V tag K.`, space-joined (two numbers per line) | correct, p 0.99 | wrong (402 / 833, p ≤ 0.06) | wrong |

The hard miss is the same with cold, RAM-cached and SSD-cached prefixes, 512- or 2048-token chunks, bf16
instead of 4-bit pools, prompt-suffix pruning on or off, fp32 index scores, and the release's in-place FP8/FP4
activation rounding (index q/k FP4, window KV and `o` FP8, compressed latent FP4/e4m3). It is not the
quantization either: attention, indexer and shared experts are 8-bit, and swapping the release's own MXFP4 routed
experts into layers 0-19 (everything the layer-20 candidate selector depends on) ranks the target lower, not higher
(candidate block 21,114 → 34,776 of 57,678; 2,048 kept). For the answer token at 300k the late index layers find
the target (layer-20 candidate block rank 178; layers 28 / 32 / 36 rank it 39 / 32 / 2) while layers 2 / 8 / 14 do
not (3,388 / 21,317 / 13,485); at 461k it is outside every top-512 and outside the layer-20 candidates. These
final-token ranks show where selection is tight but do not decide the answer alone: on the easy prompts the target
can be outside them and still be answered, because the question tokens inside the 128-token window did the lookup
during prefill. Reading: a limit of the model's sparse attention on ~29k near-identical lines, not
of the runtime, cache or quantization. A 1M context remains usable for retrieval of distinctive content.

## Known open items

- Prose decode ≈ AR (acceptance ~0.36; the governor keeps it at AR, not above).
- Regenerate `row_exact_qmv` for the installed MLX and use it for DSpark verify rows (speed +
  exact greedy identity).
- `tests/test_chunking.py` assumes 32-dim index heads, which don't fit the 64-wide groups of the
  4-bit pools; it needs real-geometry heads.
- Chunk boundaries can still flip a near-tied first token: the head is fp32 now (fp32 input, fp32 logits,
  as the reference), which removed the exact bf16 ties, but chunking changes the bf16 hidden states
  slightly (cosine ~0.999), enough to flip a top-2 margin below ~0.05 (DECODE_DSPARK_METHODS M1).
- `iogpu.wired_limit_mb` raise not tested.
