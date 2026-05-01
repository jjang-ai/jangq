# DSV4 HSA + CSA — what's left, what's worth building

> Continuation of `DSV4-HSA-CSA-JANGTQ-CODESIGN.md`. Focused on
> attention-side-only levers (no MoE / no mHC / no embed/head): what
> we can do that specifically targets HSA + CSA correctness, memory,
> and speed beyond what already shipped.

## What's done (recap, attention-side only)

| change | scope | status |
|---|---|---|
| `DSV4_LONG_CTX=1` default | runtime | ✅ shipped — 60 → 76.7 % MMLU |
| Compressor + Indexer `wkv` + `weights_proj` → bf16 | convert | ✅ shipped (next bundle) |
| F32 `attn_sink` preserved | convert | ✅ shipped (next bundle) |
| Per-layer split-theta RoPE (10 K vs 160 K) | runtime | ✅ already correct |
| Inverse RoPE on attn output | runtime | ✅ already correct |
| Per-head Q + KV-entry RMSNorm before SDPA | runtime | ✅ already correct |
| Native `mx.fast.scaled_dot_product_attention(sinks=…)` | runtime | ✅ already correct |
| Synthetic mask shape harness | tests | ✅ 12/12 PASS |

## What's NOT done — concrete attention-side levers

### A1 — Pool-side KV cache quantization (the big memory unlock)

Compressor pool entries (HSA + CSA layers) are **read-only after produce**:
no further mutation. They sit in cache for the whole decode session. At 1 M
context:

- **HSA layers** (compress_ratio=128, ~21 layers): pool grows to ~7 800 entries × head_dim=512 × bf16 ≈ 8 MB per layer × 21 = **168 MB**
- **CSA layers** (compress_ratio=4 with overlap, ~20 layers): pool grows to ~250 K entries × 512 × bf16 ≈ 250 MB per layer × 20 = **5 GB**

Quantize pool entries 4-bit affine in cache → **4× memory savings**, ~4 GB
recovered for 1 M context. Pool entries are heavily averaged (HSA pools
128 source tokens together; CSA averages 4 with overlap), so 4-bit affine
is near-lossless on them — verified bench precedent on Mistral 3.5 MXFP4
(cos = 0.996 at 4-bit g=32).

This is a **runtime-only change** in `DeepseekV4Cache` — no re-convert
needed.

### A2 — Indexer top-k correctness harness (the big correctness verifier)

We moved `indexer.weights_proj` to bf16 passthrough (P2), but never
verified the actual top-k SELECTION matches a torch reference. If our
`_indexer_score_reduction` compile graph or the `_compressed_visibility &
indexer_selected` mask has a subtle off-by-one, queries attend to slightly
the wrong pool entries every step. The model would still produce coherent
text — just the wrong text.

**Build**: a 2-input regression test that
1. Loads the JANGTQ bundle and runs Indexer for a fixed 1 024-token prompt.
2. Loads the same prompt through a torch reference Indexer.
3. Compares the top-k INDEX SET (not values) per query position. Threshold:
   ≥ 95 % overlap on top-512 selection.

If it diverges, we have a genuine bug to chase. If it agrees, we can
stop suspecting Indexer correctness and move on to other levers.

### A3 — CSA early-prefill pool short-circuit

When `pool_size < top_k=512`, Indexer still does the full math (relu,
score reduction, partial sort) only to return all available pool entries.
For prompts < 2 048 tokens this fires every CSA forward — wasted compute
on every prefill of a short prompt.

**Fix**: if `P < top_k`, skip the indexer score path; pass `topk=None`
through the existing GATHER fast path which already handles this case.
Saves ~150 matmul dispatches per token on short prefills × 20 CSA layers.
Quality unchanged.

### A4 — Sliding window seam mask

`full_kv = concat(window_kv, pool_kv)` on `compress_ratio>0` layers. Mask
is `concat(win_mask, comp_mask_extra)`. The seam — query positions in the
overlap zone (queries at offset N where some pool entries cover the same
source tokens as window keys) — could double-count. Paper §1.4(3) says
"sliding window adds n_win uncompressed KV entries" implying NO overlap
in the index space, but the masks don't enforce non-overlap explicitly.

**Verify**: synthetic prompt of length L = 256, count attention mass on
overlapping (window-and-pool) source positions. Should be ≤ 1.0 in
softmax-norm units.

### A5 — Indexer head-importance reduction precision

`_indexer_score_reduction` runs in fp32 (good). It does
`relu(scores) * scale * weights * n_heads^-0.5` then `.sum(axis=heads)`.
The `n_heads^-0.5` constant comes from `n_heads=64` → `0.125`. Verify the
constant matches the source torch; the paper's explicit reference is the
"DSA" math but the indexer scoring formula is half a sentence.

### A6 — Adaptive top-k

Paper sets top-k = 512. For VERY long contexts (1 M tokens, CSA pool ~
250 K), 512/250 K = 0.2 % of pool. For short contexts (4 K, CSA pool ~
1 000 with overlap), 512 = 50 % of pool.

**Probe**: does fixed-512 give worse quality than `min(P, 512)` *or*
`min(P//2, 512)` on long context? Worth measuring; we don't know.

### A7 — Pool-attention fused Metal kernel (the speed unlock)

Currently:
1. `compressor(x)` → `pooled` (matmul + gate + ape + RMSNorm)
2. `indexer(x, q_residual)` → `topk` (matmul + score reduce + topk)
3. `_compressed_visibility & indexer_selected` → `comp_mask_extra`
4. `concat([win_mask, comp_mask_extra])` → `sdpa_mask`
5. `concat([window_kv, pool], axis=2)` → `full_kv`
6. `mx.fast.scaled_dot_product_attention(q, full_kv, full_kv, scale, mask, sinks)`

Six dispatches per layer × 41 layers × per-token decode = **246 dispatches/token**
(plus the GQA repeat).

We control the JANGTQ kernel. We can write a single fused kernel:
**inputs**: `q`, packed `tq_packed_kv_window` (4-bit), packed
`tq_packed_kv_pool` (4-bit, after A1), `topk_idx`, `attn_sink`, RoPE freqs.
**output**: post-RoPE-inverse attn_out.

One Metal dispatch instead of six. On HSA + CSA layers, the largest win
is on the dispatch overhead; matmul time stays the same. M5 Max
estimated: ~15-25 % decode speedup on the long-ctx path.

This is the most aggressive lever and the highest engineering cost. Skip
until A1-A4 are done — they're 10× cheaper for similar payoff.

## What's CHEAPER from non-attention side that helps HSA+CSA quality

- A8: **Reasoning parser DSV4 dispatch** — already wired; verify by running
  Think mode on a tough MMLU question, observe model emits `<think>…</think>`
  and our extractor pulls correct A/B/C/D after `</think>`.
- A9: **Per-layer-class profiling** — instrument hidden-state RMS at each
  block boundary, measure per-class drift contribution. Diagnostic, not a
  fix; informs whether HSA or CSA is dragging quality more.

## Order of operations

1. **A2** Indexer correctness harness (1 day, blocks all further claims)
2. **A3** CSA short-circuit (half day, free speedup)
3. **A1** Pool KV quant (2-3 days, big memory unlock for 1 M context)
4. **A8** Reasoning parser e2e verification (half day, paper-quality unlock)
5. **A4** Window-pool seam audit (half day, cheap check)
6. **A6** Adaptive top-k probe (1 day measurement)
7. **A7** Fused Metal pool-attention kernel (1-2 weeks, last)

Verification protocol after each: codec round-trip cos ≥ 0.94 + 30 q
MMLU smoke ≥ 76 %. Anything below that is a regression caught.
