# DSV4-Flash — HSA + CSA + SWA tri-mode runtime for JANGTQ + JANG

> Goal: make the three attention modes interleaved across DSV4-Flash's 43
> layers actually work end-to-end on JANGTQ + JANG_2L bundles. Right now
> all three are bypassed at decode (`DSV4_LONG_CTX=0` default, plain
> `KVCache`) and prefill crashes with a shape bug at ≥128-token prompts
> when the env var is flipped. This is a survey + plan, not yet a fix.

## §1 The three modes — paper vs code

| paper § | name (paper) | name in our code | layers (compress_ratios=) | mechanism |
|---|---|---|---|---|
| 1.4 | **HSA** ("Heavily Compressed Attention") | `compress_ratio == 128` | indices 1, 3, 5, …, 39 (~21 layers) | DENSE attention over 128×-pooled KV + sliding window |
| 1.4 | **CSA** ("Compressed Sparse Attention") | `compress_ratio ==   4` | indices 2, 4, 6, …, 40 (~20 layers) | SPARSE attention: Indexer top-k=512 over 4×-pooled KV + sliding window |
| §2  | **SWA** (sliding window only)            | `compress_ratio ==   0` | layer 0 + last (2 layers) | plain windowed attention, n_win=128 |

(The paper uses "HCA" — we'll keep using your "HSA" for the dense-pool 128× mode going forward, but be aware the paper writes it "HCA". Our `mlx_model.py` uses the literal `compress_ratio` integer.)

For **all** layers, the K dimension after compression is:

```
full_kv = [ window_kv (n_win=128 entries, uncompressed)
          ,  pool_kv  (variable; 1 entry / compress_ratio source-tokens) ]
```

CSA differs from HSA only by the Indexer pre-selecting which pool entries the query is allowed to see (top-k=512 of all available pool entries).

## §2 Current state (post-2026-04-25 audit)

### What works
- Decode WITHOUT long-ctx: `DSV4_LONG_CTX=0` (default) → all layers run plain `KVCache`. Coherent up to ~4K tokens, drifts after that. JANGTQ pass@1 HumanEval+ ≈ 67% (T=0.6, FIM).
- Per-layer RoPE: `compress_ratio>0` layers use `compress_rope_theta=160_000` + YaRN; `compress_ratio==0` layers use base `rope_theta=10_000` + no YaRN. **Required for numerical stability** — without it middle layers drift to inf by L40.
- Inverse RoPE on attention output (`scale=-1.0`) — paper §1.4(2) requires this; bit-exact verified in `_attn_partial_rope_fused`.
- Per-head Q + KV-entry RMSNorm (paper §1.4(1)) — confirmed via `mx.fast.rms_norm` weight=None.
- Attention sinks (paper §1.4(4)) — passed to `mx.fast.scaled_dot_product_attention(sinks=...)`.
- Grouped output projection (paper §1.4) — `wo_a` (group-wise) → `wo_b` (concat-to-hidden).

### What's broken (gating long-ctx)
1. **Prefill shape bug at L≥128** (`DSV4_LONG_CTX=1` path):
   `Shapes (L,L) and (1,H,L,big) cannot be broadcast` from SDPA mask. Triggered by the long_ctx_kv_modified path concatenating `(window_kv, pool_kv)` then SDPA falling back to the external mlx-lm causal mask sized for the original window-only K.
   Fix: when `long_ctx_kv_modified=True`, build our own full-shape mask covering both window and pool (or set `mask=None` and pre-bake a manual scoring path that handles SWA + sink + pool selection).
2. **Compressor pool-accumulation across decode steps** is implemented (PR #1195 two-mode selection: S=1 GATHER / S>1 flat-mask) but only exercised when `cache` is `DeepseekV4Cache`. Plain `KVCache` skips compressor entirely → no global context for queries past window.
3. **Indexer top-k correctness** — verified at S=1 GATHER, NOT verified at S>1 flat-mask path. Bench would catch this.

### What's broken for JANGTQ specifically
1. **§433 fix (Compressor wkv + Indexer weights_proj at bf16 passthrough)** is applied at convert-time, but pre-§433 bundles on HF have these as 8-bit affine. Affects: `OsaurusAI/DeepSeek-V4-Flash-JANGTQ` + `JANGTQ2` (built before fix).
2. **`hc_*_fn / hc_*_base / hc_*_scale / attn_sink / ffn.gate.bias` — source dtype is FP32, bundle stores fp16.** Loses 13 mantissa bits on the mHC mixing matrices. Per `DSV4-LAYER-IMPORTANCE.md`: keep at fp16 was advised for bf16 source, but DSV4's source is fp32 — going to fp16 is more aggressive than intended.
3. **R2: `_hc_pre` bf16 variance saturation** — `mean(x²)` over 16,384-elem bf16 destroys variance. Verified critical bug in `mlx_model.py:975`. Already fixed in pip-installed runtime via the `_hc_premix_kernel` Metal kernel that runs the reduction in fp32; in-tree dev source still has the bf16 path.

## §3 Plan to make HSA + CSA + SWA all work end-to-end on JANGTQ

Ordered by blast radius, smallest to largest. Each step is independent and verifiable.

### P0 — Build a regression harness (1 day)
- Pick three prompt lengths: 64 tokens (window-only, all 3 modes degenerate to SWA), 512 (window + a handful of CSA selections), 4096 (window + many HSA pool entries).
- Decode 32 tokens each with `DSV4_LONG_CTX=0` and `=1`, plain bf16 source.
- Diff the per-layer hidden state at every block boundary against the reference torch impl in `layer_forward.py`.
- This is the trust baseline: any change downstream must keep the diff at <1e-3 RMS.

### P1 — Fix the prefill shape bug (2-3 days)
- Reproduce on a 256-token prompt with the bf16 source on M5 Max (no JANGTQ noise yet).
- The SDPA call inside `DeepseekV4Attention.__call__` when `long_ctx_kv_modified=True` is at `mlx_model.py:1110`-ish. We pass `mask` from mlx-lm caller, but K dim grew from `window` to `window + pool`.
- Two fixes possible:
  - (a) Build the (B, 1, L, K_total) bool mask ourselves and pass it; ignore the external mask. Cleaner.
  - (b) Pad the external mask. More fragile.
- Add a unit test: `prefill 256-token prompt, bf16, DSV4_LONG_CTX=1, expect coherent next-tok argmax matches reference`.

### P2 — Per-mode quant policy for Compressor + Indexer (1 day)
Compressor (HSA + CSA, ~41 layers):
- `compressor.wkv.weight` (BF16 source) → **bf16 passthrough** (small low-rank, 4096×512 worst case; quant noise here corrupts the pool aggregator). §433 already does this for new convert runs.
- `compressor.wgate.weight` (FP8 E4M3 source) → **8-bit affine gs=32**.
- `compressor.ape` (BF16 source, learnable positional offset) → **bf16 passthrough**.

Indexer (CSA only, ~20 layers):
- `indexer.weights_proj.weight` (BF16 source) → **bf16 passthrough** (top-k decision projection — quant noise = wrong selection).
- `indexer.wq_b.weight` (FP8 E4M3 source) → **8-bit affine gs=32**.
- `indexer.compressor.*` → reuse the Compressor rules above (it's a nested Compressor module).

Verify: pre-§433 HF bundles need to be rebuilt or post-patched. Add a patcher (`jang_tools.patch_dsv4_compressor_dtypes`) that walks an existing bundle, identifies these modules, and re-quantizes the affected tensors from a checkpointed bf16 source. Mirrors the `patch_dsv4_quant_config` tool.

### P3 — mHC pre-mix fp32 (already partially fixed)
- The pip-installed `_hc_premix_kernel` does the variance reduction in fp32 internal. Mirror that in the dev source's `_hc_pre` so a fresh editable install is also correct.
- For JANGTQ: the `hc_attn_fn / hc_attn_base / hc_attn_scale` matrices are stored as fp16 (loss vs fp32 source). Test if **promoting them to fp32 in the bundle** (~135 MB total) recovers any benchmark headroom. If yes, ship as a per-module override in the converter.

### P4 — `attn_sink` + `ffn.gate.bias` fp32 in bundle (0.5 day)
- 64 sinks/layer + 256 gate biases/layer × 43 layers ≈ ~80 KB total. Trivial to keep as fp32. Add per-module override.

### P5 — Indexer top-k correctness at S>1 prefill (1 day)
- The flat-mask path at S>1 builds `(B, 1, L, P) bool mask = causal_staircase AND indexer_selected`. Verify against a torch reference for a known prompt where we can compute the expected attention weights.
- This is the hardest correctness check; any bug here means the model attends to the wrong pool entries during prefill, and decode inherits that broken cache.

### P6 — Two end-to-end MMLU runs as proof
- bf16 source: should match the reference torch impl within sample noise. Floor for what JANGTQ can hit.
- JANGTQ + DSV4_LONG_CTX=1: target `bf16 score - 5pp` or better. Anything less than that says JANGTQ is dropping correctness somewhere in the Compressor/Indexer chain.

## §4 Files this touches

- `jang-tools/jang_tools/dsv4/mlx_model.py` — prefill mask fix, `_hc_pre` fp32 mirror.
- `jang-tools/jang_tools/dsv4/convert_dsv4_jangtq.py` — per-module overrides for Compressor/Indexer/mHC + attn_sink/gate.bias.
- `jang-tools/jang_tools/dsv4/runtime.py` — set `DSV4_LONG_CTX=1` default once §3.P1 is verified.
- New: `jang-tools/jang_tools/patch_dsv4_compressor_dtypes.py` — post-publication patcher for older HF bundles.

## §5 What NOT to chase

- **fp32 SDPA on the L==1 contraction** — already tested, regresses -7pp on DSV4. The bf16 SDPA is correct; mHC + Compressor noise dominates the SDPA precision gap. Do not re-introduce.
- **AWQ at α=0.25** for DSV4 — channel-magnitude spread is only 2.5×, AWQ is neutral. Don't waste calibration runs on it.
- **fp16 affine on routed experts** — the codebook MXTQ at 2-bit is the right floor. Going to fp16 would 4× bundle size for marginal MMLU.
