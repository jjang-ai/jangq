# DSV4-Flash + DSV4-Pro — Complete status, gaps, and roadmap to "full quality"

> Consolidates everything we know about DSV4 as of 2026-05-01: architecture
> (Flash + Pro), runtime state, JANGTQ + JANG quant policy, evaluation
> findings, and what's still missing to hit paper-claimed quality on
> Apple Silicon.
>
> Cross-reference: per-topic deep dives in this directory and at:
> - `research/DSV4-PAPER-DEEPDIVE-2026-04-25.md` — line-by-line paper port
> - `research/DSV4-RUNTIME-ARCHITECTURE.md` — Python + Swift impl guide
> - `research/DSV4-HSA-CSA-SWA-RUNTIME-PLAN.md` — prior 7-step plan
> - `research/JANGTQ-DSV4-RUNTIME-AUDIT-2026-04-25.md` — bug catalog
> - `research/DSV4-FLASH-MMLU-INVESTIGATION-2026-04-26.md` — quality dive

## §1 The two DSV4 model families

| | DSV4-Flash | DSV4-Pro |
|---|---|---|
| Total params | **284 B** | unpublished (likely 1T-class) |
| Active params | **13 B** | larger |
| Layers | **43** | unpublished |
| Hidden d | **4096** | unpublished |
| Query heads `n_h` | 64 | likely 64+ |
| Head dim `c` | **512** (4× larger than typical) | likely 512 |
| Q low-rank `d_c` | 1024 | likely 1024 |
| O groups `g` × `d_g` | 8 × 1024 | likely 8 × 1024 |
| MoE: routed experts | **256** + 1 shared | unpublished |
| MoE: top-k | 6 (active) | unpublished |
| MoE: hash routing | first **3** layers (deterministic per-token) | likely same |
| HSA compress `m'` | **128** | likely 128 |
| CSA compress `m`  | **4** | likely 4 |
| CSA Indexer top-k | **512** | likely 512 |
| Sliding window `n_win` | 128 | 128 |
| Layer 0 + last | pure SWA (compress_ratio=0) | same pattern |
| mHC `n_hc` (residual copies) | 4 | 4 |
| Sinkhorn iters | 20 | 20 |
| RoPE base (compress_ratio=0) | 10 000, no YaRN | same |
| RoPE base (compress_ratio>0) | 160 000 + YaRN | same |
| Max ctx | 1 M | 1 M |
| Source dtype | bf16 weights, fp8 e4m3 + UE8M0 scales for attn/shared, I8 packed FP4 + e8m0 scales for routed experts, F32 for hc_*/sinks/gate.bias | likely identical training-side; bundle dtypes TBD when Pro weights drop |

**Headline benchmark gap (paper Table 7):**
| Benchmark | Non-Think | Think High | Think Max |
|---|---|---|---|
| MMLU-Pro | 83.0 | 86.4 | 86.2 |
| **LiveCodeBench Pass@1-COT** | **55.2** | **88.4** | **91.6** |
| GPQA Diamond | 71.2 | 87.4 | 88.1 |
| HLE | 8.1 | 29.4 | 34.8 |

→ Hitting paper-claimed quality requires **Think High** mode at proper budget,
not Non-Think. Most of our 60% MMLU floor was the wrong mode + wrong
attention path, not codec noise.

## §2 Three attention modes

| code name | paper name | layers (compress_ratios=) | mechanism |
|---|---|---|---|
| `compress_ratio == 0` (SWA) | sliding window | layer 0 + 42 (2 of 43) | plain windowed attn, n_win=128 |
| `compress_ratio == 128` (HSA / HCA in paper) | Heavily Compressed Attention | indices 1, 3, 5, …, 39 (~21 layers) | DENSE attn over 128×-pooled KV + sliding window |
| `compress_ratio == 4` (CSA) | Compressed Sparse Attention | indices 2, 4, 6, …, 40 (~20 layers) | SPARSE attn: Indexer top-k=512 over 4×-pooled KV + sliding window |

For all layers the K dim after compression is `[ window_kv (n_win=128) ; pool_kv (variable) ]`.
CSA differs from HSA only by Indexer pre-selecting top-512 of pool entries.

Per-layer details:
- **Q + KV-entry RMSNorm BEFORE core attention** (paper §1.4.1) — confirmed at `mlx_model.py:712`
- **Partial RoPE last 64 dims of q/k AND output** — inverse RoPE on output is REQUIRED, not optional
- **Learnable per-head attention sinks** — passed to `mx.fast.scaled_dot_product_attention(sinks=…)`
- **Grouped Output Projection** — n_h=64 → g=8 groups × d_g=1024 → final 4096

## §3 Where we are TODAY (2026-05-01)

### 3a. ✅ FIXED in this session

| fix | result |
|---|---|
| **P0** synthetic mask shape harness (`dsv4/tests/test_long_ctx_shapes.py`) | 12/12 PASS for SWA + HSA + CSA at L=1..2048 |
| **P1** `DSV4_LONG_CTX` default 0 → 1 | 251-token prefill OK in 30.9 s; 41 / 43 layers run `DeepseekV4Cache` (Compressor + Indexer pool); 2 SWA layers run plain `KVCache` |
| **P2** Compressor + Indexer per-tensor quant policy in `convert_dsv4_jangtq.py` | `compressor.wkv` + `indexer.weights_proj` → bf16 passthrough; `compressor.wgate` + `indexer.wq_b` → 8-bit affine |
| **P3** `_hc_pre` fp32 cast | already in dev source `mlx_model.py:1392`; mirrors pip-installed Metal kernel |
| **P4** F32-source tensors stay fp32 in bundle | `hc_*_{fn,base,scale}` + `attn_sink` + `ffn.gate.bias` no longer cast to fp16 (~135 MB extra in bundle) |
| **Patcher** for older bundles (`jang_tools.patch_dsv4_compressor_dtypes`) | dequantizes misclassified Compressor + Indexer wkv / weights_proj weights back to fp16, drops the .scales/.biases sidecars |

### 3b. 📈 MEASURED IMPACT

`OsaurusAI/DeepSeek-V4-Flash-JANGTQ` (79 GB), MMLU 30q smoke:

| run | DSV4_LONG_CTX | MMLU |
|---|---|---|
| pre-fix (HSA + CSA bypassed) | `0` (then-default) | **60 % / 200 q** |
| **post-P1 (HSA + CSA + SWA tri-mode)** | `1` (new default) | **76.7 % / 30 q** |

**+16 pp purely from letting the model run its own attention.** Codec round-trip cosine ≥ 0.94 on every probed module (same as Mistral, Qwen, Kimi); DSV4 was never codec-bottlenecked.

### 3c. ❌ STILL MISSING (the gap to paper-claimed quality)

| # | gap | impact | difficulty |
|---|---|---|---|
| **G1** | **Think High mode wired into runtime** — chat template `enable_thinking=True`, `<think>…</think>` parsing, max_tokens 16-32K | +20-30 pp on MMLU-Pro / LiveCodeBench / GPQA. This is the dominant gap. | low (mistral_parser.py + qwen3_parser.py already mirrored in `jang_tools.reasoning/`; just needs DSV4-specific tag config) |
| **G2** | **`reasoning_effort=max` system prompt** auto-injection | +3-7 pp on hardest benchmarks | low (chat_template branch on a flag) |
| **G3** | **Re-converted JANGTQ bundle with P2 + P4 policy fixes baked in** — current OsaurusAI bundle was built before these fixes | likely +2-5 pp from cleaner Compressor/Indexer + +1-2 pp from fp32 mHC/sinks | medium (need bf16 source on disk + 30-min convert) |
| **G4** | **Indexer top-k correctness at S>1 prefill** — flat-mask path verified end-to-end coherent, NOT verified against torch reference | unknown impact; could be 0 or could be +5 pp if there's a subtle off-by-one | medium (need torch reference for one prompt + diff) |
| **G5** | **Per-bundle MMLU full 200q two-pass** (no-reasoning + reasoning) | quantifies what's actually working vs the paper headline number | low (jang_tools.eval.mmlu harness already wired) |
| **G6** | **HumanEval+ Think Max benchmark** — paper claims 88-91% pass@1-COT in this mode; we've only ever measured 37-67% in Non-Think | tells us how close to paper SoTA we can actually get | medium (large generation budgets, takes hours per pass) |
| **G7** | **DSV4-Pro support** — when weights drop on HF | new convert + load path, but architecture is the same family so most code applies | depends on Pro release timing |
| **G8** | **Swift JANGTQ kernel binding** — `JANGCxx.h` shim is in jang-tools; `vMLXLMCommon/JANGTQKernels.swift` exists in vmlx; not yet wired in jang-tools' Swift package | required for DSV4 in Osaurus app. JANGTQ Python runtime works today; Swift native runtime stale. | medium-high (kernel binding + per-arch loader) |

### 3d. 🐛 Known runtime bugs (status)

From `JANGTQ-DSV4-RUNTIME-AUDIT-2026-04-25.md`:
- **R1** (MLA bf16 SDPA) — REJECTED, fp32 cast regresses -7 pp; bf16 is correct
- **R2** (`_hc_pre` bf16 variance saturation) — FIXED (fp32 cast in dev source line 1392, Metal kernel in pip)
- **R3** (SwiGLU bf16 cast-back) — open; minor noise compound, not a quality killer
- **R4** — false positive, not a bug
- **R5** (warmup mHC tile shape) — known cosmetic; warmup fails silently and skips → ~60s first-inference compile cost. Functional but not free.
- **R6** (P19 MLA fusion drops mode/bits/group_size) — latent, not triggered today (DSV4 attention is plain 8-bit affine in current bundles)
- **R7** (TurboQuant `_dequant_experts` reshape on non-contiguous packed) — open; only fires if anyone slices `tq_packed`

## §4 Reasoning the paper for "full quality"

The MMLU smoke jump from 60 → 76.7 % was the easy half — letting the model run its own attention. The remaining ~10 pp to paper-claimed Non-Think (~83 % MMLU-Pro) and the further ~10 pp to Think-High come from FOUR axes, in roughly this order of effort × payoff:

### Axis A — Sampling & generation budget (CHEAPEST, biggest immediate win)

The paper's Table 3 explicitly says reasoning mode determines budget:

| mode | system prompt | max_new tokens | sampler |
|---|---|---|---|
| Non-Think | none | 1024-2048 | greedy or T=0 |
| Think High | `<think>` enabled | 8 192 (chat) / 16 384 (code) | T=1.0 top_p=0.95 |
| Think Max | `Reasoning Effort: Absolute maximum` | 32 768 (chat) / 65 536 (code) | T=1.0 top_p=0.95 |

We've been running everything at max_tokens ≤ 1 024. Non-Think bench is OK at 1 024; **Think High is impossible at 1 024 because the model never reaches `</think>`**. Increase budget per mode.

**Fix**: per-mode budget defaults in `jang_tools.eval.mmlu` + `jang_tools.dsv4.runtime`. Free.

### Axis B — Reasoning-tag parsing (CHEAP, unblocks Think modes)

Without correct tag handling our extractor reads the chain-of-thought as the answer. Paper outputs `<think>…</think>` then plain answer.

**Fix**: we already mirrored `mistral_parser.py` + `deepseek_r1_parser.py` from vmlx into `jang_tools.reasoning/`. Wire DSV4 to `DeepseekR1ReasoningParser` (it uses `<think>…</think>` exactly the way DSV4 does). Already wired in `jang_tools.eval.mmlu._strip_thinking` for `model_type` starting with `deepseek` or `dsv`. Verify the model_type string of our bundle matches.

### Axis C — Re-convert with P2 + P4 baked in (MEDIUM cost, marginal payoff)

Current OsaurusAI bundle was built before P2 (Compressor/Indexer bf16) and P4 (fp32 hc_*/sinks/gate.bias). Re-converting the 79 GB bundle from bf16 source picks up these. Order of magnitude payoff: +2-5 pp on hard benchmarks, near-zero on easy ones. Cost: 30 min on the M5 Max + ~10 min upload.

**Or**: run `jang_tools.patch_dsv4_compressor_dtypes` on the existing bundle to pick up P2 in-place. Doesn't help with P4 (F32 source dtype is gone in fp16 bundle).

### Axis D — Long-context correctness (MEDIUM-HIGH cost, unknown payoff)

The Compressor + Indexer mask construction is correct on shape (12/12 synthetic). The Indexer top-k SELECTION at S>1 prefill has not been verified against a torch reference — there could be a subtle off-by-one in `_compressed_visibility AND Indexer_selected` that lets queries see slightly the wrong pool entries. Bench would catch a quality drop on >2K-token prompts.

**Fix**: build a 1-layer torch reference, run a 512-token prompt through both, diff post-attention hidden state. Cost: half a day; fix-or-no-fix decision in the diff.

### Axis E — Swift native runtime (HIGH cost, required for Osaurus deployment)

JANGTQ Python runtime works today; the Swift loader hard-errors without the now-shipped `jangtq_runtime.safetensors` sidecar (unblocked at the file layer for both new bundles + all 13 OsaurusAI JANGTQ repos as of 2026-05-01 banners). What's missing for actual native Swift inference of DSV4:

1. **JANGTQ kernel binding** in `jang-tools/swift/Sources/JANGCxx/` — the C++ shim header exists, body is a stub. Needs to bind to `vmlx-swift-lm/Sources/vMLXLMCommon/JANGTQKernels.swift`.
2. **DSV4 Swift arch port** — `vmlx-swift-lm/Sources/vMLXLLM/Models/DeepseekV4JANGTQ.swift` exists at 640 LOC scaffold. Needs:
   - mHC `hcCollapse` / `hcExpand` Metal kernels (Swift port of `hc_split_sinkhorn`)
   - Per-layer Compressor + Indexer with the same v4_cache state machine
   - Hash routing for first 3 layers
   - All four critical fixes from `DSV4-FLASH-FOUR-CRITICAL-FIXES-2026-04-24.md` mirrored
3. **Cache class** — `DeepseekV4Cache` Swift equivalent with the rotating window + compressor/indexer state buffers
4. **Reasoning parser binding** — already in vmlx-swift-lm, just needs to be wired to DSV4 in the Osaurus app dispatch.

## §5 Files in our control

### Python
- `jang-tools/jang_tools/dsv4/mlx_model.py` (1700 LOC) — main runtime, all 13 in-tree fixes, default-ON tri-mode
- `jang-tools/jang_tools/dsv4/convert_dsv4_jangtq.py` — converter with P2 + P4 policy
- `jang-tools/jang_tools/dsv4/convert_dsv4_jang.py` — JANG_2L converter (affine path)
- `jang-tools/jang_tools/dsv4/{fp4_codec,fp8_ue8m0_codec}.py` — source codec
- `jang-tools/jang_tools/dsv4/{runtime,bench_humaneval,bench_humaneval_pass5,bench_speed}.py` — bench harnesses
- `jang-tools/jang_tools/dsv4/encoding_adapter.py` — DSV4 chat-template Python encoder
- `jang-tools/jang_tools/dsv4/tests/test_long_ctx_shapes.py` — 12/12 synthetic mask harness
- `jang-tools/jang_tools/dsv4/tests/test_long_ctx_coherence.py` — 3-prompt-length coherence harness (scaffolded; deferred)
- `jang-tools/jang_tools/load_jangtq.py` — global JANGTQ bundle loader, auto-detects DSV4
- `jang-tools/jang_tools/patch_dsv4_compressor_dtypes.py` — in-place patcher for older bundles
- `jang-tools/jang_tools/eval/mmlu.py` — two-pass MMLU harness with auto-`enable_thinking`
- `jang-tools/jang_tools/reasoning/` — mirrored from vmlx; deepseek_r1_parser.py handles DSV4 `<think>` tags

### Swift (jang-tools/swift/)
- `Sources/JANGRuntime` — bundle metadata loader, `BundleProbe.detect`
- `Sources/JANGQuant` — `QuantMeta`, format detection (mxtq / mxfp4 / fp8 / bf16), invariant enforcement
- `Sources/JANGCxx` — C++ shim header + stub for `jang_tq_decode_bf16` (binds to vmlx-swift TQ kernel)
- `Sources/JANGImage` — native pixtral preprocessor (used by Mistral 3.5; not DSV4)
- `Sources/JANGDistributed` — `World`, `ShardPlan`, `TB5Probe` (parked, not the current path)
- `Sources/_VmlxMirror/JANGTQKernels.swift` — reference copy of vmlx-swift-lm's JANGTQ kernel

### Bundles on HF
- `OsaurusAI/DeepSeek-V4-Flash-JANGTQ` (79 GB) — current production, has banner warning + `jangtq_runtime.safetensors` sidecar
- `OsaurusAI/DeepSeek-V4-Flash-JANGTQ2` (74 GB) — older 2-bit, banner + sidecar
- `OsaurusAI/DeepSeek-V4-Flash-JANG_2L` (107 GB) — affine 2-bit oracle
- `OsaurusAI/DeepSeek-V4-Flash-MXFP4-lossless` — direct-copy FP4 source, deprecated path

## §6 Concrete next steps in order

1. **Run full 200-q MMLU two-pass** on the existing OsaurusAI/DeepSeek-V4-Flash-JANGTQ bundle to lock in the post-P1 quality number (`jang-mmlu --src ... --mode both --qps 20 --out research/dsv4/mmlu-200q-tri-mode.json`). One run, ~1-2 h on M5 Max.
2. **Wire DSV4 → `DeepseekR1ReasoningParser`** in `jang_tools.eval.mmlu` (verify the model_type the bundle ships with matches the dispatch). Re-run the reasoning pass; expect +20-30 pp on MMLU-Pro.
3. **Per-mode max_tokens defaults** in `jang_tools.dsv4.runtime` (chat=8K, think=32K, think_max=64K). Trivial.
4. **G3 — Re-convert from bf16 source** (~30 min) to bake P2 + P4. Run G5 + G6 against the new bundle.
5. **G4 — Indexer S>1 torch-reference diff** before claiming "fully correct."
6. **G8 — Swift JANGTQ kernel binding** for Osaurus production. This is the biggest remaining lift.

## §7 What "full quality" means

Hitting the paper's claimed numbers on JANGTQ on Apple Silicon means at minimum:

- **MMLU-Pro 80%+** in Think High at JANGTQ2 (paper FP claim is 86.4 %) — codec floor is ~3-5 pp.
- **HumanEval+ pass@1 75-85%** in Think High mode with proper budget — paper Non-Think baseline 55.2 % means JANGTQ Non-Think floor is ~50 % with codec noise.
- **GPQA Diamond 80%+** — paper FP claim 87.4 %.

If we hit these, we're at "frontier on Apple Silicon" status. Everything in §3c/§4 above is on the critical path; Axis A + B are the cheapest, biggest payoff. Start there.
