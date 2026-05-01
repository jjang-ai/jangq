# DSV4 HSA + CSA + SWA × JANGTQ — codec/kernel co-design

> Companion to `DSV4-COMPLETE-STATUS-2026-05-01.md`. Focus: the JANGTQ
> codec and the JANGTQ Metal kernel are **OURS**. We are not constrained
> to a stock affine + Hadamard layout. Everything in this doc is a
> codec-side or kernel-side change we can make to fit DSV4's HSA + CSA +
> SWA tri-mode attention better.

## §1 Why this matters

The MMLU smoke jumped 60 → 76.7 % from one runtime fix (`DSV4_LONG_CTX=1`).
The remaining gap to paper-claimed quality is split across:

- **Sampling** (Think High, max_tokens budget) — addressed in the main status doc.
- **Codec/Kernel co-design** — this doc. We own JANGTQ end to end. If a
  layer's importance, distribution, or access pattern doesn't fit the
  generic 2-bit codebook, we can change the codec for THAT layer without
  touching anything else.

The audit (`research/DSV4-FLASH-MMLU-INVESTIGATION-2026-04-26.md`) flagged
several places where stock JANGTQ leaves quality on the table because we
applied a one-size codec across modules with very different roles in the
HSA / CSA / SWA pipeline. We can fix that.

## §2 The three attention modes — what they STRESS in the codec

| mode | layers | dominant op | what stresses the quant |
|---|---|---|---|
| **SWA** (compress_ratio=0) | 0, 42 | local 128-token attn | Q/K/V dot products. `wq_b` matrix (1024 → 32 768) is the heaviest matmul × 64 heads; quant noise here corrupts every Q. |
| **HSA** (compress_ratio=128) | 1,3,5,…,39 | DENSE attn over 128×-pooled KV pool + window | `compressor.wkv` (4096→512) AGGREGATES every 128 tokens into one pool entry. Quant noise here is **multiplied by 128** in effective error, because every token contributes the same per-pool-entry. |
| **CSA** (compress_ratio=4) | 2,4,6,…,40 | SPARSE attn: `indexer.weights_proj` picks top-512 of 4×-pool, then DSA | `indexer.weights_proj` is the **DECISION** projection — its quant noise causes wrong tokens to be selected. Codec noise here = silent attention drift, hardest to debug. |

This is why the audit said "compressor.wkv + indexer.weights_proj must be
bf16 passthrough." Codec noise on aggregator/decision tensors is multiplied
by the compression ratio in effective error.

## §3 What we already changed in §3a of the main doc

(Recap so this is self-contained.)

| change | before | after | impact |
|---|---|---|---|
| `attn.compressor.wkv.weight` | 8-bit affine gs=32 | **bf16 passthrough** | HSA pool aggregator clean; HSA pool entries no longer eat per-token codec noise × 128 |
| `attn.indexer.compressor.wkv.weight` | 8-bit affine gs=32 | **bf16 passthrough** | CSA pool aggregator clean |
| `attn.indexer.weights_proj.weight` | 8-bit affine gs=32 | **bf16 passthrough** | CSA top-k SELECTION decision clean — chooses correct pool entries |
| `hc_*_{fn,base,scale}` | fp16 (lossy from F32 source) | **fp32 in bundle** | mHC mixing matrices preserved bit-exact; doubly-stochastic Sinkhorn no longer drifts |
| `attn_sink` | fp16 | **fp32** | per-head learned sink logit preserved (paper §1.4.4 specifies F32 source) |
| `ffn.gate.bias` | fp16 | **fp32** | noaux_tc routing bias preserved bit-exact |

These are convert-time policy fixes. NEXT bundle will have them. The current
production OsaurusAI bundle does not.

## §4 What we can STILL do (codec/kernel co-design proposals)

We control:
- `jang_tools/turboquant/codebook.py` — codebook computation
- `jang_tools/turboquant/rotation.py` — Hadamard rotation
- `jang_tools/turboquant/linear.py` — `tq_quantize_weight` + `tq_quantize_experts`
- `jang_tools/turboquant/{tq_kernel,gather_tq_kernel,fused_gate_up_kernel,kmeans_block_kernel}.py` — Metal kernel families
- `vmlx/swift/Sources/vMLXLMCommon/JANGTQKernels.swift` — Swift native kernel

So every option below is on the table.

### §4.1 Per-layer-type bit allocation

DSV4's 43 layers are NOT homogeneous. Layer 0 + last carry the embedding/
unembedding gradient, the 21 HSA layers do dense pool attention, the 20
CSA layers do sparse selection, and the 3 hash-routed layers have
deterministic per-token routing.

**Proposal**: per-layer bit policy at convert time. Trivially expressible
in `convert_dsv4_jangtq.classify(name)`:

| layer class | layers | proposed routed-expert bits | rationale |
|---|---|---|---|
| Hash-routed (deterministic) | 0, 1, 2 | **3 bit** (bump from 2) | hash means each token sees ONE specific expert, no smoothing → quant noise compounds harder. Already exposed via `DSV4_HASH_BITS` env; default it ON. |
| First/last (compress_ratio=0) | 0, 42 | **3 bit** | embedding-adjacent; high gradient sensitivity. Costs ~0.5 GB. |
| Middle SWA-only (none in current arch) | — | — | n/a |
| HSA layers | 1,3,…,39 | 2 bit (current) | dense pool already smooths over 128 tokens; codec noise gets averaged. |
| CSA layers | 2,4,…,40 | 2 bit (current) | top-k selection forces a sparse subset; the SELECTED entries get codec noise but NON-SELECTED are pruned, partial smoothing. |

Net cost over the bundle: ~2-4 GB extra. Net quality: probably +1-3 pp on
hard benchmarks. We've never measured this — it's the cheapest experiment.

### §4.2 Codebook tuning per `(in_features, role)`

Current `compute_codebook(in_features, bits)` is a generic K-means initialization
on a uniform distribution. The actual weight distributions per layer-class differ:

- `wq_b` (1024 → 32768) is **wide-output**, learned over 43 layers — heavy-tailed
- `wkv` (4096 → 512) is **narrow-output** — different statistics
- routed experts `experts.E.{w1,w2,w3}` are **per-expert** — long-tailed gating

**Proposal**: import the actual bf16 source distributions for one
representative layer-class instance, fit K-means there, **store the
codebook in `jangtq_runtime.safetensors`** (small — 4 floats per
distinct codebook). Same machinery already exists for the global
codebook; we'd just write multiple entries:

```
codebook.<role>.<in_features>.<bits>   float32  shape=(2**bits,)
```

The Swift loader picks the role-tagged codebook when present, falls back to
generic. The `tq_packed` data layout doesn't change. Effort: half a day.

### §4.3 Hadamard rotation: per-pool-entry instead of per-row

For CSA pool entries, every entry represents 4 tokens. Currently we apply
Hadamard rotation over the SAME `in_features=512` (head_dim) regardless
of whether the row represents a single token or a pool entry. That mixes
token-aligned and pool-aligned variance into the same rotation.

**Proposal**: for `compressor.*` and `indexer.*` modules specifically,
apply a SECOND rotation along the pool-entry axis (i.e. across the 4
source tokens that pooled into one entry). Keep the per-row rotation as
before; add a fused per-block rotation in the kernel.

This is a kernel-side change. Effort: 1-2 days. Payoff: speculative;
worth a measurement on a held-out long-prompt bench.

### §4.4 Routed-expert quantization aware of expert specialization

DSV4's 256 routed experts × top-6 routing means each expert sees a small,
specialized slice of the input distribution. The hash-routed layers
(first 3) make this MORE deterministic — each `(token_id, expert)` pair is
fixed at training.

Stock JANGTQ stacks all 256 experts into one big array and quantizes
together. We could:

- **Per-expert codebook** for layer 0/1/2 (hash-routed): instead of one
  shared codebook, one codebook per expert (256 × 2-bit codepoints =
  trivial overhead). Captures per-expert distribution.
- **Skip-quant the most-used experts**: if expert hit-rate is highly
  unbalanced, top-K most-used experts could stay bf16. Need calibration
  to find K, but cost is bounded.

Both proposals require touching `tq_quantize_experts` to emit per-expert
codebook tags. The Swift kernel would need to dispatch on the per-expert
codebook tag. Effort: medium. Payoff: speculative but theoretically
strongest for hash-routed layers.

### §4.5 KV cache quantization for HSA + CSA pool entries

Pool entries for the 41 HSA + CSA layers can be quantized in-cache. The
current runtime stores them in bf16. They're **read-only** after the
Compressor produces them (no further mutation), so a codec applied at
write time stays valid for the entire decode. Quality impact: pool entries
are already heavily averaged (HSA averages 128 tokens), so 4-bit affine
on the pool would be near-lossless. Memory savings: ~4× on the pool
portion of the cache, which is the bigger half for long context.

This is a runtime/cache change, not a weight codec change. Effort: medium.
Payoff: makes 1M context actually fit in 128 GB on M5 Max.

### §4.6 Hash-routing-aware kernel

The first 3 layers use hash routing — deterministic per-token expert
assignment. Today the Metal kernel pretends it's a top-k softmax route
and computes a meaningless score. We could **specialize a hash-routed
gate kernel** that:

- skips the score computation entirely
- writes weights = scores at the deterministic indices

Marginal speedup (3 layers / 43), but it's correctness/speed-aligned and
a one-day change.

## §5 Order of operations (cheapest → most expensive)

1. **§4.1 per-layer bit policy** — half a day, low-risk, mostly already exposed via env vars (`DSV4_HASH_BITS`)
2. **§4.2 role-tagged codebooks** — sidecar already there, just add codebook entries
3. **§4.5 KV cache pool quantization** — runtime change, big memory savings, moderate effort
4. **§4.3 per-pool-entry Hadamard** — kernel change, speculative; only do if §4.1+§4.2 don't close the gap
5. **§4.4 per-expert / skip-quant on hash layers** — kernel + converter change, only if pp-on-hard-benchmarks data justifies
6. **§4.6 hash-routing kernel specialization** — speed-aligned, low priority for quality

## §6 Verification protocol (don't ship without this)

For any of §4's experiments:

1. **Codec round-trip on three representative tensors** — `wq_b L0`, `compressor.wkv L1`, `indexer.weights_proj L2`. Compute `cos(decoded, source)`. Threshold ≥ 0.94 for 2-bit, ≥ 0.99 for 4-bit. Same harness we used for Mistral and Laguna.
2. **Layer-by-layer hidden state diff** vs torch reference for a 256-token prompt. Threshold per-layer RMS diff < 1e-3 vs bf16 source. The mlx_model.py has hooks for this; just need to wire the torch ref.
3. **MMLU 30q smoke** in Non-Think mode — should be ≥ 76 % (current floor); regression caught here is fast.
4. **MMLU 200q two-pass** in Think High — should be ≥ 80 % MMLU-Pro to claim "approaching paper Non-Think baseline 83 %."
5. **Long-prompt coherence** at 4096 tokens — Compressor + Indexer pool fully engaged, both modes; output must stay coherent past the window, no `### 2.2.1.2` repetition tail.

## §7 The one thing not negotiable

**Every JANGTQ bundle that ships, including any §4 experiment, must
include `jangtq_runtime.safetensors` co-emitted by the converter.** If
we add per-role or per-expert codebooks (§4.2 / §4.4), they go INSIDE
this sidecar, named with the role tag. The Swift loader will be updated
to read the new tags; the file remains the single source of truth for
"what does the Swift native runtime need to decode this bundle." See
`feedback_jangtq_swift_sidecar.md` for the rule. This is now baked
into both `convert_laguna_jangtq.py` and `convert_mistral3_jangtq.py`.
DSV4's `convert_dsv4_jangtq.py` already had it (sidecar shipped from
day one); the new format additions go there too.
