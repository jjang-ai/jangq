# Laguna S 2.1 — Swift (vmlx-swift) runtime notes

Target: `/Users/eric/vmlx-swift` — the Laguna family model lives under
`Libraries/MLXLLM/Models/` (M.1 port) with config decode in
`LagunaConfiguration`. The 2026-06-19 source-only parity audit
(`docs/local/CODEX_LAGUNA_RESIDUAL.md` in vmlx-swift) found the durable
M.1 mismatch and it is EXACTLY the class of bug S-2.1 will re-trigger, so
this checklist is ordered by what actually broke before.

## Port checklist (ordered by risk)

### 1. Attention must be layer-indexed  ← the audited M.1 bug
Python consumes `num_attention_heads_per_layer[i]`, `layer_types[i]`, and
selects RoPE from `rope_parameters[layer_types[i]]`. The audited Swift M.1
attention was built WITHOUT `layerIndex` and hard-coded the
`full_attention` RoPE lookup. S-2.1 makes this worse than M.1: heads are
48 (full) vs **72** (SWA) and the two RoPEs differ in theta (500k/10k),
type (yarn/default), AND rotary fraction (0.5/1.0).

```swift
// per layer i:
let nHeads   = config.numAttentionHeadsPerLayer[i]      // 48 or 72
let layerT   = config.layerTypes[i]                     // full_attention | sliding_attention
let rp       = config.ropeParameters[layerT]!           // NOT ["full_attention"]!
let ropeDims = Int(Float(headDim) * (rp.partialRotaryFactor ?? 1.0)) // 64 or 128
```

`LagunaConfiguration` already decodes all of these (audit confirmed) —
the model code just has to actually use them.

### 2. Partial rotary on full-attention layers
Rotate only the first `ropeDims` (64 of 128) dims; concat the untouched
tail. SWA layers rotate all 128. YaRN: HF `attention_factor`
(1.4852030263919618) maps to the `mscale` parameter of the Swift YaRN
implementation (`RoPEUtils` matched the fixed mscale math in the audit —
reuse, don't reimplement). factor 128, original_max 8192, theta 500k.

### 3. Gating: branch on g_proj width, softplus in fp32
S-2.1 `gating="per-head"` → `g_proj` out = nHeads → one gate per head,
broadcast over headDim. M.1 is per-element (nHeads·headDim) → elementwise.
The Python runtime branches on the tensor width at call time; do the same:

```swift
let gate = softplus(gProj(x).asType(.float32)).asType(out.dtype)  // softplus, NOT sigmoid
if gate.dim(-1) == nHeads {           // S-2.1 per-head
    out = (out.reshaped(B, T, nHeads, headDim) * gate[.ellipsis, .newAxis])
        .reshaped(B, T, nHeads * headDim)
} else {                              // M.1 per-element
    out = out * gate
}
```
sigmoid here was the historic residual-blow-up bug (std 0.29 → 11 over 30
layers) — softplus is load-bearing.

### 4. Cache + masks (the bugs just fixed in the Python port, 2026-07-21)
- SWA layers: rotating cache `maxSize=512`, **keep=0** — NO attention
  sinks. The HF reference gates sinks behind `swa_attention_sink_enabled`,
  unset in every shipped Laguna config. (Python had keep=4 from the gemma
  habit; token divergence starts exactly at the window boundary.)
- Prefill masks are PER LAYER TYPE: full layers get plain causal, SWA
  layers get the banded window mask (block `j > i` and `i - j >= 512`).
  Never derive one mask from `cache[0]` and share it — layer 0 is full
  attention, so SWA layers silently attend the whole prefix on prompts
  > 512. This passes every short smoke test and fails agentic use.
- Full-attention KV grows unbounded (1M ctx) — that's the memory planning
  input: 12 full layers × 8 kv × 128, SWA layers cap at 512.

### 5. Router math (verified against S-2.1's shipped modeling_laguna.py)
sigmoid(logits_fp32) → selection scores = sigmoid + e_score_correction_bias
(bias affects WHICH experts, never the weights) → top-10 → gather UN-biased
scores → renorm (norm_topk_prob) → `routed * 2.5 + shared` (shared
unscaled). Router logit softcapping is 0.0 = disabled. Do router math in
fp32; the packed-expert `SwitchGLU` transpose/gather convention in
`SwitchLayers.swift` already matches (audit-verified).

### 6. Mixed-precision affine load
Bundles are per-module bits (`config.json[quantization]`: attention 8,
shared/dense/embed 6, routed 2/2/3 or 4/4/4). The loader must honor
per-module overrides — a single top-level bits dequantizes 2-bit packed
rows as 8-bit and errors (or worse, garbage that mimics the quant floor —
the JANGTQ config metadata bug pattern). Python derives true bits from
packed-vs-scales shapes as a cross-check; mirror that if cheap.

### 7. Chat protocol plumbing
eos [2, 24] (id 24 = end-of-turn — MUST be in the stop set or chat runs
on), bos 2 emitted by the template itself (`〈|EOS|〉` literal — do not
prepend another), thinking default ON via
`jang_config.chat.template_kwargs_defaults.enable_thinking`. Generation
prompt tails: `<assistant><think>` / `<assistant></think>`.

## Verification (Swift side)

1. Token-for-token greedy parity vs the Python runtime, same bundle, same
   prompt: short (32 in / 32 out) AND long (1500 in / 32 out — crosses the
   window; this is the one that catches §4 bugs). Python side:
   `python docs/runtime/laguna-s21/python_example.py --src <bundle> --parity`
   then compare Swift tokens to the same run.
2. Perturbation probe if parity fails past 512: change one prompt token
   ~600 back; SWA-only divergence patterns point at masks/cache, uniform
   divergence points at RoPE.
3. The 17-point runtime checklist (`research/GLM-5.1-RUNTIME-AUDIT.md`)
   before calling the port done. (feedback_runtime_before_quant)
