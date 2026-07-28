# Nanbeige 4.2-3B — Swift (vmlx-swift) port notes

Target: `/Users/eric/vmlx-swift`. New file `Libraries/MLXLLM/Models/Nanbeige.swift`,
registered in `LLMModelFactory.swift`. Read [`NOTES.md`](NOTES.md) first — this
file is only the port checklist, ordered by risk.

Good news: the per-layer math is **plain Llama**. `Libraries/MLXLLM/Models/Llama.swift`
is a near-complete starting point — GQA, SwiGLU, pre-norm RMSNorm, NeoX RoPE,
no bias, no qk-norm, no sliding window, no MoE. Copy it and change three things:
the loop, the cache count, and the head-dim derivation.

---

## 1. `newCache` must return 44 entries ← the one that will bite

`Libraries/MLXLMCommon/LanguageModel.swift:399` gives every
`KVCacheDimensionProvider` an automatic `newCache`:

```swift
public protocol KVCacheDimensionProvider { var kvHeads: [Int] { get } }
extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = kvHeads.count            // <-- cache size comes from HERE
        ...
    }
}
```

If `kvHeads` is built as `Array(repeating: nKVHeads, count: numHiddenLayers)`
you get **22** caches for a model that needs **44**, loop 1 re-enters loop 0's
slots, and the model emits confident wrong tokens with no error. Do one of:

```swift
// preferred: make the count structurally correct
public var kvHeads: [Int] {
    Array(repeating: config.numKeyValueHeads,
          count: config.numHiddenLayers * config.totalLoops)   // 44
}
```

or override explicitly (mirror `BailingHybrid.swift:1223`):

```swift
public func newCache(parameters: GenerateParameters?) -> [KVCache] {
    let n = config.numHiddenLayers * config.totalLoops
    if let maxKVSize = parameters?.maxKVSize {
        return (0 ..< n).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: 4) }
    }
    return (0 ..< n).map { _ in KVCacheSimple() }
}
```

Then **assert** against the bundle: `config.json["jang_runtime"]["cache_slots"]`.
A mismatch is a hard failure, not a warning.

`RotatingKVCache(keep: 4)` is fine here — this model has no sliding-window
layers and no attention sinks; the rotating cache is only ever a
user-set `maxKVSize` policy, not an architectural requirement.

## 2. The loop in `callAsFunction`

```swift
public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = embedTokens(inputs)
    let nLayers = layers.count                       // 22
    let slots = cache?.count ?? 0
    precondition(cache == nil || slots == nLayers * numLoops,
                 "nanbeige: expected \(nLayers * numLoops) cache slots, got \(slots)")

    // ONE mask for every loop: all slots carry the same offset because each
    // loop appends the same token span at the same positions.
    let mask = createAttentionMask(h: h, cache: cache.map { [$0[0]] } ?? [])

    for loop in 0 ..< numLoops {
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i + loop * nLayers])
        }
        if !skipLoopFinalNorm { h = norm(h) }        // EVERY loop
    }
    if skipLoopFinalNorm { h = norm(h) }
    return h
}
```

Three things people get wrong here:

* **Recomputing the mask per loop.** Harmless if done *before* that loop's
  cache updates, wrong if done after. Compute once, up front.
* **Applying `norm` only after the last loop.** `skip_loop_final_norm` is
  `false` on 4.2-3B → the norm runs at the end of **every** loop and its output
  feeds layer 0 of the next loop.
* **Advancing positions per loop.** RoPE offset comes from the *slot* of the
  loop being executed; every slot has the same offset. Positions do not double.

## 3. Head dim: read it, don't derive it

```swift
let headDim = config.headDim ?? config.kvChannels ?? (hiddenSize / nHeads)   // 128
```

`nHeads * headDim` = 48 × 128 = **6144**, `hiddenSize` = **3072**.
`hiddenSize / nHeads` = 64 — half the true head dim. `q_proj` is
`[6144, 3072]`, `o_proj` is `[3072, 6144]`. Any code that assumes
`nHeads * headDim == hiddenSize` (a very common shortcut in the Llama-family
ports) produces a shape error at best and a silent reshape at worst.

Attention scale is plain `1/sqrt(headDim)` = `128 ** -0.5`.

## 4. Configuration decoding

```swift
public struct NanbeigeConfiguration: Codable, Sendable {
    let hiddenSize: Int                 // 3072
    let numHiddenLayers: Int            // 22
    let intermediateSize: Int           // 10752
    let numAttentionHeads: Int          // 48
    let numKeyValueHeads: Int           // 8
    let headDim: Int                    // 128   (also `kv_channels`)
    let rmsNormEps: Float               // 1e-5
    let vocabSize: Int                  // 166144
    let ropeTheta: Float                // 7e7   <- not 10000, not 500000
    let maxPositionEmbeddings: Int      // 262144
    let tieWordEmbeddings: Bool         // false -> bare lm_head
    let attentionBias: Bool             // false

    // loop
    let numLoops: Int                   // 2
    let loopLossWeights: [Float]?       // []  — if non-empty, loops = count + 1
    let skipLoopFinalNorm: Bool         // false

    // must be rejected, not ignored (Nanbeige 4.5 turns these on)
    let enableDoubleLoopSplit: Bool?    // LoopSplit
    let loopShareKv: Bool?
    let enableHyperConnection: Bool?
    let enableMhc: Bool?
    let enableDepthAttention: Bool?
    let qkLayernorm: Bool?
    let embNeighborNum: Int?            // n-gram embeddings

    var totalLoops: Int {
        if let w = loopLossWeights, !w.isEmpty { return w.count + 1 }
        return max(1, numLoops)
    }
}
```

`totalLoops` must mirror `NanbeigeModel._get_num_loops`: `loop_loss_weights`
**overrides** `num_loops` when non-empty. Throw on any of the six unsupported
flags — a silent fall-through to the plain path would look like a working model.

## 5. RoPE

`theta = 70_000_000`, `rope_scaling = null`, NeoX half-rotation, **full** 128
dims rotated (no partial-rotary factor, no YaRN, no NTK). This is the plain
`RoPE(dimensions: headDim, traditional: false, base: 70_000_000)` path — no
`RoPEUtils` YaRN machinery, unlike Laguna.

Watch precision: at θ=7e7 and 262 K positions the inv-freq spread is wide;
compute cos/sin in fp32 (the reference explicitly forces fp32 for this reason —
`NanbeigeRotaryEmbedding.forward` disables autocast). MLX's fast RoPE already
does this internally; if you hand-roll it, don't do it in fp16.

## 6. Norms: NO +1

`NanbeigeRMSNorm` = `weight * x / rms(x)` — the plain Llama form, identical to
`RMSNorm` in mlx-swift. This is **not** the Qwen3.5 / Ornith `weight + 1`
convention, and not Gemma's. Bundles ship norms as raw fp16 with
`jang_runtime.norm_convention = "llama_rmsnorm_no_plus_one"`; do not shift them
on load and do not shift them in the kernel.

## 7. Weights / loader

Names are stock Llama, so `JangLoader`'s per-tensor shape walk needs nothing new:

```
model.embed_tokens.weight
model.layers.{0..21}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}
model.layers.{0..21}.mlp.{gate,up,down}_proj.{weight,scales,biases}
model.layers.{0..21}.{input_layernorm,post_attention_layernorm}.weight
model.norm.weight
lm_head.weight                       <- BARE, no `language_model.` prefix
```

`tie_word_embeddings = false` and there is no vision tower, so per
`AGENTS.md` the head must stay bare `lm_head` — do not add the VL prefix.

Mixed-bit: `config.json["quantization"]` is `{group_size, bits: 8, mode}` plus a
**per-module override for every 4-/6-bit module** (68 of them on JANG_4M/6M).
Read the overrides — a single top-level width dequantizes 8-bit attention with
the 4-bit kernel and emits garbage. MXFP8 has zero overrides (uniform).

Layers are shared across loops: **one** `layers` array of 22 blocks, referenced
twice. Do not duplicate weights to 44 blocks — that doubles resident memory for
no benefit.

## 8. Registration + capabilities

```swift
// LLMModelFactory.swift
"nanbeige": create(NanbeigeConfiguration.self, NanbeigeModel.init),
```

Capability stamp already lands in the bundle from `jang_tools/capabilities.py`:
`reasoning_parser: "qwen3"`, `tool_parser: "xml_function"`,
`think_in_template: true`, `cache_type: "kv"`, `modality: text`.

`cache_type` is deliberately `"kv"` — the per-slot codec really is ordinary KV,
so prefix/paged/L2/TQ-KV paths work unchanged. The *only* deviation is the slot
count, which is why it lives in `jang_runtime.cache_slots` and must be asserted
rather than inferred.

## 9. Prompt encoder

* Tokenize the rendered template with **`addSpecialTokens: false`**. The
  template already emits `<|im_start|>` (= `bos_token`, 166100) and the
  tokenizer's post-processor will otherwise prepend a duplicate.
* Bundles ship a standalone `chat_template.jinja` (extracted from
  `tokenizer_config.json` at conversion time) for the Swift Jinja path.
* Thinking rail: default and `enable_thinking=true` end the prompt with an
  **open** `<think>\n`; `enable_thinking=false` prefills
  `<think>\n\n</think>\n\n`. The reasoning parser therefore sees only a closing
  `</think>` in the model output — the `qwen3` parser's implicit-reasoning mode.
* Multi-turn: `preserve_thinking` defaults to **true** and every published
  benchmark was run that way. Keep historical reasoning in the rendered history
  unless the user explicitly turns it off.
* Stop ids: `166101 <|im_end|>` (eos) and `166102 <|endoftext|>`.
* Tool results: consecutive `tool` messages merge into **one** user turn with
  one `<tool_response>` block each.

## 10. Regression tests to write

1. `newCache().count == 44` and `== config.jang_runtime.cache_slots`.
2. Prefill-then-step logits ≈ full-prefill logits (rel < 1e-2).
3. **Negative control**: build 22 caches, pass `cache[i % 22]`, assert the
   argmax *differs* from the correct path. This is the regression test that
   catches the whole bug class; the Python side measured rel 9.0e-1 and a
   different argmax.
4. Config rejection: a config with `enable_double_loop_split=true` (or any of
   the other five flags) must throw, not load.
5. Single-BOS: rendered 1-turn chat tokenizes to exactly three `166100`, not
   four.
