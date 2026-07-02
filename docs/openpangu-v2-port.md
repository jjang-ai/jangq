# OpenPangu-2.0-Flash (`openpangu_v2`) — vmlx-swift port: issues & fixes

Working log of porting Huawei's **openPangu-2.0-Flash** (92B MoE, 6B active, 512K ctx)
to vmlx-swift. Branch `feat/openpangu-v2`. Model files:
`OpenPanguV2.swift` (attention + MoE), `OpenPanguV2MHC.swift` (hyper-connections),
`OpenPanguV2Model.swift` (decoder/inner/outer + sanitize), `OpenPanguV2Cache.swift`,
`OpenPanguV2Configuration.swift`, factory `dispatchOpenPanguV2`.

## THE reference (this is the ground truth — use it, don't guess)

The modeling code is **NOT in HuggingFace or vLLM** (HF ships only
`configuration_*.py` + `tokenization_*.py`; vLLM only has the older
`pangu_ultra_moe`, which has sinks but no mHC/convs/DSA). The real forward is in
the official Ascend inference repo:

**`git clone https://gitcode.com/ascend-tribe/openPangu-2.0-Infer`**
- `components/omni-npu/src/omni_npu/layers/mhc/npu_mhc.py` — the mHC (has readable
  `_mhc_pre_naive` / `_mhc_sinkhorn_naive` / `_mhc_post_naive` fallbacks).
- `components/omni-npu/src/omni_npu/v1/layers/attention/npu_pangu.py` — MLA +
  sinks + convs + DSA indexer (Ascend-fused, ~3.5k lines, but the ordering is readable).
- `components/omni-npu/src/omni_npu/v1/models/pangu/pangu_v2_moe.py` — decoder
  layer + MoE + sandwich norm.

It is Ascend/torch_npu code (can't run on Mac), but the op ORDER and math are
what you need. Read `_mhc_*_naive` and the `_forward*` q/kv streams.

## Architecture (openpangu_v2, from config + reference)

- 46 layers, hidden 2560, 48 heads, vocab 151552, rope_theta 6.4M, rope_interleave=False.
- **MLA**: q_lora 1024, kv_lora 512, qk_nope 128 + qk_rope 64, v 128. scale = qk_head_dim^-0.5 (192^-0.5), no mscale (no yarn).
- **3 causal depthwise convs** (kernel 3, stateful, Mamba-style): `qa_conv` on the
  q latent, `compresskv_conv` on the compressed-kv split, `o_conv` on the attn output.
- **128 prepended-KV attention sinks** (`param_sink_compressed_kv` [128,512],
  `param_sink_k_pe` [128,64]) — position-free.
- **DSA + SWA hybrid** at 1:2: `dsa_layers`=[0,3,6,…45] (16, full-attn + lightning
  indexer top-2048), `swa_layers`=rest (sliding window, `sliding_window_list`
  512×30 then 2048×3).
- **mHC (4-stream hyper-connections)**: `attn_mhc_module` + `mlp_mhc_module` per
  layer + global `merge_mhc_module`. Replaces the plain residual with a 4-stream one.
- **Sandwich norm** (input/post-attn/pre-mlp/post-mlp) + `block_post_layernorm` on 9 layers.
- **MoE**: 256 routed + 1 shared, first_k_dense_replace=2, sigmoid gate + biased
  top-8, routed_scaling_factor 2.5. (`use_mome=True` is a WEIGHTLESS flag — the
  MoME name = the mHC/conv "memory" machinery, no extra weights.)
- **MTP depth 3** (layers 46-48). Reasoning model (deepseek_r1 style, `<think>`).

## Bugs hit & fixes (in the order found)

Chase these in order; each was found by live-loading JANG_2L in osaurus and reading output.

### Loader / plumbing (get it to LOAD)
1. **Tokenizer**: `tokenizer_class = OpenPanguV2Tokenizer` unknown to
   swift-transformers. It's a standard byte-level BPE → map to `Qwen2Tokenizer` in
   `JangLoader.defaultTokenizerClassSubstitutions`.
2. **phi is quantized + bit-ambiguous**: `attn/mlp_mhc_module.phi` ships 2-bit
   affine (weight[24,640]+scales[24,80]+biases). The JANG shape-walk mis-infers it
   2-bit→8-bit (packed 640 satisfies both). Fix: in `sanitize`, DEQUANTIZE phi to
   dense fp16 using the KNOWN logical dim (mhcNumStream*hidden=10240 → bits exact),
   rename `…phi.weight`→`…phi`, and make phi a RAW `@ParameterInfo` (not a Linear)
   so the quant substitution can't re-quantize it. The config quant dict only
   rewrites Linear/Embedding SUBMODULES.
3. **Experts already stacked**: bundle ships `mlp.switch_mlp.{gate,up,down}_proj.*`
   (matches SwitchGLU) — do NOT stack per-expert (there are no `experts.N.*` keys).
4. **`e_score_correction_bias`** ships at `mlp.e_score_correction_bias` (one level
   up from the gate) → remap to `mlp.gate.e_score_correction_bias`.
5. **conv weight** is PyTorch `[C,1,3]` → transpose to MLX Conv1d `[C,3,1]`.
6. **`block_post_layernorm.weight` is `[4*hidden]=10240`** (the flattened 4-stream
   residual), NOT `[2560]`. Init `RMSNorm(mhcNumStream*hidden)` and apply to the
   FLATTENED residual (reshape (B,L,4,H)→(B,L,4H), norm, reshape back). A `[2560]`
   RMSNorm on a `[…,4,2560]` tensor silently collapses to a scalar → crash.

### Numerical correctness (get it COHERENT) — the money bugs
7. **mHC base params were mis-shaped**: `branch_beta` is `[24]` (= the DSV4-style
   `base[24]`, split pre[0:4]/post[4:8]/comb[8:24]), and `branch_beta_pre` is
   `[mhcNumStream]=4` — NOT 3 / 1 per-field scalars. `branch_alpha` IS `[3]` (scale).
   Use branch_beta directly as base.
8. **mHC expand transpose** (`NPUmHC._mhc_post_naive`): the residual mix is
   `new[j] = Σᵢ h_res[i,j]·residual[i]` = **`h_resᵀ @ residual`**, NOT
   `h_res @ residual`. Transpose comb's last two (stream) axes before the matmul.
   Also use `hc_eps=1e-6` (not rms_norm_eps) for the sigmoid/softmax/sinkhorn, and
   the MERGE (pre_only) gate has NO `+eps`.
9. **conv residual connection**: `npu_ai_infra_fused_causal_conv1d(...,
   residual_connection=1)` → the conv is `y = conv(x) + x`, not `conv(x)`.
10. **conv ORDER (before layernorm)**: reference is
    `q_a_proj → qa_conv → q_a_layernorm → q_b_proj` and
    `kv_a_proj → split → compresskv_conv(k_nope) → kv_a_layernorm → kv_b_proj`.
    The conv runs on the RAW projected latent BEFORE the layernorm. Applying the
    conv AFTER the layernorm mixes normalized features (wrong space) → coherent but
    factually-off output. `o_conv` is after attention, before o_proj (that one's fine).

### Confirmed CORRECT against the reference (don't "fix" these)
- rope = `rotary_mode="half"` = non-traditional / split-half (config
  rope_interleave=False). Do NOT use traditional/interleaved.
- attention scale = qk_head_dim^-0.5, no mscale.
- **sinks HELP** — removing them re-degrades output. sink_k_nope =
  `kv_a_layernorm(param_sink_compressed_kv)`; value_sink = same; key_rope_sink =
  `param_sink_k_pe` raw (NO rope). Position-free. (In the expanded-MLA formulation
  we run, expand the sink through kv_b_proj to per-head nope+v; equivalent to the
  reference's absorbed op.)
- MoE gate = sigmoid scores; select on `scores + e_score_correction_bias`; weight
  with UNBIASED scores; renormalize (norm_topk_prob) ×routed_scaling_factor.
  `use_grouped_topk=True, num_expert_group=1, topk_group=1` == plain top-k.
- DSA lightning indexer is a **no-op for prompts <2048 tokens** (it selects top-2048;
  fewer keys → selects all). So it CANNOT explain short-prompt accuracy issues.

## Engine / decode-loop discipline (do this FIRST, before caching)

Lesson from this port: **prove a simple, correct decode loop before touching prefix/
paged/SSD caching.** The right sequence:

1. **Single-turn prefill correctness**: prompt "The capital of France is", check the
   FIRST predicted token (argmax of last-position logits). If it isn't ~"Paris",
   the forward pass is wrong — do NOT blame the sampler/cache/quant. This one check
   distinguishes a forward bug from decode/cache bugs.
2. **Greedy decode loop**: temp=0, generate 20-40 tokens with the plain KV cache
   only (no prefix/paged/SSD). Verify coherent + on-topic. openpangu is a REASONING
   model — it opens `<think>` and needs high max_tokens (hundreds) to close it and
   emit `content`; at ~0.5-1.5 tok/s on M-series a 2-bit 92B is slow, so test with
   `thinking=false` (chat_template kwarg) for a fast direct answer during dev.
3. **Path-dependent state carry**: the 3 conv-states MUST round-trip with the KV
   across decode steps AND turns. A KV-only cache reuse without the conv-state is a
   silent false hit (garbled turn-2). vmlx: `OpenPanguV2Cache` carries the 3
   conv-states + is flagged `PathDependentStateCache` so `cacheContainsPathDependentState`
   is true and paged/KV-only reuse is skipped.
4. **Multiturn**: only after 1-3 are green, test turn-2/3 coherence + the SWA
   512→2048 window boundary + DSA at >2048 tokens.
5. **THEN** prefix / SSD / paged cache + quant pooling + MTP decode.

## Test harness (how it was driven)

- Build via osaurus xcodebuild (standalone `swift build` blocked by stale `.build`
  SDK cache): `Packages/OsaurusCore/Package.swift` → `.package(path:"/Users/eric/vmlx-swift")`
  local override, `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  xcodebuild -workspace osaurus.xcworkspace -scheme osaurus -configuration Debug
  -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO build`.
- Serve: symlink bundle into `~/.osaurus/models/JANGQ-AI/…`, run the built binary,
  POST `/v1/chat/completions`. RAM-gated (JANG_2L ~40GB; bf16 187GB won't fit 69GB).
- Diagnostics (env-gated in the model): `OPENPANGU_MHC_TRACE`, `OPENPANGU_MHC_BYPASS`,
  `OPENPANGU_NO_CONVS`, `OPENPANGU_NO_SINKS`, `OPENPANGU_ROPE_TRAD`. Bisection with
  these localized the bugs (but note: with a broken mHC/conv, disabling a
  still-correct component like sinks can look "better" — re-bisect after each fix).

## OPEN: coherent but weak prompt-conditioning (the current wall)

After all 10 fixes the model emits **fluent, structured, sophisticated prose**
(e.g. "2+2?" → a full multi-paragraph Chinese essay on geometry-teaching pedagogy)
but **barely conditions on the actual prompt** — different prompts give
different-but-off-topic essays; a bare "Repeat: banana banana" isn't copied.

Ruled OUT: chat template (renders correctly), tokenization (special tokens
`<|message_start|>`/`<think>` are single tokens, prompt tokenizes clean), decode/
conv-state carry (output is coherent, not garbled → streaming works), and every
architectural component (all match the omni-npu reference).

Two live hypotheses, need to separate them:
1. **2-bit experts** — JANG_2L is 3.17-bit avg but the ROUTED EXPERTS are 2-bit
   (attn/embed/lm_head are 8/6-bit). Experts do the heavy compute in an MoE; at
   2-bit, instruction-following/factual precision collapses while 8-bit attention
   keeps fluency. Plausibly THE cause.
2. A subtle prefill/first-token bug not caught by structural diffing.

**Definitive next tests** (do these, don't hand-wave "it's quant"):
- **Minimal decode loop** (per Eric's guidance): drive the model OUTSIDE osaurus's
  BatchEngine — a raw prefill + greedy TokenIterator loop, KV-cache only, no
  prefix/paged/SSD/batching — on a forced-factual prompt. Check the FIRST decode
  token argmax. If it's right there but wrong through osaurus → engine/cache bug.
  If wrong even in the minimal loop → model (quant or prefill bug).
- **Higher-bit A/B**: quantize the bf16 (187GB) down to 6-bit or 8-bit (fits ~60-90GB)
  with the jang quantizer and re-run the same prompts. If 8-bit answers correctly →
  it's the 2-bit experts, port is fully correct. If 8-bit is ALSO off → a residual
  bug. This is the cleanest separator; the only reason it wasn't done is the bf16
  download + convert cost.
- **Per-layer numerical diff** would be ideal but the reference only runs on Ascend
  (torch_npu), not on Mac.

## Open items (structural)
- DSA lightning indexer (top-2048 on the 16 dsa_layers) — needed only for >2048 ctx.
- MTP depth-3 head (layers 46-48) for spec-decode.
- osaurus catalog: isOpenPanguV2Family + isKnownHybridModel + cache_type hybrid auto-load.
- Verify accuracy vs a higher-bit bundle (only JANG_2L 2-bit + bf16 187GB exist locally).
