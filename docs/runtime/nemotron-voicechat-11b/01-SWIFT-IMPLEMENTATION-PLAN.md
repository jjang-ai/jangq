# VoiceChat 11B — Swift runtime implementation plan

**INTERNAL.** 2026-08-20. Swift is the target; upstream `mlx_vlm` Python is not
the deliverable (Eric, this session).

---

## 1. Load path: use the mlx-community MLX layout, not the NeMo original

The NVIDIA original ships a **NeMo training config** (no `model_type`, no
`architectures`) and one 44 GB fp32 `model.safetensors`. Hand-writing that
mapping is avoidable: `mlx-community/NemotronLabs-VoiceChat-11B-bf16` already
publishes an MLX-layout bundle with a proper HF-style config and a
`model.safetensors.index.json`. That is the layout Swift should load.

    model_type:    nemotron_voicechat
    architectures: [NemotronVoiceChatForConditionalGeneration]

## 2. Config spec (verbatim from the MLX bundle)

### text_config — `nemotron_h`
    vocab_size 131072 · hidden 4480 · intermediate 15680 · layers 56
    heads 40 · kv_heads 8 · head_dim 128
    mamba_num_heads 128 · mamba_head_dim 80 · ssm_state_size 128
    conv_kernel 4 · n_groups 8 · layer_norm_epsilon 1e-05
    hybrid_override_pattern:
      M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-
      M = Mamba2, - = MLP, * = attention  ->  27 / 25 / 4 = 56 ✓

Cross-checked against the raw tensor inventory: 27 `mixer.A_log`, 25
`mixer.up_proj`, 4 `mixer.q_proj`. The pattern string and the weights agree.

### audio_config
    preprocessor(12) · encoder(30) · decoder(6) · joint(6)
    output_dim 4480 · max_symbols 10
🚨 The RNN-T **decoder and joint live inside `audio_config`**, not at top level.

### tts_config
    hidden 1152 · intermediate 4608 · layers 28 · heads 16 · kv_heads 16
    head_dim 72 · sliding_window 7500 · latent_size 512
    num_quantizers 31 · codebook_size 1024 · num_delay_speech_tokens 2
    character_encoder(8) · mog_head(6)
    guidance_scale 0.2 · top_p 0.95 · noise_scale 0.001
    audio_prompt_duration 3.0 · use_gated_fusion_for_text_audio true

### codec_config
    sample_rate 22050 · base_channels 384 · channel_multipliers(3)
    downsample_rates(3) · blocks_per_stage 3 · block_kernel_size 7
    latent_dim 512 · n_fft 16 · hop_length 4
    num_quantizers 31 · codebook_size 1024

### Runtime constants
    bos 1 · eos 2 · pad 12 · **silence_token_id 11** · **rnnt_blank_id 1024**
    input 16000 Hz · output 22050 Hz · frame_duration 0.08
    function_channel_weight 2.0 · speaker "Aria"
    rnnt_vocabulary: 1024 entries shipped inline in config

## 3. 🎉 What vmlx-swift ALREADY has

| Need | Existing Swift | Status |
|---|---|---|
| Nemotron-H 56-layer hybrid | `Libraries/MLXLLM/Models/NemotronH.swift` — parses `hybridOverridePattern`, has `ssmStateSize`/`convKernelSize`, `NemotronHConfiguration` | ✅ **7.714 B / 69 % of the model** |
| JANG quant on that backbone | `NemotronHJANGTQ.swift`, `NemotronHJANGTQContext` | ✅ |
| Conformer / Parakeet encoder | `MLXVLM/Models/NemotronHOmni/Parakeet.swift` (403 ln) — `NemotronHBatchNorm1d`, conformer block with `norm_feed_forward1/2`, `norm_self_att`, `norm_conv`, `norm_out` | ✅ structure present |
| Mel preprocessor | `NemotronHOmni/Preprocessors.swift` (956 ln) | ✅ |
| Audio I/O | `NemotronHOmni/AudioIO.swift` (303 ln) | ✅ |
| Modality projector | `NemotronHOmni/Projectors.swift` | ✅ |
| Composed Omni model | `NemotronHOmni/NemotronHOmni.swift` (1161 ln) | ✅ reference for composition |
| Live audio streaming | `OmniAudioLatencyBench`, `OmniAudioChunkStabilityBench` | ✅ measured 157–219 ms first delta |

**Roughly 70 % of the parameter count and the entire audio-input pipeline
already exist.** This is a much smaller job than the raw model description
suggests.

## 4. What actually has to be written

Ordered by dependency:

1. **`NemotronVoiceChatConfiguration`** — Codable for §2. Small, do first;
   it makes everything else compile-checkable.
2. **Conformer streaming deltas** on `Parakeet.swift`:
   - `conv_norm_type: layer_norm` — current code uses `NemotronHBatchNorm1d`;
     needs a LayerNorm variant selected by config.
   - `att_context_style: chunked_limited` + `att_context_size` — current path
     is full attention; chunked needs encoder context carried across chunks.
   - `causal_downsampling: true` / `conv_context_size: causal`.
   These are exactly the three that make it run causally on a live mic.
3. **`function_head`** — one extra vocab-sized Linear (0.587 B). Mechanically
   trivial, but see §5: it is a first-class output channel.
4. **RNN-T decoder + joint** — small weights (~0.009 B) but a *different decode
   algorithm* from autoregressive LM decode. Blank id 1024, `max_symbols 10`.
5. **TTS stack** — genuinely new, ~1 B:
   - 28-layer / 1152-hidden transformer with `sliding_window 7500`
   - **MoG head** (mixture-of-Gaussians, 0.159 B)
   - **RVQ codec decoder** — 31 quantizers × 1024 codebook, `n_fft 16`,
     `hop_length 4`, 22.05 kHz out
   - `audio_prompt_latents.Aria` — speaker is a learned latent, so voices are
     swappable without retraining
6. **Duplex session layer** — the part with no precedent: concurrent text /
   user / ASR / function channels, `delay_source_text_by 15`,
   `delay_text_channel_by 2`, barge-in, turn-taking.

## 5. Two things that will bite if missed

🚨 **`function_head` is FULL VOCAB SIZE** (0.587 B — same as `lm_head`). Tool
calls are a parallel output channel with their own head, *not* text parsed
after the fact. A runtime that only reads `lm_head` silently loses all tool
calling. `function_channel_weight: 2.0` confirms it is weighted, not incidental.

🚨 Three vocab-sized matrices (`embed_tokens`, `lm_head`, `function_head`) are
**1.76 B = 16 % of the model**. Whatever we do about quantizing those dominates
the size/quality tradeoff — and `function_head` must NOT be treated as a
low-priority tensor just because it is not `lm_head`.

## 6. Quantization — deliberately deferred

JANG 2/4/6 + MXFP8 with the mandated Hessian + AWQ + imatrix trio all require
**running forward passes to capture activations**. That needs a working
runtime. Swift comes first (Eric's call this session); quants follow once
there is something to calibrate against and run on.

When we get there, note that a VoiceChat calibration corpus already exists:
`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` ships
`calibration/fixtures/*.json` + `calibration/replay.tar` — real conversational
replay data, far better suited than the text prompt corpus we use for LLMs.

## 7. Status

- [x] NVIDIA source downloaded (41 GiB fp32) + tensor inventory
- [x] MLX-layout config spec extracted
- [x] Swift reuse audit — backbone + audio input already exist
- [ ] mlx-community bf16 download (in progress, ~22 GB — the load target)
- [ ] `NemotronVoiceChatConfiguration`
- [ ] Conformer streaming deltas
- [ ] RNN-T decode
- [ ] TTS + codec
- [ ] Duplex session

---

# PROGRESS — step 1 landed (2026-08-20)

## `NemotronVoiceChatConfiguration.swift` — DONE, 40/40 tests green

    Libraries/MLXVLM/Models/NemotronVoiceChat/NemotronVoiceChatConfiguration.swift
    Tests/MLXLMTests/NemotronVoiceChatConfigTests.swift

    NemotronHTests                37/37 passed   (regression guard)
    NemotronVoiceChatConfigTests   3/3  passed

Run with the documented form (plain `swift test` cannot find the `Testing`
module on this machine):

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
      --filter "NemotronVoiceChatConfigTests|NemotronHTests" --jobs 2 \
      -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks

The test decodes the **real shipped config** from the downloaded MLX bundle, not
a fixture. That choice paid for itself immediately — see below.

## 🚨 THREE real incompatibilities a fixture would have hidden

### 1. `NemotronHConfiguration` could not decode a DENSE nemotron_h at all
Its custom decoder used required `container.decode` for four MoE fields:
`moe_intermediate_size`, `moe_shared_expert_intermediate_size`,
`n_routed_experts`, `num_experts_per_tok`. VoiceChat's backbone
(Nemotron-Nano-9B-v2) is **dense** and ships none of them, so decoding threw.

Fixed by making all four `decodeIfPresent ?? 0`. **Strictly widening**: any
config that carries them decodes exactly as before; only configs that
previously THREW now load. Safe because a dense bundle's
`hybrid_override_pattern` contains only `M` / `-` / `*` and never `E`, so no
MoE layer is constructed and the values are unused. Verified by 37/37
`NemotronHTests` still passing.

### 2. `att_context_size` is `[[Int]]`, not `[Int]`
The shipped value is `[[70, 0]]` — a LIST OF `[left, right]` PAIRS.

🚨 **Right context is 0 — the encoder never waits for future audio.** That is
the property that makes sub-200 ms duplex possible. A future revision with a
non-zero right context would add algorithmic latency to *every* response with
nothing failing, so `isFullyCausal` is now asserted, not assumed.

Exposed as `leftContextFrames` (70 — sizes the state carried across chunks),
`rightContextFrames` (0) and `isFullyCausal`.

### 3. RNN-T decoder + joint are nested inside `audio_config`
Not top level. Asserted non-nil so a future reshuffle fails loudly instead of
silently producing a model with no transducer.

## Invariants now guarded

| Assertion | Why it matters |
|---|---|
| `audio_config.output_dim == text_config.hidden_size` (4480) | perception→LLM dim mismatch is a load-time crash at best |
| `hybrid_override_pattern` = 27 M / 25 - / 4 * = 56 | pattern vs weights disagreement builds a wrong layer stack |
| `rnnt_blank_id == rnnt_vocabulary.count` (1024) | blank is one PAST the vocab, not a member |
| `tts.num_quantizers == codec.num_quantizers` (31) | RVQ depth mismatch = garbage audio |
| `tts.codebook_size == codec.codebook_size` (1024) | ditto |
| `tts.latent_size == codec.latent_dim` (512) | ditto |
| `function_channel_weight == 2.0` | tool calls are a WEIGHTED channel with their own head |
| `silence_token_id (11) != pad_token_id (12)` | silence is emitted output; pad is absence |
| `frames_per_second == 12.5` | encoder rate must match the duplex timeline |
| encoder `conv_norm_type == layer_norm`, `chunked_limited`, causal | our Parakeet is BatchNorm + full attention — the deltas to build |

## Next
Conformer streaming deltas on `NemotronHOmni/Parakeet.swift`:
LayerNorm conv variant, chunked_limited attention carrying 70 frames of left
context across chunks, causal downsampling.

---

# PROGRESS — step 2 landed (2026-08-20)

## Streaming Conformer — DONE, 8/8 green

    Libraries/MLXVLM/Models/NemotronVoiceChat/StreamingConformer.swift
    Tests/MLXLMTests/VoiceChatStreamingConformerTests.swift

    NemotronVoiceChatConfigTests       3/3
    VoiceChatStreamingConformerTests   5/5

New in its own file, NOT as flags on `NemotronHOmni/Parakeet.swift`: that
encoder is shipped and benchmarked (18/18 RunBench, 157–219 ms first delta) and
symmetric padding is CORRECT for offline Omni. Changing it in place would alter
a working model to serve a different one.

| Delta | Parakeet.swift (offline) | VoiceChat (streaming) |
|---|---|---|
| conv norm | `NemotronHBatchNorm1d` (running stats over whole utterances) | `LayerNorm` — chunk-invariant by construction |
| depthwise pad | `(K-1)/2` **both sides** — reads 4 frames of FUTURE | `K-1` **left only** — strictly causal |
| cross-chunk state | none | carries `K-1` tail frames |
| attention | full | `chunked_limited`, mask from `[[70, 0]]` |

## 🚨 `Parakeet.swift`'s "causal" comment is wrong
Line ~245 reads `// Depthwise causal conv` but the code computes
`pad = (K-1)/2` on both sides. Harmless offline — and left untouched — but the
comment is misleading and someone will eventually trust it. On a live mic that
padding means frame `t` cannot be emitted until `t+4` exists: silent added
latency no offline test can observe.

## Tests assert PROPERTIES, not outputs
A shape/smoke test on a conv passes whether or not it is causal, and whether or
not chunked inference matches whole-sequence. Both are wrong only on a live mic.

1. **Causality** — perturb the LAST input frame, assert every earlier output is
   bit-unchanged (< 1e-5). Plus a **vacuity guard**: the perturbation must still
   reach the last output (> 1e-3), so the test cannot pass on a conv that
   ignores its input. The Omni symmetric conv would FAIL this.
2. **Chunk invariance** — uneven chunks `[7,5,11,4,3]` with carried state equal
   whole-sequence to < 1e-4.
3. **Negative control** — stateless chunking must DIFFER (> 1e-3). Without this
   the invariance test could pass for a trivial reason.
4. **`[[70, 0]]` mask** — 70 frames back allowed, 71 blocked, ALL lookahead
   blocked.

Why (2)/(3) matter: without cross-chunk carry every boundary zero-pads and
injects an artifact once per chunk — periodic glitching, invisible to any
whole-sequence test.

## Status
- [x] 1. `NemotronVoiceChatConfiguration` (40/40)
- [x] 2. Conformer streaming deltas (8/8)
- [x] 3. `function_head` (VoiceChatSTTModelTests 5/5)
- [x] 4. RNN-T decode loop (same suite)
- [ ] 5. TTS: MoG head + RVQ codec
- [ ] 6. Duplex session layer

---

# PROGRESS — steps 3+4 landed (2026-08-20, vmlx-swift `voicechat-integration` cfef5f69)

    Libraries/MLXVLM/Models/NemotronVoiceChat/VoiceChatConformer.swift   full 24L encoder, REAL keys
    Libraries/MLXVLM/Models/NemotronVoiceChat/VoiceChatRNNT.swift        transducer + greedy stream decode
    Libraries/MLXVLM/Models/NemotronVoiceChat/VoiceChatSTTModel.swift    perception + backbone + BOTH heads
    Tests/MLXLMTests/VoiceChatSTTModelTests.swift                        5/5 (real bundle)

Key discoveries for whoever wires the rest:

1. **The bundle stores the backbone FLAT under `llm.*`** (`llm.layers.*`,
   `llm.norm_f.*`) with `embed_tokens`/`lm_head` as stt-level siblings —
   Swift's `NemotronHModel` nests under `backbone.` and owns embedding/head
   slots. `VoiceChatSTTModel.sanitized` remaps (and the key-parity test
   verifies BOTH directions with `verify: [.all]` against the real bundle).
2. **The RNN-T LSTM ships raw PyTorch keys** (`weight_ih_l0`…): ih→Wx, hh→Wh,
   and the TWO bias vectors SUM into one. Dropping one silently halves the
   recurrent bias — pinned by test with exact values.
3. **`chunked_limited` is CHUNK-based** (NeMo groups frames into chunks of
   `right+1`); it equals the per-frame [70,0] sliding window ONLY because
   right context is 0 on this bundle. The general form is implemented.
4. **Prior `StreamingConformer.swift` conv-module keys were prototype-only**
   (`depthwise`, `norm`); the real bundle uses `depthwise_conv` and
   `batch_norm` (a LayerNorm under NeMo's historical name). The real-key
   implementation lives in `VoiceChatConformer.swift`; the prototype file
   stays for its property tests of the causal/carry techniques.
5. RNN-T decode state must carry ACROSS chunks (LSTM h/c + last token +
   cached prediction output) — chunked == whole is pinned with the real
   transducer weights.
