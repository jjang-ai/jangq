# VoiceChat 11B — quant plan + Swift integration handoff

**INTERNAL.** 2026-08-20. Read `00-ARCH-AND-SWIFT-SCOPE.md` and
`01-SWIFT-IMPLEMENTATION-PLAN.md` first — this assumes both.

Audience: the Swift agent picking up integration later, plus whoever builds the
calibrated quants once a runtime exists.

---

## 1. What exists RIGHT NOW

| Artifact | Path | State |
|---|---|---|
| NVIDIA original (fp32) | `~/models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B` | 41 GiB. **NeMo config — unloadable by any HF-driven path.** Reference only. |
| MLX bf16 | `~/models/nemotron-voicechat-src/VoiceChat-11B-mlx-bf16` | 21 GiB, 5 shards. **This is the load target.** |
| **MXFP8** | `~/models/JANGQ-AI/NemotronLabs-VoiceChat-11B-MXFP8` | **10.91 GiB, built + verified** |
| Swift config | `Libraries/MLXVLM/Models/NemotronVoiceChat/NemotronVoiceChatConfiguration.swift` | ✅ 3/3 tests |
| Swift streaming conformer | `.../NemotronVoiceChat/StreamingConformer.swift` | ✅ 5/5 property tests |

## 2. MXFP8 — done, and why it could be done first

    python -m jang_tools.convert_voicechat_mxfp8 <bf16_src> <out> --group-size 32

    quantized 653 | fp passthrough: 808 1-D, 4 PROTECTED, 167 not group-divisible
    41 GiB fp32 -> 21 GiB bf16 -> 10.91 GiB MXFP8

MXFP8 is **uniform 8-bit and needs no calibration**, so it is pure tensor ops —
no model class, no forward pass, no runtime. That is the only reason it could
land before the Swift runtime. It also gives the Swift work a real quantized
artifact to load instead of waiting.

🚨 `mx.quantize(..., mode="mxfp8")` returns **TWO** values `(codes, scales)`,
not three — e8m0 group scales need no bias. Affine returns three. Unpacking
three raises loudly; the dangerous variant of this mistake is writing an empty
`.biases` that a loader silently reads as zeros.

## 3. 🚨 TENSORS THAT MUST NEVER BE QUANTIZED

Verified byte-identical in the shipped MXFP8 bundle:

| Tensor | Shape | Why |
|---|---|---|
| `tts_model.tts_model.rvq_embs` | [31, 1024, 512] | **RVQ codebook.** Lookup entries, not a projection — quantizing moves every centroid and corrupts decoded audio globally. Only 0.016 B; free to protect. |
| `tts_model.audio_prompt_latents.<name>` | [1, 37, 1152] | **Speaker identity**, 83 KiB. The entire voice. Also what custom voices ARE. |
| `tts_model._control_codes` | [3] int64 | control tokens |
| `tts_model.codec_silence_tokens` | [31] int64 | per-quantizer silence codes |

Same class as Ornith's vision `linear_fc2` at in=4304: tiny tensors whose
corruption is invisible to a structural check and fatal to output. The
converter enforces this via a `PROTECTED` substring list recorded into
`config.jang_protected_fp_tensors`; **carry it into every future quant.**

## 4. Parameter budget — what quantization choices actually cost

    linear (quantizable)   8.856 B  79.8 %
    vocab heads            1.174 B  10.6 %   lm_head + function_head
    embedding tables       0.611 B   5.5 %
    codec conv             0.200 B   1.8 %
    MoG head               0.159 B   1.4 %
    conv                   0.077 B   0.7 %
    RVQ codebook           0.016 B   0.1 %   PROTECTED
    1-D norms/biases       0.002 B   0.0 %
                          -------
                          11.095 B

🚨 **`function_head` is FULL VOCAB SIZE** (0.587 B, identical to `lm_head`).
Tool calls come out of their own head on a weighted channel
(`function_channel_weight: 2.0`), NOT parsed from text. Two consequences:
  * a runtime reading only `lm_head` loses tool calling **silently**;
  * it must not be treated as a low-priority tensor during bit allocation just
    because it is not `lm_head`.

Together the three vocab-sized matrices are 1.76 B = **16 %** of the model, so
how they are quantized dominates the size/quality tradeoff.

## 5. THE CALIBRATED QUANTS — ~4-bit and dynamic ~2-bit

Requested: a ~4-bit, and **the smallest possible dynamic ~2-bit**, both with
AWQ + Hessian + imatrix. Eric: *"crucial for best voices."*

### Why they are blocked (not deprioritised)
All three methods derive from the per-input-channel second moment `E[x_c²]`,
which is captured by **running forward passes**. No runtime -> no capture ->
no Hessian, no AWQ scales, no imatrix. MXFP8 is the only calibration-free lane.

### Rough size targets (linear+heads+embeddings quantized, rest fp)
    MXFP8   ~10.9 GiB   built
    ~6-bit   ~8.5 GiB
    ~4-bit   ~6.0 GiB   <- the practical default
    ~2-bit   ~3.7 GiB   <- dynamic/Hessian-allocated, NOT uniform 2-bit

### 🚨 Voice quality is NOT the same axis as text quality
This is the part that matters for the dino voices and is easy to get wrong.

The model has **two output paths sharing one backbone**: text (`lm_head`) and
speech (`tts_model` -> MoG head -> RVQ codec -> waveform). Standard KL/top-1
evaluation measures ONLY the text path. A quant can hold text perplexity and
still produce audibly degraded speech, and nothing in our usual eval would say
so.

Concrete asymmetries to respect when allocating bits:
  * **MoG head (0.159 B)** predicts continuous mixture parameters. Error there
    is not softmax-normalised away like a token distribution — it moves the
    sampled waveform directly. Treat as high-sensitivity.
  * **Codec decoder convs (0.200 B)** run at 22.05 kHz output rate; artifacts
    are periodic and audible.
  * **RVQ codebook** — protected, see §3.
  * The 31 quantizers are **residual**: an error in quantizer *k* is inherited
    by every stage after it. Early quantizers matter more than late ones.

**Recommendation:** allocate the TTS side conservatively even in the ~2-bit
build, and spend the aggressive bits on the 8.856 B of backbone linears. A
"2-bit" VoiceChat should mean a 2-bit *backbone*, not a 2-bit codec.

### Calibration corpus — already exists, use it
`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` ships
`calibration/fixtures/*.json` + `calibration/replay.tar` — real conversational
replay. Far better matched than our LLM text-prompt corpus, and the corpus
should exercise **both** paths (speech in AND speech out), or the TTS side gets
calibrated on nothing.

### Do NOT repeat these (measured this week, see memories)
  * `qwen36_imatrix_refit` reloads W from SOURCE and therefore **silently
    reverts AWQ** unless `--awq-scales` is passed. Tell-tale: rel-err identical
    with and without AWQ. (`feedback_calibrated_trio_mandatory`)
  * **GPTQ made a 35B MoE WORSE out-of-sample** despite −33.9 % in-sample
    reconstruction. Always A/B against a pre-GPTQ snapshot on held-out prompts.
    (`feedback_gptq_overfits_moe`)
  * `gptq_mlx._pack_uint32` produces non-MLX layouts at **3/5/6-bit** — only
    2/4/8 are correct. A dynamic 2-bit map WILL contain 3/5-bit modules, so any
    GPTQ pass over it must self-test the pack and keep RTN on mismatch.

## 6. Custom voices (the Osaurus dino) — likely NO training required

`audio_prompt_latents.Aria` is `[1, 37, 1152]` — **83 KiB**, fed through a
frozen `audio_prompt_projection_W` `[1152, 1152]`, with
`audio_prompt_duration: 3.0`.

The speaker is **not baked into the weights**. It is a small conditioning
latent derived from ~3 s of reference audio. So the expected path is *cloning*,
not fine-tuning:

    1. record ~3 s of the target voice
    2. encode -> audio_prompt_projection_W -> [1, 37, 1152] latent
    3. store beside Aria, e.g. audio_prompt_latents.Osaurus
    4. select by name (config `speaker` is a lookup key)

Implications:
  * multiple voices ship in ONE bundle, ~83 KiB each, switchable at runtime;
  * **the codec ENCODER is needed, not just the decoder** — earlier scoping
    treated TTS as decode-only. Both are in the bundle (`audio_codec.encoder`
    and `.decoder`, 76 tensors each);
  * speaker latents must stay fp in every quant (§3).

⚠️ UNVERIFIED, confirm before building on it: whether the 37 frames are fixed
or vary with prompt length, and whether the projection consumes codec latents
or raw mel. Check against the reference implementation.

## 7. Swift integration handoff

### Done and green
    NemotronVoiceChatConfiguration.swift   3/3
    StreamingConformer.swift               5/5

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
      --filter "NemotronVoiceChatConfigTests|VoiceChatStreamingConformerTests" \
      --jobs 2 -Xswiftc -F -Xswiftc \
      /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks

(plain `swift test` cannot find the `Testing` module on this machine)

### Already in vmlx-swift — do not rewrite
| Need | Where | Note |
|---|---|---|
| Nemotron-H 56-layer hybrid | `MLXLLM/Models/NemotronH.swift` | 7.714 B = 69 % of the model |
| JANG quant on backbone | `NemotronHJANGTQ.swift` | |
| Conformer (offline) | `MLXVLM/Models/NemotronHOmni/Parakeet.swift` | structure reusable |
| Mel preprocessor | `NemotronHOmni/Preprocessors.swift` | |
| Audio I/O | `NemotronHOmni/AudioIO.swift` | |
| Composition reference | `NemotronHOmni/NemotronHOmni.swift` | 1161 ln |
| Streaming benches | `OmniAudioLatencyBench`, `OmniAudioChunkStabilityBench` | 157–219 ms first delta measured |

### Remaining, in order
3. `function_head` — one vocab-sized Linear. Trivial mechanically; see §4.
4. RNN-T decoder + joint — nested in `audio_config`; blank id 1024,
   `max_symbols 10`. Different decode algorithm from AR LM decode.
5. TTS: 28-layer/1152 transformer (`sliding_window 7500`) + MoG head + RVQ
   codec decoder (31 × 1024, `n_fft 16`, `hop_length 4`, 22.05 kHz).
6. Duplex session: concurrent text/user/ASR/function channels,
   `delay_source_text_by 15`, `delay_text_channel_by 2`, barge-in, turn-taking.
   No precedent in our codebase.

### Traps already paid for
1. `NemotronHConfiguration` could not decode a **dense** nemotron_h — four MoE
   fields were required. Fixed to `decodeIfPresent ?? 0` (strictly widening;
   37/37 NemotronHTests still pass). VoiceChat's backbone is dense.
2. `att_context_size` is `[[Int]]` **not** `[Int]` — ships `[[70, 0]]`,
   `[left, right]`. **Right context is 0: no lookahead**, which is what makes
   sub-200 ms duplex possible. A non-zero value would add latency to every
   response with nothing failing.
3. RNN-T decoder/joint are **inside `audio_config`**, not top level.
4. `Parakeet.swift`'s depthwise is commented "causal" but pads `(K-1)/2` on
   BOTH sides — it reads 4 frames of future. Correct offline, wrong on a live
   mic. Left untouched; `StreamingConformer.swift` has the causal version.

### Test discipline that has repeatedly paid off
Point tests at the **real shipped artifact**, not fixtures — that caught three
incompatibilities in its first three runs. Assert **properties**, not outputs:
causality (perturb a late frame, assert earlier outputs unchanged, plus a
vacuity guard that the perturbation reaches somewhere) and chunk-invariance
(uneven chunks with carried state == whole sequence, plus a negative control
proving the carry is not a no-op). Both failures are invisible offline and
audible live.

## 8. Status
- [x] MXFP8 built + protected tensors verified byte-identical
- [x] Swift config + streaming conformer, 8/8 green
- [ ] ~4-bit and dynamic ~2-bit — BLOCKED on runtime (calibration)
- [ ] Swift steps 3–6
- [ ] Custom-voice (dino) pipeline — needs codec ENCODER path
