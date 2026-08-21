# VoiceChat 11B — brief for the vmlx-swift / Osaurus integration agent

**INTERNAL.** 2026-08-20. Hand this to the agent doing the Swift work.
Copy-pasteable prompts are in §7.

---

## 1. What you are integrating, in one paragraph

NemotronLabs VoiceChat 11B is a **full-duplex speech-to-speech model**: it
listens on a continuous 16 kHz timeline, transcribes with an RNN-T, answers in
text, emits tool calls on a *separate head*, and synthesizes 22.05 kHz speech —
handling barge-in and turn-taking. It is **not an LLM with a TTS bolted on**;
text and speech are two output paths over one `nemotron_h` backbone, running on
a shared frame clock (0.08 s = 12.5 fps).

Load target is the MLX-layout bundle. The NVIDIA original ships a **NeMo
training config** (no `model_type`, no `architectures`, one 44 GB fp32 blob) and
cannot be dispatched by any HF-config-driven loader.

## 2. Ground state — what already exists (do NOT rebuild)

### In vmlx-swift already
| Need | Where | Note |
|---|---|---|
| `nemotron_h` 56-layer hybrid | `Libraries/MLXLLM/Models/NemotronH.swift` | **7.714 B = 69 % of the model**, parses `hybridOverridePattern` |
| JANG quant on that backbone | `NemotronHJANGTQ.swift` | |
| Conformer encoder (offline) | `MLXVLM/Models/NemotronHOmni/Parakeet.swift` | reusable structure |
| Mel preprocessor | `NemotronHOmni/Preprocessors.swift` | 956 ln |
| Audio I/O | `NemotronHOmni/AudioIO.swift` | |
| Composition reference | `NemotronHOmni/NemotronHOmni.swift` | 1161 ln |
| Live-audio benches | `OmniAudioLatencyBench`, `OmniAudioChunkStabilityBench` | 157–219 ms first delta measured |

### Landed this session (green)
    Libraries/MLXVLM/Models/NemotronVoiceChat/NemotronVoiceChatConfiguration.swift
    Libraries/MLXVLM/Models/NemotronVoiceChat/StreamingConformer.swift
    Tests/MLXLMTests/NemotronVoiceChatConfigTests.swift          3/3
    Tests/MLXLMTests/VoiceChatStreamingConformerTests.swift      5/5

### Published quants to test against
    OsaurusAI/NemotronLabs-VoiceChat-11B-MXFP8    11.03 GiB
    OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4    7.92 GiB   <- default
    OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_2    5.71 GiB

### Python reference to mirror (build-time only, NOT our deliverable)
`mlx_vlm/models/nemotron_voicechat/` (upstream >= 0.6.15):
`config.py` 184 · `model.py` 162 · `session.py` 319 · `streaming.py` 575 ·
`tts.py` 619. **`streaming.py` is the duplex/barge-in reference.**

## 3. Remaining work, in dependency order

3. **`function_head`** — one vocab-sized Linear (0.587 B). Mechanically trivial.
4. **RNN-T decoder + joint** — nested **inside `audio_config`**; blank id 1024,
   `max_symbols 10`. A different decode algorithm from AR LM decode.
5. **TTS stack** — 28-layer/1152 transformer (`sliding_window 7500`) + MoG head
   + RVQ codec decoder (31 x 1024, `n_fft 16`, `hop 4`, 22.05 kHz).
6. **Duplex session** — concurrent text / user / ASR / function channels,
   `delay_source_text_by 15`, `delay_text_channel_by 2`, barge-in, turn-taking.
   No precedent in our codebase; mirror `streaming.py`.

## 4. 🚨 Traps already paid for — do not rediscover these

1. **Two RMSNorm conventions in ONE model.** Backbone uses plain `nn.RMSNorm`
   (applies `weight`); TTS uses `OffsetRMSNorm` (Gemma-style, `1.0 + weight`).
   Getting this wrong emptied text output while imatrix rel-err (identical
   0.0488), a clean NaN scan, and correct ASR all read healthy.
2. **`NemotronHConfiguration` could not decode a DENSE nemotron_h** — four MoE
   fields were required. Fixed to `decodeIfPresent ?? 0` (strictly widening;
   37/37 NemotronHTests still pass). VoiceChat's backbone is dense.
3. **`att_context_size` is `[[Int]]`, not `[Int]`** — ships `[[70, 0]]` =
   `[left, right]`. **Right context 0 = NO lookahead**, which is what allows
   sub-200 ms duplex. A non-zero value silently adds latency to every response.
4. **RNN-T decoder/joint live inside `audio_config`**, not top level.
5. **`Parakeet.swift`'s depthwise is commented "causal" but pads `(K-1)/2` on
   BOTH sides** — it reads 4 frames of future. Fine offline, wrong on a live
   mic. `StreamingConformer.swift` has the causal version; leave Parakeet alone.
6. **`function_head` is FULL VOCAB SIZE** (0.587 B, same as `lm_head`, and
   measured `tr(H)` **identical**). Tool calls are a weighted parallel channel
   (`function_channel_weight: 2.0`), not parsed text. A runtime reading only
   `lm_head` loses tool calling **silently**.
7. **A quantized bundle must be `nn.quantize`d BEFORE loading weights**, using
   the per-module map in `config.json["quantization"]`, returning "don't
   quantize" for anything absent (those were fp passthroughs).
8. **These tensors are fp and must stay fp**: `rvq_embs` (RVQ codebook),
   `audio_prompt_latents.*` (speaker identity), `mog_head.proj_mus` (read raw),
   `embed_subword.embed_tokens` (its dtype is read).

## 5. Test discipline that has repeatedly caught real bugs

* **Point tests at the real shipped artifact, not fixtures.** The config test
  found three incompatibilities in its first three runs.
* **Assert properties, not outputs.** Causality: perturb a late input frame,
  assert earlier outputs are unchanged — *plus a vacuity guard* that the
  perturbation reaches somewhere. Chunk-invariance: uneven chunks with carried
  state must equal whole-sequence — *plus a negative control* proving stateless
  chunking differs.
* **For audio, "a wav was produced" is not evidence.** Silence and noise both
  produce wavs. Check RMS, zero-crossing rate, spectral centroid.
* **Cross the axes.** On Ornith, every linear row passed on all 8 quants while
  the variating pattern found one quant silently losing tool calls when an
  image shared the turn. The voice analogue is **barge-in during a tool call**.

## 6. Objective duplex test fixtures (already characterised)

NVIDIA ships three **stereo 24 kHz** wavs where the channels are separate
speakers — left = user, right = agent. Measured (50 ms frames, RMS > 0.01):

    interruptions.wav  30.0s   user 19.8%  agent 29.2%  BOTH 4.2%   <- barge-in
    turn_taking.wav    41.1s   user 11.1%  agent 29.1%  BOTH 0.0%   <- never overlaps
    tool_call.wav      84.5s   user 18.5%  agent 20.4%  BOTH 0.4%

This makes duplex correctness **measurable**, not a listening opinion:
* turn-taking PASS = our agent/user overlap ≈ 0 % (reference is exactly 0.0 %)
* barge-in PASS = agent energy falls within ~300 ms of user energy resuming
* tool-call PASS = assert `toolCalls == 1` **on the function channel**, never
  the transcript

⚠️ Model I/O is 16 kHz in / 22.05 kHz out; the fixtures are 24 kHz. Resample.
`jang/examples/voicechat/prepare_fixture.py` does the extraction.

Full ladder L0–L9 in `03-LIVE-TEST-PROTOCOL.md`. **L9 is the Osaurus GUI
harness with a real mic** — per house rule, proof goes through the running dev
app's GUI, never curl/HTTP/CLI.

## 7. Copy-pasteable prompts

### 7.1 Kick-off
> You are integrating NemotronLabs VoiceChat 11B into vmlx-swift. Read
> `jang/docs/runtime/nemotron-voicechat-11b/` 00 through 05 first — especially
> 05 §4 (traps) — then confirm the current state by running:
>
>     DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
>       --filter "NemotronVoiceChatConfigTests|VoiceChatStreamingConformerTests" \
>       --jobs 2 -Xswiftc -F -Xswiftc \
>       /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks
>
> Expect 8/8. Do not start writing until that passes — it is the baseline.

### 7.2 Next unit of work (step 3+4)
> Implement `function_head` and the RNN-T decoder/joint for VoiceChat in
> `Libraries/MLXVLM/Models/NemotronVoiceChat/`.
>
> `function_head` is a full vocab-sized Linear over the same hidden state as
> `lm_head` — its measured Hessian trace is IDENTICAL to `lm_head`'s. Expose the
> function channel as a first-class output, NOT as text to be parsed; the
> reference returns it as `VoiceChatResult.function_tokens`.
>
> RNN-T config is nested inside `audio_config` (`decoder`, `joint`), blank id
> 1024, `max_symbols 10`. Mirror
> `mlx_vlm/models/nemotron_voicechat/model.py` + `mlx_audio .../rnnt.py`.
>
> Add tests that assert the tool call itself, never the transcript — a model
> can say "I checked the weather" while emitting zero tool calls, and we have
> shipped exactly that bug before.

### 7.3 TTS (step 5)
> Implement the VoiceChat TTS: 28-layer/1152 transformer with
> `sliding_window 7500`, the MoG head, and the RVQ codec decoder
> (31 quantizers x 1024 codebook, `n_fft 16`, `hop_length 4`, out 22.05 kHz).
> Mirror `mlx_vlm/models/nemotron_voicechat/tts.py`.
>
> 🚨 `mog_head.proj_mus` is read RAW and reshaped (`tts.py:136`), and
> `embed_subword.embed_tokens.weight.dtype` is read to allocate a buffer
> (`tts.py:342`). Both must remain unquantized — the published bundles already
> keep them fp; your loader must not assume every `.weight` is quantized.
>
> Verify with `jang/examples/voicechat/offline_turn.py` output as the reference,
> and check the audio is speech-shaped (RMS / ZCR / spectral centroid), not
> merely non-empty.

### 7.4 Duplex session (step 6)
> Implement the duplex session layer, mirroring
> `mlx_vlm/models/nemotron_voicechat/streaming.py`. Concurrent text / user /
> ASR / function channels on a 12.5 fps shared clock, with
> `delay_source_text_by 15` and `delay_text_channel_by 2`.
>
> Validate against the NVIDIA fixtures using the OBJECTIVE criteria in
> `03-LIVE-TEST-PROTOCOL.md` §2: turn-taking overlap must approach 0 %
> (reference is exactly 0.0 %), and on `interruptions.wav` the agent's energy
> must fall within ~300 ms of the user resuming. Then run the Osaurus GUI
> harness with a real mic — interrupt it mid-sentence and confirm it yields and
> resumes coherently.

### 7.5 Standing instruction for the agent
> Never report a step complete on a structural signal alone. In this codebase
> six separate defects this week reported success while broken: AWQ silently
> reverted through three clean rebuilds; four MoE tools reported success while
> covering 8 % of a model; a published bundle dequantized every Linear correctly
> and emitted garbage; GPTQ improved every in-sample metric and got worse
> out-of-sample; one quant passed every single-shape probe and lost tool calls
> only where two axes crossed; and a wrong norm-fold emptied text output while
> rel-err, NaN scan and ASR all read healthy. Generate, listen, and cross the
> axes.

## 8. Key config values (from the shipped MLX bundle)

    model_type            nemotron_voicechat
    text_config           nemotron_h, hidden 4480, 56 layers, vocab 131072
                          pattern M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-
                          (M=Mamba2 x27, -=MLP x25, *=attention x4)
    audio_config.encoder  Conformer 24L d=1024 h=8, rel_pos, dw_striding x8,
                          conv_norm_type layer_norm, att_context_style chunked_limited,
                          att_context_size [[70, 0]], causal_downsampling true
    audio_config          decoder/joint nested here; output_dim 4480; max_symbols 10
    tts_config            28L, hidden 1152, heads 16, head_dim 72,
                          sliding_window 7500, latent 512,
                          num_quantizers 31, codebook 1024, num_delay_speech_tokens 2
    codec_config          22050 Hz, base_channels 384, n_fft 16, hop 4, latent 512
    tokens                bos 1, eos 2, pad 12, silence 11, rnnt_blank 1024
    rates                 in 16000 Hz, out 22050 Hz, frame 0.08s (12.5 fps)
    function_channel_weight 2.0        speaker "Aria"

## 9. Custom voices (Osaurus dino) — later, but design for it now

`audio_prompt_latents.Aria` is `[1, 37, 1152]` = **83 KiB**, through a frozen
`audio_prompt_projection_W [1152,1152]`, `audio_prompt_duration: 3.0`. The
speaker is a **learned latent, not baked into the weights** — so voices are
likely clonable from ~3 s of reference audio rather than trained.

Design implications for the runtime:
* support **multiple named speaker latents in one bundle** (~83 KiB each),
  selectable at runtime — `config.speaker` is a lookup key;
* the codec **ENCODER** is needed, not just the decoder (both ship, 76 tensors
  each), because making a voice means audio -> latent.

⚠️ UNVERIFIED: whether the 37 frames are fixed or vary with prompt length, and
whether the projection consumes codec latents or raw mel. Confirm before
building on it.

---

## 10. 🚨 BUILD: plain `swift build` FAILS — use the Xcode toolchain

    Libraries/MLXVLM/Models/NemotronVoiceChat/VoiceChatCodec.swift:17:8:
    error: cannot load module 'MLXFFT' built with SDK 'macosx26.5'
           when using SDK 'macosx26.4'

`VoiceChatCodec.swift` imports **MLXFFT** (the codec needs an inverse STFT).
Prebuilt module artifacts in `.build/` were compiled against a newer SDK than
plain `swift build` selects, so it fails on a *toolchain* mismatch, not a code
error. Same root cause as `swift test` being unable to find the `Testing`
module on this machine.

**Always build and test through the Xcode toolchain:**

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
      xcrun swift build -c debug --target MLXVLM

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
      xcrun swift test --filter "NemotronVoiceChatConfigTests|VoiceChatStreamingConformerTests" \
      --jobs 2 -Xswiftc -F -Xswiftc \
      /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks

With that form the target builds clean. If you see an SDK-mismatch error,
you are on the wrong toolchain — do not "fix" it by editing imports.

## 11. Implementation status (updated 2026-08-20, later)

`Libraries/MLXVLM/Models/NemotronVoiceChat/` — 2560 lines:

    NemotronVoiceChatConfiguration.swift   364   config (+ MoG, char-encoder)
    StreamingConformer.swift               152   causal conv + chunked mask
    VoiceChatConformer.swift               430   perception encoder
    VoiceChatRNNT.swift                    248   transducer decode
    VoiceChatSTTModel.swift                160   stt composition
    VoiceChatTTS.swift                     448   TTS transformer + MoG
    VoiceChatTTSModel.swift                343   tts composition
    VoiceChatCodec.swift                   415   RVQ codec (imports MLXFFT)

Steps 3–5 of §3 are substantially implemented. **Step 6 (duplex session) and
the live ladder L3–L9 remain**, and nothing here has been validated against the
objective fixture criteria in §6 yet — building and passing config/unit tests is
not the same as producing correct audio on a live timeline.
