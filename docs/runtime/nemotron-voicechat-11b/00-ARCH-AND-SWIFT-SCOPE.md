# NVIDIA NemotronLabs VoiceChat 11B — architecture + Swift runtime scope

**INTERNAL.** Created 2026-08-20. Source: `nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`
(41.3 GiB, ungated). Downloading to `~/models/nemotron-voicechat-src/`.

---

## 1. This is NOT a single LLM — it is a four-part duplex speech system

🚨 `config.json` is a **NeMo training config** (`data` / `trainer` /
`exp_manager` / `model`), not an HF transformers config. There is no
`architectures` key, no `model_type`. Nothing in our HF-config-driven tooling
will recognize it.

| Component | What | Source |
|---|---|---|
| **LLM backbone** | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` — nemotron_h hybrid (Mamba2 + attention) | `model.stt.model.pretrained_llm` |
| **Perception** | `AudioPerception` = mel preprocessor → ConformerEncoder → IdentityConnector, `output_dim 4480` | `model.stt.model.perception` |
| **Speech generation** | codec-based TTS, same 9B LM as its `pretrained_lm_name` | `model.speech_generation.model` |
| **RNN-T** | `RNNTDecoder` + `RNNTJoint`, vocab **1024** | `_rnnt_merge_info` |

Speaker: `inference_speaker_name: Aria`. Audio in **16 kHz**, out **22.05 kHz**,
frame length 0.08 s.

## 2. Perception encoder — mostly ALREADY IMPLEMENTED in MLX

`jang_tools/nemotron_omni/parakeet.py` is a full MLX Conformer: subsampling,
relative-position multi-head attention, conv module, blocks, and a weight
mapper. Parakeet *is* NVIDIA FastConformer, and this encoder is the same family.

| VoiceChat needs | parakeet.py has | Delta |
|---|---|---|
| `ConformerEncoder` d_model 1024, n_heads 8, **n_layers 24** | ✅ `ParakeetEncoder` | layer count is config |
| `self_attention_model: rel_pos` | ✅ `RelativeMultiHeadAttention` + `_rel_shift` | — |
| `subsampling: dw_striding`, factor **8** | ✅ `ParakeetSubsampling` | check factor 8 vs Parakeet's |
| `conv_kernel_size: 9` | ✅ kernel 9, depthwise causal | — |
| mel: 128 feат, n_fft 512, 16 kHz, win .025 / hop .01 | ✅ `audio_features.py` (n_fft 512, win 400, hop 160, n_mels 128) | **exact match** |
| **`conv_norm_type: layer_norm`** | ❌ uses `BatchNorm1d` | **must add LayerNorm variant** |
| **`att_context_style: chunked_limited`** + `att_context_size` | ❌ full attention only | **must add chunked/limited context** |
| **`causal_downsampling: True`**, `conv_context_size: causal` | partial (depthwise conv is causal) | **must make subsampling causal** |
| `use_bias: False`, `untie_biases: True`, `xscaling: False` | verify | — |

**The three deltas are all streaming concerns.** Offline Conformer is done;
what is missing is the machinery that lets it run causally on a live mic.

## 3. What makes this hard — full duplex, not turn-based

This is the real work, and it is orchestration rather than kernels. The model
runs **multiple simultaneous channels** with deliberate time offsets:

    duplex_text_channel_weight     1.0
    duplex_user_channel_weight     1.0
    duplex_asr_text_weight         1.0
    duplex_function_channel_weight 2.0     <- tool calls are their own channel
    delay_source_text_by           15
    delay_text_channel_by          2

`use_function_head: True` — tool calling is a dedicated head, not parsed out of
text. The repo ships `interruptions.wav`, `turn_taking.wav`, `tool_call.wav`,
which is a fair statement of what the runtime must handle: the user talking
over the agent, deciding when to take/yield the turn, and emitting a tool call
mid-stream.

None of our existing runtimes do any of this. vMLX/Osaurus are request-response
text engines; there is no audio input path, no streaming encoder state, no
barge-in, and no concurrent output channel.

## 4. Swift runtime scope — honest estimate

Ordered by dependency, not by size:

1. **Backbone.** Nemotron-Nano-9B-v2 is nemotron_h (Mamba2 + attention hybrid).
   We have prior nemotron_h work (`convert_nemotron_*`, Nemotron 3.5 Lightning)
   — check whether the Swift side already has the Mamba2 block, since that is
   the single biggest reuse question.
2. **Mel front-end in Swift.** `audio_features.py` is numpy; needs a Swift/Metal
   or Accelerate STFT + Slaney mel filterbank. Deterministic and testable
   against the Python one — do that first, it is the cheapest confidence win.
3. **Streaming Conformer.** Port `parakeet.py` to Swift *and* add the three
   deltas (layer_norm conv, chunked_limited attention, causal downsampling).
   Chunked attention requires carrying encoder KV/context state across chunks.
4. **RNN-T decoder + joint.** Small (vocab 1024) but a new decode loop —
   greedy/beam RNN-T is not the same as autoregressive LM decode.
5. **Speech generation / codec TTS.** Least explored; `codec_config` +
   `tts_config` need mapping before this can be scoped honestly.
6. **Duplex session layer.** Channel scheduling, the 15/2-frame delays,
   barge-in, turn-taking, function-channel dispatch. This is the part with no
   precedent in our codebase.

## 5. Open questions before committing to an implementation

- Do the shipped safetensors actually contain all four components, or does it
  expect to pull `Nemotron-Nano-9B-v2` separately? (41.3 GiB for "11B" suggests
  fp32 or bundled-everything — check once the download lands.)
- Is there an HF-transformers path at all, or is NeMo the only reference
  implementation? That decides whether we can diff against a known-good runtime.
- `pipecat-ai/...-Spark` and `mlx-community/...` variants exist — worth reading
  the mlx-community conversion before writing our own, they may have solved the
  NeMo→MLX weight mapping already.

## 6. Status

- [x] Architecture mapped from config
- [x] Confirmed large reuse from `nemotron_omni/parakeet.py` + `audio_features.py`
- [ ] Source download (4.9 / 41.3 GiB)
- [ ] Tensor inventory — the decisive input for weight mapping
- [ ] Swift-side audit: does vmlx-swift have a Mamba2 block yet?

---

# ADDENDUM — real tensor inventory + corrected Swift scope (2026-08-20)

Download complete: **41 GiB, a single `model.safetensors`, no index.**

## 7. Inventory — everything is bundled

    1632 tensors · 11.095 B params · ALL fp32 (1626 F32 + 6 I64)

    stt_model   997 tensors   10.098 B
    tts_model   635 tensors    0.997 B

✅ Answers open question #1: the 9B backbone is **inside this file**. Nothing
needs to be pulled from `Nemotron-Nano-9B-v2` separately. fp32 is why "11B" is
44 GB — a bf16 conversion alone halves it to ~22 GB before any quantization.

### stt_model (10.098 B)
| Component | Tensors | Params |
|---|---|---|
| `llm.layers` | 338 | 7.714 B |
| `perception.encoder` | 636 | 0.609 B |
| `embed_tokens.weight` | 1 | 0.587 B |
| **`function_head.weight`** | 1 | **0.587 B** |
| `lm_head.weight` | 1 | 0.587 B |
| `rnnt_decoder.prediction` | 9 | 0.007 B |
| `perception.proj` | 2 | 0.005 B |
| `rnnt_joint.{joint_net,enc,pred}` | 6 | 0.002 B |

🚨 **`function_head` is FULL VOCAB SIZE** — same 0.587 B as `lm_head`. Tool
calling is a genuine parallel output channel with its own vocabulary head, not
text parsed after the fact. Three vocab-sized matrices (embed + lm_head +
function_head) are **1.76 B params = 16 % of the model**, so how they are
quantized dominates the size/quality tradeoff.

### tts_model (0.997 B)
| Component | Tensors | Params |
|---|---|---|
| `tts_model.backbone` | 365 | 0.595 B |
| `tts_model.mog_head` | 21 | 0.159 B |
| `audio_codec.decoder` | 76 | 0.092 B |
| `audio_codec.encoder` | 76 | 0.092 B |
| `tts_model.embed_subword` | 20 | 0.023 B |
| `audio_codec.prvq` | 62 | 0.016 B |
| `tts_model.rvq_embs` | 1 | 0.016 B |
| `tts_model.gated_fusion_audio_text` | 7 | 0.003 B |
| **`audio_prompt_latents.Aria`** | 1 | — |

The TTS side is its own mini-LM (0.595 B) + **Mixture-of-Gaussians head** +
neural audio codec with residual VQ. `audio_prompt_latents.Aria` means the
speaker is a **learned latent** — voice is swappable by substituting latents,
not by retraining.

## 8. Backbone confirmed: Nemotron-H hybrid, 56 layers

All blocks use `mixer.*` naming (Nemotron-H convention):

    27 x Mamba2 SSM   A_log, D, conv1d, dt_bias, in_proj, out_proj, norm
    25 x MLP          down_proj, up_proj
     4 x attention    q/k/v/o_proj
    -----
    56 layers

## 9. 🎉 Swift scope is MUCH smaller than section 4 assumed

`vmlx-swift` already ships the hard parts, built and benchmarked:

| Need | Already in vmlx-swift |
|---|---|
| Nemotron-H 56-layer hybrid | ✅ `Libraries/MLXLLM/Models/NemotronH.swift` (+ `NemotronHJANGTQ.swift`) |
| Mamba2 / SSM cache | ✅ `MambaCacheCompileProbeTests`, `SSMReDeriveParityTests`, hybrid SSM warm-pass |
| Conformer audio encode (Parakeet) | ✅ live-voice path, pre-encode **43.9–50.1 ms** |
| Streaming audio chunks | ✅ `OmniAudioChunkStabilityBench`, `OmniAudioLatencyBench` |
| Raw PCM + pre-encoded through BatchEngine/TokenIterator | ✅ measured, first delta **157–219 ms** |

Evidence: `docs/NEMOTRON-OMNI-LIVE-VOICE-2026-05-12.md` — 18/18 RunBench rows
incl. audio, media-salt isolation and hybrid SSM warm-pass.

### Revised remaining work
1. **NeMo → MLX weight mapping** (`stt_model.*` / `tts_model.*` → our layout).
   No HF config, so this is hand-written. Biggest single task.
2. **Conformer streaming deltas**: `conv_norm_type: layer_norm` (ours is
   BatchNorm), `att_context_style: chunked_limited`, `causal_downsampling`.
3. **RNN-T decode loop** — small weights, but a different decode algorithm.
4. **TTS: MoG head + RVQ codec decoder** — genuinely new, ~1 B params.
5. **Duplex session layer** — channel scheduling, `delay_source_text_by: 15` /
   `delay_text_channel_by: 2`, barge-in, turn-taking, function-channel dispatch.
   Still the part with no precedent.
6. **fp32 → bf16/quant conversion** — 44 GB is unshippable; bf16 alone gives
   ~22 GB.

Sections 4.1–4.3 are effectively **done**; the real work is 1, 4 and 5.
