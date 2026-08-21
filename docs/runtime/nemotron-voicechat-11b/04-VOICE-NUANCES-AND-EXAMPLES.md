# VoiceChat 11B — voice nuances, working examples, quant traps

**INTERNAL.** 2026-08-20. Everything here was hit in practice, not anticipated.

Scripts live in `jang/examples/voicechat/`.

---

## 1. Working examples (all verified end-to-end)

| Script | Does |
|---|---|
| `prepare_fixture.py` | Extracts the USER channel from NVIDIA's stereo wavs as 16 kHz mono, and reports channel activity/overlap |
| `offline_turn.py` | Full turn: audio -> ASR transcript + reply text + 22.05 kHz speech + function tokens |
| `compare_speech.py` | Objective speech-path comparison (mel distance, envelope corr, per-stage RVQ agreement) |

### Verified output — bf16 reference

    user transcript : 'Hi, do you know any good cookie recipes? Maybe a peanut butter one'
    agent text      : "Oh my, yes! I love this topic-there's something magical about ..."
    audio out       : 16.00s @ 22050 Hz     audio codes (200, 31)
    function tokens : (200,)                rms=0.0112 peak=0.1990
    generate        : 55.8s (0.29x realtime on M5 Max, bf16)

### Verified output — MXFP8 (11.03 GiB)

    quantized structure: 563 modules (8-bit mxfp8)
    load: 0.5s   (bf16 was 1.7s)
    user transcript : 'Hi, do you know any good cookie recipes? Maybe a peanut butter one'   <- identical ASR
    agent text      : "Oh, I love cookies! I can give you a simple recipe if you'd like."
    generate        : 48.2s (0.33x realtime)     rms=0.0088 peak=0.1631

Every path alive: ASR (RNN-T), text (`lm_head`), speech (200 frames x 31 RVQ),
and the function channel as a separate 200-token stream.

### Objective "is it actually speech" check
Do not trust "it produced a wav". Noise and silence both produce wavs.

    speech-active frames   18.4%   (noise ~100%, silence 0%)
    spectral centroid    2322 Hz   (white noise at this sr would be ~5512)
    zero-crossing rate    0.0430   (speech 0.02-0.10; noise ~0.5)

## 2. 🚨 QUANT TRAPS — every one of these was hit for real

The pattern: VoiceChat reads several weights **raw**, bypassing the module
forward. Quantization packs those into uint32, so any raw read gets a tensor
that is 4x too small in the last dim (8-bit) with the wrong dtype.

### 2.1 `mx.quantize(mode="mxfp8")` returns TWO values, not three
affine returns `(codes, scales, biases)`; **mxfp8 returns `(codes, scales)`** —
the e8m0 group scale needs no bias. Unpacking three raises loudly. The
dangerous variant is writing an empty `.biases` a loader then reads as zeros.

### 2.2 `ndim >= 2` is NOT "is a Linear weight"
The Conformer carries bare 2-D parameters — `self_attn.pos_bias_u` /
`pos_bias_v` `(n_heads, head_dim)`. Quantizing them swaps an array for a
quantized-module dict:

    ValueError: Cannot perform addition on an mlx.core.array and dict

**Rule: require a `.weight` suffix AND `ndim == 2`.** That also excludes 3-D
conv kernels, which MLX quantizes differently.

### 2.3 `mog_head.proj_mus` is READ RAW — must stay fp
`tts.py:136`:

    mus = self.proj_mus.weight.reshape(num_predictions, low_rank, -1)[component]

It never calls `proj_mus(x)`, so a packed weight yields:

    ValueError: [matmul] (1,64,288) must match (1,1152,1)      # 288 = 1152/4

`[65536, 1152]` = **75 M params**, the largest thing deliberately left in fp.

### 2.4 `embed_subword.embed_tokens` — its DTYPE is read
`tts.py:342`:

    out = mx.zeros(..., dtype=self.embed_tokens.weight.dtype)

Quantized, that is `uint32`, so float values get written into an integer buffer
and **silently truncated**. No exception. Tiny table (257x1152) — protect it.

### 2.5 Tiny lookup tables
`subword_flag_emb.cont_emb` is `[2, 1152]`; `bos_eos_emb.special_emb` is
`[3, 1152]`. Group statistics over 2-3 rows are meaningless. Rule:
**do not quantize a 2-D weight with fewer than 64 rows.**

### 2.6 Never quantize these (verified byte-identical in the shipped MXFP8)
| Tensor | Shape | Why |
|---|---|---|
| `tts_model.rvq_embs` | [31,1024,512] | RVQ **codebook** — lookup entries, not a projection. Quantizing moves every centroid; corrupts all decoded audio. |
| `audio_prompt_latents.*` | [1,37,1152] | **Speaker identity**, 83 KiB. Also what custom voices ARE. |
| `mog_head.proj_mus` | [65536,1152] | read raw (§2.3) |
| `mog_head.low_mat` | [1024,512,64] | 3-D, indexed directly |
| `embed_subword.embed_tokens` | [257,1152] | dtype read (§2.4) |
| `_control_codes`, `codec_silence_tokens` | int64 | control tokens |

### 2.7 A quantized bundle must be `nn.quantize`d BEFORE `load_weights`
Otherwise packed uint32 codes load as ordinary weights. Here it fails loudly
(`addmm` shape mismatch) — but the same mistake on a shape that happens to
match would load silently and emit garbage. `offline_turn.py` builds the
`class_predicate` from the per-module map in `config.quantization`, and returns
`False` for anything absent so fp passthroughs stay fp.

### Final MXFP8 accounting

    quantized 563 | fp: 1060 non-Linear, 6 PROTECTED, 3 tiny tables
    41 GiB fp32 -> 21 GiB bf16 -> 11.03 GiB MXFP8

## 3. 🚨 THE BIG ONE — comparing speech between quants

**Our standard judge (KL / top-1 on `lm_head`) measures the TEXT path only.**
VoiceChat has two output paths on one backbone. A quant can hold text
perplexity and sound degraded, and nothing in the usual eval reports it.

But the naive fix is also wrong. First real comparison:

    mel log-spectral distance : 5.3439
    energy-envelope correlation: 0.2457

That looks catastrophic. It is **meaningless** — the two bundles produced
different *text*:

    bf16 : "Oh my, yes! I love this topic-there's something magical about ..."
    mxfp8: "Oh, I love cookies! I can give you a simple recipe if you'd like."

The TTS faithfully rendered whatever the LM sampled. Comparing the audio
compares **two different utterances**, i.e. sampling divergence, not
quantization damage.

🚨 **A speech comparison is valid only when both bundles emit the SAME TEXT.**
Either decode greedily so tokens match, or drive the TTS directly with a fixed
token sequence. `compare_speech.py` now takes `--text-ref/--text-test` and
REFUSES when they differ, rather than printing a number that invites a wrong
conclusion.

### Metrics to use once text is held constant
* **mel log-spectral distance** — timbre/artifact divergence (0 identical, >1.0 audible)
* **energy-envelope correlation** — dropouts, truncation, rhythm drift (<0.9 suspect)
* **per-stage RVQ code agreement** — 🚨 report **per stage**, never pooled. The
  31 quantizers are RESIDUAL: divergence in an early stage is inherited by every
  later one, so stage 0 at 70 % is far worse than stage 30 at 70 %.
* duration drift — a length difference is itself a defect.
* and a human listen before shipping.

## 4. Bit-allocation guidance for the calibrated quants

Requested: ~4-bit and the smallest possible dynamic ~2-bit, with AWQ + Hessian
+ imatrix. *"Crucial for best voices."*

    linear (quantizable)   8.856 B  79.8 %   <- spend aggression HERE
    vocab heads            1.174 B  10.6 %   lm_head + function_head
    embedding tables       0.611 B   5.5 %
    codec conv             0.200 B   1.8 %   22.05 kHz output rate
    MoG head               0.159 B   1.4 %   continuous params
    conv                   0.077 B   0.7 %
    RVQ codebook           0.016 B   0.1 %   PROTECTED

**A "2-bit VoiceChat" should mean a 2-bit BACKBONE, not a 2-bit codec.**

* The **MoG head** predicts continuous mixture parameters. Error is not
  softmax-normalised away like a token distribution — it moves the sampled
  waveform directly. High sensitivity, and 75 M of it is already fp anyway.
* **Codec decoder convs** run at the 22.05 kHz output rate; artifacts there are
  periodic and audible.
* **Early RVQ stages** matter more than late ones (residual).
* `function_head` is FULL VOCAB SIZE (0.587 B, same as `lm_head`) on a weighted
  channel. Do not deprioritise it just because it is not `lm_head` — losing it
  loses tool calling silently.

### Calibration corpus
`pipecat-ai/NVIDIA-NemotronLabs-VoiceChat-11B-Spark` ships
`calibration/fixtures/*.json` + `calibration/replay.tar` — real conversational
replay. It must exercise **both** directions (speech in AND speech out) or the
TTS side is calibrated on nothing.

## 5. Custom voices (Osaurus dino) — cloning, probably not training

`audio_prompt_latents.Aria` is `[1, 37, 1152]` = **83 KiB**, through a frozen
`audio_prompt_projection_W [1152,1152]`, with `audio_prompt_duration: 3.0`.

The speaker is **not in the weights**. Expected path:

    1. record ~3 s of the target voice
    2. encode -> audio_prompt_projection_W -> [1, 37, 1152]
    3. store beside Aria as audio_prompt_latents.Osaurus
    4. select by name (config `speaker` is a lookup key)

Implications: many voices in ONE bundle at ~83 KiB each, switchable at runtime;
**the codec ENCODER is required, not just the decoder** (both ship, 76 tensors
each); and speaker latents must stay fp in every quant.

⚠️ UNVERIFIED — confirm before building on it: whether the 37 frames are fixed
or vary with prompt length, and whether the projection consumes codec latents or
raw mel.

## 6. Reference implementation to mirror in Swift

`mlx_vlm/models/nemotron_voicechat/` (upstream >= 0.6.15) is a COMPLETE
reference — used here only as a build-time harness, not as our deliverable:

    config.py 184 · convert.py 312 · model.py 162
    session.py 319 · streaming.py 575 · tts.py 619

`VoiceChatSession.generate(audio) -> VoiceChatResult(text, audio, sample_rate,
text_tokens, audio_codes, function_tokens, user_transcript)`.

Note `function_tokens` is a **first-class field** — assert tool calls against
it, never against the transcript. (Ornith-2D shipped a model that *said* it
called a tool while emitting zero calls.)

`streaming.py` (575 ln) is the duplex/barge-in reference for step 6.

---

# MEASURED HESSIAN — bit-allocation evidence (2026-08-20)

Capture: `examples/voicechat/calibrate.py`, 562 modules, **3,383,317 row-samples**
over all three NVIDIA fixtures (turn-taking + barge-in + tool-call, 155 s of real
conversational audio, both directions exercised).

    ~/models/Logs/voicechat-calib/vc-calib.safetensors

## Mean tr(H) by component — higher = more sensitive

    TTS subword enc        n=  8   mean 169,885   max 1,281,301   <- MOST sensitive
    lm_head                n=  1   mean  22,157
    function_head          n=  1   mean  22,157                   <- IDENTICAL to lm_head
    LLM backbone           n=120   mean  18,397   max   539,633
    TTS backbone           n=199   mean   2,380   max    41,757
    perception (Conformer) n=218   mean     934   max   140,123
    TTS MoG head           n= 12   mean     411   max       698
    RNN-T                  n=  3   mean      61   max       161

## Top single modules

    1,281,301  TTS subword enc  embed_subword.backbone.encoder.layers.0.mlp.down_proj
      539,633  LLM backbone     llm.layers.54.mixer.out_proj
      450,261  LLM backbone     llm.layers.55.mixer.down_proj
      148,758  LLM backbone     llm.layers.52.mixer.out_proj
      140,123  perception       perception.encoder.pre_encode.out
      105,626  LLM backbone     llm.layers.38.mixer.out_proj

## What this changes

### 🚨 CORRECTION — I was wrong about the MoG head
Section 4 above argued the MoG head is "high sensitivity" from reasoning about
continuous outputs. **Measured, it has the second-LOWEST trace in the model**
(mean 411 vs lm_head 22,157).

Caveat that keeps it from being a free win: `tr(H)` measures INPUT activation
magnitude and proxies *reconstruction* error. The MoG head's output IS the
waveform distribution, so a low trace means low reconstruction contribution, not
necessarily low **perceptual** impact. Treat the trace as evidence it is not the
top priority — not as licence to crush it. Confirm by listening, per §3.

### ✅ CONFIRMED — `function_head` must match `lm_head`
Both measure **22,157.1**, identically: they read the same hidden state. This is
direct measured support for what was previously an argument from architecture.
Allocate them the same width. Deprioritising the tool channel because it is
"not lm_head" has no evidence behind it and loses tool calling silently.

### NEW — the TTS subword encoder is the most sensitive thing in the model
`embed_subword` converts text into the TTS representation and sits at
**8x the mean trace of `lm_head`**, with the single highest module anywhere.
It is only **8 modules** — protecting it is nearly free and should be the first
floor set in any aggressive map.

### Backbone: late layers dominate
Layers 54, 55, 52, 50, 48, 47 (`out_proj` / `down_proj`) carry the largest
traces. A uniform backbone width wastes bits on early layers and starves late
ones.

### Perception is low-mean with one outlier
Mean 934 across 218 modules, but `perception.encoder.pre_encode.out` is 140,123.
The subsampling output projection is a genuine bottleneck — every downstream
frame flows through it. Floor it; leave the rest of the Conformer aggressive.

## Proposed allocation for ~4-bit and dynamic ~2-bit

    floor high, spend low:
      embed_subword (8 mods)          -> 8-bit    most sensitive, nearly free
      lm_head + function_head          -> equal, 6-8 bit
      perception.pre_encode.out        -> 8-bit    single bottleneck
      llm late layers (~46-55)         -> +1-2 bits over base
      llm early/mid layers             -> base     (the 8.856 B to spend on)
      TTS backbone                     -> base
      MoG head / codec / RNN-T         -> base, but LISTEN before shipping
      RVQ codebook, speaker latents,
      proj_mus, embed_tokens           -> fp, never quantized (section 2.6)

🚨 Validate with the SPEECH metrics in §3, holding text constant, plus a human
listen. Text KL alone cannot see the TTS path.
