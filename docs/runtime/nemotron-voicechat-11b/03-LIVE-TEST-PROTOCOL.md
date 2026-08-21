# VoiceChat 11B — live test protocol

**INTERNAL.** 2026-08-20. The gate that decides whether anything ships.

A duplex speech model breaks in ways no assertion catches. Text models fail
*visibly* — wrong answer, garbage tokens, exception. Voice models fail
*audibly*: a chunk-boundary click, a half-second of added latency, the agent
talking over the user, the wrong voice. Every one of those passes a shape test,
a load test, and a KL eval.

Structural green means nothing here. It meant nothing on Ornith either — see
the record at the bottom.

---

## 1. What ONLY live testing can catch

| Failure | Why offline testing misses it |
|---|---|
| Chunk-boundary artifacts | Whole-sequence inference never crosses a boundary. Audible as periodic clicking, once per chunk. |
| Added algorithmic latency | A non-causal encoder is numerically *correct* — it just needs future frames. Nothing fails; everything is late. |
| Barge-in not honoured | Requires two speakers overlapping in real time. |
| Turn-taking collapse | Agent talks over user, or never yields. Only exists on a shared timeline. |
| Tool call lost mid-stream | The `function_head` channel is separate from text — a text-only assert cannot see it. **This exact class of bug shipped on Ornith-2D.** |
| Wrong/degraded voice | Speaker latent damage. Text output stays perfect. |
| TTS degradation from quantization | KL/top-1 measure the TEXT path only. See §5. |

## 2. 🎉 NVIDIA ships GROUND-TRUTH fixtures — measured, not demos

`~/models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/*.wav`
are **stereo 24 kHz**, and the channels are *separate speakers*:

    file                dur     L(user) act   R(agent) act   BOTH active
    interruptions.wav   30.0s      19.8%         29.2%          4.2%   <- barge-in
    turn_taking.wav     41.1s      11.1%         29.1%          0.0%   <- NEVER overlaps
    tool_call.wav       84.5s      18.5%         20.4%          0.4%

(measured: 50 ms frames, RMS > 0.01 = active)

This turns "does duplex work" into an **objective measurement** instead of a
listening opinion:

* `turn_taking.wav` overlap is **exactly 0.0 %** — the reference for correct
  turn-taking. Our agent channel should also approach 0 on the same input.
* `interruptions.wav` has **4.2 %** deliberate overlap — barge-in exists and is
  bounded. Agent energy must *drop* shortly after user energy starts.
* Sample rate note: these are 24 kHz rendered demos. Model I/O is **16 kHz in /
  22.05 kHz out**; resample, do not assume.

## 3. The live test ladder — run in order, stop at first failure

### L0 — load + shapes (necessary, proves nothing)
Bundle loads, tensor count/dtypes as expected, protected tensors byte-identical
(`rvq_embs`, `audio_prompt_latents.*`). Already automated for MXFP8.

### L1 — offline WAV in, WAV out
Feed `turn_taking.wav` left channel (user) as 16 kHz mono. Assert:
* ASR text is non-empty and plausible (RNN-T path alive)
* an audio buffer is produced at 22.05 kHz (TTS path alive)
* output is not all-silence and not clipping

### L2 — chunk-boundary integrity 🚨
Same input twice: once whole, once in uneven chunks (e.g. 7/5/11/4/3 frames).
* encoder output must match to < 1e-4 (already unit-tested in
  `VoiceChatStreamingConformerTests`)
* **decoded audio must have no periodic energy spike at chunk period** — FFT
  the frame-energy envelope; a peak at 1/chunk_duration is the artifact.
Without cross-chunk carry this fails audibly and silently passes L1.

### L3 — latency, measured not felt
* time from first audio-in sample to **first audio-out sample**
* reference: our existing Omni live path measures **157–219 ms first delta**
* `att_context_size [[70, 0]]` means right context 0 — **no lookahead**. If
  measured latency exceeds the frame budget by ~4 frames (0.32 s), suspect a
  non-causal conv silently waiting for future frames.

### L4 — turn-taking (objective)
Feed `turn_taking.wav` user channel. Compute overlap % between our agent output
and the user input using the §2 method.
* **PASS: overlap ≈ 0 %** (reference is 0.0 %)
* FAIL: agent speaks while user is speaking

### L5 — barge-in (objective)
Feed `interruptions.wav` user channel.
* agent must START speaking (it is a conversation)
* when user energy resumes mid-agent-turn, **agent energy must fall within
  ~300 ms**
* PASS: measurable energy drop after each interruption onset
* FAIL: agent continues at full energy — barge-in not wired

### L6 — tool call on the function channel 🚨
Feed `tool_call.wav` user channel.
* a tool call must be EMITTED on the function channel
* 🚨 **assert the call itself, not the transcript.** The Ornith-2D failure was
  a model *saying* "Checked the weather in Tokyo." while emitting **zero** tool
  calls. A text assert passes that; a `toolCalls == 1` assert does not.

### L7 — voice identity
* default `Aria` renders and is stable across runs
* after a custom latent is installed (dino), output audibly differs while text
  output is unchanged
* swapping the latent must NOT change ASR/text behaviour at all

### L8 — VARIATING, not linear 🚨
Do not walk one happy path. Cross the axes in ONE session:

    speech-only -> speech+tools -> tools alone -> speech again
    barge-in DURING a tool call
    silence (silence_token_id 11) between turns
    long turn -> short turn -> replay an earlier turn verbatim

The Ornith lesson: every linear row passed on all 8 quants; the variating
pattern found 2D loses tool calls **only** when an image shares the turn. The
voice analogue is *barge-in during a tool call* — two axes crossing.

### L9 — GUI harness (the only proof that counts for osaurus)
Per house rule: proof AND exploration go through the **running dev app's GUI**
— never curl, never HTTP, never CLI, not even for diagnosis. Drive it with
AppleScript / System Events + `screencapture`.
* real mic in, real speakers out
* interrupt it mid-sentence with your voice
* confirm it yields, and resumes coherently

## 4. Per-quant coverage — every quant, not one

Run L1–L8 on **every** quant. Ornith proved a defect can exist in exactly one
of eight bundles and in no other.

    MXFP8    10.91 GiB   built — test first, it is the quality reference
    ~6-bit    ~8.5 GiB   pending runtime
    ~4-bit    ~6.0 GiB   pending runtime
    ~2-bit    ~3.7 GiB   pending runtime — MOST likely to fail L5/L6/L7

Expect the aggressive quants to fail the *interaction* rows (L5, L6, L8) before
they fail the basic ones. That is the pattern: 2D passed text, vision, video and
code individually, and failed only where two axes crossed.

## 5. 🚨 Quantization eval must cover the SPEECH path

Our standard judge — KL / top-1 vs a reference bundle — measures the **text**
path only (`lm_head` logits). VoiceChat has two output paths on one backbone.
**A quant can hold text KL perfectly and sound degraded**, and nothing in the
usual eval would report it.

Minimum additional speech-path metrics per quant:
* **mel-spectral distance** between quant output and MXFP8 output on identical
  input (objective, no listening)
* **codec token agreement** — RVQ code indices vs reference, per quantizer.
  The 31 quantizers are *residual*: divergence in an early quantizer is
  inherited by all later ones, so report agreement **per stage**, not pooled.
* **energy-envelope correlation** — catches artifacts and dropouts
* and a human listen before shipping. There is no substitute for the last one.

## 6. Status

- [x] Fixtures characterised (objective overlap baselines established)
- [x] L2 encoder half unit-tested (`VoiceChatStreamingConformerTests`, 5/5)
- [ ] L1, L3–L9 — **blocked on the Swift runtime** (steps 3–6)
- [ ] Speech-path quant metrics (§5) — needed before any calibrated quant ships

## 7. Why this document is strict

Every defect found in this codebase this week reported success first:

* AWQ silently reverted through three clean rebuilds — only an implausibly
  identical rel-err gave it away
* four separate MoE tools reported success while covering **8 %** of a model
* published Ornith-1.0 MXFP8 dequantized every Linear correctly and emitted
  `'5$ut\n\n'`
* GPTQ improved every in-sample metric and got **worse** out-of-sample
* Ornith-2D passed text/vision/video/code individually and loses tool calls
  where two axes cross — while *claiming the tool ran*

None of them raised an exception. For a model whose output is sound, the gap
between "no error" and "correct" is wider still.
