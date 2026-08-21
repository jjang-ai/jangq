# 07 — Voice control, direct TTS, and what the checkpoint will and won't do

Status: measured 2026-08-20 against `NemotronLabs-VoiceChat-11B` (MXFP8 /
JANG_4 / JANG_2) with the Swift runtime in `vmlx-swift`.

This document exists because three separate things in this area *looked* like
they worked and did not, and each one cost a day. Read §1 and §2 before you
try to give this model a different voice.

---

## 1. Voice cloning is NOT possible with this checkpoint

**Do not spend time on it. This is settled by measurement, not opinion.**

### The mechanism

`RVQEARTTSModel.warmup` builds the speaker prompt like this:

```python
projected_prompt = mx.matmul(code_embed, self.audio_prompt_projection_W)
if audio_prompt_latent is not None:
    projected_prompt = audio_prompt_latent.astype(code_embed.dtype)
code_embed = mx.where(pre_bos[..., None], projected_prompt, code_embed)
```

So when you supply an explicit latent you get the shipped speaker, and when you
don't, every prompt frame becomes a projection of the reference recording's own
codec code embeddings. That reads like a voice-cloning entry point. It isn't.

### Measurement 1 — the projection is unscaled

`audio_prompt_projection_W` is 1152×1152 with **std 1.000149**: an unscaled
random normal. A trained projection would carry fan-in scaling around
1/√1152 ≈ 0.029.

Applied raw, a prompt frame comes out at **RMS 44.56** while the shipped
`audio_prompt_latents.Aria` frames it replaces sit at **RMS 1.69** — 26× hot.
The backbone receives prompt conditioning far outside its trained range. This
is why cloned voices changed timbre but stopped forming words: clones read back
`""`, `"Hello"`, `"Mm."`.

Scaling by 1/√hidden restores intelligibility completely — after the fix all
clones speak full sentences ("Hello, how can I help you today?", 13 tokens).
**That fix is in the Swift port** (`VoiceChatTTSModel.warmup`, overridable with
`VMLX_VOICECHAT_PROMPT_SCALE`). It is worth keeping because it turns a garbage
path into a working one. It does not, however, buy voice cloning:

### Measurement 2 — the projection is untrained (the decisive one)

`audio_prompt_latents.Aria` is what the model uses for its own voice. Projecting
a recording **of that same voice** should therefore reproduce it. Test:
`VoiceChatVoiceCloningTests.testProjectedPromptAgainstShippedAriaLatent`.

```
[prompt-cal] raw projection RMS  44.5556
[prompt-cal] shipped Aria RMS    1.6893
[prompt-cal] RMS-matching scale  0.037915
[prompt-cal] 1/sqrt(hidden) scale  0.029463
[prompt-cal] per-frame cosine to Aria: mean -0.0229  min -0.0806  max +0.0560
```

**Cosine ≈ −0.02.** Random vectors in 1152 dimensions sit at ±1/√1152 ≈ 0.029.
The projection's output is statistically indistinguishable from random relative
to the thing it is supposed to reproduce. No scale fixes a random direction.

Consistent with all of it: the checkpoint ships exactly one latent tensor
(`tts_model.audio_prompt_latents.Aria`) and **no reference encoder**, and the
Python reference never takes this branch — `session._tts_prompt` always passes
Aria.

Observable consequences, all confirmed:
- clones differ from built-in (a random but deterministic perturbation)
- clone pitch does *not* follow the reference: a 319.6 Hz reference produced a
  182.2 Hz clone, *below* the 225 Hz built-in
- clone HNR drops (6.1 → 4.2–4.9 dB): rougher, breathier

**Conclusion:** different voices must come from post-processing (§3), or from a
future checkpoint that actually ships a trained speaker encoder.

---

## 2. Direct text-to-speech: the text channel emits a BURST

The duplex loop speaks whatever the backbone decided. An avatar also needs
"say exactly this". The speech tower supports it — `step` takes one subword
token per 0.08 s frame and does not care whether it came from `lm_head`.

`NemotronVoiceChatModel.synthesize(frameTokens:voice:)` and
`frameSchedule(subwordIds:...)` do this in the Swift port.

### The trap

The obvious schedule — spread tokens at a natural speaking rate — is **wrong**
and produces convincing-sounding garbage. Dumping the model's own per-frame
text channel for "Hello! How can I help you today?" over 76 frames gives:

```
......... | .................................. <s> Hello ! -How -can -I
-help -you -today ? ....................
```

Nine content tokens on nine **consecutive** frames after `<s>`, then PAD for the
rest of the turn. The speech tower reads the burst out over the trailing PAD
frames, so the audio runs far longer than the burst that produced it.

Measured degradation from guessing instead (asking for "Hi there! I'm so happy
to see you today."):

| frames/token | ASR read-back |
|---|---|
| 3 | "Hi, I think there are MKane" |
| 4 | "Hi, I can say that Kurt and XM Four oh" |
| 5 | "Hi" |
| 6 | "Hi, is a hi is faintable" |
| 7 | "Hi, comma dries for Zapathy. I'm happy too" |
| **1 (burst)** | **"Hi there, I'm so happy to see you today" — 100%** |

Note every wrong setting still produces fluent, confident-sounding audio that
begins with the right word. Energy statistics do not distinguish them. Only the
ASR read-back does.

Working recipe: `[PAD]×4 + <s> + ids (one per frame) + PAD×(2.5·len + 16)`.
The tail is the speaking time; too short clips the final words.

### Second trap: invented words

The model mangles onomatopoeia and contractions it has not seen written that
way: "Nom nom nom" → "Nam", "Hee hee" → "He's", "Wanna" → "Mana". Scripted
avatar lines should use real words. The ASR gate catches this.

---

## 3. Making different voices by post-processing

Two independent dials, and you need both:

- **Pitch** (`asetrate` + `atempo`) shifts pitch *and* formants together. On its
  own it gives one voice at several heights, not different characters.
- **Formants** (cepstral envelope warp, `scratchpad/formant.py`) is
  vocal-tract length — the dial that makes characters distinct. Below 1.0 reads
  as bigger, above as smaller. It leaves harmonic positions alone, so pitch is
  untouched.

Shipped cast (all ASR-verified), source voice 242 Hz:

| character | pitch | formant | tempo | f0 |
|---|---|---|---|---|
| Pip — tiny, squeaky | +6 st | ×0.98 | 1.05 | 345 Hz |
| Bubbles — bright cartoon | +5 st | ×1.00 | 1.06 | 320 Hz |
| Munch — baby pitch, round tone | +4 st | ×0.80 | 0.98 | 286 Hz |
| Rex — large, warm | −3 st | ×0.88 | 0.95 | 202 Hz |
| Sunny — energetic | +6 st | ×0.90 | 1.02 | 334 Hz |

### 🚨 Never use fixed-hop overlap-add for the pitch shift

A hand-written OLA pitch shifter shipped once and the clips were described as
"two voices at once… alien". That was real and measurable: every file carried
amplitude modulation at **66.8 Hz — the identical frequency in all of them
regardless of shift amount** — at 3–4× the source's level.

22050 ÷ 66.8 ≈ 330 samples = the shifter's hop. It stitched at points where
pitch periods did not line up; the buzz landed in the male-voice pitch range,
i.e. a 66.8 Hz tone under a 286 Hz voice.

Use correlation-aligned overlap-add (ffmpeg's `atempo` is WSOLA). After the
change: no 66.8 Hz component, AM back to source level (0.008–0.014 at 20–25 Hz,
natural prosody), and HNR *better* than the source (7.5–8.3 vs 6.4 dB).

**QC any generated audio with `scratchpad/voiceqc.py`** — it reports f0, HNR,
and AM strength/frequency. An AM peak that is the same frequency across
different pitch shifts is a stitching artifact, full stop.

### Intelligibility ceiling

Pitch shifting has a limit and it is abrupt. At +7 st with formants ×1.13, a
clip transcribed to **nothing at all** while still passing every energy check.
The table above is the highest setting per character that reads back cleanly.
Push further only with the gate running.

---

## 4. The gate — use it, nothing ships without it

`VoiceChatIntelligibilityTests.testEveryClipInDirectoryContainsWords`
transcribes every wav in a directory with the model's own ASR in one model
load, and fails on any clip with zero tokens.

```bash
VOICECHAT_QUANT_PROOF=1 \
VOICECHAT_TRANSCRIBE_DIR=/path/to/clips \
VOICECHAT_TRANSCRIBE_EXPECT="the sentence they should say" \
xcrun swift test --filter "VoiceChatIntelligibilityTests/testEveryClipInDirectoryContainsWords"
```

This exists because a folder of "voice samples" was once delivered that was
entirely babble. It had passed RMS, dynamic-range and zero-crossing checks —
**those statistics are satisfied by structured noise and certify nothing about
speech.** Post-processing is exactly the step that destroys words while leaving
the statistics intact.

Related env hooks:
- `VOICECHAT_TTS_SCRIPT` — JSON `[{name, text, ids}]` to synthesize + verify
  (`testSynthesizeChosenLines`); ids come from the bundle's own tokenizer
- `VOICECHAT_ARIA_REFERENCE` — projection calibration (§1)
- `VMLX_VOICECHAT_PROMPT_SCALE` — override the prompt projection scale
- `VMLX_VOICECHAT_GUIDANCE=1` — re-enable classifier-free guidance (see §5)

---

## 5. Still open

**Classifier-free guidance is off by default** and that is deliberate. Split by
guidance half, a decode step is exact on the conditional half at all 28 layers
(cosine 0.99995+) and collapses on the unconditional half (0.998 → 0.334 →
0.224; layer 1 norm 93.26 vs 65.99). With guidance on the decoder emits audio
that passes every energy statistic and contains no words. The reference runs
`guidance_scale` 0.2, so what this gives up is small next to the difference
between speaking and not.

Nothing else. **JANG_2 is resolved — see §6.**

---

## 6. A 2-bit VoiceChat bundle is not viable; ship JANG_3 instead

JANG_2 transcribed the fixture perfectly and never said a word. The cause is
not a load defect and not the runtime: it is the turn-taking decision.

### What actually happens

`testTextChannelLogitDiagnostic` prints the text channel's top candidates per
frame. On JANG_2 the logits are healthy — finite, sharp, max +24…+35 — and the
model simply chooses `<PAD>` on **61 of 62 frames at p=1.000**. On a working
build the decision to start speaking is a near-coin-flip:

```
JANG_4   f030   <s> 0.643   <PAD> 0.357     -> speaks
JANG_3   f031   <s> 0.705   <PAD> 0.295     -> speaks
JANG_2   f040   <PAD> 1.000  <s> 0.000      -> never speaks
```

Quantization noise biases that narrow margin, and once it is biased the model
is silent for the whole turn. Everything else about the bundle looks fine,
which is why it presents as "broken runtime".

### What does and does not fix it (all measured)

| build | change | weight bytes | rel-err | speaks? |
|---|---|---|---|---|
| JANG_2 | baseline 2-bit | 5.7 GB | 0.207 | no |
| JANG_2b | `--llm-floor 3` | 6.44 GB | 0.187 | no (`<s>` 0.006) |
| JANG_2c | `embed_tokens` floored to 8 | 6.12 GB | 0.207 | no |
| VC-test-P | perception floored to 4 | 6.25 GB | 0.136 | no |
| VC-test-L | `--llm-floor 4` | 7.75 GB | 0.175 | no (`<s>` 0.011) |
| VC-test-S | whole `stt_model.` floored to 4 | 7.87 GB | 0.104 | **yes** |
| **JANG_3** | **uniform `--base-bits 3`** | **7.16 GB** | **0.101** | **yes** |

Two things to take from this. First, **no single component is the culprit** —
flooring the backbone alone fails, flooring the perception encoder alone fails,
and only raising the entire speech-understanding tower together works. Second,
the outcome tracks overall reconstruction error, not any one module: every
build at rel-err ≲ 0.104 speaks and every build ≥ 0.136 is silent.

`embed_tokens` at 2 bits looked like an obvious cause — it is the embedding the
duplex loop feeds back into itself every frame, and `FLOORS` protects the TTS's
`embed_subword` at 8 while never mentioning it. Flooring it to 8 changed
nothing. It is still worth having as a floor, but it was not the bug.

Crucially, VC-test-S reaches 7.87 GB — **the same size as JANG_4 (7.9 GB)**. A
2-bit build that works is not smaller than the 4-bit build, so the 2-bit tier
has no reason to exist for this model. A uniform 3-bit build is the real answer:
it speaks, and at 7.16 GB it is genuinely smaller.

### Shipping matrix

Harness, 8 runs each (LCS overlap), plus live runs in the app's
Speech-to-Speech panel:

| tier | size | harness x8 | live in app |
|---|---|---|---|
| MXFP8 | 11 GB | 100 x3 | 32 tok, 32 tok — verbatim both |
| **JANG_4** | **7.9 GB** | 100 97 100 100 98 89 89 100 | 55 tok verbatim |
| JANG_3 | 7.2 GB | 100 100 **59** 100 100 100 100 100 | 33, **NOTHING**, 8, 33, 32 |
| JANG_2 | 5.7 GB | 0 x3 | (said nothing) — do not ship |

🚨 **JANG_3 speaks, but not reliably.** Its text channel is correct and
identical every single run; the SPEECH is what varies. That is the MoG head,
and it is faithful to the reference — component choice is a Gumbel-max over
1024 mixture components (`noise_scale` is only 0.001, so the Gaussian term is
negligible). At 3 bits the mixture logits are noisy enough that the pick lands
on unintelligible audio a noticeable fraction of the time: one 59% in eight
harness runs, and two poor reads in five live runs.

**Recommendation: JANG_4 is the small tier.** JANG_3 saves 0.7 GB and costs
reliability. Do not judge either from three runs — an earlier promotion of
JANG_3 rested on exactly that and the live panel disproved it within five turns.
Any future tier needs a pass-RATE, not a pass.

Built with `python -m jang_tools.convert_voicechat_jang <bf16> <out>
<calib.json> <calib.safetensors> --base-bits 3 --group-size 32`. The converter
gained `--llm-floor` and a general `--floor SUBSTR=BITS` for this bisection;
both default to the previous behaviour.

### 🚨 The gate's own metric had to be fixed first

The overlap score was a single forward walk that allowed skips only in the
heard text, so one missing letter early stalled the pointer for the rest of the
string. It scored a JANG_3 run at **11%** whose speech was plainly correct
("That sounds delicious. I'd be happy to help you make some peanut butter
cookies", 32 tokens) purely because the ASR dropped a leading "Oh,", and scored
a correct clip at 30% over "favourite" vs "favorite". It is now a longest
common subsequence, which allows skips on both sides, pinned by
`testOverlapMetricDoesNotInventFailures`.

A gate that invents failures is worse than no gate — it sends you hunting a
runtime bug that does not exist. Two of the numbers in this document were
originally misread for exactly that reason.

Note also that the MoG head samples, so the duplex gate varies run to run
(100 / 70 / 100 on identical input at one point). Treat it as a floor, never as
a byte-exact comparison, and re-run before believing a single low score.
