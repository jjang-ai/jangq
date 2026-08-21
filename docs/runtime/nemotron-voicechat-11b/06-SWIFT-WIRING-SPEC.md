# VoiceChat 11B — Swift wiring spec (self-contained)

**INTERNAL.** 2026-08-20. Everything a worker needs to wire the remaining
pieces without re-deriving. Updated as steps land. Ground truth for what IS
landed: vmlx-swift branch `voicechat-integration` (35bd87c5 ground state,
cfef5f69 steps 3+4) and `Tests/MLXLMTests/VoiceChatSTTModelTests.swift`.

## 0. The one-paragraph architecture

One `nemotron_h` 56-layer hybrid backbone (4480 hidden) fed by mixed
embeddings on a 12.5 fps frame clock. INPUT: 16 kHz mel → 24-layer causal
FastConformer (×8 subsample) → proj 1024→4480. OUTPUT heads over the SAME
hidden state: `lm_head` (text/speech channel), `function_head` (tool calls —
first-class channel, never parsed text). Side branch: RNN-T (LSTM predict +
joint) transcribes the user from UNPROJECTED 1024-d encoder frames. SPEECH
OUT: a separate 28-layer Gemma3-style transformer (1152 hidden, sliding
window 7500, pattern 6) consumes RVQ code embeddings fused with
character-aware subword conditioning, a MoG head iteratively refines 31
residual codebook indices, and a ConvNeXt codec decoder + iSTFT renders
22.05 kHz audio. Duplex = four concurrent channels with fixed frame delays.

## 0b. STATUS 2026-08-20 — steps 3–6 landed, audio-to-audio proven on all 3 quants

Branch `voicechat-integration` (vmlx-swift). **The full runtime path works
end to end on the real shipped bundles**: real WAV in → mel → causal conformer
→ nemotron_h → both heads → EAR-TTS → RVQ codes → codec → 22.05 kHz WAV out.

    quant    load   turn   frames  samples  rms      zcr     asr
    MXFP8    1.1s   3.4s   26      45864    0.0096   0.242   2 tok
    JANG_4   0.5s   6.7s   26      45864    0.0109   0.251   2 tok
    JANG_2   0.8s   4.1s   26      45864    0.0224   0.154   2 tok

Input: NVIDIA `turn_taking.wav` user (left) channel, 24 kHz → 16 kHz mono,
2.0 s. Output WAVs verified independently as valid 22.05 kHz Float32, 2.08 s,
with real voiced-frame structure (103/103, 57/103, 81/103 voiced 20 ms frames).
**Input-dependence check: 95.0 % of output samples differ between two different
2 s windows of the same conversation** (and the ASR transcript changes) — the
model is demonstrably listening, not emitting a canned response that would
satisfy every RMS/ZCR measurement.

Reproduce:

    VOICECHAT_QUANT_PROOF=1 \
    VOICECHAT_QUANT_DIR=~/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4 \
    VOICECHAT_PROOF_SECONDS=2.0 VOICECHAT_PROOF_DIR=/tmp \
    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
      --filter VoiceChatQuantAudioProofTests --jobs 2 -Xswiftc -F -Xswiftc \
      /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks

🚨 **MXFP8 is NOT on the Hub**: `OsaurusAI/NemotronLabs-VoiceChat-11B-MXFP8`
contains only `.gitattributes` — the weights were never pushed. The working
copy is local at `~/models/JANGQ-AI/NemotronLabs-VoiceChat-11B-MXFP8` (11 GB).
Publishing it is Eric's call.

### Three real defects the live run found (all fixed, all pinned)

1. **`joint_net` as a Swift array broke every QUANTIZED load.** NeMo ships only
   index 2 (0/1 are parameter-free). Plain weight loading tolerates that (MLX
   pads with `.none`), but `quantize()` builds module children keyed `{"2": …}`
   and cannot merge them into an array. The slot is now named `out`, with the
   activation applied inline and `joint_net.2` rewritten in sanitize.
2. **`embed_tokens`/`lm_head` remap matched only `.weight`**, orphaning a
   quantized bundle's `.scales`/`.biases` as unhandled keys.
3. **The quantization map is keyed by BUNDLE paths**, while the module tree
   renames several of them — so every per-module recipe missed and modules got
   the GLOBAL default. On mixed-bit JANG that is a hard shape error
   (`w (4480,2450)` vs `scales (4480,490)`). Fixed with a shared remapper plus
   `VoiceChatQuantizationMapPathTests`, which asserts every mapped module in
   all three real bundles resolves in the model tree.

### What is NOT done

The osaurus app has no duplex-speech host: `Services/Voice` is an
ASR → text → TTS pipeline (transcription overlay, TTS mode), not a full-duplex
session. Hosting VoiceChat there — mic streaming into the frame loop, 22 kHz
chunk playback, barge-in wired to the UI — is an app-side feature build, and
it is what the L9 GUI/real-mic proof and the Dinoki avatar surface depend on.

## 1. Landed Swift surface (do NOT rebuild)

| File (Libraries/MLXVLM/Models/NemotronVoiceChat/) | What | Proven by |
|---|---|---|
| NemotronVoiceChatConfiguration.swift | full config incl. nested decoder/joint, `[[Int]]` att_context, blank_as_pad, pos_emb_max_len | NemotronVoiceChatConfigTests 3/3 vs real config |
| StreamingConformer.swift | prototype causal-conv/mask property pieces (keys NOT bundle-real; kept for its tests) | VoiceChatStreamingConformerTests 5/5 |
| VoiceChatConformer.swift | REAL-KEY full 24L encoder: causal subsampling, rel-pos attn (`stream()` included for cache-aware step), chunk-based `chunked_limited` mask | key-parity via STT full-load test |
| VoiceChatRNNT.swift | PredictNetwork/JointNetwork/greedy streaming decode + PyTorch-LSTM sanitize | chunked==whole with real weights; bias-sum pinned |
| VoiceChatSTTModel.swift | perception+backbone+BOTH heads composition, `sanitized()` full stt_model remap | `update(verify: [.all])` against the real 997-tensor stt_model |
| VoiceChatTTS.swift | OffsetRMSNorm, MoG head (raw proj_mus), T5Gemma char encoder, flag embeddings, gated fusion | VoiceChatTTSTests |
| VoiceChatTTSModel.swift | 28L speech transformer (global/sliding alternation), warmup w/ Aria latent, 8-round RVQ refinement | same |
| VoiceChatCodec.swift | ConvNeXt enc/dec, 31-stage PRVQ, iSTFT vocoder, streaming caches, STFT encoder | same (incl. seam negative control) |
| VoiceChatSpeechDecoder.swift | codec + TTS + speaker latents; `sanitized()` full tts_model remap | 635-tensor `verify: [.all]` |
| NemotronVoiceChat.swift | the duplex turn loop, control-code substitution, RNN-T user transcript | VoiceChatEndToEndTests |
| VoiceChatLoader.swift | real-bundle load incl. per-module quantization map | quant audio proof + map-path tests |
| VoiceChatMel.swift | NeMo-exact log-mel front-end (preemph, Slaney, normalize NA) | quant audio proof |
| (MLXLLM) NemotronH.swift | dense decode widening + `hiddenStatesFromEmbeddings` public | NemotronHTests 37/37 |

Run everything:

    DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun swift test \
      --filter "NemotronVoiceChatConfigTests|VoiceChatStreamingConformerTests|VoiceChatSTTModelTests|NemotronHTests" \
      --jobs 2 -Xswiftc -F -Xswiftc /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/Library/Frameworks

## 2. Load-time key remaps (ALL verified against the shipped bundle)

Input keys below are bundle keys with `stt_model.` / `tts_model.` stripped by
the loader before calling each side's `sanitized`.

STT side (`VoiceChatSTTModel.sanitized`):
    embed_tokens.weight        → llm.backbone.embeddings.weight
    lm_head.weight             → llm.lm_head.weight
    llm.<X>                    → llm.backbone.<X>
    perception.preprocessor.*  → DROPPED (deterministic mel buffers, rebuilt)
    perception ndim-4 weights  → transpose(0,2,3,1)   (torch OCHW → MLX OHWC)
    perception ndim-3 weights  → transpose(0,2,1)     (torch OCK  → MLX OKC)
    llm …conv1d.weight ndim-3  → transpose(0,2,1)
    …dec_rnn.lstm.weight_ih_lN → …dec_rnn.lstm.N.Wx
    …dec_rnn.lstm.weight_hh_lN → …dec_rnn.lstm.N.Wh
    …dec_rnn.lstm.bias_ih_lN + bias_hh_lN → SUMMED → …dec_rnn.lstm.N.bias  🚨

TTS side (`SpeechDecoder.sanitize` in the reference — port verbatim):
    _control_codes                     → control_codes
    audio_codec.prvq._variance_list.N  → audio_codec.prvq.variance_list.N
    audio_codec decoder stage-start ConvTranspose1d weights (layer index % (blocks_per_stage+1) == 0
      within the stage region, not dwconv) → transpose(1,2,0); all other 3D codec weights → transpose(0,2,1)

## 3. TTS stack — complete porting map (step 5, NOT yet landed)

Reference: `mlx_vlm/models/nemotron_voicechat/tts.py` (619 ln) at
`/Users/eric/.cache/uv/archive-v0/JbCKtphtiAA0araM/...` and codec at
`/Users/eric/.cache/uv/archive-v0/oIlkbImmg8GZ1YQl/mlx_audio/codec/models/nemotron_voicechat/codec.py`.

### 3.1 Config values NOT in config.json (dataclass defaults — must be Swift defaults)
    tts: rms_norm_eps 1e-6 · query_pre_attn_scalar 256.0 · sliding_window_pattern 6
         rope_global_base_freq 1e6 · rope_local_base_freq 1e4
         num_iterations 8 · exponent 3.0
    character_encoder (nested in tts_config, partially shipped): 1 layer, hidden 1152,
         heads 16/16, head_dim 72, attn_logit_softcapping 50.0, rope_base 1e4,
         query_pre_attn_scalar 256.0, char_vocab_size 257
    mog_head (shipped): intermediate 4608 · low_rank 64 · min_log_std -4.0 ·
         num_layers 3 · num_predictions 1024 · eps 1e-6
    codec derived: stft_channels = (n_fft/2+1)*2 = 18 · frame_rate 22050/(hop·prod(rates)?) —
         read config.py property, waveform_to_token_ratio = hop_length · prod(downsample_rates)

### 3.2 Module tree & keys (all under bundle `tts_model.`)
    tts_model.backbone.layers.{0..27}   Gemma3-style block, keys:
        input_layernorm · self_attn.{q,k,v,o}_proj + {q,k}_norm ·
        post_attention_layernorm · pre_feedforward_layernorm ·
        mlp.{gate,up,down}_proj · post_feedforward_layernorm
        — 🚨 ALL norms here are OffsetRMSNorm (Gemma 1+weight). The nemotron_h
          backbone uses PLAIN RMSNorm. Mixing them empties text output while
          every structural check reads healthy (trap #1, already paid for).
        — attention: every (idx+1) % 6 == 0 layer is GLOBAL (rope 1e6, KVCache);
          others LOCAL sliding window 7500 (rope 1e4, RotatingKVCache(keep: 0)).
        — vocab_size 1 + embed_tokens deleted: inputs are ALWAYS embeddings.
    tts_model.backbone.norm.weight       final OffsetRMSNorm
    tts_model.{bos_emb,null_emb}         (1152,) raw vectors
    tts_model.embed_code.weight          Linear 512→1152 no bias
    tts_model.rvq_embs                   (31, 1024, 512) — 🚨 STAYS FP
    tts_model.audio_prompt_projection_W  (1152, 1152) frozen matmul (raw param)
    tts_model.embed_subword.*            CharAwareSubwordEncoder:
        backbone.encoder.layers.{0}.…    T5Gemma layer (pre/post_self_attn_layernorm,
                                         pre/post_feedforward_layernorm, self_attn q/k/v/o, mlp)
        backbone.encoder.norm            OffsetRMSNorm; encoder scales input by sqrt(1152)
        embed_tokens.weight              (257, 1152) — 🚨 dtype is READ (tts.py:342) — STAYS FP
        proj_embedding.weight            1152→1152 no bias
        subword_flag_emb.{cont_emb.weight,is_continuation,pad_tensor}   int buffers
        bos_eos_emb.{special_emb.weight,special_flags,pad_tensor}       int buffers
        — int buffers must survive load untouched (cast_predicate excludes
          special_flags / is_continuation / pad_tensor endings, and A_log).
    tts_model.gated_fusion_audio_text    audio_proj+text_proj (BIASED Linears),
                                         gate (1152,), residual_scale scalar, final_norm Offset.
        fuse(audio,text) = final_norm(sigmoid(residual_scale) * (g*audioProj(audio/31) + (1-g)*textProj(text)))
    tts_model.mog_head                   mlp_stack.{0,1,2} MLPLayer(pre_norm/mlp/post_norm) +
                                         mlp_stack.3 OffsetRMSNorm; proj_logits 1152→1024;
                                         proj_mus 1152→65536 — 🚨 read RAW+reshaped(1024,64,1152)
                                         (tts.py:136) — STAYS FP, loader must not assume quantized;
                                         proj_logs 1152→1 (clamped ≥ -4.0); proj_else 1152→512;
                                         low_mat (1024, 512, 64) raw param.
    audio_prompt_latents.Aria            (1, 37, 1152) — 🚨 STAYS FP (speaker identity)
    control_codes (3,) int · codec_silence_tokens (31,) int
    audio_codec.encoder/decoder/prvq     see §3.4

### 3.3 Generation algorithm (port of tts.py, exact order)
    warmup(code, subword_ids, subword_mask, audio_mask, prompt_latent):
      shifted = [0, code[:-1]]; embed = embed_code(depthsum(shifted))
      bos_mask = audio & !prev_audio; pre_bos = cumsum(bos)==0
      pre-BOS frames get the PROMPT LATENT (Aria) instead of code embeds; add bos_emb at bos
      guidance doubles the batch (cond/uncond, uncond text = null_emb)
      inputs = gated_fusion(code_embed, char_condition); backbone(inputs, cache)
    step(): same fusion for ONE frame → hidden → generate_codes(hidden)
    generate_codes(): iterative RVQ refinement, num_iterations 8, schedule
      masked_i = ceil((1-rate^3)^(1/3) * 31); each round: depthsum(current code) →
      MoG infer (guidance_scale 0.2, top_p 0.95, Gumbel component pick,
      mu = low_mat[k] @ (proj_mus_k @ x), out = mu*exp(logs)+proj_else) →
      residual + noise_scale*N(0,1)*exp(logs) → nearest-neighbour RVQ encode of
      quantizers [completed, completed+count).
    depthsum(code): sum over 31 codebooks of rvq_embs[q, code_q] with an extra
      PAD row (index 1024 = codebook_size) appended as zeros.
    make_cache(): (i+1)%6==0 → KVCache else RotatingKVCache(maxSize: 7500, keep: 0)

### 3.4 Codec (port of codec.py)
    encoder.layers: [Conv1d 18→384 k1 nobias] + per stage {3× ConvNeXtBlock1d(k7 causal)} +
      Conv1d stage_ch→next rate=stride (downsample_rates [7,7,9], channels 384/768/1536, latent 512)
    decoder.layers: reversed — ConvTranspose1d(rate=stride, nobias) then 3 blocks per stage,
      final Conv1d 384→18 k1
    ConvNeXtBlock1d: dwconv(k7 groups=ch, CAUSAL left pad k-1, streaming cache carries k-1) →
      ChannelLayerNorm(w+b) → pwconv1 ch→4ch → GELU → pwconv2 → +residual
    prvq: mus_list.{0..30} (1024,512) — encode = iterative NN on residual; decode = sum of table rows
    decode_latents: features → (B,18,T): magnitude = 100*exp(-softplus(-logits+log(100)));
      real/imag from magnitude·cos/sin(phase); DC+Nyquist imag zeroed; iSTFT
      (n_fft 16, hop 4, periodic Hann, center=False), trim (n_fft-hop)/2 each side.
      Streaming: cache 4 spectrogram frames ("istft_real/imag"), trim half_wave each side;
      flush appends half_spec zero tail.
    🚨 Swift has no ready iSTFT: implement with MLXFFT irfft per frame + overlap-add
      with window-sum normalization (constrain to ≥1e-8). n_fft 16/hop 4 is tiny —
      a simple loop is fine at 22 kHz frame counts.

### 3.5 Tokenizer coupling
    CharAwareSubwordEncoder.set_vocabulary(tokenizer.get_vocab()): single-char
    tokens sorted by id → char table (must equal 257-1 entries); subword id →
    tuple of char ids. Session must call this BEFORE first TTS use.

## 4. Duplex session (step 6) — port map

Reference `streaming.py` (575 ln): `VoiceChatSession`/`streaming` drive
per-frame: mel stream (StreamingLogMelSpectrogram), ConformerStreamingState
(window = left+right+1 = 71 frames, uses attention `stream()` — Swift hook
already exists on VoiceChatRelPositionAttention), language cache =
stt_model.makeCache(), TTS cache + CausalConv1dCache for codec.
Frame loop: user audio frame → perception step → mix embeddings (user audio,
agent text w/ delay_text_channel_by 2, source text delay 15) → backbone step →
sample text token + function token → TTS step on agent frames → codec
decode_step → 22.05 kHz chunk out. Fixtures + objective PASS criteria in
03-LIVE-TEST-PROTOCOL.md §2 (turn-taking overlap ≈0%, barge-in ≤300 ms,
toolCalls==1 on the function channel).

## 5. Quants to prove (Eric's gate: LIVE audio-to-audio on ALL THREE)

    ~/models/OsaurusAI/NemotronLabs-VoiceChat-11B-MXFP8   (downloaded)
    ~/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_4  (downloaded, default)
    ~/models/OsaurusAI/NemotronLabs-VoiceChat-11B-JANG_2  (downloaded)

Quantized loading: `nn.quantize` BEFORE weight load using the per-module map
in config.json["quantization"]; absent module = fp passthrough (mirrors the
DFlash2Loader pattern already in vmlx). Protected-fp tensors: rvq_embs,
audio_prompt_latents.*, mog_head.proj_mus, embed_subword.embed_tokens.
Speech-path metrics per quant (§5 of 03): mel-spectral distance vs MXFP8,
per-stage RVQ code agreement, energy-envelope correlation — text KL alone is
blind to the speech path.

## 6. Live-proof harness notes (house rules)

L9 = the osaurus dev app GUI with a real mic — never curl/CLI. Background
GUI drive: `OSU_MODELS_DIR=/Users/eric/models` env for the model library in
isolated roots; picker rows via prooftool `axshowpress` (AXScrollToVisible);
verify the LOADED bundle from osaurus.log mmap lines, never the chip text.
Fixtures live at `~/models/nemotron-voicechat-src/NVIDIA-NemotronLabs-VoiceChat-11B/*.wav`
(stereo 24 kHz, L=user R=agent; resample to 16 kHz mono for input).
`jang/examples/voicechat/prepare_fixture.py` extracts channels.
