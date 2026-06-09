# Gemma 4 QAT MXFP4 / JANG_4M Live Proof Notes

Date: 2026-06-09

## Local Bundle Inventory

Authoritative local bundle root:

- `/Volumes/EricsLLMDrive/jangq-ai`

Structurally complete bundles found there:

| Bundle | Format | Size | Shards | Modalities in metadata | Structural status |
| --- | --- | ---: | ---: | --- | --- |
| `gemma-4-E2B-it-qat-JANG_4M` | `jang_affine` | 7.27 GiB | 4 | text, vision, audio | PASS |
| `gemma-4-E4B-it-qat-JANG_4M` | `jang_affine` | 10.05 GiB | 6 | text, vision, audio | PASS |
| `gemma-4-12B-it-qat-JANG_4M` | `jang_affine` | 9.44 GiB | 10 | text, vision, audio | PASS |
| `gemma-4-26B-A4B-it-qat-JANG_4M` | `jang_affine` | 17.22 GiB | 18 | text, vision | PASS |
| `gemma-4-31B-it-qat-JANG_4M` | `jang_affine` | 24.69 GiB | 25 | text, vision | PASS |
| `gemma-4-E2B-it-qat-MXFP4` | `mxfp4` | 3.73 GiB | 4 | text, vision, audio | PASS |
| `gemma-4-E4B-it-qat-MXFP4` | `mxfp4` | 5.50 GiB | 5 | text, vision, audio | PASS |
| `gemma-4-12B-it-qat-MXFP4` | `mxfp4` | 7.37 GiB | 7 | text, vision, audio | PASS |
| `gemma-4-26B-A4B-it-qat-MXFP4` | `mxfp4` | 14.56 GiB | 15 | text, vision | PASS |
| `gemma-4-31B-it-qat-MXFP4` | `mxfp4` | 18.19 GiB | 18 | text, vision | PASS |

Common structural checks:

- `config.json`, `jang_config.json`, `model.safetensors.index.json`, tokenizer, processor, chat template, and generation config are present.
- Safetensor shard references resolve with `missing_shards=0`.
- `JANG_4M` metadata declares top-level 8-bit affine with 4-bit MLP/expert overrides.
- `MXFP4` metadata declares `bits=4`, `group_size=32`, `mode=mxfp4`.

## Runtime / Test Evidence

Post-audio-order vMLX fix:

- Root cause found: vMLX normalized all media placeholders before text, rendering Gemma audio prompts as `<|audio|>prompt`.
- Gemma guidance requires image before text, audio after text.
- Fixed runtime extraction so audio placeholders render after text: `prompt<|audio|>`.
- Second root cause found: vMLX batched prefill treated audio-only media requests as text-only because the fast path checked only `pixel_values`. Audio lives in `extra_kwargs` as `input_features` / `input_features_mask`, so the request was incorrectly sent through `language_model` directly and bypassed Gemma4 `Model.get_input_embeddings()`, where audio features are encoded and scattered into token embeddings.
- Fixed the batched generator so audio requests count as media and use the full VLM wrapper path.

vMLX focused regression checks:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m pytest tests/test_jang_loader.py tests/test_mllm.py::TestExtractMultimodalMessages tests/test_multimodal_routing.py -q
```

Result: `50 passed` before and after the audio-order fix.

JANG Gemma converter/template checks:

```sh
cd /Users/eric/jang
PYTHONPATH=jang-tools jang-tools/.venv/bin/python -m pytest jang-tools/tests/test_gemma4_template_patch.py -q
```

Result: `10 passed`.

Syntax check:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m py_compile \
  vmlx_engine/models/mllm.py \
  vmlx_engine/engine/batched.py \
  vmlx_engine/mllm_scheduler.py \
  vmlx_engine/mllm_batch_generator.py \
  vmlx_engine/utils/jang_loader.py \
  vmlx_engine/server.py
```

Result: PASS.

## Live API Proof

Runtime command shape:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m vmlx_engine.server \
  --model <bundle> \
  --host 127.0.0.1 \
  --port 8772 \
  --mllm \
  --default-temperature 0 \
  --default-top-p 1 \
  --default-enable-thinking false
```

### `gemma-4-E4B-it-qat-JANG_4M`

Load path:

- `Loading JANG VL model`
- `JANG v2 VLM detected`
- `JANG v2 VLM loaded in 4.5s`
- Runtime cache layout: 24 Gemma4 layers with rotating/full KV pattern.

Live results:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Text | `Answer only the number: 2+3=` | `5` | PASS |
| Vision | 96x48 red/blue PNG | `Red, Blue` | PASS |
| Speech transcription | `say "hello world"` WAV, text before audio | `Hello world` | PASS |

Audio note: the earlier tone-classification probe was removed as a release gate because direct `mlx_vlm.generate` also labels the synthetic tone as `Silence`. Speech transcription is the stronger gate for this runtime path.

### `gemma-4-E2B-it-qat-MXFP4`

Load path:

- `Loading JANG VL model`
- `JANG v2 VLM detected`
- `JANG v2 VLM loaded in 1.1s`
- Runtime cache layout: 15 Gemma4 layers.

Live results:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Text | `Answer only the number: 2+3=` | `5` | PASS |
| Vision | 96x48 red/blue PNG | `Red, Blue` | PASS |
| Audio tone | 440 Hz WAV | `Silence` | FAIL semantic |
| Audio loud tone | 880 Hz WAV | `Silence` | FAIL semantic |

Audio note: prompt token counts increased to 48 and 103 on audio requests and media diagnostics showed `input_audio`, so audio was not dropped. The semantic classification was wrong.

Post batched-audio-wrapper fix:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Speech transcription | `say "hello world"` WAV, text before audio | `Hello world` | PASS |
| Speech classification | same speech WAV, text before audio | `Speech` | PASS |

Rebuilt after BF16 passthrough fix:

| Tensor | Dtype / shape |
| --- | --- |
| `vision_tower.patch_embedder.input_proj.weight` | `BF16`, `[768, 768]` |
| `embed_audio.embedding_projection.weight` | `BF16`, `[2560, 1536]` |

Old broken bundle retained as `.broken-20260609-131632`; canonical local path now points at the rebuilt fixed bundle.

### `gemma-4-E2B-it-qat-JANG_4M`

Load path:

- `Loading JANG VL model`
- `JANG v2 VLM detected`
- `JANG v2 VLM loaded in 1.0s`
- Runtime cache layout: 15 Gemma4 layers.

Live audio recheck:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Audio tone | 440 Hz WAV | `Silence` | FAIL semantic |
| Audio loud tone | 880 Hz WAV | `Silence` | FAIL semantic |
| Speech transcription | `say "hello world"` WAV | claims no audio | FAIL semantic |
| Speech classification | same speech WAV | `Speech` | PASS coarse |

Direct processor inspection showed `input_features` are finite for the tone fixtures despite the Gemma4 audio feature extractor warnings. The audio path is wired, but audio coherency is only partially proven.

Post audio-order fix:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Audio tone classification | 440 Hz WAV, text before audio | `Speech` | FAIL semantic |
| Speech classification | `say "hello world"` WAV, text before audio | `Speech` | PASS coarse |
| Speech transcription | same speech WAV, text before audio | claims no audio | FAIL semantic |

The fix corrected the rendered chat-template order from:

```text
<|audio|>Listen and classify.
```

to:

```text
Listen and classify.<|audio|>
```

This improves the prompt contract and preserves coarse speech detection, but it does not prove full audio coherency or transcription.

Direct `mlx_vlm.generate` comparison after prompt-order fix:

| Probe | Output |
| --- | --- |
| Speech classification | `Speech` |
| Speech transcription | `Hello world` |
| Tone classification | `Silence` |

This proved the JANG bundle and underlying MLX model could transcribe audio, while the vMLX server still failed. The remaining vMLX mismatch was the batched text-only fast path bypassing the VLM wrapper for audio requests.

Post batched-audio-wrapper fix:

| Probe | Prompt / fixture | Output | Status |
| --- | --- | --- | --- |
| Speech transcription | `say "hello world"` WAV, text before audio | `Hello world` | PASS |
| Speech classification | same speech WAV, text before audio | `Speech` | PASS |
| Tone classification | 440 Hz WAV, text before audio | `Silence` | FAIL semantic, matches direct `mlx_vlm.generate` |

Rebuilt after BF16 passthrough fix:

| Tensor | Dtype / shape |
| --- | --- |
| `vision_tower.patch_embedder.input_proj.weight` | `BF16`, `[768, 768]` |
| `embed_audio.embedding_projection.weight` | `BF16`, `[1536, 1536]` |

Old broken bundle retained as `.broken-20260609-131356`; canonical local path now points at the rebuilt fixed bundle.

### `gemma-4-12B-it-qat-JANG_4M`

Rebuilt after BF16 passthrough fix and promoted to the canonical local path. Old broken bundle retained as `.broken-20260609-132139`.

Fixed structural checks:

| Tensor | Dtype / shape |
| --- | --- |
| `vision_embedder.patch_dense.weight` | `BF16`, `[3840, 6912]` |
| `embed_audio.embedding_projection.weight` | `BF16`, `[3840, 640]` |
| `language_model.model.embed_tokens.weight` | `F16`, `[262144, 3840]` |

Runtime status:

| Path | Probe | Output | Status |
| --- | --- | --- | --- |
| vMLX `load_jang_model` text-only | Chat-template `2+3` | `<|channel>thought...<channel|>5` | PASS |
| Standard `mlx_vlm` VLM load | `gemma4_unified` | `No module named 'mlx_vlm.models.gemma4_unified'` | BLOCKED |

Implementation note: `vmlx_engine/utils/jang_loader.py` now promotes `gemma4_unified` / `gemma4_unified_text` to the Gemma4 text graph only for the text loader path. This does not claim 12B multimodal readiness. Vision/audio for 12B require a real `gemma4_unified` VLM implementation.

### `gemma-4-26B-A4B-it-qat-JANG_4M`

Initial failure:

- Direct JANG and vMLX server text/vision were incoherent.
- Root cause 1: Gemma4 MoE expert tensors were written as `experts.gate_up_proj` / `experts.down_proj`, while the MLX runtime model expects `experts.switch_glu.gate_proj` / `up_proj` / `down_proj`. The expert modules stayed random/unloaded, producing token soup.
- Root cause 2: multimodal passthrough tensors from the BF16 source, including `vision_tower.patch_embedder.input_proj.weight`, were forced to fp16 in the converter. The 26B source vision path passes `red, blue`; the fp16-passthrough converted bundle looped `0/1/...`.

Fixes applied:

- `vmlx_engine/utils/jang_loader.py` now splits/remaps existing Gemma4 `experts.gate_up_proj` and `experts.down_proj` quantized triplets into `experts.switch_glu.*` so old staged bundles can load experts correctly.
- `jang_tools/convert_gemma4_jang.py` now emits split `experts.switch_glu.gate_proj`, `up_proj`, and `down_proj` keys directly for new JANG_4M bundles.
- `jang_tools/convert_gemma4_jang.py` and `convert_gemma4_mxfp.py` now write multimodal passthrough tensors with `mx.save_safetensors` and preserve those tensors as BF16 instead of forcing fp16.
- Rebuilt `/Volumes/EricsLLMDrive/jangq-ai/gemma-4-26B-A4B-it-qat-JANG_4M`; old broken bundle retained as `.broken-20260609-130223`.

Fixed structural checks:

| Tensor | Dtype / shape |
| --- | --- |
| `language_model.model.layers.0.experts.switch_glu.gate_proj.weight` | `U32`, `[128, 704, 352]` |
| `language_model.model.layers.0.experts.switch_glu.up_proj.weight` | `U32`, `[128, 704, 352]` |
| `language_model.model.layers.0.experts.switch_glu.down_proj.weight` | `U32`, `[128, 2816, 88]` |
| `vision_tower.patch_embedder.input_proj.weight` | `BF16`, `[1152, 768]` |

Live results after rebuild:

| Path | Probe | Output | Status |
| --- | --- | --- | --- |
| Direct `mlx_vlm.generate` | Text | `<|channel>thought\n<channel|>5` | PASS |
| Direct `mlx_vlm.generate` | Vision red/blue PNG | `<|channel>thought\n<channel|>red, blue` | PASS |
| vMLX server | Text | `5` | PASS |
| vMLX server | Vision red/blue PNG | `red, blue` | PASS |

### `gemma-4-31B-it-qat-JANG_4M`

Rebuilt with the same BF16 multimodal passthrough fix. This model has no routed expert key split issue, but the old bundle still used fp16 for BF16 source vision passthrough tensors.

Fixed structural checks:

| Tensor | Dtype / shape |
| --- | --- |
| `vision_tower.patch_embedder.input_proj.weight` | `BF16`, `[1152, 768]` |
| `language_model.model.layers.0.mlp.gate_proj.weight` | `U32`, `[21504, 672]` |

Live results after rebuild:

| Path | Probe | Output | Status |
| --- | --- | --- | --- |
| Direct `mlx_vlm.generate` | Text | `<|channel>thought\n<channel|>5` | PASS |
| Direct `mlx_vlm.generate` | Vision red/blue PNG | `<|channel>thought\n<channel|>Red, blue` | PASS |
| vMLX server | Text | `5` | PASS |
| vMLX server | Vision red/blue PNG | `Red, blue` | PASS |

Old broken bundle retained as `.broken-20260609-130722`; canonical local path now points at the rebuilt fixed bundle.

## HF Stage Status

The JANGQ-AI and OsaurusAI stage copies for all five JANG_4M bundles were refreshed from the fixed canonical local bundles, preserving existing README/logo branding files. Old staged model shards were removed before copying replacement shards so stale shard layouts do not remain in the repo folders.

Verified staged artifact checks:

| Stage bundle | Check | Status |
| --- | --- | --- |
| `JANGQ-AI/gemma-4-12B-it-qat-JANG_4M` | `vision_embedder.patch_dense.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |
| `OsaurusAI/gemma-4-12B-it-qat-JANG_4M` | `vision_embedder.patch_dense.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |
| `JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `experts.switch_glu.gate_proj.weight` exists; stale `experts.gate_up_proj.weight` missing | PASS |
| `OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `experts.switch_glu.gate_proj.weight` exists; stale `experts.gate_up_proj.weight` missing | PASS |
| `JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16` | PASS |
| `OsaurusAI/gemma-4-31B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16` | PASS |
| `JANGQ-AI/gemma-4-E2B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |
| `OsaurusAI/gemma-4-E2B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |
| `JANGQ-AI/gemma-4-E4B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |
| `OsaurusAI/gemma-4-E4B-it-qat-JANG_4M` | `vision_tower.patch_embedder.input_proj.weight` is `BF16`; `embed_audio.embedding_projection.weight` is `BF16` | PASS |

Focused post-refresh checks:

```sh
cd /Users/eric/mlx/vllm-mlx
.venv/bin/python -m py_compile vmlx_engine/utils/jang_loader.py
.venv/bin/python -m pytest tests/test_jang_loader.py -q

cd /Users/eric/jang
PYTHONPATH=jang-tools jang-tools/.venv/bin/python -m py_compile \
  jang-tools/jang_tools/convert_gemma4_jang.py \
  jang-tools/jang_tools/convert_gemma4_mxfp.py
PYTHONPATH=jang-tools jang-tools/.venv/bin/python -m pytest \
  jang-tools/tests/test_gemma4_template_patch.py -q
```

Results: `py_compile` PASS, `tests/test_jang_loader.py` `36 passed`, Gemma4 template tests `10 passed`.

## HF Upload Status

Uploaded and remote-verified JANGQ-AI repos:

| Repo | Files | Safetensors shards | Remote bytes | Verification |
| --- | ---: | ---: | ---: | --- |
| `JANGQ-AI/gemma-4-E2B-it-qat-JANG_4M` | 15 | 4 | 7,837,800,841 | PASS |
| `JANGQ-AI/gemma-4-E4B-it-qat-JANG_4M` | 17 | 6 | 10,821,750,107 | PASS |
| `JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` | 29 | 18 | 18,525,443,446 | PASS |
| `JANGQ-AI/gemma-4-31B-it-qat-JANG_4M` | 36 | 25 | 26,548,519,014 | PASS |
| `JANGQ-AI/gemma-4-12B-it-qat-JANG_4M` | 21 | 10 | 10,167,836,440 | PASS |

Remote checks confirmed each JANGQ-AI repo has `README.md`, `config.json`, `jang_config.json`, `model.safetensors.index.json`, tokenizer files, safetensor shards, and `jangq-logo-dark.png`.

Additional remote sanity checks:

- All five uploaded repos declare `weight_format: jang_affine` in both `config.json` and `jang_config.json`.
- `JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` contains `language_model.model.layers.0.experts.switch_glu.gate_proj.weight`.
- `JANGQ-AI/gemma-4-26B-A4B-it-qat-JANG_4M` does not contain stale `language_model.model.layers.0.experts.gate_up_proj.weight`.
- `JANGQ-AI/gemma-4-12B-it-qat-JANG_4M` README includes the text-only current vMLX runtime caveat for `gemma4_unified`.

Upload transport note: the default HF/Xet path stalled on the 26B upload at zero completed pre-uploads. Retrying with `HF_HUB_DISABLE_XET=1` and one worker completed successfully; the same non-Xet path was used for 31B and 12B.

Uploaded and remote-verified OsaurusAI repos:

| Repo | Files | Safetensors shards | Remote bytes | Verification |
| --- | ---: | ---: | ---: | --- |
| `OsaurusAI/gemma-4-E2B-it-qat-JANG_4M` | 15 | 4 | 7,837,801,248 | PASS |
| `OsaurusAI/gemma-4-E4B-it-qat-JANG_4M` | 17 | 6 | 10,821,750,514 | PASS |
| `OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M` | 29 | 18 | 18,525,443,853 | PASS |
| `OsaurusAI/gemma-4-31B-it-qat-JANG_4M` | 36 | 25 | 26,548,519,421 | PASS |
| `OsaurusAI/gemma-4-12B-it-qat-JANG_4M` | 21 | 10 | 10,167,836,847 | PASS |

Remote checks confirmed each OsaurusAI repo has `README.md`, `config.json`, `jang_config.json`, `model.safetensors.index.json`, tokenizer files, safetensor shards, and `osaurus-x-banner.png`.

Additional remote sanity checks:

- All five uploaded OsaurusAI repos declare `weight_format: jang_affine` in both `config.json` and `jang_config.json`.
- `OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M` contains `language_model.model.layers.0.experts.switch_glu.gate_proj.weight`.
- `OsaurusAI/gemma-4-26B-A4B-it-qat-JANG_4M` does not contain stale `language_model.model.layers.0.experts.gate_up_proj.weight`.
- `OsaurusAI/gemma-4-12B-it-qat-JANG_4M` README includes the text-only current vMLX runtime caveat for `gemma4_unified`.
- OsaurusAI uploads used the Keychain token for HF identity `Osaurus-AI`, verified with access to `OsaurusAI`; raw token values were not printed or stored in the repo.

## Current Classification

`JANG_4M`: built and structurally correct for all five Gemma QAT sizes. Live text/vision proof exists for E2B, E4B, rebuilt 26B, and rebuilt 31B. Audio prompt ordering is fixed, audio now uses the full VLM wrapper in batched vMLX, and E2B/E4B JANG_4M pass speech transcription. 12B has text-only vMLX proof after `gemma4_unified` text promotion, but multimodal remains blocked on missing `mlx_vlm.models.gemma4_unified`. Tone classification remains a weak probe and returns `Silence`, matching direct `mlx_vlm.generate`.

`MXFP4`: built and structurally correct for all five Gemma QAT sizes. Live text/vision proof exists for E2B. After the same vMLX batched-audio fix, E2B MXFP4 passes speech transcription/classification.

Do not claim all Gemma QAT bundles are fully audio-coherent across every task yet. Current live proof supports text, vision, and speech-audio coherence on E2B/E4B JANG_4M, text/vision coherence on rebuilt 26B/31B JANG_4M, text-only coherence on rebuilt 12B JANG_4M, and speech-audio coherence on E2B MXFP4. 12B JANG_4M remains a separate `gemma4_unified` early-fusion runtime lane and must not be claimed multimodal-ready through the standard `mlx_vlm` Gemma4 VLM graph.
