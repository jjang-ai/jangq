# Muse Glimmer 30B quant and runtime handoff

Status: all three retained quant artifacts pass structural and dequantization checks. JANG_2D
also has coherent text-generation proof in the real Osaurus/vmlx-swift target
runtime. Image grounding, video grounding, ATEM tool behavior, multi-turn behavior,
and cache reuse remain unverified.

## Pinned local artifacts

| Artifact | Revision | Local path | Relationship |
|---|---|---|---|
| BF16 base | `f84ecc3a0ea984a4c04542a84269e3d065350a6e` | `~/models/meta-models/Muse-Glimmer-30B` | Quant source |
| Five-layer assistant | `2c86316d689027b91123638739743fef1d425233` | `~/models/meta-models/Muse-Glimmer-30B-assistant` | Separate future DFlash artifact; not part of any base quant |
| JANG_2D | base revision above | `~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_2D` | Dense-safe sub-16-GB base candidate only |
| JANG_4M | base revision above | `~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M` | Base model only |
| JANG_6M | base revision above | `~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M` | Base model only |

Do not merge, rename, or copy assistant tensors into the base bundle. The
assistant is five decoder layers (`layers.0` through `layers.4`) plus its own
encoder/norm. A later DFlash implementation must load it as a separately keyed
artifact and must include the assistant revision in its cache/runtime identity.

## Source contract

Muse Glimmer is a dense multimodal VLM, not an MoE model. The verified source
contains 1,436 BF16 tensors: a 52-layer, 6,656-hidden text decoder and a
50-layer ViT-G/14 frontend. The decoder uses 32 query heads, two KV heads,
128-wide heads, a 131,072-token maximum, and this exact repeated schedule:

```text
sliding, sliding, sliding, full
```

There are 39 sliding layers with a 2,048-token window and 13 full-attention
layers at indices `3, 7, ..., 51`. Sliding layers use RoPE theta 500,000; full
layers have `layer_rope_theta=0` and therefore use NoPE. Preserve the centered
RMSNorm behavior, Q/K normalization and `qk_scale_factor=3.87`, attention gate,
`output_multiplier=0.19611613513818404`, and final logit soft cap 20.

The multimodal source namespaces are:

```text
model.vision_tower.*
model.vision_adapter.*
model.vision_projection.*
```

All 809 tensors in those namespaces are FP16 passthrough in all JANG bundles.
This deliberately avoids a text-only calibration damaging image or video
quality. The VL bundle maps the untied bare source `lm_head.weight` to
`language_model.lm_head.{weight,scales,biases}`; that wrapped head is correct
for this VL artifact and must not be generalized to text-only models.

## Quant results and proof level

| Bundle | Effective bits | Indexed size | Quantized modules | Dequant rel-L1: layer-0 K | Dequant rel-L1: layer-0 MLP down |
|---|---:|---:|---:|---:|---:|
| JANG_2D | 2.96 | 14.80 GiB / 15.89 GB | 418 | 0.100382 | 0.206680 |
| JANG_4M | 4.63 | 20.20 GB | 418 | 0.005954 | 0.098860 |
| JANG_6M | 6.31 | 25.67 GB | 418 | 0.005954 | 0.023647 |

The verifier also checks finalized shard names, index/header equality, source
revision, exact processor/template sidecars, deployment generation metadata,
capability stamps, FP16 vision passthrough, and absence of assistant/MTP tensors.

Run any gate from the repository root:

```bash
PYTHONPATH=jang-tools uv run --no-project \
  --with mlx --with numpy --with safetensors --with tqdm \
  python jang-tools/scripts/verify_muse_glimmer_artifact.py \
  ~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_2D \
  --profile JANG_2D --dequant

PYTHONPATH=jang-tools uv run --no-project \
  --with mlx --with numpy --with safetensors --with tqdm \
  python jang-tools/scripts/verify_muse_glimmer_artifact.py \
  ~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_4M \
  --profile JANG_4M --dequant

PYTHONPATH=jang-tools uv run --no-project \
  --with mlx --with numpy --with safetensors --with tqdm \
  python jang-tools/scripts/verify_muse_glimmer_artifact.py \
  ~/models/JANGQ-AI/Muse-Glimmer-30B-JANG_6M \
  --profile JANG_6M --dequant
```

These are artifact gates, not generation proof. JANG_2D separately closed the
minimum pre-upload text gate described below. Do not label any bundle fully
runtime-ready until the target runtime also proves grounded image and video
output, ATEM tools, multi-turn behavior, and cache reuse.

## QAT, GPTQ, imatrix, and AWQ

The source is BF16 and exposes no QAT tensors, scale metadata, or QAT checkpoint.
Therefore `source_qat=not_present` is stamped in both bundles. A PTQ conversion
must never be relabeled as QAT. The official dynamic/K-quant GGUF releases are
also PTQ references, not proof of a reusable QAT source.

Current method support is deliberately conservative:

| Method | Current status | Reason |
|---|---|---|
| QAT | unavailable | No QAT source checkpoint exists. Producing one requires training, not a converter flag. |
| GPTQ/Hessian | not applied | The current generic JANG GPTQ path is limited to expert/3-D MoE tensors. Glimmer is dense; claiming GPTQ would be false. |
| imatrix | not applied | Fixed `JANG_2D`/`JANG_4M`/`JANG_6M` allocation is tier-based and does not consume imatrix scores. The converter rejects an ignored Glimmer imatrix. |
| AWQ | not applied | The generic collector loads text-only `mlx_lm`, which cannot run this VLM and cannot cover media-conditioned language activations. The converter now rejects this unsafe path. |

For a calibrated follow-up, use the official Muse Glimmer model path and collect
statistics after media token replacement from a balanced text/image/video set.
The calibration artifact must be keyed by source revision, processor revision,
modality mix, prompt-template digest, and sequence lengths. Add blockwise dense
GPTQ or a correctly folded AWQ implementation first; then compare it against
these baselines on all three modalities. A language-only win is insufficient.

## Generation, reasoning, and tools

Deployment sampling follows the pinned model card's explicit best-practice
recommendation: `do_sample=true`, `temperature=1.0`, `top_p=0.95`, and
`top_k=64`. The source token contract remains BOS 200000, EOS
`[200001, 200008]`, pad 200018, and maximum length 131072. These values must be
identical in `generation_config.json`, `jang_config.chat.sampling_defaults`,
and `jang_config.chat.generation_defaults`.

The native chat-template control is `reasoning_strength`, with values `low`,
`medium`, `high`, and `xhigh`; omission means `high`. Do not translate this to a
generic `enable_thinking` toggle. Reasoning is emitted in an assistant-to-self
channel:

```text
<|start|>assistant to=self<|message|>...<|eom|>
```

Visible content uses `to=user`. A streaming parser must accept fragmented
prefixes and suffixes, retain an incomplete control token between chunks, route
only the complete self-channel payload to `reasoning_content`, and never leak
channel markers into visible content.

Tools use ATEM. A call is carried in an assistant turn addressed to the tool:

```text
<atem:function_calls>
<atem:invoke name="namespace.function">
<atem:parameter name="argument">value</atem:parameter>
</atem:invoke>
</atem:function_calls>
```

Parameter text is not strict XML: strings may contain spaces/newlines, while
lists and objects use JSON. Parse incrementally, wait for the closing ATEM block,
validate the called name against the request's tool schema, and then emit one
structured tool call. Tool results return as `<|start|>tool NAME<|message|>` with
a `<tool_output name="NAME">` body. The required live matrix is: no tools,
single call, fragmented call, multiple calls, tool result, reasoning then tool,
tool then final answer, invalid/unavailable function, and no raw marker leakage.

## Image and video handoff

Image placeholders expand to image-start, patch tokens, and image-end. Video
uses video-start/end, per-frame timestamps, video patch tokens, and frame
separators. The shipped video processor samples at up to 2 fps, caps at 96
frames, uses temporal patch size 2, and caps each frame at 144 tokens. Do not
implement video as repeated independent images: preserve temporal patch order,
frame timestamps, separators, and the shared vision adapter/projection path.

## Prefix, partial-block, and suffix-prefill cache contract

Allocate one cache per text layer: rotating KV for the 39 sliding layers and
unbounded KV for the 13 full layers. Never apply one uniform cache type to all
52 layers.

For a repeated or extended prompt:

1. Match the longest rendered-token prefix under the same base revision,
   quant profile, tokenizer/template digest, media digest, and cache policy.
2. Restore all complete cache blocks before the match boundary.
3. If the last stored block is only partially reusable, slice every layer's K/V
   state to the exact matched token count. Do not restore the unmatched suffix.
4. Prefill only tokens after the restored boundary. Do not recompute matched
   tokens and do not skip new suffix tokens.
5. Preserve the absolute position/offset when trimming rotating caches. A
   2,048-token sliding payload does not mean the conversation position resets.
6. Store only after all 52 layer caches represent the same committed boundary;
   a half-written mixed full/SWA cache is invalid.

Media bytes/frames and the fully rendered prompt must affect identity. Changes
to reasoning strength or tool schemas need no extra salt if they already change
rendered tokens; otherwise they require an explicit salt. Compare both paths so
an unnecessary salt does not fragment identical prefixes. Parser state itself
is not reusable KV state: on restore, begin parsing fresh at the new generated
suffix and never replay a prior partial reasoning/ATEM fragment.

The five-layer assistant is excluded from this contract. When DFlash is added
later, assistant revision, mode, depth, and acceptance policy must join the cache
identity, and cache commits must remain base-model authoritative.

## Publication handoff

`JANG_2D` has 15,890,602,696 indexed weight bytes and remains below 16 GB
decimal as a complete bundle while retaining
all 809 vision tensors in FP16. Its language
policy is 4-bit for all 260 gated-attention modules and the untied lm head,
3-bit for token embeddings and dense MLP gate/down projections, and 2-bit for
dense MLP up projections. This follows the asymmetric protection principle of
Muse's published K-Quant releases but uses MLX-native affine weights; it is not
byte- or algorithm-equivalent to GGUF K-Quant.

The mandatory pre-upload text gate closed on 2026-08-10 against the exact local
JANG_2D files. Osaurus/vmlx-swift shape-walked 418 per-module quantization
overrides. A `/v1/chat/completions` request containing `Reply with exactly: FOUR`
and no sampling or reasoning override returned visible `FOUR`, a separate
non-empty `reasoning_content`, `finish_reason=stop`, and 36.87 tokens/s. The
running process had all 21 local shard paths open. This proves coherent text,
the native default reasoning path, reasoning-channel separation, and EOS for
that request; it does not prove media, tools, multi-turn, or cache behavior.

The staged targets are `OsaurusAI/Muse-Glimmer-30B-JANG_4M` at revision
`24a68502d68554cd8b596be1b7703d16d7f8eb49` and
`OsaurusAI/Muse-Glimmer-30B-JANG_6M` at revision
`dd625c14cea6b63b6b558b262bbce8cf53a83afc`. Both repositories are private and
contain all required model-card, license, policy, banner, config, tokenizer,
processor, template, index, and shard files. Keep them private until the
mandatory target-runtime coherence gate closes.

Byte-identical copies also live on `erics-m5-max.local` under
`/Volumes/EricsLLMDrive/jangq-ai/`. They were streamed over the direct
Thunderbolt link using a bridge-bound HTTP server. Sorted SHA-256 manifests
match for all 44 files in 4M and all 49 files in 6M; local/remote byte totals
are 21,721,789,553 and 27,595,790,919 respectively. The temporary HTTP listener
was stopped after verification.

After vmlx-swift proves coherent text generation, run and record image, video,
reasoning-stream, ATEM tool-round-trip, and cache-reuse rows before publishing.
When the runtime matrix passes, change visibility deliberately and verify the
public revision, file inventory, config/README bytes, and a fresh download.
