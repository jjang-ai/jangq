# Muse Glimmer 30B quant and runtime handoff

Status: quant artifacts pass structural and dequantization checks. Generation,
reasoning/tool streaming, image grounding, video grounding, and cache reuse are
deferred to the vMLX Swift/Python runtime owners and remain unverified.

## Pinned local artifacts

| Artifact | Revision | Local path | Relationship |
|---|---|---|---|
| BF16 base | `f84ecc3a0ea984a4c04542a84269e3d065350a6e` | `~/models/meta-models/Muse-Glimmer-30B` | Quant source |
| Five-layer assistant | `2c86316d689027b91123638739743fef1d425233` | `~/models/meta-models/Muse-Glimmer-30B-assistant` | Separate future DFlash artifact; not part of either quant |
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

All 809 tensors in those namespaces are FP16 passthrough in both JANG bundles.
This deliberately avoids a text-only calibration damaging image or video
quality. The VL bundle maps the untied bare source `lm_head.weight` to
`language_model.lm_head.{weight,scales,biases}`; that wrapped head is correct
for this VL artifact and must not be generalized to text-only models.

## Quant results and proof level

| Bundle | Effective bits | Indexed size | Quantized modules | Dequant rel-L1: layer-0 K | Dequant rel-L1: layer-0 MLP down |
|---|---:|---:|---:|---:|---:|
| JANG_4M | 4.63 | 20.20 GB | 418 | 0.005954 | 0.098860 |
| JANG_6M | 6.31 | 25.67 GB | 418 | 0.005954 | 0.023647 |

The verifier also checks finalized shard names, index/header equality, source
revision, exact processor/template/generation sidecars, capability stamps,
FP16 vision passthrough, and absence of assistant/MTP tensors.

Run either gate from the repository root:

```bash
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

These are artifact gates, not generation proof. Do not upload or label either
bundle runtime-ready until the real target runtime produces coherent text,
grounded image output, grounded video output, reasoning, and a tool round trip.

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
| imatrix | not applied | Fixed `JANG_4M`/`JANG_6M` allocation is tier-based and does not consume imatrix scores. The converter now rejects an ignored Glimmer imatrix. |
| AWQ | not applied | The generic collector loads text-only `mlx_lm`, which cannot run this VLM and cannot cover media-conditioned language activations. The converter now rejects this unsafe path. |

For a calibrated follow-up, use the official Muse Glimmer model path and collect
statistics after media token replacement from a balanced text/image/video set.
The calibration artifact must be keyed by source revision, processor revision,
modality mix, prompt-template digest, and sequence lengths. Add blockwise dense
GPTQ or a correctly folded AWQ implementation first; then compare it against
these baselines on all three modalities. A language-only win is insufficient.

## Generation, reasoning, and tools

The shipped `generation_config.json` is authoritative: greedy decoding
(`do_sample=false`), BOS 200000, EOS `[200001, 200008]`, pad 200018, maximum
length 131072. The model card's temperature/top-p/top-k suggestions are
recommendations, not bundle defaults; do not stamp them over the shipped file.

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
