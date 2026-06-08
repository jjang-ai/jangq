# Nemotron 3 Ultra Long Coherence And VL Boundary Proof

Date: 2026-06-04

Artifact:
`/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

This note records the bounded long-decode smoke test and the media/VL boundary
for vMLX agents. It is intentionally separate from source conversion proof:
load success is not the same thing as coherent chat/runtime behavior.

## Probe Command

```sh
PYTHONPATH=jang-tools JANGTQ_WIRED_LIMIT_GB=105 \
  jang-tools/.venv/bin/python \
  jang-tools/examples/nemotron_ultra/long_decode_coherence_probe.py \
  --rows full \
  --max-tokens 96 \
  --sampler greedy \
  --out docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json
```

The probe loads the real 98G bundle with `skip_params_eval=True`, so the first
row includes cold JIT TTFT. Warm request rows are the useful steady-state
decode rows.

## Result Summary

Log:
`docs/runtime/logs/2026-06-04-nemotron-ultra-long-coherence-greedy-probe.json`

| Row | Result | Decode | Notes |
| --- | --- | --- | --- |
| `factual_japan` | PARTIAL | `8.178 tok/s` | Correct `Tokyo`/`Japan`, EOS reached, but visible `</think>` leak and high 4-gram repetition. |
| `arithmetic_brief` | PARTIAL | `8.136 tok/s` | Correct `42`, EOS reached, but visible `</think>` leak and repeated answer text. |
| `reasoning_apples` | PARTIAL | `8.133 tok/s` | Reasoning reaches `5`, but no EOS within 96 tokens and visible `</think>` remains in decoded text. |

Interpretation:

- The runtime is not token salad. It can preserve simple factual/arithmetic
  content across longer decode.
- Coherence is still partial because no-thinking rows leak `</think>` and
  repeat the visible answer text.
- Thinking output should be parsed into reasoning/content fields before visible
  chat display. The raw decoded text is not yet acceptable as final assistant
  content.
- The speed row is consistent with the current BF16 activation-retention path:
  about `8.1 tok/s` on warm decode.

## Acceptance Rules

Do not call this production-coherent until these rows pass:

- no-thinking row: expected answer present, EOS reached, no visible
  `<think>`/`</think>` markers, low repetition
- thinking row: reasoning captured separately, final content captured after
  `</think>`, truncated reasoning explicitly marked when max tokens cut it off
- XML tool row: `<tool_call><function=...>` parsed into structured tool calls
  with no marker spill into visible text
- long decode row: at least one 256-token budget row without runaway repeated
  phrases or parser-state drift
- streaming row: incremental chunks preserve parser state across chunk
  boundaries and reset cleanly between requests

The current greedy probe does not satisfy these gates. It is a diagnostic row,
not a release row.

## VL / Media Boundary

This JANGTQ_1L artifact is text-only.

Source-file check:

- present: `config.json`, `tokenizer_config.json`,
  `model.safetensors.index.json`
- absent: `processor_config.json`, `preprocessor_config.json`

Source tensor-key check:

```sh
jq -r '.weight_map | keys[]' \
  /Volumes/EricsLLMDrive/sources/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4/model.safetensors.index.json |
  rg -i 'vision|visual|image|audio|speech|encoder|projector|mm_projector|multi_modal|multimodal|vl|video'
```

Observed result: no matching tensor keys.

vMLX policy:

- reject image/audio/video inputs for this artifact, or route them to a
  different multimodal bundle
- do not allocate a vision encoder cache lane
- do not allocate an audio encoder cache lane
- do not add media token expansion cache entries
- do not salt this artifact with a fake processor hash

Future Nemotron Omni/VL/audio bundles need a different accepted-prefix tuple
that includes processor config identity, media expansion tokens, media encoder
state, projector state, and any pre-encoded audio/image/video cache state.
Those states are not optional sidecars; they are cache correctness inputs just
like attention K/V and Mamba companion state.

## Runtime Handoff

For this text-only artifact, the accepted prefix is complete only when all of
these match:

- token ids
- tokenizer and chat template revision
- parser/template mode and streaming parser state
- model path or immutable revision
- quant profile
- MTP mode, currently disabled
- modality policy, currently `text`
- 12 attention K/V cache entries
- 48 Mamba companion states

TurboQuant KV cache can only prove the 12 attention K/V entries. It does not
prove Mamba SSM state, parser state, tool-call state, or media state.
