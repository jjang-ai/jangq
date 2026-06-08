# Nemotron 3 Ultra Cache Block Contract

Date: 2026-06-04

Artifact:
`/Volumes/EricsLLMDrive/jangq-ai/NVIDIA-Nemotron-3-Ultra-550B-A55B-JANGTQ_1L`

This is the cache-specific handoff for vMLX Python and Swift engine work.

## Topology

The model has 108 layers but only 60 cache-bearing layers:

- 48 Mamba layers: recurrent companion state
- 12 attention layers: normal K/V cache
- 48 MoE layers: no cache entry

Cache entry order follows `layers_block_type` while skipping MoE layers. Do not
index caches by raw layer number.

## Cache Components

Treat a reusable prefix as this tuple:

```text
accepted_prefix =
  token_ids
  tokenizer/template identity
  parser mode
  parser streaming state version
  model revision/path
  quant profile
  mtp mode
  modality/media policy
  attention_kv_blocks[12]
  mamba_companion_states[48]
```

Attention K/V and Mamba state are peers. K/V alone is not a valid hit.

## Attention K/V Component

The attention component has 12 layers:

- q heads: 64
- kv heads: 2
- head dim: 128
- no sliding window
- no MLA
- no rotating KV

TurboQuant KV encoding applies only here. Policy salt must include:

- key bits
- value bits
- group size
- max KV length
- raw prompt-boundary policy
- cache codec version
- model path or immutable model revision
- quant profile
- parser/template mode
- MTP mode

If raw prompt-boundary K/V is used by vMLX, persist that raw boundary before
generated-token TQ KV blocks. Do not silently mix raw and TQ blocks without a
salt change.

Recommended storage record per attention layer:

- layer cache ordinal `0..59`, not raw layer index
- raw layer index for diagnostics
- token start/end
- block parent hash
- key/value codec and bit settings
- raw-boundary flag
- dtype/shape checksum
- model/cache salt

## Mamba Companion Component

Each Mamba layer stores an `ArraysCache(size=2)` equivalent:

- slot 0: convolution rolling state, `[batch, conv_kernel - 1, conv_dim]`
- slot 1: SSM recurrent state, implementation-defined MLX `ssm_update` shape

Ultra values:

- `conv_kernel=4`
- Mamba `n_groups=8`
- `ssm_state_size=128`
- local Mamba intermediate width `256 * 64 = 16384`
- `conv_dim=18432`
- `time_step_limit=[0.0, 1e20]`

Do not derive this from MoE `n_group=1`; that is a different field.

Recommended storage record per Mamba layer:

- layer cache ordinal `0..59`, not raw layer index
- raw layer index for diagnostics
- token boundary exactly matching attention K/V
- conv state bytes or tensor reference for slot `0`
- SSM recurrent state bytes or tensor reference for slot `1`
- `lengths` / offset state if present
- dtype/shape checksum for both slots
- model/cache salt

This companion record is mandatory for a cache hit. Do not store it as a
best-effort sidecar that can be silently absent.

## Accept / Reject Rules

Valid hit:

- token prefix matches exactly
- prompt template and parser mode match
- attention K/V block chain verifies
- all 48 Mamba companion states exist for the same prefix boundary
- TQ KV codec salt matches
- no draft/speculative token state is included

Partial hit:

- attention K/V exists but Mamba state missing
- Mamba state exists but K/V missing
- either component stops at a different prefix length

A partial hit must downgrade to the longest complete prefix or rederive the
missing component before decode. It must not decode with stale or absent Mamba
state.

## Async Rederive

Async rederive is allowed only for verified prompt prefixes:

1. Identify the longest complete accepted prefix.
2. Re-prefill from that prefix to the desired prompt boundary.
3. Capture both updated attention K/V and all Mamba companion states.
4. Store the new complete tuple atomically.
5. Mark later requests eligible only after both components are present.

Do not capture after rejected MTP/draft tokens. This bundle drops MTP; if a
future bundle preserves it, draft K/V and draft SSM state must be private until
acceptance.

## Parser And Streaming State Salt

Prefix cache validity depends on parser/template state, not only token ids.
Include these in the cache salt or request-state key:

- chat template revision / tokenizer revision
- `enable_thinking`
- `medium_effort`
- `force_nonempty_content`
- reasoning parser alias (`deepseek_r1`, `nemotron_v3`, `nemotron_3`, etc.)
- tool parser alias (`nemotron`, `xml_function`, `qwen3_coder_xml`, etc.)
- streaming parser state version
- whether a thinking span is currently open/truncated
- whether a tool-call span is currently open/truncated

Cache entries should be captured at clean prompt/assistant boundaries. Do not
reuse a prefix captured in the middle of an open `<think>`, `<tool_call>`, or
`<tool_response>` span unless the parser state object is captured and restored
as part of the same accepted prefix.

## Modality / VL Boundary

This JANGTQ_1L artifact is text-only. Cache policy must explicitly record that
no media processors are active:

- no vision encoder state
- no audio encoder state
- no image/audio projector cache
- no media token expansion cache
- no processor config hash

Future VL/audio Nemotron bundles must add those media states to the accepted
prefix tuple. This bundle must not open a media cache lane just because
Nemotron Omni code exists elsewhere in the repo.

## Required Proof Rows

Before vMLX calls cache support working for this artifact, collect logs for:

- cold prefill with 60 cache entries allocated
- first request stores 12 attention K/V block chains plus 48 Mamba states
- second identical prompt hits both components at the same token boundary
- artificial Mamba-state miss rejects or rederives instead of using K/V only
- TurboQuant KV codec logs show only 12 attention K/V layers encoded
- no media cache gates are opened because this artifact is text-only
- parser salt mismatch rejects cache reuse
- open/truncated reasoning/tool parser state is not cached as a clean prefix
