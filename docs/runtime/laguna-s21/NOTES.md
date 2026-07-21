# Laguna S 2.1 — runtime + quant notes (2026-07-21)

> **STATUS: SHIPPED 2026-07-21.** Both bundles AWQ-based, verified, public:
> `OsaurusAI/Laguna-S-2.1-JANG_2L` + `-JANG_4M` and `JANGQ-AI/` mirrors.
> Local: `~/.mlxstudio/models/JANGQ-AI/Laguna-S-2.1-JANG_{2L,4M}`.
> This file is the single reference for running/porting this model.

## TL;DR — the 10 things you MUST get right to inference Laguna

1. **bf16 activation stream, never fp16** — residual peaks ~942k at L47;
   fp16 NaNs at L46 and the model emits 〈|UNK|〉 forever (§dtype below).
2. **Wired limit** — without it decode halves. Runtime auto-stamps
   `min(bundle*1.2+8GB, 118GB)`.
3. **Per-layer-type prefill masks** — full layers causal, SWA layers banded
   (window 512). One shared mask silently corrupts prompts > 512.
4. **`RotatingKVCache(keep=0)`** — NO attention sinks; keep=4 diverges from
   the reference exactly at the window boundary.
5. **Layer-indexed attention** — heads are 48 (full) / 72 (SWA); RoPE
   differs per layer type (YaRN θ500k partial-rotary 0.5 vs default θ10k
   full rotary). Never read `num_attention_heads` alone.
6. **softplus g_proj gate, branch on width** — S-2.1 per-head (n_heads),
   M.1 per-element (n_heads·head_dim). sigmoid here = residual blow-up.
7. **Router**: sigmoid, bias for SELECTION only, weights from un-biased
   scores, renorm over top-10, routed×2.5 + shared unscaled, fp32 math.
8. **eos [2, 24]** — 24 is end-of-turn; missing it = chat runs on. The
   template emits its own 〈|EOS|〉 (bos 2): never prepend another.
9. **Thinking defaults ON** via `default_chat_template_kwargs` — the
   template's own jinja fallback is OFF; engines that drop the kwargs
   silently no-think. Prompt tails: `<assistant><think>` / `<assistant></think>`.
10. **Per-module quantization bits** from `config.json[quantization]` —
   a single top-level width mis-dequantizes the 2/3/4-bit experts.

## Session changelog (2026-07-21, all in jang-tools working tree)

| Change | File | Why |
|---|---|---|
| NEW converter w/ profiles + gates | `jang_tools/convert_laguna_jang.py` | 2L/3L/4M policies; EOS-consistency, chat round-trip, single-BOS, capabilities verify — all fatal |
| NEW AWQ capture (MLX streaming) | `jang_tools/laguna/awq_capture.py` | runs the REAL LagunaLayer; seq-len must stay ≤ window |
| Per-layer-type masks | `jang_tools/laguna/model.py` | SWA layers attended whole prefix past 512 |
| keep=4 → keep=0 | `jang_tools/laguna/model.py` | no sinks in any shipped Laguna config |
| fp16 → bf16 stream | `jang_tools/laguna/runtime.py` | L46 inf/NaN on 4M (fp16 65504 ceiling) |
| Wired-limit auto-stamp | `jang_tools/laguna/runtime.py` | 14→32.6 tok/s (4M), 31.9→48.3 (2L) |
| `laguna` FAMILY_MAP row | `jang_tools/capabilities.py` | deepseek_r1 + glm47 + think_in_template=True + kv |
| awq_search emit loop | `jang_tools/hy3/awq_search.py` | drive off stat keys, not first_k_dense_replace |
| Tests: policy + chat block | `tests/test_laguna_jang_affine_policy.py` | 7 tests |
| Tests: attention/cache proofs | `tests/test_laguna_hybrid_attention_cache.py` | 5 tests, negative-control-verified |

AWQ artifacts (keep — rebuilds reuse them): `~/models/poolside/
Laguna-S-2.1-awq-{stats,scales,scales-4bit}.safetensors` (+ meta.json).
Source pinned: `~/models/poolside/Laguna-S-2.1/.jang_source_pin.json`
(sha a50e85e). Winners: absmean α=0.25 at BOTH 2-bit (+2.4%) and 4-bit
(+2.9%), 0 inert channels.


Source: `poolside/Laguna-S-2.1` → `~/models/poolside/Laguna-S-2.1` (BF16, 46 shards, ~235 GB).
118B total / ~8B active. Text-only (verified from the tensor index: zero
vision/audio/video tensors — not just the card claim).

## Architecture cheat sheet

| thing | value | trap |
|---|---|---|
| layers | 48; layer 0 dense MLP (inter 12288), 1..47 MoE | dense layer detect = absence of `mlp.gate.weight` |
| attention | 12 full (48 heads) : 36 SWA (72 heads), GQA 8 kv, head_dim 128 | **per-layer head count** — never read `num_attention_heads` alone |
| SWA | window 512, RoPE default theta 10k, partial_rotary 1.0 | NO attention sinks (`swa_attention_sink_enabled` unset) → `RotatingKVCache(keep=0)` |
| full attn | YaRN theta 500k, factor 128, orig 8192, mscale=`attention_factor` 1.4852, partial_rotary **0.5** | rotate only first 64 of 128 dims |
| MoE | 256 routed top-10 + 1 shared (inter 1024 both) | sigmoid router; bias (`e_score_correction_bias`) picks selection, weights come from UN-biased scores; norm_topk_prob renorm; routed×2.5, shared unscaled |
| gating | `gating="per-head"` → g_proj emits n_heads gates, softplus (NOT sigmoid) | M.1 is per-element (n_heads·head_dim); runtime branches on gate width |
| router softcap | `moe_router_logit_softcapping: 0.0` | 0 = disabled, no-op |
| tokens | bos 2, eos [2, 24], pad 9, vocab 100352 | template leads with literal `〈|EOS|〉` (= id 2) — watch for double BOS |
| context | 1,048,576 | YaRN only active on full-attn layers |

## Chat protocol (GLM-style think tags)

- Prompt ends `<assistant><think>` (thinking) or `<assistant></think>` (no-think).
- **Vendor serving default is thinking ON** via
  `generation_config.default_chat_template_kwargs.enable_thinking=true`;
  the template's own jinja fallback is `false`. Engines that drop the kwargs
  silently run no-think. Bundles stamp this in `jang_config.chat`.
- Sampling (vendor, verbatim): temp 1.0, top_p 1.0, min_p 0.0, top_k 20.
  No loop audit has been run on the quantized tail yet — if the 2-bit tail
  loops (cf. hy3 2026-07-10 audit), floor top_p/min_p THEN, with data.
- Parsers: `poolside_v1` (reasoning + tool calls). Tool calls:
  `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.

## Parser mapping (capabilities / vmlx routing)

`FAMILY_MAP["laguna"]` = reasoning `deepseek_r1`, tools `glm47`,
`think_in_template=True`, cache `kv`. Rationale: the template is a GLM
derivative and the tool format is byte-compatible with the glm47 parser
(`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
— name is everything before the first `<arg_key>`, exactly what
GLM4ToolCallParser.swift extracts). The vendor's own vLLM parser name for
both is `poolside_v1` — recorded in `jang_config.chat.vendor_parsers`,
never in capabilities.

## Known warnings (checked, do not chase)

- transformers "incorrect regex pattern … fix_mistral_regex=True" on
  tokenizer load: heuristic detection of a Mistral-family pretokenizer
  regex artifact. poolside shipped this exact tokenizer.json and serves
  with it (vLLM loads the same file the same way), so consistent use
  matches vendor behavior. Do NOT set the flag one-sided — that would make
  our encode diverge from the reference serving stack.
- transformers "Unrecognized keys in rope_parameters": transformers
  version noise; our runtime reads rope_parameters directly.

## Runtime dtype: bf16 stream is MANDATORY (fixed 2026-07-21)

The activation stream must run bf16 (source torch_dtype), not the fp16 the
bundle sidecars are stored in. S-2.1's residual grows ~4 → ~1200 by L40;
at fp16 the L46 MoE intermediate overflows 65504 → inf → NaN logits →
argmax 0 → 〈|UNK|〉 spam. runtime.py casts all fp16 params to bf16 at load.
Diagnostic signature: UNK spam + teacher-forced parity 1.000-pre/0.000-post
= NaN logits, NOT a cache bug. bf16 costs nothing: 2L measured 31.9 tok/s
bf16 vs 30.5 fp16.

## Wired limit (mandatory for real speed)

The runtime auto-stamps `mx.set_wired_limit(min(bundle*1.2 + 8GB, 118GB))`
at load (118 GB ceiling = Ornith-proven on the 128 GB M5 Max). Without it
the GPU working set gets evicted under decode: 4M measured 14.0 tok/s
default vs 32.6 wired; even 2L was silently throttled (31.9 → 48.3).
Benchmark nothing without checking the `[laguna] wired_limit=` line.

Verified numbers (M5 Max 128 GB, greedy, cached, wired):
- JANG_2L 41 GB: **48.3 tok/s**, load 3.3s, parity 1.000/1.000
- JANG_4M 63 GB: **32.6 tok/s**, parity 0.998/0.999

## The attention/cache bugs this port had (fixed 2026-07-21)

Both invisible to short smoke prompts; both bite only past the 512 window:

1. **Shared prefill mask.** `LagunaForCausalLM.__call__` built ONE causal
   mask from `caches[0]` and fed it to every layer. Layer 0 is
   full-attention (plain KVCache) → mlx_lm returns the `"causal"` fast-path
   mask → SWA layers attended the whole prefix on prompts > 512.
   Fix: per-layer-type masks, `create_attention_mask(h, cache_of_own_type,
   window_size=512)` for SWA (returns a bool band mask), mirrors the HF
   reference (`create_causal_mask` / `create_sliding_window_causal_mask`
   mapping) and the mlx_lm gemma3 pattern.
2. **Attention sinks that don't exist.** `make_cache` used
   `RotatingKVCache(keep=4)` (gemma habit). The HF reference only keeps
   sink tokens behind `swa_attention_sink_enabled`, which NO shipped Laguna
   config sets. keep=4 let decode attend the first 4 prompt tokens forever.
   Fix: `keep=0`.

Proofs: `jang-tools/tests/test_laguna_hybrid_attention_cache.py` (5 tests,
tiny random weights, run in ~1s):
- SWA receptive field == window (perturbation outside window changes nothing)
- full-attention control (perturbation far back MUST change logits)
- structural mask dispatch on the hybrid stack (records the mask each layer
  receives; catches the shared-mask bug — verified by negative control)
- cached greedy == no-cache greedy across the window boundary, prompt 4 and
  20, decode past window (catches keep=4 — verified by negative control)

**Run these after ANY change to laguna/model.py or the Swift port math.**

## Verification protocol for a new bundle (in order)

1. `pytest tests/test_laguna_jang_affine_policy.py tests/test_laguna_hybrid_attention_cache.py`
2. Converter self-gates (run automatically at convert time): eos
   config-vs-generation_config match, chat-template think/no-think
   round-trip from the WRITTEN bundle, single-BOS check, capabilities
   verify_directory.
3. Short greedy smoke, no cache (pure T>1 path):
   `python -m jang_tools.laguna.runtime --src <bundle> --prompt 'def fibonacci(n):' --max-new 32 --no-cache`
4. Same WITH cache — text must match no-cache text exactly (greedy).
5. **Long-prompt cache parity** (the one that actually exercises SWA):
   `python docs/runtime/laguna-s21/python_example.py --src <bundle> --parity`
   (~1500-token prompt > window; cached and no-cache tokens must be identical)
6. Chat-mode coherence, thinking on (vendor default):
   `python docs/runtime/laguna-s21/python_example.py --src <bundle> --chat "Explain GQA in two sentences."`
7. Only then: benchmarks / ship. (feedback_verify_runtime_before_ship)

Nuances when interpreting results:
- **Greedy cached vs no-cache token divergence is EXPECTED on quantized
  bundles** — bf16 kernel-order noise flips near-ties (measured: p=0.41 vs
  0.31 flip at step 9 on 2L, both continuations coherent). Exact-match
  parity is only valid on the fp32 tiny-weight unit tests. The real-bundle
  gate is teacher-forced per-position agreement split at the 512 boundary;
  only POST-window collapse indicates a mask/cache bug.
- Diagnostic signature: 〈|UNK|〉 spam + parity 1.000-pre/0.000-post =
  NaN logits (dtype/overflow), NOT a cache bug.
- Ops: never pipe a converter/verify run through `| tail` or `grep` in a
  way that masks its exit code, and always read actual PASS markers from
  background-task output before promoting/deleting bundles — one bg
  "success" this session had run nothing (wrong uv project after cwd
  reset).

## Conversion

```bash
# JANG_2L (~44 GB), then JANG_4M (~68 GB) — SEQUENTIALLY, never concurrent MLX
python -m jang_tools.convert_laguna_jang \
    --src ~/models/poolside/Laguna-S-2.1 \
    --out ~/.mlxstudio/models/JANGQ-AI/Laguna-S-2.1-JANG_2L --profile JANG_2L
python -m jang_tools.convert_laguna_jang \
    --src ~/models/poolside/Laguna-S-2.1 \
    --out ~/.mlxstudio/models/JANGQ-AI/Laguna-S-2.1-JANG_4M --profile JANG_4M
```

Profiles: 2L = routed 2/2/3, attn+g_proj 8, shared/dense/embed 6, lm_head 8,
gs 64 (exact shipped M.1 recipe). 4M = routed 4/4/4, shared/dense 8.
bf16 source (235 GB) cannot fit in 128 GB RAM for a full-precision sanity
run — the port is proven on M.1/XS.2 + the tiny-weight tests above; the 2L
bundle no-cache greedy is the runtime gate.

## Speculative decode (later)

`poolside/Laguna-S-2.1-DFlash` is a trained draft model (vendor serving
uses `num_speculative_tokens: 15`). Separate repo, not part of this
campaign; candidate for a jang-spec style add-on once 2L/4M ship.
