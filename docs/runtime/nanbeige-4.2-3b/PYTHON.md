# Nanbeige 4.2-3B — Python / vMLX engine integration notes

Targets: `/Users/eric/mlx/vllm-mlx` (vMLX engine + server + panel) and
`jang-tools`. Read [`NOTES.md`](NOTES.md) first. The MLX model already exists and
is verified — this file is about wiring it into the engine, scheduler and API
surface without tripping the loop.

Working runtime: `jang_tools/nanbeige/model.py`, registered by
`jang_tools/nanbeige/mlx_register.py`.

```python
from jang_tools.nanbeige import mlx_register  # noqa: F401  (registers "nanbeige")
from mlx_lm import load
model, tok = load("~/models/JANGQ-AI/Nanbeige4.2-3B-JANG_4M")
assert model.cache_slots == 44
```

Registration is the mimo_v2 pattern: inject the module as
`sys.modules["mlx_lm.models.nanbeige"]` and add it to
`mlx_lm.models._MODEL_MAPPING`. If vMLX has its own model registry, register
there too rather than relying on the monkey-patch.

---

## 1. Cache: 44 slots, and every code path must agree

`Model.make_cache()` returns `num_hidden_layers * total_loops` entries. mlx_lm
honours this automatically — `cache.make_prompt_cache` defers to `make_cache`
when present (`mlx_lm/models/cache.py:32`). What does **not** defer:

| pattern | status |
|---|---|
| `make_prompt_cache(model)` | ✅ uses `make_cache` → 44 |
| `[KVCache() for _ in model.layers]` | ❌ 22 — silently wrong |
| `[KVCache() for _ in range(cfg.num_hidden_layers)]` | ❌ 22 — silently wrong |
| `prompt_cache[: len(model.layers)]` (speculative-decoding split, `generate.py:526`) | ❌ truncates to 22 — **do not enable draft/spec-decode for this family until that split is made loop-aware** |

`Model.layers` deliberately returns the 22 real blocks (module tree, quant walk,
and weight loading all depend on that). Use `model.cache_slots` — never
`len(model.layers)` — anywhere a cache length is needed.

Engine-side assert at load:

```python
slots = json.load(open(f"{path}/config.json"))["jang_runtime"]["cache_slots"]
assert len(cache) == slots == model.cache_slots, "nanbeige loop cache mismatch"
```

### Prefix cache / paged / L2

The per-slot codec is ordinary `KVCache` — plain GQA K/V, no typed state, no
compression, no path dependence. So prefix hits, paged blocks, cache-salt
bypass, L2 block writes and fresh-process L2 restore all work **unchanged**,
provided the block/record schema stores 44 entries and the restore path
reconstructs 44. Two concrete requirements:

* Any serialized cache record must persist the slot **count** (or read it from
  `jang_runtime.cache_slots`) — a record written by a 44-slot process and
  restored into a 22-slot allocation is the same corruption in slow motion.
* All 44 slots advance in lockstep and share one `offset`, so trimming
  (`trim_prompt_cache`) and `maybe_quantize_kv_cache` need no special casing —
  but do verify `len({c.offset for c in cache}) == 1` in tests; a split offset
  set means a loop was skipped or double-fed.

This model is **not** path-dependent — it does not need the
`path-dependent-cache-restore` N−1 re-prefill pattern used for DSV4/ZAYA/hybrid
SSM. Clean per-slot restore is correct.

### KV memory is the real constraint

176 KB/token at fp16 (2048 values/slot × 44 slots × 2 B). See the table in
`NOTES.md`: 22.5 GB at 128 K, 46.1 GB at the 262 144 max — against a 2.9 GB
JANG_4M bundle. Pick `max_kv_size` / quantized-KV defaults from that table, and
size the Metal working-set guard against KV, not against bundle size. The usual
"wired limit ≈ bundle × 1.2 + 8 GB" heuristic under-provisions this model badly
at long context.

## 2. Model registry / autodetect

Add `nanbeige` to the vMLX registry as **text-only, dense, kv-cache**:

* family `nanbeige`, reasoning parser `qwen3`, tool parser `xml_function`,
  `think_in_template=True`, cache `kv`, modality `text`.
* No `vision_config` / `audio_config` / `video_config` in the config and zero
  modal tensors in the index → text-only, and the weight-gate in
  `jang_tools/capabilities.py` already resolves that correctly.
* No MoE → **no trained active-K metadata**. Do not stamp `top_k`; there is no
  router. (Distinct from API sampling `top_k`, which `generation_config.json`
  sets to 20.)
* No MTP, no draft head, no speculative-decoding metadata.

`build_capabilities` already emits the right stamp — verified on all three
bundles:

```json
{"reasoning_parser": "qwen3", "tool_parser": "xml_function",
 "think_in_template": true, "supports_tools": true, "supports_thinking": true,
 "family": "nanbeige", "modality": "text"}
```

## 3. Prompt encoder

* **`add_special_tokens=False`** when tokenizing a rendered template. The
  template emits `<|im_start|>` (166100 = `bos_token`) per turn and the
  tokenizer's post-processor prepends another. Measured: a one-turn chat
  renders 3 × 166100; `tok(rendered)` yields 4.
  Do **not** branch on `tokenizer.add_bos_token` — it reads `False` on the
  loaded object while the post-processor still injects BOS.
* Stop ids: `166101` (`<|im_end|>`, eos) **and** `166102` (`<|endoftext|>`).
* `generation_config.json` defaults: `temperature 0.6, top_p 0.95, top_k 20,
  do_sample true`. Greedy is off-distribution for a reasoning model — use these
  defaults, matching the DSV4 lesson (`memory/project_dsv4_eval_nuances`).
* Reasoning rails:
  * default / `enable_thinking=True` → prompt ends `<|im_start|>assistant\n<think>\n`
    (open tag; model output begins *inside* reasoning and the first tag seen is
    `</think>`).
  * `enable_thinking=False` → `<think>\n\n</think>\n\n` prefilled.
* `preserve_thinking` (default **true**) keeps each historical assistant turn's
  reasoning. Every published Nanbeige benchmark used `preserve_thinking=true`.
  If the API stores assistant turns as visible-content-only, multi-turn quality
  drifts from the published setup — expose it as an explicit setting.
* System prompt: absent → the template injects a Chinese default
  (`你是南北阁…`); with `tools` and no system message it injects a *different*
  Chinese tool-calling system prompt. Injecting a product system prompt is fine
  but is a deliberate deviation.

## 4. Tool calling

Default `tool_call_format='xml'` → `xml_function` shape:

```
<tool_call>
<function=get_weather>
<parameter=city>
SF
</parameter>
</function>
</tool_call>
```

Same shape as mimo_v2 / nemotron-ultra; the existing `xml_function` parser
applies. Parameter values are raw text between the `<parameter=…>` lines
(multi-line allowed), **not** JSON — non-string arguments are `tojson`-encoded
by the template, so the parser must handle both bare strings and JSON scalars.

Tool results: consecutive `role: "tool"` messages are merged into **one** user
turn containing one `<tool_response>…</tool_response>` block per result. Emitting
one user turn per tool result does not match the template.

`tool_call_format='json'` switches to the Qwen `<tool_call>{"name":…}` shape —
support it as an opt-in, but default to xml.

## 5. Quantized bundles

| bundle | size | mode | attn / mlp / embed | per-module overrides |
|---|---|---|---|---|
| `Nanbeige4.2-3B-MXFP8` | 4.0 GB | `mxfp8` | 8 / 8 / 8 | 0 (uniform) |
| `Nanbeige4.2-3B-JANG_6M` | 3.6 GB | `affine` | 8 / 6 / 6 | 68 |
| `Nanbeige4.2-3B-JANG_4M` | 2.9 GB | `affine` | 8 / 4 / 4 | 68 |

All under `~/models/JANGQ-AI/`. Source: `~/models/Nanbeige/Nanbeige4.2-3B`
(pinned in `.jang_source_pin.json`).

`config.json["quantization"]` = top-level `{group_size: 32, bits: 8, mode}` plus
a per-module override for each 4-/6-bit module. The loader must apply the
overrides; a single top-level width dequantizes 8-bit attention with the 4-/6-bit
kernel (`memory/project_jangtq_config_metadata_bug`).

There is no JANGTQ sidecar and no `mxtq_bits` / `routed_expert_bits` here —
those are TurboQuant/MoE fields and this is dense affine/MXFP. Do not stamp them.

### Measured quality (logit divergence vs bf16 source, 5 prompts)

| bundle | top-1 agreement | mean KL | max KL |
|---|---|---|---|
| MXFP8 | 4/5 | 0.1446 | 0.6844 |
| JANG_6M | **5/5** | **0.0010** | **0.0030** |
| JANG_4M | 5/5 | 0.0192 | 0.0398 |

Note the ordering: **JANG_6M is the most faithful, and JANG_4M beats MXFP8.**
MXFP8's e4m3 elements carry only ~3 mantissa bits per value, so "8-bit" MX is
*not* strictly better than 6-bit or even 4-bit affine with per-group scale+bias
on this weight distribution. Recommend JANG_6M as the quality bundle and
JANG_4M as the small one; MXFP8 exists mainly for format coverage.

## 6. Speed / resource notes

Every token runs 44 layer passes and touches 44 cache slots, so this 4.17 B
model costs roughly the compute of an 8 B dense model at the memory footprint of
a 4 B one. Source bf16 on M5 Max: prefill 258.7 tok/s, decode 32.7 tok/s, peak
8.43 GB. Do not benchmark it against 3–4 B peers without noting the loop.

## 7. Tests worth adding

1. `model.cache_slots == 44 == config.jang_runtime.cache_slots`.
2. Full-prefill vs incremental vs token-by-token logits agree (measured rel
   9.2e-4 / 6.1e-4 / 3.3e-6 vs HF).
3. **Negative control**: 22 caches shared across both loops must produce a
   *different* argmax (measured rel 9.0e-1). Without this test the bug class is
   invisible.
4. `len({c.offset for c in cache}) == 1` after any prefill/decode.
5. Config guard: each of `enable_double_loop_split`, `loop_share_kv`,
   `enable_hyper_connection`, `enable_mhc`, `enable_depth_attention`,
   `qk_layernorm`, `emb_neighbor_num` raises on load (Nanbeige 4.5 will set
   them).
6. Single-BOS on a rendered template.
7. Cache save → fresh-process restore → continued decode matches uninterrupted
   decode.

## 8. Reference scripts

Kept under `docs/runtime/nanbeige-4.2-3b/`:

| script | what |
|---|---|
| `parity_hf.py` | HF reference forward **with the `inv_freq` repair** (see NOTES §Verification) |
| `parity_mlx.py` / `cache_proof.py` | MLX vs HF logits, incremental + token-by-token, negative control |
| `bisect_layer0.py` | per-sublayer localisation; this is what caught the zeroed `inv_freq` |
| `divergence.py` | per-bundle KL vs bf16 source |
| `gate.py` | multi-turn coherence gate (4 turns × 2 rails × 3 bundles, persistent cache) |
