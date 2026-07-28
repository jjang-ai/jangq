# Nanbeige 4.2-3B — runtime + quant reference (2026-07-27)

> Master reference for `Nanbeige/Nanbeige4.2-3B`. Read this before touching any
> loader, cache, parser, or converter for this family.
> Companions: [`SWIFT.md`](SWIFT.md) (vmlx-swift port), [`PYTHON.md`](PYTHON.md)
> (vMLX Python engine / server integration).

Source: `Nanbeige/Nanbeige4.2-3B` rev `fab06df20f278e1d8afe2a64f0f8b38b7cae03ee`
→ `~/models/Nanbeige/Nanbeige4.2-3B` (bf16, 2 shards, 8.34 GB, 201 tensors).
Apache-2.0. 4.17 B total params / ~3.15 B non-embedding. Text-only (verified
from the tensor index: zero vision/audio/video tensors). en + zh.

---

## TL;DR — the 8 things you MUST get right

1. **It is a Looped Transformer.** `num_loops = 2`: the same 22 decoder layers
   run **twice**, sharing weights. Effective depth 44. Nothing about the weight
   file hints at this — only `config.json["num_loops"]`.
2. **The KV cache has 44 slots, not 22.** Slot index is
   `layer_idx + loop_idx * num_hidden_layers`. Each loop pass attends only to
   the keys *its own pass* wrote. Sizing the cache from `num_hidden_layers` is
   the single highest-risk bug — see the negative control in §Verification: it
   still *runs*, still emits fluent-looking text, and is wrong from token 1.
3. **The final norm (`model.norm`) runs at the end of EVERY loop**, not once at
   the end (`skip_loop_final_norm = false`). The normed output of loop 0 is the
   input to layer 0 of loop 1.
4. **`position_ids` are identical in both loops.** The loop adds depth, not
   positions. One causal mask, computed once, is valid for every loop.
5. **Plain Llama RMSNorm — NO +1 shift.** `NanbeigeRMSNorm` is
   `weight * x / rms(x)`. This is *not* the Qwen3.5/Ornith `+1` convention. A
   +1 here is silent garbage.
6. **`n_heads * head_dim != hidden_size`** (48 × 128 = 6144 vs 3072). `q_proj`
   and `o_proj` are "wide". Never derive `head_dim` from
   `hidden_size / num_attention_heads` — read `head_dim` (or `kv_channels`).
7. **Double-BOS trap.** The chat template already emits `<|im_start|>` (id
   166100 = `bos_token`), and the tokenizer's post-processor prepends another
   one. Tokenize the rendered template with `add_special_tokens=False`.
8. **Thinking is ON by default.** The generation prompt ends with an *open*
   `<think>\n`. Only `enable_thinking=False` prefills a closed
   `<think>\n\n</think>\n\n`.

---

## Architecture

Per-layer math is byte-for-byte Llama. The loop is the whole story.

| Field | Value |
|---|---|
| `model_type` / arch | `nanbeige` / `NanbeigeForCausalLM` |
| `hidden_size` | 3072 |
| `intermediate_size` | 10752 (SwiGLU: `down(silu(gate(x)) * up(x))`) |
| `num_hidden_layers` | 22 |
| **`num_loops`** | **2** |
| **effective depth** | **44** |
| `num_attention_heads` | 48 |
| `num_key_value_heads` | 8 (GQA, 6 q per kv) |
| `head_dim` / `kv_channels` | 128 |
| `attention_bias` | false (no qkv/o bias) |
| `qk_layernorm` | absent → false |
| `rms_norm_eps` | 1e-5 |
| `rope_theta` | **70 000 000** (7e7) |
| `rope_scaling` | `null` — no YaRN, no NTK, no partial rotary |
| RoPE style | NeoX half-rotation (`rotate_half`), full 128 dims |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 166144 |
| `tie_word_embeddings` | **false** → bare `lm_head.weight` |
| `loop_loss_weights` | `[]` (training-only; if non-empty it OVERRIDES `num_loops` with `len(w)+1`) |
| `skip_loop_final_norm` | false |

### The loop, exactly

```
h = embed_tokens(x)
for loop_idx in range(num_loops):                 # 2
    for i in range(num_hidden_layers):            # 22, SHARED weights
        h = layers[i](h, cache=cache[i + loop_idx * num_hidden_layers])
    if not skip_loop_final_norm:
        h = norm(h)                               # EVERY loop
if skip_loop_final_norm:
    h = norm(h)
logits = lm_head(h)
```

Reference: `modeling_nanbeige.py::NanbeigeModel.forward` (loop at L2217) and
`_get_loop_cache_layer_idx` (L133) — `return layer_idx + loop_idx * num_hidden_layers`.

### Tensor inventory (201)

```
model.embed_tokens.weight                             [166144, 3072]
model.layers.{0..21}.self_attn.{q,k,v,o}_proj.weight   q/o [6144,3072]/[3072,6144], k/v [1024,3072]
model.layers.{0..21}.mlp.{gate,up,down}_proj.weight    [10752,3072] x2, [3072,10752]
model.layers.{0..21}.{input_layernorm,post_attention_layernorm}.weight
model.norm.weight
lm_head.weight                                        [166144, 3072]   (bare — untied)
```

No MoE, no MTP, no SSM, no vision/audio, no expert stacking, no sidecars.

### Features present in `modeling_nanbeige.py` but OFF here

`modeling_nanbeige.py` is a superset shipped ahead of Nanbeige 4.5. All of the
below are disabled in the 4.2-3B config and **must be hard-rejected**, not
ignored, by any loader:

`enable_double_loop_split` (LoopSplit), `loop_share_kv`, `enable_hyper_connection`,
`enable_mhc` (Sinkhorn manifold hyper-connections), `enable_depth_attention`,
`emb_neighbor_num` / n-gram embeddings (`NgramCache`), `qk_layernorm`.

Both `jang_tools/nanbeige/model.py` and `jang_tools/nanbeige/convert.py` raise on
each of these. Keep it that way — Nanbeige 4.5 will turn them on and a silent
fallback would look like a working model that is quietly wrong.

---

## Cache contract

```
cache_slots       = num_hidden_layers * num_loops = 44
slot(layer, loop) = layer + loop * num_hidden_layers
```

* Per-slot codec is **ordinary KV** — plain GQA K/V, no compression, no typed
  state. Generic KV / TurboQuant-KV / prefix / paged / L2 machinery all apply
  unchanged **per slot**. `capabilities.cache_type` is therefore `"kv"`.
* All 44 slots advance in lockstep (same tokens, same positions), so `offset`
  is uniform across slots, one mask serves all loops, and trim / restore /
  quantize operate uniformly.
* **The only thing that differs from a plain model is the slot count.** Any code
  path that writes `for i in range(num_hidden_layers)` or
  `cache = [KVCache() for _ in model.layers]` is wrong here.

`config.json` in every converted bundle carries the contract explicitly:

```json
"jang_runtime": {
  "architecture": "looped_transformer",
  "cache_layout": "looped_kv_v1",
  "num_loops": 2,
  "num_hidden_layers": 22,
  "cache_slots": 44,
  "cache_slot_formula": "layer_idx + loop_idx * num_hidden_layers",
  "loop_final_norm": "every_loop",
  "norm_convention": "llama_rmsnorm_no_plus_one"
}
```

Runtimes should **assert** `len(cache) == jang_runtime.cache_slots` at load, not
infer it.

### KV memory

Per token per slot: `8 kv_heads × 128 head_dim × 2 (K,V)` = 2048 values.
Across 44 slots: 90 112 values/token → **176 KB/token at fp16/bf16**
(2× what the 22 layer count suggests; equivalent to a 44-layer, 8-kv-head model).

| context | KV (fp16) | KV (8-bit quantized) |
|---|---|---|
| 8 K | 1.4 GB | 0.7 GB |
| 32 K | 5.6 GB | 2.8 GB |
| 128 K | 22.5 GB | 11.3 GB |
| 262 144 (max) | 46.1 GB | 23.1 GB |

Long context on this model is KV-bound, not weight-bound. A 2.7 GB JANG_4M
bundle at 128 K needs ~8× the bundle size in KV. Plan `max_kv_size` /
quantized-KV defaults accordingly.

---

## Prompt / chat contract

Tokenizer: `LlamaTokenizer` (sentencepiece + `tokenizer.json`), no `auto_map`,
so bundles need **no remote code**.

| token | id |
|---|---|
| `<|im_start|>` (bos) | 166100 |
| `<|im_end|>` (eos) | 166101 |
| `<|endoftext|>` | 166102 |
| `<think>` / `</think>` | 166103 / 166104 |
| `<tool_call>` / `</tool_call>` | 166105 / 166106 |
| `<unk>` (unk + pad) | 0 |

`generation_config.json`: `do_sample=true, temperature=0.6, top_p=0.95, top_k=20`,
`eos_token_id=166101`.

### BOS

The rendered template contains one `<|im_start|>` per turn (3 for a one-turn
chat). The tokenizer **also** prepends BOS when `add_special_tokens=True`:

```
apply_chat_template(..., tokenize=True)      -> [166100, ...]   3 x <|im_start|>   OK
encode(rendered, add_special_tokens=True)    -> [166100, 166100, ...]  4 x         BUG
encode(rendered, add_special_tokens=False)   -> [166100, ...]   3 x                OK
```

Note `tokenizer.add_bos_token` reads **False** on the loaded object even though
`tokenizer_config.json` says `true` — the BOS comes from `tokenizer.json`'s
post-processor. Do not gate on `add_bos_token`; always use
`add_special_tokens=False` on a rendered template.

### Thinking rails

| kwargs | assistant prefix emitted |
|---|---|
| *(none)* → default | `<|im_start|>assistant\n<think>\n` — **thinking ON, tag left open** |
| `enable_thinking=True` | same as default |
| `enable_thinking=False` | `<|im_start|>assistant\n<think>\n\n</think>\n\n` — closed, direct answer |

Because `<think>` is pre-opened, the model's visible output starts *inside*
reasoning and the first tag the runtime sees is `</think>`. The `qwen3`
reasoning parser handles exactly this ("implicit reasoning mode"); a parser that
requires a leading `<think>` in the output will classify all reasoning as
content.

### `preserve_thinking` — multi-turn, and it matters

For assistant history turns the template renders
`<think>\n{reasoning}\n</think>\n\n{content}`. `preserve_thinking` (default
**true**) keeps each historical turn's reasoning; `preserve_thinking=false`
blanks reasoning for every turn before the last user query
(`<think>\n\n</think>\n\n{content}`).

The model card states **all reported benchmarks were run with
`preserve_thinking=true`**. A server that strips reasoning from history is
running a different prompt distribution than the published numbers. Default to
preserving it, and make it an explicit setting rather than an accident of how
the API stores assistant turns.

### System prompt

With no system message the template injects a Chinese default:
`你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。` — and when `tools` are
passed with no system message it injects a *different* Chinese tool-calling
system prompt. If the product injects its own English system prompt, that is a
deliberate deviation from the trained default; make it a conscious choice.

### Tools — `xml_function`

Default `tool_call_format='xml'`:

```
<tool_call>
<function=get_weather>
<parameter=city>
SF
</parameter>
</function>
</tool_call>
```

Same shape as mimo_v2 / nemotron-ultra → parser `xml_function`.
Passing `tool_call_format='json'` switches to the Qwen `<tool_call>{"name":...}`
shape; the default (xml) is what to implement.

Consecutive `role: "tool"` messages are **merged into a single user turn** with
one `<tool_response>…</tool_response>` block per result. A runtime that emits one
user turn per tool result does not match the template.

Capability stamp (`jang_tools/capabilities.py`):
`"nanbeige": ("nanbeige", "qwen3", "xml_function", True, "kv")`.

---

## Quantization

Dense model → the MoE allocation rules do not apply. Per
`memory/project_dense_vs_moe.md`, dense JANG is safe at the 4-bit and 6-bit
tiers and must **not** be shipped at 2–3 bit (one MLP per layer, no expert
redundancy). Only `JANG_4M`, `JANG_6M` and `MXFP8` are accepted by the converter.

| profile | attention q/k/v/o | mlp gate/up/down | embed + lm_head | mode |
|---|---|---|---|---|
| MXFP8 | 8 | 8 | 8 | `mxfp8` |
| JANG_6M | 8 | 6 | 6 | `affine` |
| JANG_4M | 8 | 4 | 4 | `affine` |

Norms (45 tensors) pass through fp16 **unchanged — no +1**.
`group_size=32`; every quantized dim (3072 / 6144 / 1024 / 10752 / 166144) is
divisible by it.

`config.json["quantization"]` carries the top-level default (`bits=8`) **plus a
per-module override for every module that differs** — without the overrides a
loader dequantizes 8-bit attention with the 4-bit kernel and emits garbage (the
config-metadata bit bug, `memory/project_jangtq_config_metadata_bug.md`).

Build:

```sh
python -m jang_tools.nanbeige.convert \
    ~/models/Nanbeige/Nanbeige4.2-3B \
    ~/models/JANGQ-AI/Nanbeige4.2-3B-JANG_4M \
    --profile JANG_4M
```

### Built bundles + measured quality

Logit divergence vs the bf16 source over 5 fixed prompts (`divergence.py`):

| bundle | size | overrides | top-1 agreement | mean KL | max KL |
|---|---|---|---|---|---|
| `Nanbeige4.2-3B-MXFP8` | 4.0 GB | 0 (uniform) | 4/5 | 0.1446 | 0.6844 |
| `Nanbeige4.2-3B-JANG_6M` | 3.6 GB | 68 | **5/5** | **0.0010** | **0.0030** |
| `Nanbeige4.2-3B-JANG_4M` | 2.9 GB | 68 | 5/5 | 0.0192 | 0.0398 |

**JANG_6M is the most faithful and JANG_4M beats MXFP8** — the opposite of what
the bit counts suggest. MXFP8's e4m3 elements carry ~3 mantissa bits each, so
"8-bit MX" is not strictly better than 6-bit or 4-bit affine with a per-group
scale + bias on this weight distribution. It shows up in behaviour too: in the
gate, MXFP8 dated Tokyo's capital move to "1936" where JANG_6M and JANG_4M both
said 1868.

Recommendation: **JANG_6M** as the quality bundle, **JANG_4M** as the small/fast
one (also the fastest at 43–47 tok/s), MXFP8 for format coverage only.

---

## Verification (2026-07-27, M5 Max)

Harness: `jang_tools/nanbeige/model.py` + `mlx_register`, `mlx_lm` 0.31.3,
MLX 0.31.3, fp32 on both sides.

### HF reference had to be repaired first

Under `transformers` 5.7 the checkpoint's custom code loads **broken**:
`NanbeigeRotaryEmbedding.inv_freq` is a non-persistent buffer registered in
`__init__`, and transformers ≥5 does not restore it — it arrives as **all
zeros**, i.e. cos=1 / sin=0 / **no positional information at all**. The model
still produces plausible short completions, which is exactly why this is
dangerous as a baseline.

Also: `_init_rope` does `config.rope_scaling["type"]`, and transformers ≥5
injects `{"rope_type": "default"}` instead of `None` → `KeyError: 'type'`.

To get a usable reference: set `config.rope_scaling = None` before
`from_pretrained`, then rebuild every layer's
`inv_freq = 1 / base**(arange(0, dim, 2) / dim)`.
Script: `docs/runtime/nanbeige-4.2-3b/parity_hf.py`.

### Results

| check | result |
|---|---|
| MLX vs HF, full prefill, "…capital of Japan is" (13 tok) | rel 1.05e-3, argmax + top-5 **identical** (`Tokyo, Kyoto, Osaka, Tokyo, Seoul`) |
| MLX vs HF, full prefill, python code prompt (22 tok) | rel 2.30e-3, argmax + top-5 **identical** |
| MLX 44-slot cache, prefill N−1 + 1 step vs HF | rel 6.14e-4, argmax match |
| MLX 44-slot cache, token-by-token decode vs HF | rel 3.31e-6, argmax match, all offsets = 22 |
| **negative control**: 22 slots shared by both loops | rel 9.04e-1, **argmax 2113 vs 4182 — wrong** |

The negative control is the point: the wrong cache does not crash, does not
warn, and produces confident wrong tokens. Any port must reproduce the
negative control as a regression test.

Layer-by-layer localisation (`mlx_hidden.py` + `bisect_layer0.py`) is what found
the zeroed `inv_freq`; keep both scripts for the next port.

### Multi-turn coherence gate — PASS on all three bundles

Full artifacts: `docs/internal/release-gates/20260727_nanbeige42_3b/SUMMARY.md`
(+ `_think4k/`). Protocol: one 4-turn conversation per (bundle, rail) driven
through a **persistent 44-slot cache** — each turn feeds only its delta, as a
server does. Turns 2–4 require recall of facts stated only in turn 1. Sampling
at the model defaults (temp 0.6 / top_p 0.95 / top_k 20).

* **Direct rail**: all 4 turns `finish=stop` on all three bundles. Correct
  arithmetic (126), "Eric" and "JANG" recalled in turns 2–4, 4 one-sentence
  bullets, valid Python + valid trailing JSON. Tails read in full — no
  repetition, language switching, prompt copying, leaked tags or fake roles.
* **Thinking rail**: turns 1–3 `finish=stop` with cleanly closed `</think>` on
  all three. Turn 4 needed >5000 tokens; a 14 000-token re-run finished it
  correctly both with and without the prompt ambiguity that lengthened the
  reasoning. No coherence defect.
* **Cache evidence**: `make_cache()` returned 44 on every bundle and matched
  `jang_runtime.cache_slots`; after every turn `{c.offset for c in cache}` was a
  **single** value (all 44 slots in lockstep), advancing monotonically
  (e.g. 199 → 413 → 590 → 1061). Correct turn-2/3/4 recall over a delta-fed
  cache is direct evidence history was carried, not re-prefilled.
* **Thinking-rail budget is a production concern**: simple turns cost 600–2400
  reasoning tokens. A `max_tokens` sized for non-reasoning models truncates
  mid-`<think>`, leaving an unterminated tag in history. Budget generously or
  close the tag before appending the turn.

### bf16 smoke, source weights

`"What is 84 * 3 / 2?"` (thinking off) → `**126**  (84 × 3 = 252; 252 ÷ 2 = 126)`.
Prompt 258.7 tok/s, decode 32.7 tok/s, peak 8.43 GB.

Decode is slower than a 4 B model of the same weight size because every token
runs 44 layer passes and touches 44 cache slots — budget roughly the compute of
an 8 B dense model at the memory footprint of a 4 B one.

---

## Repo artifacts

| path | what |
|---|---|
| `jang-tools/jang_tools/nanbeige/model.py` | MLX runtime (loop + 44-slot `make_cache`) |
| `jang-tools/jang_tools/nanbeige/mlx_register.py` | registers `nanbeige` with `mlx_lm` |
| `jang-tools/jang_tools/nanbeige/convert.py` | MXFP8 / JANG_6M / JANG_4M converter |
| `jang-tools/jang_tools/capabilities.py` | `nanbeige` FAMILY_MAP row |
| `docs/runtime/nanbeige-4.2-3b/` | this dir — notes, Swift + Python handoffs, parity scripts |
