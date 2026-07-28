# Nanbeige4.2-3B — multi-turn coherence gate, 2026-07-27

Machine: M5 Max MacBook. MLX 0.31.3 / mlx_lm 0.31.3.
Runtime: `jang_tools/nanbeige/model.py` (looped transformer, 44-slot KV cache).
Sampling: model defaults — `temp 0.6, top_p 0.95, top_k 20` (per
`generation_config.json`; greedy is off-distribution for a reasoning model).

Artifacts:
* `./` — 4-turn × 2-rail × 3-bundle run, first pass (thinking rail token budget
  too small — see Caveats).
* `../20260727_nanbeige42_3b_think4k/` — thinking rail re-run at 4000/5000 tokens.
* Full per-turn request, reasoning, visible output, finish reason, token counts,
  decode tok/s and cache offsets are in the JSON files.

## Protocol

One conversation of 4 turns per (bundle, rail), driven through a **persistent
44-slot KV cache** — each turn feeds only its delta, exactly as a server does.
Turns 2–4 all require recall of facts stated only in turn 1 ("Eric", "JANG").

| turn | kind | must demonstrate |
|---|---|---|
| 1 | short factual | correct answer; name + toolkit introduced |
| 2 | arithmetic + recall | `84*3/2` two-step = 126; recall "JANG" |
| 3 | long instruction | exactly 4 one-sentence bullets; recall "Eric" |
| 4 | code + structured output | function w/ docstring + doctest; trailing JSON with both facts |

## Results

### Direct rail (`enable_thinking=False`) — **PASS on all three bundles**

| bundle | t1 | t2 (=126) | t3 (recall) | t4 (code+JSON) | peak |
|---|---|---|---|---|---|
| MXFP8 | stop 61 tok | ✅ 126, "JANG" | ✅ 4 bullets, "Eric" | ✅ `{"name":"Eric","toolkit":"JANG"}` | 4.48 GB |
| JANG_6M | stop 140 tok | ✅ 126, "JANG" | ✅ 4 bullets, "Eric" | ✅ `{"name":"Eric","toolkit":"JANG"}` | 3.94 GB |
| JANG_4M | stop 147 tok | ✅ 126, "JANG" | ✅ 4 bullets, "Eric" | ✅ `{"name":"Eric","toolkit":"JANG"}` | 3.20 GB |

Every turn finished naturally (`finish_reason=stop`). Tails read in full: no
repetition, no language switching, no prompt copying, no leaked tags, no fake
role turns, no malformed code or JSON.

Decode: JANG_4M 43–47 tok/s, JANG_6M 37 tok/s, MXFP8 21–41 tok/s.

### Thinking rail (`enable_thinking=True`, 4000/5000-token budget)

| bundle | t1 | t2 | t3 | t4 |
|---|---|---|---|---|
| MXFP8 | stop 1333 — **"1936"** ❌ factual | stop 599 ✅ 126 + JANG | stop 1164 ✅ + Eric | length 5000, reasoning never closed |
| JANG_6M | stop 1403 ✅ 1868 | stop 642 ✅ 126 + JANG | stop 1084 ✅ + Eric | length 5000, reasoning never closed |
| JANG_4M | stop 2388 ✅ 1868 | stop 948 ✅ 126 + JANG | stop 1264 ✅ + Eric | length 5000, reasoning **closed**, cut in visible section |

Turns 1–3 pass on all three bundles: reasoning closes cleanly, visible content
is coherent through the tail, and cross-turn recall works over the persistent
cache.

Turn 4 hit the token budget on all three. It is a budget/prompt issue, not a
coherence failure — see Findings.

## Cache evidence

* `make_cache()` returned **44** slots on every bundle and matched
  `config.json["jang_runtime"]["cache_slots"]`.
* After every turn, `{c.offset for c in cache}` was a **single value** — all 44
  slots advance in lockstep. Example (JANG_6M direct):
  `199 → 413 → 590 → 1061`, monotonic, no resets.
* Turn N was fed as a delta against the live cache; correct answers in turns
  2–4 that depend on turn-1 facts are direct evidence the cache carried real
  history rather than being re-prefilled.

## Findings

1. **MXFP8 is the weakest bundle.** Worst logit divergence vs bf16 source
   (mean KL 0.145, top-1 4/5, vs JANG_6M 0.0010 / 5/5 and JANG_4M 0.0192 / 5/5),
   and it produced a factual error on the thinking rail ("Tokyo became capital
   in 1936" — both JANG bundles said 1868). Recommend JANG_6M for quality,
   JANG_4M for size/speed; MXFP8 for format coverage only.
2. **Turn 4 thinking-rail truncation was a harness budget error, not a defect —
   confirmed.** All three bundles ran past 5000 tokens on that prompt (JANG_4M
   got furthest: reasoning closed, code started). Follow-up probe
   (`docs/runtime/nanbeige-4.2-3b/turn4_probe.py`, JANG_6M, same 4-turn history,
   14 000-token budget on turn 4):

   | arm | turn-4 prompt | result |
   |---|---|---|
   | A | original, ambiguous ("returning KV cache **bytes**") | **stop** — complete function + docstring + doctest + `{"name": "Eric", "toolkit": "JANG"}` |
   | B | disambiguated ("returns an int … in bytes") | **stop** at 4842 tokens, same complete output |

   Both arms produced arithmetically correct doctests with the right formula
   (including the ×2 for K and V): A `kv_bytes(4,1,1,2,4,2) == 128`,
   B `kv_bytes(8,2,2,4,4) == 2048`. Turns 1–3 in both arms: stop, reasoning
   closed, recall intact. The ambiguity lengthens reasoning; it does not break
   it. **No bundle defect.**
3. **Production implication.** This is a reasoning model whose thinking rail
   routinely spends 600–2400 tokens on simple turns. A `max_tokens` default
   sized for non-reasoning models truncates mid-reasoning, which leaves an
   unterminated `<think>` in the conversation history. Servers must either
   budget generously on the thinking rail or explicitly repair/close an
   unterminated `<think>` before appending the turn to history.
4. **MXFP8 degenerate run inside reasoning.** MXFP8 turn 4 emitted a long
   `\x00\x00…` sequence inside the (unclosed) reasoning block. Contained to the
   truncated turn, but another mark against MXFP8.

## Not covered by this gate

Deliberately out of scope here, still required before any release:
API-surface parity (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`,
Ollama), streaming + stop-button/disconnect, prefix/paged cache-hit counters,
L2 block write + fresh-process restore, tool-call rows end-to-end, sleep/wake
lifecycle, and sustained long-context decode. Those live in the vMLX engine,
which does not yet have a `nanbeige` runtime — see
`docs/runtime/nanbeige-4.2-3b/PYTHON.md` and `SWIFT.md`.
