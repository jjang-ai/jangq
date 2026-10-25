# DeepSeek-V4.1-Flash — decode & DSpark exact methods

Canonical, cross-machine blueprint for the validated V4.1 decode/speculative methods. Reference
implementation: this directory (`jang_tools/dsv41/`, jangq). Mirror these verbatim into (a) the vMLX
engine's V4.1 SingleBatchGenerator spec-decode path and (b) vmlx-swift
(`Libraries/MLXLMCommon/SpecDec`). All numbers measured on an M5 Ultra (256 GB) against the
`DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX` bundle.

## M1 — DSpark speculative decode loop (`dspark.py::DSparkRunner.generate`)

Output is **token-identical to plain greedy**: `emit` is composed only of backbone verify-forward
argmaxes, so the draft changes the *acceptance rate* (the speedup), never the tokens.

State: `c` = frontier (last position whose hidden is committed to the draft windows); `pending` =
backbone logits predicting position `c+1`. Per step:

1. `p0 = argmax(pending)` — backbone truth for `c+1` (free).
2. `draft_ids, conf = draft(p0, c)` → `[p0@c+1, d1@c+2, …, dW@c+1+W]`.
   **CRITICAL off-by-one:** seed the draft with the backbone's already-predicted `p0` (position
   `c+1`) and draft `c+2` onward, with `main_hidden` taken at position `c`. Seeding the committed
   token at `c` (drafting `c+1`) gives ~35 % accept + divergence.
3. `feed = [p0, d1..dW]` (W+1 tokens at `c+1..c+1+W`); `ck = checkpoint()` (O(1) ref snapshot).
4. `vlog, vmh = forward(feed, all_logits=True, return_hidden=True)` — **one** backbone forward.
   `vlog` has W+1 rows; `vlog[j]` predicts position `c+2+j`, so `truth[j] = argmax(vlog[j])`, and the
   bonus `truth[W]` exists.
5. Accept: `k` = longest prefix where `draft_ids[1+j] == truth[j]` (`j = 0..W-1`).
6. **NO-REFEED FAST PATH — `k == W` (~2× on high-accept content):** the verify forward already wrote
   exactly-correct *causal* KV for every accepted position (no rejected tail to undo), so **do not
   restore+refeed**. Keep the verify cache: `seed_windows(vmh, c+1..c+1+W)`; `c += W+1`;
   `pending = vlog[W]`; `emit = [p0,d1..dW]`. `truth[W]` becomes the next step's `p0`. → **1 forward/step.**
7. **PARTIAL ACCEPT — `k < W`:** `emit = [p0, d1..dk, truth[k]]`; `restore(ck)`;
   `flog, fmh = forward(emit)`; `seed_windows(fmh, …)`; `c += len(emit)`; `pending = flog[-1]`.
   → 2 forwards/step. (Removing this second forward needs exact cross-layer cache trim of the rejected
   tail; the engine's `DeepseekV41Cache` supports `trim`/rollback, so the engine *can* do no-refeed
   here too. Low marginal value in the standalone runtime since high-accept content hits the fast path.)

Why it works: a 6-row verify forward ≈ 62 ms ≈ 2× a 1-row decode (32 ms) — the backbone forward
scales with rows — so eliminating the second (commit) forward on fully-accepted blocks ≈ 2×.

**Measured (n=80, adaptive, WITH fast path):** code **1.88× / 55.7 t/s**, structured **2.00× / 57.1**,
low_accept **1.93× / 50.1**, prose 1.04× / 31.5. MIN 1.04× (never-below-AR PASS). Output
token-identical to greedy on code (59/59) and structured (29/29). = parity with antirez `ds4.c`
56.8 t/s serial.

## M2 — Adaptive never-below-AR (`DSparkRunner(adaptive=True)`)

- `accept_ema = 0.7*ema + 0.3*(k/W)` each step; draft width `W = max(1, round(self.width * ema))`
  (shrink as acceptance falls so low-accept content drafts — and rejects — less).
- **LATCH:** after `latch_after = 2` consecutive zero-accept drafts, run **bare greedy** (no
  draft/verify/hidden/window upkeep) for the rest of the generation. Latch state is per-`generate()`
  (resets each request). Guarantees spec throughput ≥ AR on sustained low-accept content.
- **Gap for vmlx-swift to improve:** the latch is permanent *within* a generation (no re-entry), so
  mixed reasoning→code loses spec after one lull. Swift should add re-entry hysteresis (e.g. re-probe
  at `16·2^k` intervals) — the Python reference lacks it.

## M3 — Why single-stream AR decode is at the floor (so MTP is the only lever)

- 10.6 GB read/token; decode 31 t/s = **333 GB/s of ~800 peak (42 % util)**.
- MLX `quantized_matmul` qmv is only ~50 %-bw-efficient at M=1 (fully-fused-qmv ceiling = 398 GB/s =
  **1.19×**); `gather_qmm` = 0.29× of qmv; `lane_qmm` (TensorFold) is **11 % slower at M=1** (wins only
  at 2–16 rows); `mx.compile` = no help (cannot merge independent GEMVs).
- **DEAD ENDS for single-stream (all measured):** fused MLA decode-attn kernel, mx.compile, GEMV
  fusion, gather_qmm swap, lane_qmm. Do not re-attempt.
- **The lever is MTP (M1):** a wide verify costs ≈ one forward (M=1 and M=16 take ~the same per-GEMV
  time — weight-read-bound), so accepting a block amortizes. Secondary lever: fewer bytes/token (the
  LM head alone is 703 MB/token; a lighter head or uniform-Q2 raises AR proportionally — a quant
  tradeoff, not an engine change).
- Prefill is already at parity (865 vs ds4.c 790 pp/s); no prefill work needed.

## M4 — MLA decode attention kernel (`metal_decode_attn.py`, default OFF)

Port of the engine's `glm5_mla_decode_attn` (= mlx-serve PR #911 fused decode): 64 heads × rank-512
shared latent, bf16 rows read in place by index (no `mx.take` gather, no fp32 latent copy), fp32
accumulation (bf16 over the 512-dim contraction drifts into repetition loops). V4.1 adds the learned
per-head `attn_sink` into the **combine** pass only (extra `exp(sink−M)` denominator term, no value
row; the partial pass is the GLM kernel verbatim). Equivalent to the stock gather path (cos 0.999999).
**Default OFF — a wall-clock wash at decode** (attention is not the decode cost; see M3). Kept as a
building block: the same in-place-indexed fp32-accumulate structure is what a head_dim-512 fused
*prefill* attention kernel would use (not pursued — prefill already at parity).

## Engine (vmlx-py) wiring status — DONE (vmlx `bf1ce697`)

V4.1 DSpark speculative decode is wired into the engine serving path and verified live:
- **Draft load** (`loaders/load_jangtq_dsv41.py`): `_maybe_load_dsv41_dspark` calls
  `jang_tools.dsv41.dspark.load_dspark(runtime.args, runtime.model.embed, runtime.model.head)` and
  attaches `model._dspark`. Gated by `VMLX_DSV41_DSPARK` (default OFF).
- **Generator** (`utils/dsv41_dspark_generator.py::DSV41DSparkGenerator`): subclasses
  `SingleBatchGenerator`; overrides `_start_request`/`_yield_current_and_schedule_next` to drive
  `DSparkRunner.stream()` on `model.runtime` (its own runtime `Session` + checkpoint/restore + the
  no-refeed fast path), yielding one committed token per scheduler step and reusing the inherited
  finish-reason/Response/stop machinery. Text-only; VL requests fall back to the inherited AR path.
- **Selection** (`scheduler.py::_create_batch_generator`): a branch before the SBG fallback returns
  `DSV41DSparkGenerator` iff `model._dspark` is attached — so the default AR path is never affected.
- **Verified live** (`vmlx serve`, warmed, thinking-off code request): coherent output, `finish=stop`,
  **decode-only ~65.9 tok/s vs AR ~31** (2x+); generator selection confirmed in logs; output
  token-identical to greedy by construction.
- Note: the engine's older `vmlx_engine/dspark/` (drafter/generate/session) is Qwen3/MiniCPM DSpark,
  unrelated to V4.1. Future: the engine's `DeepseekV41Cache` exposes `trim`, so a later version can do
  M1-step-7 no-refeed for *partial* accept too (the standalone runtime skips it).

### vmlx-swift mirror (Libraries/MLXLMCommon/SpecDec)
Same shape: load the bundle `dspark/` draft, keep a draft-windows + backbone `Session` per sequence,
and in the decode step run M1 (draft → verify forward → accept → fast-path|commit) + M2 (adaptive
width + AR latch, **plus** the re-entry hysteresis the Python reference lacks).
