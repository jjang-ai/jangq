# DeepSeek-V4.1-Flash — decode & DSpark exact methods

Canonical, cross-machine blueprint for the validated V4.1 decode/speculative methods. Reference
implementation: this directory (`jang_tools/dsv41/`, jangq). Mirror these verbatim into (a) the vMLX
engine's V4.1 SingleBatchGenerator spec-decode path and (b) vmlx-swift
(`Libraries/MLXLMCommon/SpecDec`). All numbers measured on an M5 Ultra (256 GB) against the
`DeepSeek-V4.1-Flash-1017-JANG-Q2Q3-IMATRIX` bundle.

## M1 — DSpark speculative decode loop (`dspark.py::DSparkRunner.generate`)

Output is **greedy by construction**: `emit` is composed only of backbone verify-forward argmaxes,
so the draft changes the *acceptance rate* (the speedup), not the decision rule. It is identical to
single-token AR **except at near-ties**: a W+1-row verify forward (and a chunked prefill) computes
bf16 hidden states that differ in the last bits from a 1-row decode, which can flip a top-2 whose
margin is below ~0.05. Measured (256 tokens, fp32 logits): structured identical; code / code2 /
prose first diverge at AR margins 0.0056 / 0.0045 / 0.040. With bf16 logits the top-2 tied exactly
every ~50-100 tokens, so the head now produces fp32 logits as the reference does.

**Sampled (temperature > 0): exact too, on by default in vMLX** (`dspark.SpecSampler`). The token at
each position is `argmax(f(logprobs) / T + g_pos)`: f is mlx_lm's top_p, min_p, top_k filters in the
sampler's order, and `g_pos` is Gumbel noise keyed by the request seed and the ABSOLUTE position, so it
is a draw from the AR sampler's distribution and a re-sampled rejected position replays the same draw.
The drafter's Markov chain uses the same noise (coupled drafting), and a draft is kept only when it
equals that sample; steps 1, 4 and 5 below use `sampler.pick` in place of argmax.
`tests/test_spec_sampler.py` checks the sequence distribution against exact sequential sampling over
the real `decode_stream`. vMLX: `VMLX_DSV41_DSPARK_SAMPLED=0` sends sampled requests to the AR path.

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
7. **PARTIAL ACCEPT — `0 < k < W` (in-place commit, 1 forward/step):** the verify forward ran with
   `capture_verify=True`, so every layer holds references to its per-token cache inputs (post-RoPE
   window KV; the compressor's `(kv, score)` projections, or the normed latent at ratio 1).
   `emit = [p0, d1..dk]`; `DSV41.commit_prefix(sess, ck, k+1)` restores the checkpoint and replays only
   those k+1 tokens' cache updates (window append; compressor pooling → pending tail + completed
   groups → compressed / index pool rows with the same RoPE positions; history + position);
   `seed_windows(vmh[:k+1], …)`; `c += k+1`; `pending = vlog[k]`, so `truth[k]` is the next p0.
   **`k == 0`** keeps the old path: `emit = [p0, truth[0]]`, `restore(ck)`, `forward(emit)` (2 forwards,
   but cheaper per token than a 1-token step). The refeed path for partial accepts (and its A/B switch) is
   removed: it ran a second forward over the kept tokens every step. Measured (256 tokens, chat prompts): code 44.9 → 57.5 tok/s, code2
   41.3 → 48.7, Rust 50.9 → 59.1, prose unchanged. `tests/test_commit_prefix.py` proves the replay exact
   (`commit_prefix(W+1)` == the capture forward's own state, bit for bit).

Why it works: a 6-row verify forward ≈ 62 ms ≈ 2× a 1-row decode (32 ms) — the backbone forward
scales with rows — so eliminating the second (commit) forward ≈ 2× per step, on full and partial accepts alike.

**Measured (n=80, adaptive, WITH fast path):** code **1.88× / 55.7 t/s**, structured **2.00× / 57.1**,
low_accept **1.93× / 50.1**, prose 1.04× / 31.5. MIN 1.04× (never-below-AR PASS). Output
identical to greedy on code (59/59) and structured (29/29) over those short runs (see the near-tie note in M1). = parity with antirez `ds4.c`
56.8 t/s serial.

## M2 — Adaptive never-below-AR (`DSparkRunner(adaptive=True)`)

- `accept_ema = 0.7*ema + 0.3*(k/W)` each step; draft width `W = clamp(round(self.width * ema), 1,
  self.width)` — depth is **acceptance-governed**, shrinking as acceptance falls AND growing back as
  it recovers, always within the draft head's physical size `bs` (a capacity bound, not a policy depth).
- **Wall-clock governor (replaces the zero-accept streak trigger, 2026-10-25 audit).** Acceptance
  alone cannot keep speculation at or above AR speed: a step that accepts 1 of 5 drafts resets any
  zero-accept streak yet still pays draft + verify + commit forwards. `_Governor` keeps an EMA of
  seconds per emitted token for speculative steps and for AR steps (step start to next step start,
  every sync included) and speculates only while that is cheaper than AR. AR mode issues a
  speculative probe every `probe_every = 16` AR tokens (a probe replaces the stale estimate, so one
  good probe switches back); spec mode re-measures AR every `ar_probe_every = 64` steps (AR cost
  grows with context). Neither mode is terminal. Synthetic-timing test:
  `jang-tools/tests/dsv41_dspark_governor_test.py` (low acceptance stays within a few % of AR,
  high acceptance beats it, acceptance returning mid-stream recovers within one probe period).
- **Verified (`tests/verify_dspark_adaptive.py`):** output matches plain greedy (up to near-ties, M1)
  AR, and on a low-accept→easy prompt drafting re-engages (accepted > 0), proving recovery.
- Swift parity: the vmlx-swift port should mirror this recoverable mode (no permanent latch).

## M3 — Where single-stream AR decode time goes (it was not at the floor)

- 10.6 GB read/token; stock decode 29–31 t/s = **~330 GB/s of ~800 peak**. The earlier reading
  ("MLX qmv is ~50 % bandwidth-efficient, so AR is at the floor and MTP is the only lever") was wrong:
  the gap was dispatch count and host/GPU serialization, not the GEMVs. M5 below moves AR decode
  29.0 → 39.2 tok/s (2k context) and 26.0 → 34.3 (100k), token-identical.
- Still true: `mx.compile` does not merge independent GEMVs; `gather_qmm` is slower than qmv at
  M=1; `lane_qmm` (TensorFold) is 11 % slower at M=1; the MLA decode-attention kernel alone (M4) is a
  wash. Each is one dispatch out of ~150 per layer; only removing most of them pays.
- MTP (M1) still multiplies whatever AR is: a wide verify costs about one forward.
- Prefill: the 865 pp/s figure was measured on repetitive text, where the Engram rows stay
  page-cache warm. On unique text cold prefill is bound by Engram row reads; see
  `RUNTIME_SERVING.md`.

## M4 — Removed: standalone decode attention kernel, MoE expert dedup

Both were measured and deleted so nothing dead stays in the runtime:
- `metal_decode_attn.py` (port of the engine's `glm5_mla_decode_attn`: 64 heads x rank-512 latent, rows
  read in place by index, fp32 accumulation, sink in the combine pass). Equivalent (cos 0.999999) but a
  wall-clock wash (31.08 vs 31.42 tok/s): attention is one of ~150 dispatches per layer, and replacing
  one dispatch with another does not move a dispatch-bound step. M5 fused the rest instead.
- MoE expert dedup across verify rows (ds4 `moe_dedup_plan`): bit-identical, but slower than M5's fused
  MoE at 2/4/8 rows (0.351/0.509/0.794 vs 0.330/0.433/0.668 ms per layer). The fused MoE is already bound
  by the distinct experts' bytes (repeated tiles hit the GPU caches); the plan's pair loop only costs
  parallelism.

## M5 — Fused decode / verify kernels (`metal_decode.py`, default ON)

Four bit-exact changes for forwards of at most 8 rows (decode, DSpark verify); prefill is untouched.
Each kernel copies MLX's accumulation order (quantized GEMV walk, f32 GEMV tiling, row reduction,
RMSNorm), so greedy decode is bit-equal with each flag off or on.

| change | flag (`=0` disables) | per layer (synthetic, real dims) |
|---|---|---|
| per-layer `mx.async_eval` (<= 16 rows): the GPU runs early layers while later ones are encoded | `VMLX_DSV41_LAYER_ASYNC` | ~-0.14 ms |
| router logits from the bf16 gate weight (MLX f32 GEMV order, no per-call f32 copy); only where MLX uses the same tiling (K < 16 E) | `VMLX_DSV41_ROUTER_GEMV` | 1 row 86 → 68 µs |
| 6 routed + shared expert in 2 dispatches (mlx-serve `MOE_GU`/`MOE_DN`) instead of ~25 | `VMLX_DSV41_FUSED_MOE` | 1 row 0.279 → 0.226 ms |
| hyper-connection collapse + mixes + Sinkhorn split + expand in 3 dispatches per sublayer instead of ~14; self-checks once against the stock path | `VMLX_DSV41_FUSED_HC` | 1 row 0.183 → 0.137 ms |

Real model (M5 Ultra, AR greedy, all four on vs off): 2k context 29.0–29.3 → 39.1–39.2 tok/s, 256 greedy
tokens identical; 100k context 26.0–26.3 → 34.3 tok/s, 64 tokens identical. Verify rows (2–8) now each
compute what a 1-row decode of that token computes, so verify is closer to sequential decode than the
stock batched path (tiny model: max logit gap 1.61 → 0.35); it is still not bit-identical to it.

Note: MLX runs f32 matmuls as TF32 on M5 by default (`MLX_ENABLE_TF32=1`); the hyper-connection einsum
carries ~4e-4 relative error from it. The fused HC kernel reproduces that rounding to stay bit-equal.

Tests: `tests/dsv41_decode_fusion_test.py` (op-level bit-equality, greedy identity on random-weight
models from `dsv41/tests/synth.py`); benchmark harness `dsv41/tests/bench_decode.py`.

## Engine (vmlx-py) wiring status — DONE (vmlx `bf1ce697`)

V4.1 DSpark speculative decode is wired into the engine serving path and verified live:
- **Draft load** (`loaders/load_jangtq_dsv41.py`): `_maybe_load_dsv41_dspark` calls
  `jang_tools.dsv41.dspark.load_dspark(runtime.args, runtime.model.embed, runtime.model.head)` and
  attaches `model._dspark`. Default ON for bundles that ship `dspark/`; `VMLX_DSV41_DSPARK=0` disables.
- **Generator** (`utils/dsv41_dspark_generator.py::DSV41DSparkGenerator`): subclasses
  `SingleBatchGenerator`; overrides `_start_request`/`_yield_current_and_schedule_next` to drive
  `DSparkRunner.stream()` on `model.runtime` (its own runtime `Session` + checkpoint/restore + the
  no-refeed fast path), yielding one committed token per scheduler step and reusing the inherited
  finish-reason/Response/stop machinery. Text requests, greedy or sampled (temperature > 0 runs
  `SpecSampler`, M1); requests with images still to prefill, logits processors (repetition / frequency /
  presence penalty, logit_bias) or logprobs take the inherited AR path. A VL prefix-cache hit that
  covers every image leaves a text-only tail, which DSpark serves.
- **Selection** (`scheduler.py::_create_batch_generator`): a branch before the SBG fallback returns
  `DSV41DSparkGenerator` iff `model._dspark` is attached — so the default AR path is never affected.
- **Verified live** (`vmlx serve`, warmed, thinking-off code request): coherent output, `finish=stop`,
  **decode-only ~65.9 tok/s vs AR ~31** (2x+); generator selection confirmed in logs; output
  greedy by construction (identical except at near-ties, M1).
- Note: the engine's older `vmlx_engine/dspark/` (drafter/generate/session) is Qwen3/MiniCPM DSpark,
  unrelated to V4.1. Partial accepts already commit in place in the engine too (M1 step 7: the
  generator drives `DSparkRunner` on the engine's `_SeqState`, and `DSV41.commit_prefix` replays the
  kept tokens). Rollback never goes through `DeepseekV41Cache.trim`, which is a no-op
  (`is_trimmable()` is False: window rows and the compressor's pending group cannot be rebuilt).

### vmlx-swift mirror (Libraries/MLXLMCommon/SpecDec)
Same shape: load the bundle `dspark/` draft, keep a draft-windows + backbone `Session` per sequence,
and in the decode step run M1 (draft → verify forward → accept → fast-path|commit, greedy or
`SpecSampler`) + M2 (acceptance-tracked width + the wall-clock governor: AR steps while speculation is
slower, a speculative probe every 16 AR tokens, an AR re-measure every 64 spec steps; no permanent latch).
