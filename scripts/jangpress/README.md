# Kimi-K2.6 + JangPress runtime scripts

Tooling to load any Kimi-K2.6 JANGTQ bundle with `vmlxctl` + JangPress mmap
eviction, and run MMLU 200q (or any other API workload) against it.

JangPress itself is the cold-tier MoE memory policy implemented in
[osaurus-ai/vmlx-swift-lm](https://github.com/osaurus-ai/vmlx-swift-lm).
See [`docs/JANGPRESS.md`](../../docs/JANGPRESS.md) for the API surface
and policy details.

## Files

| File | Purpose |
|---|---|
| `build_kimi_shadow.py` | Builds a shadow dir with flat config (text_config promoted to top-level + `has_vision=false`) — works around two vMLX routing/decoding bugs. Idempotent; safe to re-run. |
| `kimi_serve.sh` | Wraps `vmlxctl serve` with JangPress mmap, memory-budget override, and the shadow-dir prep. |
| `kimi_mmlu.sh` | Runs `benchmark_mmlu_api.py` (200q) against an already-running serve. Saves a tee'd transcript + JSONL of full prompts/answers. |

## Quickstart

Two-terminal flow (set `VMLXCTL` and `PY` if not on `$PATH`):

```bash
# Terminal 1 — serve
cd scripts/jangpress
./kimi_serve.sh ~/.mlxstudio/models/JANGQ-AI/Kimi-K2.6-Med-JANGTQ 100 8082

# Terminal 2 — MMLU 200q (after the serve says "ready")
./kimi_mmlu.sh Kimi-K2.6-Med-JANGTQ chat 8082
```

For thinking mode (DeepSeek-V3-style `<think>` reasoning):
```bash
./kimi_mmlu.sh Kimi-K2.6-Med-JANGTQ thinking 8082
```

## Variants & sizing

Bundles published at [`huggingface.co/JANGQ-AI`](https://huggingface.co/JANGQ-AI):

| Variant | Size | Routed kept | Recommended pct |
|---|---|---|---|
| `Kimi-K2.6-Small-JANGTQ` | 153 GB | 211 of 384 | 100 |
| `Kimi-K2.6-Med-JANGTQ` | 167 GB | 250 of 384 | 100 |
| `Kimi-K2.6-Large-JANGTQ` | ~190 GB | 288 of 384 | 100 |

`pct=100` lets JangPress evict the maximum routed-expert mass — right
choice on a 128 GB Mac. Drop to 50–70 if you have 256 GB+.

## Environment knobs

| Env var | Default | Purpose |
|---|---|---|
| `VMLXCTL` | `/path/to/vmlx/swift/.build/arm64-apple-macosx/release/vmlxctl` | Path to the JangPress-aware vmlxctl. Build from `osaurus-ai/vmlx-swift-lm` if not installed. |
| `PY` / `KIMI_PY` | `/path/to/jang/.venv/bin/python` | Interpreter with `huggingface_hub`, `httpx`, `pandas`. `KIMI_PY` overrides stale placeholder `PY` values. |
| `SHADOW_ROOT` | `/tmp/kimi-shadow` | Where shadow dirs live. Internal SSD recommended. |
| `VMLX_MEMORY_BUDGET_OVERRIDE` | `274877906944` (256 GB) | Bypass the load-gate's "model requires ≈X GB peak" check. |
| `JANGPRESS_PRESTACK` | `1` | Force prestack regen (default-on; set `0` to disable). |
| `KIMI_LOW_RAM` | `1` | Adds low-RAM serve flags: disables prefix/memory/disk KV caches and idle, sets `kv-cache-quantization=none`, and defaults thinking off. |
| `KIMI_JANGPRESS_FORCE_MODE` | `soft` | Sets `--jang-press-force-mode`. Use `force` only when first inference needs more aggressive reclaim; expect slowdown. |
| `KIMI_ROUTER_ADVICE` | `0` | Adds `--enable-jangpress-router-advice` and exports `JANGPRESS_ROUTER_ADVICE=1` when `1`. The wrapper forces `0` by default so old shell env does not leak in. |
| `KIMI_ROUTE_TELEMETRY` | `0` | Controls `VMLX_JANGPRESS_ROUTE_TELEMETRY`. Keep off for Kimi; route readback has peaked at 130 GB physical footprint before token 1. |
| `KIMI_LAYER_EVAL` | `0` | Diagnostic only. Enables `VMLX_JANGTQ_LAYER_EVAL=1` to force layer-boundary eval in the Swift Kimi/DSV3 JANGTQ model. Tested on Small; did not reduce the 130 GB footprint. |
| `KIMI_LAYER_RECLAIM` | `1` | When `KIMI_LAYER_EVAL=1`, advises each completed layer's canonical routed expert ranges as DONTNEED. |
| `LOG` | `/tmp/kimi_serve_<bundle>.log` | Where serve logs go. |
| `OUT_DIR` | `$JANG_ROOT/jangpress-mmlu-runs` | Where MMLU runner writes log + JSONL. |
| `MMLU_API_BASE` | `http://127.0.0.1:<port>/v1` | API base used by `kimi_mmlu.sh` / `benchmark_mmlu_api.py`. |
| `MMLU_HTTP_TIMEOUT` | `600` | Per-request timeout for very slow JangPress decode/refault. |

## Operational notes

- Idle RSS after load is about 1 GB with canonical mmap active, while VSZ
  is about 600 GB. Activity Monitor's large number is virtual/mmap
  reservation, not resident RAM.
- `JANGPRESS_PRESTACK=1` regenerates the prestack overlay (~150 GB) on
  first load. Subsequent loads of the same bundle are fast.
- Current Kimi status on the 128 GB host: Small and Med can load to
  `/health`, but neither has produced a single coherent token through
  `vmlxctl serve`. First prefill reaches ~130 GB process footprint and
  severe system memory pressure before token 1. Do not start MMLU until a
  one-token probe returns content.

## Troubleshooting

**"Loading VLM weights" → "Unsupported model type: kimi_k25"**
Bundle's `jang_config.json` is missing `has_vision: false`. The shadow
script adds this automatically; if you bypass it, patch the bundle:
```bash
python3 -c "import json; p='<bundle>/jang_config.json'; c=json.load(open(p)); c['has_vision']=False; json.dump(c,open(p,'w'),indent=2)"
```

**"Loading LLM weights" → "Unsupported model type: kimi_k2"**
The shadow script wasn't run (or its output is stale). Re-run
`build_kimi_shadow.py <bundle>` and verify `$SHADOW_ROOT/<bundle>/config.json`
has `model_type: kimi_k25` at top-level (not inside `text_config`).

**"insufficient memory: model requires ≈234 GB peak"**
Either `VMLX_MEMORY_BUDGET_OVERRIDE` isn't set or you're using a vmlxctl
build that pre-dates the JangPress memory bypass. The serve script sets
the env var; if you call `vmlxctl serve` directly, export it first:
```bash
export VMLX_MEMORY_BUDGET_OVERRIDE=274877906944
```

**MMLU hangs after a few questions**
Decode under JangPress is slow under heavy eviction. Default `httpx`
timeout in `benchmark_mmlu_api.py` is 120 s — bump if needed:
```python
client = httpx.Client(base_url=API_BASE, timeout=600.0)
```

**JangPress prestack overlay is too big for `~/Library/Caches`**
Set `JANGPRESS_PRESTACK_CACHE_DIR=<path on a larger drive>` before
launching. The overlay is ~150 GB per Kimi variant.
