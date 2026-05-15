# MiniMax JANGTQ Prefill MPP/NAX Path

Date: 2026-05-15

## Target

MiniMax JANGTQ keeps the compact TQ storage contract:

- `*.tq_packed`
- `*.tq_norms`
- `*.tq_bits`
- deterministic codebooks/signs from the runtime sidecar/cache

The speed target is to make long-prompt routed MoE prefill use the same M5 Max
TensorOps class of work that makes affine JANG2L prefill fast, without
expanding JANGTQ weights into affine storage and without changing decode.

## Difference From JANG2L

JANG2L is MLX affine quantization. It reaches high prefill throughput through
MLX's native quantized matmul and gather-qmm kernels.

JANGTQ is codebook plus row-norm quantization. It cannot call the affine kernel
directly because the runtime operands are `tq_packed` codebook indices plus
`tq_norms`, not affine `weight/scales/biases`. The matching JANGTQ path is the
MPP/NAX kernel in `jang_tools.turboquant.mpp_nax_kernel`, which unpacks
codebook values directly into TensorOps fragments.

## Runtime Policy

Sorted routed prefill now defaults to MPP/NAX `auto`.

This is intentionally narrower than global `JANGTQ_MPP_NAX=auto`:

- sorted routed prefill uses MPP/NAX when the shape is large enough;
- single-token decode remains on the existing low-overhead P15/P17 path;
- broadcast/non-sorted paths remain opt-in unless `JANGTQ_MPP_NAX` is set;
- global `JANGTQ_MPP_NAX=0` disables the accelerated path;
- `JANGTQ_MPP_NAX_PREFILL=0` disables only the default prefill path.

The effective prefill mode is:

1. `JANGTQ_MPP_NAX_PREFILL`, if set;
2. global `JANGTQ_MPP_NAX`, if set;
3. otherwise `auto`.

`auto` currently routes when:

- sorted dispatch rows are at least 512; or
- dispatch rows are at least 256 and the projection is large enough to matter.

For MiniMax long prompts, sorted rows are approximately `prompt_tokens * top_k`,
so a 2048-token prompt with top-8 routing gives roughly 16384 sorted rows.

## Files

- `jang-tools/jang_tools/turboquant/mpp_nax_kernel.py`
  - owns the prefill policy helper and grouped TensorOps kernels.
- `jang-tools/jang_tools/turboquant/fused_gate_up_kernel.py`
  - routes sorted fused gate/up/SwiGLU prefill through grouped MPP/NAX by
    default.
- `jang-tools/jang_tools/turboquant/gather_tq_kernel.py`
  - routes sorted down-proj gather prefill through grouped MPP/NAX by default.
- `jang-tools/tests/test_turboquant_mpp_nax_kernel.py`
  - covers default dispatch, opt-out, grouped kernel correctness, and full
    expert-cluster equivalence against the existing JANGTQ path.

## Verification Commands

Unit and kernel correctness:

```sh
cd /Users/eric/jang/jang-tools
uv run --extra mlx --with pytest python -m pytest tests/test_turboquant_mpp_nax_kernel.py -q
```

Focused default-dispatch check:

```sh
cd /Users/eric/jang/jang-tools
uv run --extra mlx --with pytest python -m pytest tests/test_turboquant_mpp_nax_kernel.py \
  -k 'defaults_to_grouped_nax or can_disable_default_nax' -q
```

Live MiniMax comparison should be run with the same prompt, fresh process, and
fresh cache for both modes:

```sh
cd /Users/eric/jang/jang-tools
uv run --extra mlx python jangtq_live_nax_probe.py \
  /Users/eric/models/JANGQ/MiniMax-M2.7-Small-JANGTQ \
  --prompt-tokens 2048 \
  --max-tokens 8 \
  --json-out /tmp/minimax_small_jangtq_prefill_nax_probe_2048.json
```

Do not count a load-only test as success. The proof needs prompt-processing
tokens per second, decode tokens per second, real generated text, and a cache
freshness guard.

## Local Probe Notes

On this branch, the local 37 GB MiniMax-M2.7-Small-JANGTQ bundle showed the
new default path improving prefill without moving decode materially:

| Prompt target | Legacy prefill | Default prefill | Decode delta |
|---:|---:|---:|---:|
| 512 body tokens | 148.6 pp/s | 221.6 pp/s | 39.1 -> 38.8 tok/s |
| 2048 body tokens | 138.1 pp/s | 230.8 pp/s | 35.7 -> 35.4 tok/s |

This proves the sorted routed prefill path is active and beneficial. It does
not yet prove JANG2L-class prefill. The remaining gap is expected to be in the
TQ-specific grouped path: it still has codebook unpack work and currently builds
same-expert tile metadata from sorted expert indices. Closing the gap to the
affine JANG2L baseline likely needs a no-CPU-sync tile planner or an affine-like
TQ tile sidecar that preserves the JANGTQ storage contract.
