# MiMo-V2.5 JANG Runtime Examples

These examples are for local MiMo-V2.5 JANG bundles under MLX/vMLX bring-up.

Current coherent bundle:

```sh
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

That bundle is audio+vision preserved and MTP removed. It passed structural
verification and short coherent text smokes on 2026-05-28. It is the promoted
`JANG_2L_322_D3E16` CPU-affine build. The old broken canonical directory was
renamed to `MiMo-V2.5-JANG_2L-broken-20260528-pre-qkv-cpuaffine`.

Runtime boundary on the local M5 Max:

- loaded model bytes: `106102 MiB`
- MLX recommended working set: `110100 MiB`
- peak memory during short cached decode: about `114.5 GB`
- coherent cached decode is around `2 tok/s` generation rate on short prompts,
  not `30 tok/s`

## Text Smoke

Normal path:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/text_smoke.py \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Sink ablation path:

```sh
JANG_MIMO_DISABLE_SINK=1 \
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/text_smoke.py \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Interpretation:

- If normal gibberish becomes coherent with `JANG_MIMO_DISABLE_SINK=1`, fix
  `_sdpa_with_sink` before touching quantization.
- If both cached and `--no-cache-greedy` are gibberish, the failure is before
  cache/prefix/L2 behavior. Audit the routed MoE path and quantization error.
- Do not add hidden sampler clamps, forced `<think>` text, or close-token bias
  to make the smoke look coherent.

No-cache isolation:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/text_smoke.py \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L \
  --thinking off \
  --max-tokens 8 \
  --no-cache-greedy
```

Known 2026-05-28 coherent output:

```text
The capital city of France is **Paris**
4.
```

The script prints loaded model size, MLX's recommended working-set limit,
prompt t/s, generation t/s, peak memory, and wall-clock t/s. `mlx_lm` already
sets the wired limit to MLX's `max_recommended_working_set_size`; setting a
larger limit on this machine raises
`ValueError: Setting a wired limit larger than the maximum working set size is not allowed`.

## Expert Quantization Probe

This compares selected layer-1 routed experts against source FP8-dequant math:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/expert_quant_probe.py \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --bundle /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Observed during the 2026-05-28 pass:

- the source qkv checkpoint layout is TP=4 rank-interleaved and must be
  deinterleaved before conversion
- routed experts must be CPU-packed min/max affine; `mx.quantize` low-bit
  expert packing did not match the source-side quantization probe
- `3/2/3` globally was source-promising but OOMed on this machine
- `3/2/2` with early layers `1..16` using `down_proj=3` is the current coherent
  fit candidate

## Layer Diff Probe

This streams source FP8 weights and compares hidden states after each layer:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/layer_diff_probe.py \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --bundle /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L \
  --layers 12 \
  --thinking off
```

Use this before claiming a bundle is coherent. A successful load or a single
generated token is not enough for MiMo.

## Bundle Audit

This reads safetensor headers only and reports exact payload by role and layer:

```sh
PYTHONPATH=/Users/eric/jang/jang-tools \
  /Users/eric/jang/jang-tools/.venv/bin/python \
  jang-tools/examples/mimo_v2/bundle_audit.py \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Promoted bundle summary:

- payload: `113.236 GB`, `106G` on disk
- routed experts: `102.073 GB`
- text attention qkv: `3.052 GB`
- text attention o_proj: `3.221 GB`
- visual: `1.457 GB`
- audio: `0.522 GB`

Rejected speed candidate:

- path: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L_322_CPUAFFINE-candidate`
- payload: `108.942 GB`, `102G` on disk
- runtime: coherent on France at `generation_tps=5.626`, but fails arithmetic
  smoke with `45.` for `2 + 2`
- status: not final

## Slim Build Candidate

`JANG_2S` is the smaller affine JANG candidate:

```sh
python -m jang_tools.mimo_v2.convert_jang \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --dst /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2S \
  --profile 2s \
  --drop-mtp
```

Policy after the 2026-05-28 correction:

- routed experts: `4/2/3` affine floors for the 256-expert rule
- attention qkv and layer-0 dense: 6-bit affine
- embeddings and lm_head: BF16 passthrough unless a profile explicitly changes
  the source-allocation policy
- text attention o_proj: 8-bit affine
- routers, norms, sink biases: passthrough
- visual/audio: passthrough
- MTP: dropped

This is still JANG affine. It is not JANGTQ and does not use TurboQuant
codebooks.

## Candidate Profiles

- `2` / `2l`: routed expert projections `4/2/3`, group 128.
- `2g32`: routed expert projections `4/2/3`, group 32.
- `322d3eN`: routed experts `3/2/2`, with layers `1..N` using `down_proj=3`.
- `233`: routed expert projections `2/3/3`, group 128.
- `2k` / `224`: gate/up 2-bit, down 4-bit, group 128.
- `422`: gate 4-bit, up/down 2-bit, group 128.
- `242`: gate/down 2-bit, up 4-bit, group 128.
- `333`: all routed expert projections 3-bit, group 128.
