# JANGT: trellis-coded MoE experts for MLX

JANGT stores routed MoE experts with a bitshift trellis code instead of a scalar codebook. Each weight row is rotated
(blockwise Hadamard-128 with per-layer random signs), normalized by its RMS, and encoded by a Viterbi search over an
L-bit state (L = 12, so a row's whole cost array fits in threadgroup memory). Every step shifts 2K new bits into the
state and emits two weights (V = 2) through a pure-ALU bit-pattern code, so decoding needs no lookup table. K = 2, 2.5
or 3 bits per weight. Massive input channels are carved out before the rotation and kept as fp16 columns. Down
projections additionally get column-RMS scaling with a least-squares column refit.

The design and constants are our own (`codes.py` / `codesearch*.py` search the code family); no code or constants are
taken from ExLlamaV3 or QTIP.

A JANGT bundle can mix formats per unit: some (layer, gate/up | down) units in JANGT, the rest in
[JANGH](../jangh/README.md). `config.json` then carries a per-module `quantization` entry (`"mode": "jangt", "kbits"`
or `"mode": "jangtq2", "bits"`) plus a `jangt` block (`version`, `code`, `state_bits`, `rotation`,
`outlier_columns`).

No released runtime loads JANGT bundles yet. `moe_kernels.py` and `switch.py` are the reference decode kernels and MoE
module for runtime authors.

## Layout

| module | what it is |
|---|---|
| `codes.py`, `codesearch.py`, `codesearch_v2.py`, `trellis.py` | code families, CPU code search, Gaussian MSE / rate-distortion references |
| `v2core.py`, `v2l.py` | Viterbi encoder and GEMV kernels (L = 16 and configurable-L variants), state packing |
| `minitest.py` | dense-model encoder (`encode_matrix`, `JANGTLinear`) and a small-model KL / speed harness |
| `affine_gptq.py` | GPTQ on MLX's affine grid: the affine baseline `minitest` compares against |
| `moe_encode.py` | encoder for the routed experts of one MoE layer (per-expert Hessians, outliers, refit) |
| `moe_kernels.py`, `switch.py` | gathered-expert decode kernels and the mixed JANGT/JANGH MoE module |
| `plan.py`, `curves_jt.py`, `curves_jt2.py`, `expert_alloc.py`, `allocate.py` | byte accounting, error curves, and per-unit format allocation under a byte budget |
| `n05_sens.py` | routing-flip sensitivity per (layer, group), the allocator's weighting |
| `n05_pick_downs.py`, `n05_assemble.py` | collect JANGH down projections and assemble an aligned, mixed bundle |
| `n05_streval.py`, `stage_io.py` | layer-streamed KL / top-1 / NLL evaluation against the BF16 reference, straight from stage files |
| `bench_b2.py` | decode-kernel microbenchmark |

## Building a mixed bundle (Naive-N0.5-Flash)

The pipeline reuses the JANGH N0.5 capture, Hessians and recipe (`jang_tools.jangh.n05`), so run the JANGH steps
first: `stream_capture` (reference + Hessians), `calib`, `curves`, `plan`, and `convert --keep-stage` for any layers
whose JANGH down projections you re-quantize. Every step below needs:

```sh
pip install -e "jang-tools[jangh]"   # from the repository root; Apple Silicon
export MLX_ENABLE_TF32=0 JANGT_HB=128 JANGT_L=12
```

```sh
# 1. sensitivity and JANGT error curves
python -m jang_tools.jangt.n05_sens --help
python -m jang_tools.jangt.curves_jt2 --model SRC --headers headers.json --curves curves.json --stats stats.safetensors \
    --hess-dir HESS --out curves_jt2.json
# 2. per-unit allocation for one or more byte targets (and optional per-expert widths)
python -m jang_tools.jangt.allocate --help
python -m jang_tools.jangt.expert_alloc --help
# 3. encode the JANGT units, collect the JANGH downs the plan keeps
python -m jang_tools.jangt.moe_encode --model SRC --headers headers.json --hess-dir HESS --layers 3,5,8 \
    --plan jangh_plan.json --out jt_stage
python -m jang_tools.jangt.n05_pick_downs --todo todo.json --have-dir jh_stage --conv-dir jh_conv/.stage \
    --alloc alloc.json --key "97.55:free" --out jh_pick
# 4. score before building: streamed KL against the BF16 reference, experts read from the stage dirs
python -m jang_tools.jangt.n05_streval --model SRC --headers headers.json --seqs corpus.jsonl --ref ref.safetensors \
    --nonexpert affine8 --experts "stages:jt_stage+jh_pick+JANGH2_BUNDLE" --sets klref,agentic_ref --out scores.json
# 5. assemble (non-expert tensors, tokenizer and template files come from the JANGH base bundle)
python -m jang_tools.jangt.n05_assemble --jt jt_stage --pick jh_pick --base JANGH2_BUNDLE \
    --plan alloc.json --plan-key "97.55:free" --out BUNDLE
```

Streamed scores are not a substitute for generation in the target runtime: verify coherent output, tool calls and
reasoning close in the runtime before publishing a bundle.
