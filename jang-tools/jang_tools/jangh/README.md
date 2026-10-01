# JANGH: codebook-quantized MoE experts for MLX

JANGH is the expert format used by the `*-JANGH2` bundles on the JANGQ-AI Hub org (GLM-5.3-Flash, Naive-N0.5-Flash).
Routed experts are stored at 2-4 bits with a fixed odd-cubic codebook per bit width, one fp16 scale per output row and a
blockwise Hadamard-32 rotation of the input dimension. Everything else in a bundle stays 8-bit (MXFP8 or affine) or full
precision. On disk the identifiers are `jangtq` (`version: 2`), per-module `"mode": "jangtq2"`, and the tensors
`*.tq2_packed` / `*.tq2_scales`; the packing reuses MLX's own bit layout so the runtime can reuse MLX kernel structure.

## Layout

| module | what it is |
|---|---|
| `format.py` | codebook, bit packing/unpacking, Hadamard-32 rotation, row encode/decode |
| `encode.py` | row-scale search + nearest-level assignment (RTN with the codebook) |
| `gptq.py` | GPTQ with the codebook quantizer |
| `kernels.py`, `switch.py` | Metal kernels and the `TQSwitchGLU` drop-in for mlx_lm's `SwitchGLU` |
| `install.py` | installs JANGH experts into a constructed model before `nn.quantize` (runtime hook) |
| `glm53/` | the GLM-5.3-Flash pipeline: capture, calibration, bit allocation, conversion, audit, KL evaluation, finalize |
| `n05/` | the Naive-N0.5-Flash pipeline (SWA + DSA hybrid attention, chaotic top-8 routing): reference model, capture, Hessians, recipe, conversion, evaluation |
| `tools/` | `rescale_bundle*.py`: scale correction of a finished bundle (see below); `deq_check.py`: decode bundle tensors and compare with the source experts |
| `tests/` | `test_jangtq2.py`, `test_switch.py`: kernel and module correctness gates (must print `ALL PASS` before any build) |

## Installation and entry points

For an installed release containing these modules, use `pip install "jang[jangh]"`.
The extra includes MLX, MLX-LM, Transformers and SciPy. Pipeline commands use
`python -m jang_tools.jangh.<family>.<module> --help`; they are separate from
`jang convert`. Package installation does not establish model quality or runtime compatibility.

## Building a bundle (GLM-5.3-Flash)

Prepare the BF16 source, calibration corpus, imatrix and tensor grid census before
running this example. `grid_census.json` is a JSON list of per-tensor objects with
`k` (tensor name) and `grid` (`mx8` for eligible MXFP8 tensors). It must describe
the exact source tensor grid; this GLM pipeline consumes that producer artifact
and does not generate it. Do not substitute an unrelated model census.

```sh
pip install -e "jang-tools[jangh]"  # from the repository root; Apple Silicon
export MLX_ENABLE_TF32=0             # calibration and evaluation only; never let TF32 into Hessians
python -m jang_tools.jangh.tests.test_jangtq2
python -m jang_tools.jangh.tests.test_switch
# 1. reference capture: layer-streamed BF16 forward over the calibration corpus (reservoir rows, routing, per-expert E[x^2])
python -m jang_tools.jangh.glm53.stream_capture --model SRC --seqs corpus.jsonl --stats-out diag.safetensors --ref-out klref.safetensors
# 2. unit curves (error vs bits per layer) and the bit allocation under a byte budget
python -m jang_tools.jangh.glm53.calib_tq --model SRC --diag diag.safetensors --out calib/ --experts 32 --rotation hadamard32
python -m jang_tools.jangh.glm53.plan --model SRC --census grid_census.json --calib calib/calib_tq_h32.json --out plan.json
# 3. convert (GPTQ on gate/up, held-out-gated GPTQ on down, imatrix prior), audit, evaluate, finalize
python -m jang_tools.jangh.glm53.convert_tq --model SRC --plan plan.json --diag diag.safetensors --imatrix imatrix.safetensors --out BUNDLE --rotation hadamard32
python -m jang_tools.jangh.glm53.audit BUNDLE
python -m jang_tools.jangh.glm53.kl_eval_tq --bundle BUNDLE --klref klref.safetensors --metrics-out kl.json
```

The Naive-N0.5 pipeline follows the same steps with `jang_tools.jangh.n05.*` (`build_corpus`, `stream_capture`,
`calib`, `curves`, `plan`, `convert`, `expert_audit`, `stream_eval`, `finalize`). Scripts take the BF16 source and the
tensor-header index through `JANGH_SOURCE` / `JANGH_HEADERS` or explicit arguments; nothing is hard-coded.

## Scale correction (read before shipping a low-bit bundle)

Error-minimizing row scales shrink every quantized matrix along its source row: `<w,q>/<w,w>` is about 0.88 at 2 bits
and 0.965 at 3 bits. Three matrices per expert and dozens of layers deep, strong late decisions are attenuated; the
decision to end a long reasoning block is one of them, and a bundle can then fail to emit `</think>` after a few
thousand reasoning tokens. `tools/rescale_bundle.py` rewrites only the `tq2_scales` tensors to unit gain along the source
row (`scale *= <w,w>/<w,q>`, clipped to [0.5, 2]); codes, size, format and speed are unchanged. `tools/rescale_bundle_data.py`
measures per-expert gains on real routed activations instead (smaller correction, best token-level fidelity). Whether a
model needs the correction must be measured: score `</think>` at the positions where a full-precision reference ends
its reasoning, on transcripts generated by a working engine, at several reasoning lengths.

## Rules that every step here follows

- Verify in the real runtime before calling a bundle done; a Python loader smoke test is not verification.
- Calibration Hessians are computed with TF32 off, per expert, validated for finiteness and positive-definiteness.
- Every bundle ships aligned shards (`jang_tools.format.aligned_safetensors`) and a `jang_config.json` that records
  calibration, per-layer expert bits and any scale correction.
