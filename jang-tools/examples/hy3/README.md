# Hy3-preview Runtime Prep

This folder tracks the JANGTQ workflow for `tencent/Hy3-preview`
(`model_type=hy_v3`).

## Current Source

Expected local source directory:

```sh
/path/to/models/Tencent/Hy3-preview
```

Download is large. Use one downloader only:

```sh
uvx --from huggingface-hub hf download tencent/Hy3-preview \
  --repo-type model \
  --local-dir /path/to/models/Tencent/Hy3-preview \
  --max-workers 4
```

As of the first local pass, the active download was already running. Do not
start a second copy; check with `ps` and `du -sh` first.

## Architecture Snapshot

From `config.json` and the Hugging Face model card:

- text-only; no `vision_config` or processor sidecar expected
- `architectures=["HYV3ForCausalLM"]`
- `model_type=hy_v3`
- 295B total / 21B active
- 80 decoder layers plus 1 MTP layer
- hidden size 4096
- GQA attention: 64 Q heads, 8 KV heads, head dim 128
- q/k RMSNorm before RoPE attention
- context length 262144
- RoPE theta 11158840
- MoE: 192 routed experts, top-8, sigmoid router, expert-bias correction
- one always-active shared expert per MoE layer
- first decoder layer is dense FFN (`first_k_dense_replace=1`)
- `enable_lm_head_fp32=true`

## Files

| File | Purpose |
|---|---|
| `00_inspect_source.py` | Low-RAM config/index/header audit. Handles partial download state. |
| `01_python_reference_smoke.py` | OpenAI-compatible smoke client for vLLM or SGLang reference servers. |
| `02_python_runtime_contract.py` | Emits the cache, attention, routing, MTP, and quantization contract as JSON. |
| `Hy3RuntimeContract.swift` | Standalone Swift contract printer for runtime implementation notes. |
| `RUNTIME_AND_QUANTIZATION_NOTES.md` | Runtime, converter, cache, and publish gates for this architecture. |
| `MTP_COMPATIBILITY.md` | Exact MTP startup, runtime, cache, metadata, and model-card contract. |
| `python_runtime/` | Python runtime/parser skeletons for future `../vmlx` work. |
| `swift_runtime/` | Swift runtime/parser skeletons for future `../vmlx-swift-lm` work. |

## Runtime Implications

Hy3 is not a VL model. It is a text MoE model with dense GQA attention plus
MTP. Cache topology is standard causal KV, not MLA/SSM/CCA, but runtime support
still needs custom MoE routing and MTP policy:

- Router scoring must use sigmoid, then add expert correction bias for expert
  choice.
- Selected routed weights are normalized by their sum and multiplied by
  `router_scaling_factor`.
- Shared expert output is always added to routed expert output.
- Q and K projections use per-head RMSNorm before RoPE.
- MTP must be explicit: either implement speculative decode or strip/ignore it
  with a documented quality/perf boundary.

## Conversion Policy

Active JANGTQ profile priority:

- `JANGTQ2`: first 128 GB release candidate. Routed expert gate/up/down
  projections use 2-bit MXTQ, and the non-routed core stays 8-bit affine or
  passthrough.
- `JANGTQ1`: experimental size-floor candidate. Routed expert gate/up/down
  projections use 1-bit MXTQ. It is expected to be memory-comfortable on
  128 GB class devices, but quality is unproven and likely worse than
  `JANGTQ2`.
- `JANGTQ_K`: quality-first candidate. Routed expert gate/up projections use
  2-bit MXTQ, down projections use 4-bit MXTQ. This is not a proven
  comfortable 128 GB target.
- `JANGTQ4`: quality/reference fallback if `JANGTQ_K` is not coherent enough.

MXFP4 is out of the active lane for this model unless explicitly re-added.

Precision floors for the first safe pass:

- router gate and expert-bias tensors passthrough
- q/k norms and all RMSNorms passthrough
- `lm_head` 8-bit affine or fp16 until coherence proves lower precision
- shared expert 8-bit affine first, not 2-bit TQ
- MTP tensors included with an explicit 8-bit affine policy where tensor names
  match. Runtime docs must state whether speculative MTP decode is enabled or
  normal decode is being used with MTP disabled.

## Publish Gate

No Osaurus upload until all of these are true:

- source download complete and checksum verified
- tensor-name census complete from `model.safetensors.index.json`
- converter maps dense layer, sparse MoE layers, shared expert, router bias,
  q/k norms, and MTP tensors explicitly
- generated `JANGTQ2` bundle passes `verify_directory`
- generated `JANGTQ1` bundle, if built, is labeled experimental and passes the
  same sidecar/index/capability gates before any coherence smoke
- JANGTQ bundles include `jangtq_runtime.safetensors`
- Swift/Python runtime has a real generation proof, cache behavior proof, and
  MTP compatibility note
- any 128 GB claim is backed by measured bundle size and runtime load proof
