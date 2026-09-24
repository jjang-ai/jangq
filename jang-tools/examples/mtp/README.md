# MTP Runtime Examples

Small, low-RAM helpers for checking whether a bundle has MTP and whether a
profile is realistic for a target memory size.

## Inspect A Bundle

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py /path/to/model-or-bundle
```

This reads config and safetensors index metadata only. It does not load model
weights.

For the active Qwen3.6 27B MTP lane:

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py \
  /path/to/models/Sources/Qwen/Qwen3.6-27B
```

Expected source-side signal before building `Qwen3.6-27B-JANG_4M-MTP`:

- `configured_mtp_layers` is `1`;
- `artifact_has_mtp_weights` is `true`;
- `mtp_tensor_count` is nonzero.
- `artifact_has_vision_weights` is `true`;
- `visual_tensor_count` is nonzero.

If config advertises MTP but `artifact_has_mtp_weights` is `false`, runtime MTP
cannot be activated from that artifact; recover a source with `mtp.*` weights or
stop.

For Qwen conditional-generation sources, also stop if vision metadata exists but
`artifact_has_vision_weights` is `false`; that means the VL tower was stripped or
the wrong source tree is being inspected.

## Qwen3.6 MTP/VL Probe

```sh
python3 jang-tools/examples/mtp/qwen36_mtp_runtime_probe.py \
  /path/to/models/Sources/Qwen/Qwen3.6-27B \
  --strict
```

This is still a metadata/header proof, not a generation proof. In strict mode it
fails if Qwen MTP config exists without `mtp.*` weights, if conditional-
generation metadata exists without visual tensors, or if a converted bundle
still advertises the stale `runtime.mtp_mode=preserved_disabled`.

For the built local `Qwen3.6-27B-JANG_4M-MTP` artifact:

```sh
python3 jang-tools/examples/mtp/qwen36_mtp_runtime_probe.py \
  /path/to/models/dealign.ai/Qwen3.6-27B-JANG_4M-MTP \
  --strict
```

Expected converted-bundle signal:

- `ok=true`;
- `runtime.mtp_mode=preserved_enabled`;
- `runtime.total_weight_gb=16.6`;
- `mtp_tensor_count=31`;
- `visual_tensor_count=333`;
- `has_preprocessor_config=true`;
- `has_video_preprocessor_config=true`.

## Qwen3.6 MXFP4 MTP Patch

For the local MXFP4 sibling artifact:

```sh
cd /path/to/jang/jang-tools
uv run python examples/mtp/patch_qwen36_mxfp4_mtp.py --replace
```

This copies the known-good `Qwen3.6-27B-MXFP4-CRACK` bundle, appends native
MTP tensors from `/path/to/models/Sources/Qwen/Qwen3.6-27B`, quantizes 2D
MTP matmuls with MLX `mode="mxfp4"`, and stamps
`runtime.mtp_mode=preserved_enabled`.

Expected local output:

- `/path/to/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP`;
- `runtime.total_weight_gb=14.38`;
- `mtp_tensor_count=23`;
- `visual_tensor_count=333`;
- text smoke output: `4`;
- image smoke output for a red square: `red`.

Current autoregressive runtime filters `mtp.*` tensors while loading because
today's `mlx-vlm` Qwen3.6 class does not expose MTP modules. The tensors remain
in the bundle for an MTP-aware speculative runtime.

## Qwen3.6 35B JANG/MXFP Build Lanes

The Qwen3.6 35B A3B source has real MoE MTP tensors and vision tensors:

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py \
  /path/to/models/JANGQ/Qwen3.6-35B-A3B
```

The two local build targets are deliberately JANG/MXFP, not JANGTQ:

```sh
python3 -m jang_tools --progress json convert \
  /path/to/models/JANGQ/Qwen3.6-35B-A3B \
  -o /path/to/models/JANGQ/Qwen3.6-35B-A3B-JANG_2K-MTP \
  -p JANG_2K \
  -b 128 \
  --force-dtype bf16
```

`JANG_2K` is the 2-bit routed-expert lane. Routed expert MLP tensors, including
`mtp.layers.0.mlp.experts.*`, stay at 2-bit. Attention, shared experts, and
other high-impact control tensors may be protected above 2-bit. The output is
the standard JANG affine `weight/scales/biases` shape produced by `mx.quantize`,
so vMLX should route it through MLX affine quantized matmul.

```sh
jang-convert-qwen35-mxfp4 \
  /path/to/models/JANGQ/Qwen3.6-35B-A3B \
  /path/to/models/JANGQ/Qwen3.6-35B-A3B-MXFP4-MTP \
  --progress json
```

The MXFP4 lane emits `weight_format=mxfp4`, quantizes language and MTP matmuls
as 4-bit affine tensors, and keeps the vision tower in fp16 passthrough form.

## Estimate Hy3 Fit

```sh
python3 jang-tools/examples/mtp/estimate_jangtq_fit.py \
  /path/to/models/Tencent/Hy3-preview \
  --profile JANGTQ_K \
  --device-gb 128
```

The estimator is intentionally conservative. It should guide conversion order,
not replace measured runtime memory proofs.
