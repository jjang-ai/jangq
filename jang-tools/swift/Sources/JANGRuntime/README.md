# JANGRuntime (Swift)

Mirrors the Python `jang_tools.jangrt` package. Three layers:

1. **JANGQuant** — pure types: `QuantMeta`, `BundleProbe.detect`. No MLX dep.
2. **JANGRuntime** — bundle metadata loader (`BundleLoader.open`).
   Weight realization plugs into vmlx-swift's `vMLXLMCommon` (TQCodebook,
   QuantizedLinear) once we publish those targets here.
3. **JANGDistributed** — World/Hostfile/ShardPlan + a TB5Probe shim that
   drives `mlx.launch` via subprocess for now. Switches to a native
   mlx-swift distributed group when that binding is exposed.

When porting MiMoV2 / Mistral3 / Laguna kernels, the Swift code lives in
`vmlx/swift/Sources/vMLXLMCommon/` (existing project). This package is the
staging ground for the *new* primitives (distributed, JANG/JANGTQ probe),
not the per-model arch. Per-model Swift ports will land in vmlx itself.
