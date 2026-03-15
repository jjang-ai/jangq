# Experiment 008: Metal Shader Compilation

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS — all shaders compile clean

## Setup

- **Xcode**: macOS Tahoe, Metal 3.0 target
- **Metal Toolchain**: 17C7003j (704.6 MB, downloaded fresh)
- **Shaders**: MXQDequant.metal, MXQCompute.metal

## Results

| Shader | Kernels | Warnings | Errors | Status |
|--------|---------|----------|--------|--------|
| MXQDequant.metal | 3 (dequant, gemv, gemm) | 0 | 0 | PASS |
| MXQCompute.metal | 7 (rmsnorm, rope, softmax, silu, silu_mul, add, embedding) | 0 | 0 | PASS |
| mxq.metallib | 10 total | 0 | 0 | 44,993 bytes |

### Kernels Compiled

**MXQDequant.metal:**
1. `mxq_dequantize` — standalone dequant to float16 buffer
2. `mxq_dequant_gemv` — fused dequant + matrix-vector multiply (token generation)
3. `mxq_dequant_gemm` — fused dequant + matrix-matrix multiply (prefill)

**MXQCompute.metal:**
4. `mxq_rms_norm` — RMSNorm
5. `mxq_rope` — Rotary Position Embeddings
6. `mxq_softmax` — Numerically stable softmax
7. `mxq_silu` — SiLU activation
8. `mxq_silu_mul` — Fused SiLU + multiply (SwiGLU)
9. `mxq_add` — Residual connection
10. `mxq_embedding` — Token embedding lookup

## Implementation Notes

- All kernels use float32 accumulation for numerical stability
- Dequant uses fast paths for 2-bit and 4-bit (direct bit shift, no general extraction)
- General extraction handles 3, 5, 6-bit via bit offset calculation
- GEMV uses SIMD reduction (simd_sum) for thread cooperation
- GEMM uses tiled approach with threadgroup shared memory
- RoPE supports position offset for KV cache continuation

## Not Yet Tested

- Actual GPU execution (compilation only verifies syntax and types)
- Performance benchmarking
- Numerical correctness vs CPU reference
- These are next steps — requires Swift runtime to dispatch kernels

## Required for Metal Toolchain

Had to run `xcodebuild -downloadComponent MetalToolchain` (704.6 MB download)
before `xcrun metal` would work. This is a macOS Tahoe requirement.
