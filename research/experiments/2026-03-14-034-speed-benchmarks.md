# Experiment 034: Speed Benchmarks on M4 Max

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: BASELINE ESTABLISHED

## Hardware
- Apple M4 Max
- 107 GB unified memory
- macOS Tahoe

## Model: Qwen2.5-3B

| Metric | 8-bit | 4-bit | Theoretical Max |
|--------|-------|-------|----------------|
| Model size | 3.3 GB | 1.8 GB | — |
| Load time | 0.66s | 0.39s | — |
| Prefill | 28.1 tok/s | 27.6 tok/s | compute-bound |
| **Decode** | **16.1 tok/s** | **15.4 tok/s** | bandwidth-bound |

### Theoretical decode speed

Decode is bandwidth-bound:
```
tokens/sec ≈ memory_bandwidth / model_size

M4 Max bandwidth: 546 GB/s
8-bit model: 3.3 GB → theoretical: 546/3.3 = 165 tok/s
4-bit model: 1.8 GB → theoretical: 546/1.8 = 303 tok/s
```

Our actual speed (16 tok/s) is ~10% of theoretical. The gap is due to:
1. Per-encoder dispatch overhead (creating new command encoder per kernel)
2. No kernel fusion (each operation is a separate dispatch)
3. Debug build (not optimized)
4. float16 intermediate writes between kernels
5. Inefficient attention kernel (per-head threadgroup, not fused)

### Optimization opportunities

1. **Single command buffer, fewer encoders** — llama.cpp uses one buffer per forward
2. **Release build** — `swift build -c release`
3. **Fused kernels** — combine RMSNorm+GEMV, SiLU+mul+GEMV
4. **Flash Attention** — tile-based attention avoids materializing N×N matrix
5. **SIMD-group matrix ops** — use simdgroup_float8x8 for hardware matmul

## Output Quality at 8-bit

Prompt: "Explain quantum computing in one sentence."

Output: "Quantum computing leverages the principles of quantum mechanics
to process information using qubits, enabling parallel computation and
potential exponential speedup over classical computers for certain tasks."

**Perfect, coherent, accurate response.**
