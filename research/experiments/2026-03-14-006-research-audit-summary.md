# Experiment 006: Research Document Audit

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: COMPLETE — errors found and documented

## Purpose

Verify accuracy of all research documents before building on them.
Launched 5 parallel fact-checking agents to audit each document.

## Summary of Findings

| Document | Errors Found | Uncertain Claims | Assessment |
|----------|-------------|-----------------|------------|
| 01 - Quantization Fundamentals | 13 (mostly minor) | 0 | SQNR constant wrong (4.35 should be 1.76), granularity MSE formula has factor of 4 error |
| 02 - llama.cpp & GGUF | 14 | 10 | Some struct definitions may be outdated, K-quant selection logic needs verification |
| 03 - Quantization Methods | 10 | 9 | Algorithm descriptions mostly correct, some formulation details uncertain |
| 04 - Extreme 2-bit | 0 errors | 5 uncertain | Cleanest doc, benchmark numbers flagged as unverified |
| 05 - Apple Metal GPU | 17 | 21 | MOST ERRORS — GPU specs wrong (M4 Ultra doesn't exist yet), bandwidth numbers off, bfloat16 Metal support claims incorrect |

## Critical Errors Requiring Correction

### Doc 01 — Quantization Fundamentals
1. SQNR formula uses constant 4.35 dB — should be 1.76 dB (sinusoidal) or 0 dB (uniform)
2. Granularity MSE term has denominator 3 instead of 12 (factor of 4 too large)
3. Clipping error integral has dimensional issue with 2σ² prefactor

### Doc 05 — Apple Metal GPU (MOST PROBLEMATIC)
1. M4 Ultra referenced — this chip does not exist as of March 2026
2. M3 Pro core counts wrong (doc says 11-18, actual varies by SKU)
3. M4 base GPU core count wrong (doc says 10, actual is 8-10)
4. Several memory bandwidth numbers incorrect
5. Max memory per chip incorrect for M3, M3 Ultra, M4 Pro
6. bfloat16 Metal support claims incorrect — Metal doesn't have native bfloat16 in the way described
7. Atomic float add claim incorrect for Metal 3.0
8. Page size claim (16KB) needs verification

### Doc 02 — llama.cpp & GGUF
1. Some C struct definitions may be outdated vs current llama.cpp main
2. K-quant selection logic may not match current implementation
3. Perplexity benchmarks are from unknown version — flag all as [VERIFY]

## Action Items
- [ ] Correct SQNR and MSE formulas in doc 01
- [ ] Remove M4 Ultra references from doc 05
- [ ] Verify and correct all GPU spec numbers in doc 05
- [ ] Verify bfloat16 Metal support status
- [ ] Flag all benchmark numbers across all docs as requiring citation
- [ ] Cross-reference llama.cpp structs against current source

## Conclusion

Eric was right to flag this. The Metal GPU document had the most errors —
hardware specs are easy to get wrong because they vary by SKU and Apple
changes them frequently. The math documents (01, 04) were cleaner.
For the paper, every hardware spec number needs a citation to Apple's
official documentation.
