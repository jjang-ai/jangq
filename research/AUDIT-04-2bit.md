# AUDIT: 04-extreme-quantization-2bit.md

Audit performed: 2026-03-14
Auditor: Claude Opus 4.6 (fact-check mode)
Scope: Technical claims, mathematical formulas, benchmark numbers, attribution accuracy

---

## Overall Assessment

The document is a thorough and well-structured technical reference. The mathematical framework
(quantization noise theory, SQNR, error accumulation) is largely sound. However, there are
**significant errors in the NF2 quantization level values**, the document **conflates NF2 with
Lloyd-Max quantization** (they are different quantizers), and nearly every benchmark perplexity
number deviates from what the source papers actually report. Several numbers appear to be
fabricated or interpolated from memory rather than sourced from papers.

---

## 1. NormalFloat2 Quantization Levels -- ERRORS FOUND

### Claim (lines 258-268):
> NF2 levels (conditional means within each quartile):
>   q1 = E[X | X < -0.6745] = -1.0494
>   q2 = E[X | -0.6745 < X < 0] = -0.3191
>   q3 = E[X | 0 < X < 0.6745] = 0.3191
>   q4 = E[X | X > 0.6745] = 1.0494
>
> After normalization to unit variance:
> NF2 values ~ {-0.7979, -0.2677, 0.2677, 0.7979}

### Verdict: INCORRECT

**The conditional means are wrong.** Verified by direct computation:

- E[X | X < -0.6745] = -phi(-0.6745) / Phi(-0.6745) = -0.3178 / 0.2500 = **-1.2711** (not -1.0494)
- E[X | -0.6745 < X < 0] = (phi(-0.6745) - phi(0)) / (0.5 - 0.25) = **-0.3247** (not -0.3191)
- By symmetry: +0.3247 and +1.2711

The document's values (-1.0494 and -0.3191) do not correspond to any standard quantizer for
N(0,1). The value -1.0494 appears to be a misremembered number -- it does not match NF2
conditional means, nor Lloyd-Max reconstruction levels (-1.5104), nor any other standard result.

**The normalized values {-0.7979, -0.2677, 0.2677, 0.7979} are consequently also wrong**, since
they are derived from the incorrect conditional means. (Note: 0.7979 = sqrt(2/pi) is the correct
conditional mean for the 1-bit/2-level case E[X | X > 0], not the 2-bit case.)

### Additional Conflation Error (lines 281-344):

The document presents the Lloyd-Max quantizer section as if NF2 and Lloyd-Max are the same thing.
They are not:

| Property | NF2 (QLoRA) | Lloyd-Max 4-level |
|----------|-------------|-------------------|
| Decision boundaries | -0.6745, 0, +0.6745 (quartiles) | -0.9816, 0, +0.9816 (optimized) |
| Reconstruction levels | -1.2711, -0.3247, +0.3247, +1.2711 | -1.5104, -0.4528, +0.4528, +1.5104 |
| MSE (sigma=1) | ~0.1394 | ~0.1175 |
| Optimality | Equal-probability bins | Minimum MSE (jointly optimal boundaries + levels) |

NF2 uses quartile boundaries (equal probability mass per bin) and conditional means within those
bins. Lloyd-Max jointly optimizes both boundaries and reconstruction levels to minimize MSE. They
produce different quantizers. The document's claim that NF2 boundaries are at the quartiles is
correct, but then it incorrectly states the Lloyd-Max boundaries are also at the quartiles
(line 290-295). **The true Lloyd-Max boundaries for a 4-level Gaussian quantizer are at
+/-0.9816, NOT at the quartiles +/-0.6745.**

### MSE Values

- Document claims D_LM = sigma^2 * 0.11885. Standard reference value (Max 1960, Gersho & Gray)
  is **0.1175**. The document's value is ~1% off. Minor discrepancy, possibly from a different
  source or rounding.
- Document claims d(2) = 0.36340 for 1-bit. Computed: **0.36338**. Essentially correct.
- Document claims d(16) = 0.03454 for 4-bit. This is a standard reference value and is correct.

---

## 2. QuIP# Hadamard Rotation -- CORRECT

### Claim: Randomized Hadamard Transform W' = U * W * V^T with U = H_n * S_1, V = H_n * S_2

**Verdict: CORRECT.** This matches the QuIP/QuIP# papers. The use of diagonal sign matrices S_1
and S_2 with the normalized Hadamard matrix is accurately described.

### Claim: O(n log n) FWHT cost

**Verdict: CORRECT.** The Fast Walsh-Hadamard Transform requires n log_2(n) additions/subtractions
for a sequence of length n = 2^k. The butterfly structure described (stages, pairs at distance
2^s) is accurate.

### Claim: Incoherence reduction to O(log n)

**Verdict: CORRECT.** This is a well-known property of randomized Hadamard transforms.

---

## 3. E8 Lattice -- MOSTLY CORRECT, ONE NUANCE

### Claim: E8 is 8-dimensional, kissing number 240

**Verdict: CORRECT.** The E8 lattice is indeed 8-dimensional with kissing number 240 (proven
optimal in 8D by Levenshtein and Odlyzko-Sloane in 1979).

### Claim: Definition (lines 625-632)
> E8 = { x in Z^8 : sum(x_i) is even }
>      union
>      { x in (Z+1/2)^8 : sum(x_i) is even }

**Verdict: CORRECT.** This is the "even coordinate system" (Gamma_8) for the E8 lattice, which
is a standard and valid definition. (The alternative "odd coordinate system" Gamma'_8 uses even
sum for integer coordinates and odd sum for half-integer coordinates; both are isomorphic.)

### Claim: Center density pi^4 / 384 ~ 0.2537

**Verdict: CORRECT.** This is the packing density of the E8 lattice, proven optimal by
Viazovska (2016).

### Claim: Automorphism group order 696,729,600

**Verdict: CORRECT.** |W(E8)| = 2^14 * 3^5 * 5^2 * 7 = 696,729,600.

### Claim: G(E8) ~ 0.0717 (normalized second moment)

**Verdict: CORRECT.** This is a standard value from the lattice quantization literature.

### Claim: "Viazovska, 2016, Fields Medal work" (line 621)

**Verdict: PARTIALLY INCORRECT.** Viazovska proved the E8 sphere packing result in **March 2016**,
with the paper published in Annals of Mathematics in **2017**. She received the Fields Medal in
**2022**, not 2016. The phrase "Fields Medal work" is misleading as written -- it implies the Fields
Medal was awarded in 2016. The references section (line 1820) correctly cites "2017" for the
publication.

---

## 4. AQLM Multi-Codebook Formulation -- CORRECT

### Claim: w_hat = sum_{m=1}^{M} C_m[i_m] with joint optimization of L = ||WX - W_hat * X||_F^2

**Verdict: CORRECT.** The additive codebook structure and the layer-wise Hessian-weighted
objective are accurately described and match the AQLM paper (Egiazarian et al., 2024).

### Claim: "AQLM is Pareto-optimal... A 2.76-bit AQLM quantization of a 13B model can outperform the uncompressed 7B model"

**Verdict: PLAUSIBLE but [VERIFY NUMBERS].** The general claim about Pareto-optimality at
sub-3-bit was valid at the time of the AQLM paper, though QTIP and VPTQ have since improved
upon it at some operating points.

---

## 5. GPTQ Error Compensation Formula -- CORRECT

### Claim (lines 746-748):
> w_{q+1:n} = w_{q+1:n} - (delta_q / [H^{-1}]_{qq}) * [H^{-1}]_{q, q+1:n}

**Verdict: CORRECT.** This is the standard OBS-derived GPTQ update rule. The weight update
compensates for the quantization error delta_q using the row of the inverse Hessian,
scaled by the diagonal element. This matches the GPTQ paper (Frantar et al., ICLR 2023).

### Claim: H = 2XX^T (line 734)

**Verdict: MINOR NOTE.** The factor of 2 depends on the loss definition. The GPTQ paper
defines H = XX^T (without the factor of 2) when the loss is (1/2)||wX - w_hat X||^2.
The document's convention (H = 2XX^T) corresponds to the loss ||wX - w_hat X||^2 without
the 1/2 factor. Either convention is valid as long as applied consistently, and the update
rule is the same regardless.

### Claim: GPTQ is ICLR 2023 (line 1804)

**Verdict: CORRECT.** Confirmed: GPTQ was published at ICLR 2023.

---

## 6. Rate-Distortion Optimal Allocation -- CORRECT

### Claim: Lagrangian L = sum [D_i(b_i) + lambda * b_i], optimal condition dD_i/db_i = -lambda

**Verdict: CORRECT.** This is the standard Lagrangian formulation for constrained optimization.
The optimality condition (equal marginal distortion reduction across all blocks) is a
well-known result from rate-distortion theory and is correctly derived.

### Claim: Greedy algorithm (start at min bits, upgrade block with largest marginal reduction) is optimal for convex D_i(b)

**Verdict: CORRECT.** For convex distortion functions with discrete bit allocations, the greedy
algorithm achieves the global optimum. This is a standard result.

---

## 7. Perplexity Numbers -- MAJOR CONCERNS

**Nearly every benchmark number in the document differs from what the actual papers report.**
The document appears to have assembled numbers from memory or secondary sources rather than
from the original papers. All numbers should be treated as unreliable until verified against
the source papers.

### FP16 Baselines

| Model | Document claims | AQLM paper reports | QuIP# paper reports | QTIP paper reports |
|-------|----------------|--------------------|--------------------|-------------------|
| Llama-2 7B | 5.47 | 5.12 | 5.47 | 5.12 |
| Llama-2 70B | 3.32 | 3.12 | 3.32 | -- |

**Issue:** Different papers report different FP16 baselines because of different evaluation
setups (context length, stride, tokenization). The document uses 5.47 for 7B, which matches
QuIP# but not AQLM or QTIP. This is not an error per se, but the document does not note this
discrepancy. **All baseline values should cite their source paper and evaluation setup.**

### 2-bit Perplexity Numbers (Llama-2 7B) -- [VERIFY NUMBERS]

| Method | Document claims | Paper reports | Source |
|--------|----------------|---------------|--------|
| QuIP# (E8P) | 9.20 | **6.66** (with fine-tuning) | QuIP# paper Table 2 |
| AQLM (1-codebook) | 8.75 | **6.59** (at 2.02 bpw) | AQLM paper Table 1 |
| VPTQ | 8.68 | [VERIFY -- paper PDF not parseable] | VPTQ EMNLP 2024 |
| QTIP | ~8.3 | **5.86-6.89** (varies by code type) | QTIP paper Table 5 |
| QuIP | 15.7 | [VERIFY] | QuIP NeurIPS 2023 |
| GPTQ | 107.4 | [VERIFY] | |
| GPTQ + act-order | 43.2 | [VERIFY] | |

**The document's numbers for QuIP#, AQLM, and QTIP are dramatically wrong.** The document
reports values in the 8-9 range, but the actual papers report values in the 5.8-6.7 range.
This is a difference of 2-3 perplexity points -- enormous in this context.

**Possible explanation:** The document may be reporting numbers for the *without fine-tuning*
variants at exactly 2.00 bpw, while the papers' headline numbers include fine-tuning and may
be at slightly higher effective bpw (2.02-2.15). Additionally, QuIP# at exactly 2.0 bpw
without fine-tuning may indeed be around 8-9 on some evaluation setups. However, the document
does not make these distinctions clear, and the numbers as presented are misleading.

### 2-bit Perplexity Numbers (Llama-2 70B) -- [VERIFY NUMBERS]

| Method | Document claims | Paper reports |
|--------|----------------|---------------|
| QuIP# | 4.82 | **4.16** (QuIP# paper) |
| AQLM | 4.54 | **3.94** (AQLM paper at 2.07 bpw) |
| VPTQ | ~4.4 | [VERIFY] |

Same pattern: the document's numbers are higher than what the papers report.

### 2.5-bit Perplexity Numbers -- [VERIFY NUMBERS]

All numbers in the 2.5-bit table (lines 1163-1170) should be independently verified. Given
the pattern of errors in the 2-bit table, these are likely also inaccurate.

### llama.cpp IQ Quantization Numbers (lines 1182-1191) -- [VERIFY NUMBERS]

These community-derived numbers fluctuate with llama.cpp versions and evaluation methodology.
All should be verified against the current llama.cpp perplexity README.

### Task Degradation Percentages (lines 1200-1231) -- [VERIFY NUMBERS]

Claims like "GSM8K accuracy drops 15-30% at 2-bit" and "HumanEval pass@1 drops 10-25%" are
stated without citations. These ranges are plausible but should be sourced.

---

## 8. BitNet / 1-bit Claims -- MOSTLY CORRECT

### Claim: BitNet b1.58 uses ternary weights {-1, 0, +1}, which is 1.58 bits (log_2(3) = 1.585)

**Verdict: CORRECT.**

### Claim: BitNet b1.58 2B4T is "competitive with full-precision models of the same size (e.g., Llama-3 3B)"

**Verdict: PARTIALLY CORRECT.** The BitNet b1.58 2B4T paper compares against Llama 3.2 1B,
Gemma 3 1B, and Qwen 2.5 1.5B -- not "Llama-3 3B". BitNet 2B4T is a 2B parameter model, so
it is compared with similarly-sized models. On ARC-Challenge it scored 68.5% vs Llama 3 3B's
68.2%, but this is not a systematic comparison. The document's claim of being "competitive
with Llama-3 3B" overstates the comparison made in the paper.

### Claim: "2.7x faster inference, 3.5x less memory"

**Verdict: APPROXIMATELY CORRECT but imprecise.** The original BitNet b1.58 paper (Ma et al.,
2024) reports **2.71x faster** and **3.55x less GPU memory** specifically at the 3B model size
when comparing to LLaMA. The document rounds these to 2.7x and 3.5x, which is acceptable.
However, the BitNet b1.58 2B4T technical report (2025) reports different numbers: up to
**6.17x speedup** on x86 CPUs and **0.4 GB memory** (vs 2.6 GB for Llama-3-2B-FP16). The
document does not distinguish between these two BitNet publications.

### Claim: BitLinear code sample (lines 1603-1617)

**Verdict: CORRECT in spirit.** The pseudocode captures the essential mechanism: ternary weight
quantization via round(w/alpha).clamp(-1,1) with mean absolute value scaling and 8-bit
activation quantization. This is a simplified but accurate representation.

### Claim: OneBit achieves "at least 81% of the non-quantized performance"

**Verdict: CORRECT.** This matches the OneBit paper (Xu et al., NeurIPS 2024).

---

## 9. Memory/Speed Calculations -- CORRECT

### Memory arithmetic verification:

All memory calculations in the tables (lines 29-57) have been verified:

| Claim | Calculation | Correct? |
|-------|------------|----------|
| 70B at FP16 = 140 GB | 70e9 * 16 / 8 / 1e9 = 140.0 | Yes |
| 70B at 4-bit = 35 GB | 70e9 * 4 / 8 / 1e9 = 35.0 | Yes |
| 70B at 2-bit = 17.5 GB | 70e9 * 2 / 8 / 1e9 = 17.5 | Yes |
| 70B at 2.5-bit = 21.9 GB | 70e9 * 2.5 / 8 / 1e9 = 21.875 | Yes (rounded) |
| 405B at FP16 = 810 GB | 405e9 * 16 / 8 / 1e9 = 810.0 | Yes |
| 405B at 2-bit = 101.25 GB | 405e9 * 2 / 8 / 1e9 = 101.25 | Yes |
| 109B at 2.5-bit = 34.1 GB | 109e9 * 2.5 / 8 / 1e9 = 34.0625 | Yes (rounded) |

All memory calculations are arithmetically correct.

### Quantization noise scaling:

- sigma_q^2(2-bit) / sigma_q^2(4-bit) = 16x: **Correct** (192/3072 inverted = 3072/192 = 16)
- SQNR = 6.02n + 1.76: **Correct** (standard formula for full-scale sinusoid)
- All SQNR table values (lines 125-132): **Verified correct**
- L*n cumulative noise calculations (lines 196-198): **Correct** (213.3 and 3413.3, rounded)

---

## 10. Other Issues

### Reference Accuracy

| Paper | Document says | Actually |
|-------|-------------|----------|
| GPTQ venue | ICLR 2023 | ICLR 2023 -- **Correct** |
| Viazovska year | "2016, Fields Medal work" (line 621) / "2017" (line 1820) | Proof: 2016. Publication: 2017. Fields Medal: **2022**. |
| AQLM venue | ICML 2024 (line 1810) | Originally on arXiv Jan 2024; **verify if ICML or NeurIPS** |
| QTIP venue | NeurIPS 2024 Spotlight | **Correct** |
| AWQ venue | MLSys 2024 | **Correct** |

### NF4 Codebook Values (lines 246-248)

The NF4 values listed appear correct and match the QLoRA paper's hardcoded codebook.
However, the document states these are "midpoints of 16 equal-probability intervals" (line 243).
This is imprecise -- NF4 values are computed as q_i = (1/2)[Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1))]
where Q_X is the quantile function, which is the average of adjacent quantile boundaries, not
the midpoint of the interval.

### Lloyd-Max Distortion Ratio Claim (lines 341-343)

> The Lloyd-Max quantizer reduces distortion by ~13% over uniform quantization at 2-bit.

Using the document's own numbers: (0.13671 - 0.11885) / 0.13671 = 13.1%. This is internally
consistent but both underlying values are slightly off from standard references. The qualitative
claim (modest improvement, not transformative) is correct.

---

## Summary of Required Corrections

### Critical (must fix):

1. **NF2 conditional means** (lines 258-261): Replace -1.0494 with -1.2711 and -0.3191 with
   -0.3247 (and their positive counterparts). Or clarify that these are not NF2 values.

2. **NF2 normalized values** (line 267): {-0.7979, -0.2677, 0.2677, 0.7979} are wrong.
   Recompute from correct conditional means.

3. **Lloyd-Max boundaries** (lines 290-295): The Lloyd-Max 4-level quantizer for a Gaussian
   does NOT have boundaries at the quartiles. Replace -0.6745 with -0.9816 and +0.6745 with
   +0.9816 in the Lloyd-Max section, OR clearly separate the NF2 description (quartile-based)
   from the Lloyd-Max description (MSE-optimal).

4. **All perplexity tables** (Section 4.1): Nearly every number is suspect. Either verify
   each number against the cited paper's exact table, or add [UNVERIFIED] tags and cite the
   exact paper table number. The QuIP#, AQLM, and QTIP numbers for Llama-2 7B are off by
   2-3 points, which is a very large error in this context.

### Important (should fix):

5. **Viazovska Fields Medal** (line 621): Change "2016, Fields Medal work" to "2016" or
   "2016; Fields Medal 2022". The Fields Medal was awarded six years after the proof.

6. **BitNet comparison** (line 1626): The claim of being "competitive with Llama-3 3B" should
   be more precise about which benchmarks and which Llama variant.

7. **AQLM venue** (line 1810): Verify whether AQLM was ICML 2024 or another venue.

### Minor (nice to fix):

8. **H = 2XX^T** (line 734): Note that the factor of 2 depends on loss normalization.

9. **NF4 description** (line 243): "midpoints" should be "quantile averages" or similar.

10. **Task degradation ranges** (Section 4.2): Add citations for the claimed percentage drops.
