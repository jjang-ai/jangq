# Audit: 03-importance-aware-quantization-methods.md

Auditor: Claude Opus 4.6 (1M context)
Date: 2026-03-14

This audit checks every mathematical formulation, algorithmic description, and benchmark
number in the document. Claims are rated:

- **CORRECT** -- verified against primary sources or well-established knowledge
- **INCORRECT** -- contradicted by primary sources; correction provided
- **UNCERTAIN** -- plausible but could not be fully verified; may be an approximation or simplification
- **VERIFY NUMBERS** -- benchmark numbers that should be cross-checked against the original papers

---

## 1. Foundational Concepts (Section 1)

### 1.1 Uniform Quantization Baseline

**CORRECT.** The formulas for Q(w), Dequant(q), scale s, and zero point z are standard
and correctly stated. The maximum error bound |e| <= s/2 is correct.

### 1.2 Layer-Wise Quantization Objective

**CORRECT.** The objective `minimize ||WX - Q(W)X||_F^2` is the standard layer-wise
reconstruction objective used across all methods. The distinction between weight error
and output error is correctly noted.

### 1.3 The Hessian Connection

**CORRECT.** The Taylor expansion `dW^T * H * dW` and `H = 2 * X^T * X` for the linear
layer squared-error case are standard results. The connection between Hessian diagonal
entries and weight importance is correctly stated.

Minor note: The factor of 2 in `H = 2 * X^T * X` is a convention tied to the specific
loss formulation (squared error without the 1/2 factor). Some papers drop the factor of
2. This is internally consistent within the document.

### 1.4 Group Quantization

**CORRECT.** The group quantization description, overhead calculations, and effective
bit-width formulas are all correct. The 32-bit overhead per group (16-bit scale +
16-bit zero) is the standard convention.

---

## 2. AWQ (Section 2)

### 2.1 Core Insight

**CORRECT.** The observation that ~0.1%-1% of weights are disproportionately important,
the mixed-precision impracticality argument, and the per-channel scaling solution are
all faithful to the AWQ paper (Lin et al., arXiv:2306.00978).

### 2.2.1 Importance Metric

**[UNCERTAIN]** The document defines the importance metric as:

```
importance(c) = ||X_{:,c}||_2 = sqrt(sum_i X_{i,c}^2)
```

The AWQ paper actually uses the **average magnitude** of activations per channel
(mean of absolute values), not the L2 norm. Multiple sources confirm the paper uses
"average magnitude of input activations on a per-channel basis." The L2 norm and the
mean absolute magnitude are related but not identical:

- L2 norm: sqrt(sum_i x_{i,c}^2)
- Mean absolute magnitude: (1/n) * sum_i |x_{i,c}|

The document's claim that this equals the square root of the Hessian diagonal is
correct for the L2-norm interpretation (since [H]_{cc} = sum_i X_{i,c}^2), but the
actual AWQ implementation likely uses mean absolute magnitude. The conceptual point is
the same -- channels with larger activations are more important -- but the precise
formula as stated may not exactly match the paper.

### 2.2.2 Per-Channel Scaling

**CORRECT.** The per-channel scaling formulation `Q(W * diag(s)) * diag(s^{-1})` and
the mathematical equivalence at full precision are correct. The explanation of how
scaling redistributes quantization precision is accurate.

### 2.2.3 Optimal Scale Derivation

**CORRECT in spirit, UNCERTAIN in detail.** The document states:

```
s_c = (||X_{:,c}||_2)^alpha
```

Based on the paper, the actual formula is `s = s_X^alpha` where `s_X` is the average
activation magnitude per channel. Multiple sources confirm this power-law grid search
formulation. However, as noted above, the exact definition of `s_X` may be mean
absolute magnitude rather than L2 norm.

The grid search over alpha in [0, 1] with ~20 values is consistent with descriptions of
the AWQ algorithm. The claim that alpha ~ 0.5 is typical is **[UNCERTAIN]** -- I could
not find the specific default alpha value in the sources, though it is plausible.

### 2.2.4 Complete Algorithm

**CORRECT** in structure. The three-step process (collect activation statistics, grid
search alpha, apply optimal scaling) accurately describes the AWQ pipeline.

### 2.3.1 - 2.3.3 Implementation Details

**CORRECT.** The two approaches for scale application at inference (absorb into
activations or into subsequent layer) are correctly described. The calibration
requirements (128 sequences of 512 tokens) are **[UNCERTAIN]** -- the paper may use
different sequence lengths in different configurations.

### 2.4-2.5 Strengths/Weaknesses

**CORRECT.** The characterization of AWQ's strengths (speed, 4-bit quality, simplicity)
and weaknesses (uniform bit width, struggles below 3-bit) are accurate.

### Paper Citation

**CORRECT.** AWQ by Lin et al. was published at MLSys 2024 (won Best Paper), arXiv:2306.00978.

---

## 3. GPTQ (Section 3)

### 3.1 Lineage: OBS -> OBQ -> GPTQ

#### 3.1.1 OBS

**CORRECT.** The OBS error formula:

```
delta_L = (w_q)^2 / (2 * [H^{-1}]_{qq})
```

is correctly stated. The compensating weight update:

```
delta_w = -w_q / [H^{-1}]_{qq} * H^{-1}_{:,q}
```

is also correct. Confirmed against the original Hassibi & Stork (1993) paper. The OBS
saliency measure identifies which weight to prune by minimizing `w^2 / [H^{-1}]_{pp}`.

#### 3.1.2 OBQ

**CORRECT.** The extension from pruning to quantization is accurately described. The
error formula `(w_q - Q(w_q))^2 / (2 * [H^{-1}]_{qq})` and the compensating update are
correct adaptations of OBS.

The complexity claim O(d^3) per row for OBQ is **CORRECT** -- this cubic complexity in
the input dimension is confirmed by the Frantar & Alistarh (2022) paper and is what
motivated GPTQ's improvements.

**[INCORRECT] Minor:** The document attributes OBQ to "Frantar & Alistarh (2022)" --
the paper is actually titled "Optimal Brain Compression" (OBC), not "Optimal Brain
Quantization" (OBQ). The paper was published at NeurIPS 2022 with authors Frantar,
Singh, and Alistarh (three authors, not two). The quantization component is sometimes
called OBQ, but the paper title is OBC. This is a minor naming issue.

### 3.2.1 Row-Independence

**CORRECT.** The decomposition into independent rows sharing the same Hessian is
correctly stated.

### 3.2.2 Column-Wise Processing

**CORRECT.** The GPTQ update rule:

```
q_j = Q(w_j)
e_j = (w_j - q_j) / [H^{-1}]_{jj}
w_{j+1:} -= e_j * H^{-1}_{j, j+1:}
```

is correctly stated. This is the core GPTQ loop.

### 3.2.3 Derivation of the Update Rule

**CORRECT.** The derivation from the constrained optimization problem through the Schur
complement identity to the final update rule is mathematically sound.

However, there is a minor notational issue on line 475:

```
e(w) = ||(w - q) * X||^2 = (w - q) * X * X^T * (w - q)^T = (w - q) * H/2 * (w - q)^T
```

The inner product should use `X * X^T` which equals `H/2`, but the row vector
convention used here requires careful attention. This is consistent within the document
but could be confusing -- the Hessian for a row of W acting on inputs X is indeed
`X * X^T` (where X is n x d_in), but this is `H^T/2` under the column convention. The
factor of 2 and transpose are handled consistently in the document.

### 3.2.4 The Cholesky Decomposition Trick

**[INCORRECT]** The document states:

```
Let H^{-1} = L * L^T (Cholesky factorization)
```

This is misleading. The GPTQ paper actually uses the **Cholesky decomposition of
H^{-1}** to precompute the needed rows, but the key point is more subtle. Multiple
sources clarify that GPTQ uses the Cholesky rows of H^{-1} to read off the needed
values `[H^{-1}]_{jj}` and `H^{-1}_{j, j+1:}` during the column sweep. Some
descriptions say GPTQ uses LDL^T factorization rather than pure Cholesky (LL^T).

The claim that "the Cholesky decomposition 'encodes' the sequential elimination
structure that matches the left-to-right processing order" is essentially correct --
this is the key insight. But the specific formula `[H^{-1}]_{jj} = ||L_{j,:}||^2` is
only correct if H^{-1} = L * L^T with L lower triangular, and even then the row access
pattern is more nuanced because GPTQ needs the *remaining* sub-block of H^{-1} at each
step, not just individual elements.

In practice, GPTQ takes the Cholesky factorization of H^{-1} (after damping and
inversion) and uses the Cholesky rows directly. The document's high-level description is
correct but the specific formula presentation slightly oversimplifies.

### 3.2.4 Damping Term

**CORRECT.** The damping term `lambda ~ 0.01 * mean(diag(H))` is consistent with the
GPTQ implementation. Sources confirm `damp_percent = 0.01` as the typical value.

### 3.2.5 Block-Wise Quantization

**CORRECT.** The block processing with B=128 columns, accumulating errors within the
block and applying a single matrix multiply for cross-block updates, is accurately
described.

### 3.3 Act-Order

**CORRECT.** The activation-order heuristic (processing columns in decreasing order of
Hessian diagonal) is correctly described, including the intuition for why it helps.

### 3.4 Complete Algorithm (Pseudocode)

**CORRECT** in overall structure. One detail to note:

The pseudocode divides the error by `L[col, col]` (the Cholesky diagonal), which is
consistent with the document's own Cholesky formulation. In implementations, this
corresponds to dividing by `[H^{-1}]_{jj}` or equivalently using the Cholesky rows.

### 3.5 Complexity Analysis

**CORRECT.** The complexity breakdown is accurate.

### 3.6-3.7 Strengths/Weaknesses

**CORRECT.** The claimed perplexity of "within 0.05-0.10 perplexity of fp16" at 4-bit
is **[VERIFY NUMBERS]** -- this is a commonly stated figure but exact numbers depend on
the model, calibration data, and group size.

### Paper Citation

**CORRECT.** GPTQ by Frantar et al. was published at ICLR 2023. arXiv:2210.17323.

---

## 4. EXL2 (Section 4)

### 4.1 Core Innovation

**CORRECT.** EXL2 is indeed the first practical mixed-precision quantization scheme for
LLMs, and it is authored by turboderp for the ExLlamaV2 project. The description of
per-block bit allocation is accurate.

### 4.2.1 Stage 1: Sensitivity Measurement

**CORRECT** in concept. The trial-quantization approach at multiple bit widths and the
Hessian-weighted block reconstruction error are accurately described.

**[UNCERTAIN]** The exact candidate bit widths listed as {2, 3, 4, 5, 6, 8} are
plausible but since EXL2 has no formal paper, these are based on code and community
documentation. Some implementations may support different sets.

### 4.2.2 Stage 2: Bit Allocation

**CORRECT** in concept. The greedy knapsack algorithm using marginal benefit per bit
cost is a reasonable description. The concavity argument for why greedy works well is
mathematically sound.

**[UNCERTAIN]** The specific implementation details (MaxHeap, upgrade paths like
2->3->4->5->6->8) are a plausible reconstruction but since there is no formal paper,
these are inferred from code behavior. The actual implementation may differ in details.

### 4.2.3 Stage 3: Final Quantization

**CORRECT.** The use of GPTQ-style optimal rounding within each block and cross-block
compensation is consistent with how EXL2 works.

### 4.3 Bit Packing and Storage

**CORRECT.** The variable-width packed format description is accurate. The overhead
calculation (including uint8 for bit width) is reasonable:

```
Effective bits = 3.0 + (16 + 16 + 8) / 128 = 3.3125
```

This is correct arithmetic, though the actual overhead may differ slightly depending on
implementation details (e.g., whether the bit-width indicator is stored per group or
amortized differently).

### 4.5 Quality Results

**[VERIFY NUMBERS]** The entire perplexity table:

```
fp16:           5.20
GPTQ 4-bit:     5.26  (+0.06)
GPTQ 3-bit:     5.72  (+0.52)
GPTQ 2-bit:     11.4  (+6.2)
EXL2 4.0 bpw:   5.23  (+0.03)
EXL2 3.0 bpw:   5.41  (+0.21)
EXL2 2.5 bpw:   5.62  (+0.42)
EXL2 2.0 bpw:   6.85  (+1.65)
```

These numbers should all be verified. Key concerns:

1. **fp16 baseline of 5.20**: The Llama 2 70B fp16 WikiText-2 perplexity is commonly
   cited in the range of ~3.3 to ~5.5 depending on the context length and evaluation
   setup. With context length 2048 (which most quantization papers use), the value 5.20
   is plausible but should be verified against the specific evaluation protocol. The
   original GPTQ paper did not evaluate Llama 2 (it was published before Llama 2).

2. **GPTQ numbers**: The original GPTQ paper (Frantar et al., ICLR 2023) tested on OPT
   and BLOOM, NOT Llama 2. Any GPTQ numbers for Llama 2 come from later reproductions
   (AutoGPTQ, GPTQ-for-LLaMA). Specific values depend heavily on group size, act-order,
   and calibration data.

3. **EXL2 numbers**: Since EXL2 has no formal paper, these numbers likely come from
   community benchmarks or turboderp's own measurements. They are plausible but not
   peer-reviewed.

**All benchmark numbers in this table should be treated as approximate/illustrative
rather than canonical results.**

---

## 5. SpQR (Section 5)

### 5.2.1 The Sensitivity Metric

**CORRECT.** The sensitivity metric:

```
sensitivity_{ij} = w_{ij}^2 * [H]_{jj}
```

This is confirmed by the SpQR paper. The derivation connecting weight magnitude squared
with Hessian diagonal entry is correctly stated.

**[UNCERTAIN]** The derivation section claims:

```
sensitivity_{ij} ~ E[delta_{ij}^2] * [H]_{jj} ~ w_{ij}^2 * [H]_{jj}
```

The argument that `w_{ij}^2` is a proxy for expected squared quantization error is a
simplification. The actual rounding error depends on the weight's position within the
quantization grid, not directly on its magnitude. However, for outlier weights (which
have extreme values), the squared magnitude is indeed correlated with quantization
error, so the intuition is correct even if the exact derivation is an approximation.

### 5.2.2 Outlier Detection

**CORRECT.** The threshold-based detection with either quantile or mean+k*std methods
is consistent with the paper. The ~1% outlier fraction is confirmed.

### 5.2.3 Quantization Procedure

**CORRECT.** The dense+sparse decomposition and storage analysis are accurately
described. The effective bits calculation:

```
3.0 * 0.99 + (16 + 13) * 0.01 = 2.97 + 0.29 = 3.26 bpw
```

is correct arithmetic.

### 5.2.4 GPTQ Integration

**CORRECT** in concept. The modified GPTQ loop that fixes outliers at fp16 and only
compensates non-outlier weights is accurately described.

### Paper Citation

**[INCORRECT] Minor.** The document cites SpQR as "Dettmers et al., 2023." The paper
was actually published at **ICLR 2024** (not 2023), though the arXiv preprint appeared
in June 2023. The correct venue citation is ICLR 2024.

---

## 6. SqueezeLLM (Section 6)

### 6.2.2 Sensitivity-Weighted K-Means

**CORRECT.** The sensitivity-weighted k-means objective:

```
minimize sum_i sensitivity_i * (w_i - c_{a(i)})^2
```

is a faithful description of the paper's approach.

### 6.2.3 Fisher Information as Sensitivity

**CORRECT.** The Fisher information formulation:

```
F_{ij} = E_x [(d log p(x) / d w_{ij})^2]
```

and its approximation via empirical gradient variance are correctly stated. The claim
that Fisher information equals the expected Hessian at the optimum is a standard result.

**[UNCERTAIN]** The document says SqueezeLLM uses the Fisher information "instead of
the Hessian diagonal (as in GPTQ/SpQR)." The SqueezeLLM paper actually uses squared
gradients (gradient-squared sensitivity), which is the empirical approximation to the
Fisher information. The distinction is essentially correct but the document could be
clearer that it is the diagonal of the empirical Fisher (i.e., squared gradients
averaged over data), not the full Fisher matrix.

### 6.2.4 The Weighted K-Means Algorithm

**[UNCERTAIN]** The pseudocode shows the assignment step using **unweighted** distance:

```
assignments = [argmin_j (w_i - centers[j])^2 for w_i in weights]
```

It is not clear from the paper whether the assignment step uses sensitivity-weighted or
unweighted distance. Standard sensitivity-weighted k-means uses unweighted assignment
(nearest center) but weighted centroid updates, which is what the document shows. This
is plausible but I could not verify this specific detail against the paper.

### 6.2.5 Lookup Table Overhead

**CORRECT.** The LUT overhead calculations are correct:

- 4-bit, group size 128: 16 * 16 / 128 = 2.0 bits per weight, effective = 6.0 bpw
- 3-bit, group size 128: 8 * 16 / 128 = 1.0 bit per weight, effective = 4.0 bpw

This overhead is a well-known drawback of non-uniform quantization.

### 6.3 Dense-and-Sparse Decomposition

**CORRECT.** The decomposition and outlier detection formula using Fisher information
(`F_{ij} * w_{ij}^2 > tau`) are consistent with the paper.

### Paper Citation

**CORRECT.** SqueezeLLM by Kim et al. was published at ICML 2024. arXiv:2306.07629.

---

## 7. QuIP# (Section 7)

### 7.2.1 Coherence and Quantization Error

**CORRECT.** The coherence definition:

```
mu(W) = (n / ||W||_F^2) * max_j ||W_{:,j}||_2^2
```

is standard. The minimum (mu=1) and maximum (mu~n) values are correct.

**[UNCERTAIN]** The error bound:

```
E[||W - Q(W)||_F^2] <= C * mu(W) * ||W||_F^2 / 2^{2b}
```

This captures the spirit of the coherence-error relationship from the QuIP/QuIP# papers
but the exact form of the theorem may differ. The claim that reducing coherence from 100
to 1 is "equivalent to ~3.3 more bits" is an informal calculation (100 = 2^{2*3.3}),
which is approximately correct (2^6.6 ~ 97).

### 7.2.2 Incoherence via Random Orthogonal Transforms

**CORRECT.** The rotation formulation W_rotated = U * W * V^T and the inference
computation y = U^T * W_rotated * (V * x) are correctly stated.

### 7.2.3 Hadamard Matrices

**CORRECT.** The recursive Hadamard definition and the randomized Hadamard matrix
R = H_n * D with random diagonal signs are standard.

**[INCORRECT] Minor:** The document states "Every entry of H_n is +/- 1/sqrt(n)."
This is correct for the **normalized** Hadamard matrix. The unnormalized version has
entries +/- 1, and the normalization factor is 1/sqrt(n). This is fine as long as it is
clear the document refers to the normalized form.

### 7.2.4 Fast Walsh-Hadamard Transform

**CORRECT.** The butterfly-style algorithm is correctly described. The O(n log n)
complexity is standard.

### 7.3.1 The E8 Lattice

**CORRECT.** The E8 lattice achieves the densest sphere packing in 8 dimensions (proven
by Viazovska, 2016). The 240 shortest vectors (roots) are correct -- confirmed by
multiple mathematical references.

The normalized second moment:

```
G(E8) = 0.0717
G(Z^8) = 1/12 = 0.0833
```

**CORRECT.** The G(E8) = 0.0717 value (more precisely 0.07168) is confirmed by lattice
quantizer literature. The G(Z^8) = 1/12 = 0.0833 is a standard result. The improvement
ratio 0.0833/0.0717 = 1.16 (16% reduction) is correct arithmetic.

### 7.3.2 E8 Quantization

**[INCORRECT]** The document states:

> "QuIP# groups weights into vectors of 8 and quantizes each vector to the nearest E8
> lattice point"

and

> "For 2-bit quantization (4 levels per weight, 4^8 = 65536 total for 8 weights), the
> codebook contains 65536 E8 lattice points"

This is partially correct but misses key details. QuIP# actually uses a codebook called
**E8P**, which is a specific subset/variant of the E8 lattice. The E8P codebook has
2^16 entries but exploits E8 symmetries to require only a lookup into a size-2^8
codebook (1 KiB instead of 1 MiB). The codebook is not simply "65536 E8 lattice points
selected to cover the most likely weight vectors" -- it is a carefully constructed
codebook using the half-integer lattice D8-hat and union structure of E8 = D8-hat union
(D8-hat + 1/2).

Additionally, the document states the encoding uses "16 bits per group of 8 weights =
2 bits per weight." This is correct for the 2-bit case.

### 7.4 Complete QuIP# Pipeline

**[INCORRECT]** The pseudocode describes the quantization procedure as:

```
# Step 4: LDLQ (LDLT-based quantization with Hessian compensation)
# Similar to GPTQ but with E8 lattice rounding
```

But then the compensation step reads:

```
W_rot[:, j+8:] -= error * H_rot[j:j+8, j+8:]^{-1}
```

This compensation formula is incorrect. The GPTQ-style update should use the Hessian
**inverse**, not `H^{-1}` applied to the error-Hessian product. The actual QuIP# paper
uses **Block LDLQ**, which is based on a g-block LDL decomposition (NOT Cholesky
decomposition of H^{-1}). Block LDLQ extends the LDLQ procedure to support vector
quantization by rounding g columns together with linear feedback from already-rounded
columns. The pseudocode oversimplifies this significantly.

### 7.5 Quality Results

**[VERIFY NUMBERS]** The table:

```
Llama-2-70B perplexity (wikitext):
  fp16:           5.20
  GPTQ 4-bit:     5.26
  GPTQ 2-bit:     11.4
  QuIP# 4-bit:    5.22
  QuIP# 2-bit:    5.75
```

These numbers should be verified against the QuIP# paper (Table 2). The QuIP# paper
reports results for Llama 1 and Llama 2 on WikiText-2 and C4 with context length 2048.
The exact values depend on the evaluation setup. The relative ordering (QuIP# 2-bit
dramatically better than GPTQ 2-bit) is confirmed by the paper.

---

## 8. AQLM (Section 8)

### 8.2.1 Multi-Codebook Additive Quantization

**CORRECT** in general formulation. The representation:

```
w_group ~ sum_{m=1}^{M} C_m[i_m]
```

is the standard additive quantization formulation.

**[INCORRECT]** The document uses the example:

> "with d = 8, M = 2, k = 8 (256 codewords per codebook)"
> "Bits per weight: 16 / 8 = 2 bits per weight"

The AQLM paper does NOT use K = 256 (k = 8 bits) codebooks for 2-bit quantization.
According to the actual paper and HuggingFace documentation:

- For 2-bit: 1 codebook with 2^15 or 2^16 entries, group size 8
- For 3-bit: 2 codebooks of 2^12 entries each, group size 8
- For 4-bit: 2 codebooks of 2^15 or 2^16 entries each, group size 8

The codebook sizes are much larger (thousands to tens of thousands of entries), not 256.
The document's example of M=2, K=256 is a simplified illustration that does not match
the actual paper's configurations. This matters because the codebook search complexity
with K=256 would be 256^2 = 65536 (feasible by enumeration), whereas with K=2^16 it
would be (2^16)^2 = 2^32 (infeasible by enumeration), which is why the paper uses
**beam search in a Markov Random Field formulation**, not exhaustive enumeration.

### 8.2.2 Codebook Learning

**[INCORRECT]** The document describes the assignment step as:

> "For M = 2 with K = 256, this is a search over 256^2 = 65536 combinations, which is
> feasible by enumeration."

As noted above, the actual codebook sizes are much larger and the paper uses beam search
over an MRF formulation, NOT exhaustive enumeration. This is a significant error in the
algorithmic description.

The codebook update step (weighted least-squares) is **CORRECT** in formulation.

### 8.3 Complete Algorithm

**[INCORRECT]** The pseudocode shows a greedy sequential assignment:

```
For m in 1..M:
  i_{g,m} = argmin_j ||residual - C_m[j]||_H^2
  residual -= C_m[i_{g,m}]
```

The actual AQLM paper uses beam search over the MRF-structured joint assignment space,
not simple greedy residual quantization. The greedy approach shown here would give much
worse results than the actual beam search approach.

### 8.4 Storage Analysis

**[INCORRECT]** The storage analysis uses M=2, K=256, which as noted does not match the
actual paper. With the actual configurations (e.g., 1 codebook with K=2^16 for 2-bit),
the storage would be:

- Indices: 1 * 16 bits / 8 = 2.0 bits per weight (this happens to work out similarly)
- Codebook: 1 * 2^16 * 8 * 16 = 8,388,608 bits = 1 MB per layer

The codebook overhead per weight for a 67M-weight layer: 8M / 67M ~ 0.125 bits per
weight, which is more significant than the 0.001 claimed in the document (based on the
incorrectly small K=256).

### 8.5 Quality Results

**[VERIFY NUMBERS]** The table:

```
AQLM 2-bit:     5.58
```

**[LIKELY INCORRECT]** One search result referencing the AQLM paper suggests the Llama 2
70B 2-bit result may be around 3.94 perplexity on WikiText-2, which would be much
better than 5.58. However, this could depend on the specific AQLM configuration
(with/without fine-tuning, codebook sizes, etc.) and evaluation protocol (context
length, sequence stride). The value 5.58 is suspicious and should be verified against
Table 1 of the AQLM paper directly.

---

## 9. Comparative Analysis (Section 9)

### 9.1 Quantitative Comparison Table

**[VERIFY NUMBERS]** The entire comparison table should be treated as approximate. Key
concerns:

1. **All perplexity numbers** are for Llama-2-70B on WikiText-2, but the original GPTQ
   paper did not test Llama-2. Numbers come from various reproduction efforts with
   different configurations. The claimed fp16 baseline of 5.20 is plausible but varies
   by evaluation setup.

2. **RTN 3-bit = 6.80**: [VERIFY NUMBERS] -- plausible but unverified.

3. **AWQ 3-bit = 5.60**: [VERIFY NUMBERS] -- plausible but AWQ is not typically used at
   3-bit.

4. **SpQR 2-bit = ~7.0, SqueezeLLM 2-bit = ~8.0**: Marked as estimated (*) in the
   document. These are rough approximations.

5. **Calibration speed estimates**: The estimates (AWQ <10min, GPTQ 1-4hr, AQLM 8-24hr)
   are reasonable ballpark figures but depend heavily on hardware and implementation.

6. **Inference speed estimates**: The relative speed column (e.g., AQLM = 0.5x) is
   plausible but should be verified. The claim that quantized models are "typically
   faster than fp16 due to reduced memory bandwidth" is correct for weight-only
   quantization at batch size 1, but the relative speeds in the table are stated as
   <= 1.0x, implying they are slower than fp16, which contradicts this claim.

### 9.2 Quality vs Compression Frontier

**CORRECT** in overall narrative. The key observation that methods converge at 4 bits
and differentiate at 2-3 bits is well-established.

### 9.4 Method Decomposition

**CORRECT.** The decomposition table accurately characterizes each method's techniques.

### 9.5 Hardware Compatibility

**CORRECT** in general. The note about GPTQ format being loadable by llama.cpp and MLX
through format conversion is accurate.

---

## 10. Implications for MXQ (Section 10)

This section contains design recommendations, not factual claims, so there is less to
audit. The recommendations are sensible and consistent with the survey.

---

## Paper Citation Audit

| Paper | Cited As | Correct? |
|-------|----------|----------|
| AWQ | Lin et al., 2024, MLSys | **CORRECT** (MLSys 2024, Best Paper) |
| GPTQ | Frantar et al., 2023, ICLR | **CORRECT** (ICLR 2023) |
| OBC/OBQ | Frantar & Alistarh, 2022, NeurIPS | **INCORRECT**: 3 authors (Frantar, Singh, Alistarh); title is "Optimal Brain Compression" not "Optimal Brain Quantization" |
| SpQR | Dettmers et al., 2023 | **INCORRECT**: Published at ICLR 2024, not 2023. ArXiv preprint is 2023. |
| SqueezeLLM | Kim et al., 2024 | **CORRECT** (ICML 2024) |
| QuIP# | Tseng et al., 2024 | **CORRECT** (ICML 2024) |
| AQLM | Egiazarian et al., 2024 | **CORRECT** (ICML 2024) |
| OBS | Hassibi & Stork, 1993 | **CORRECT** (NeurIPS 1993) |

---

## Summary of Issues by Severity

### Definite Errors (INCORRECT)

1. **AQLM codebook sizes**: The document uses K=256 throughout, but the actual paper
   uses K=2^15 or K=2^16 (32768-65536 entries), not 256. This invalidates the
   storage analysis, the claim about exhaustive enumeration feasibility, and the
   codebook overhead calculation.

2. **AQLM assignment algorithm**: The paper uses beam search over an MRF formulation,
   not exhaustive enumeration or simple greedy residual quantization as the document
   describes.

3. **QuIP# pipeline pseudocode**: The compensation formula is wrong. QuIP# uses Block
   LDLQ (g-block LDL decomposition), not a Cholesky-based approach with the formula
   shown.

4. **QuIP# E8P codebook**: The document oversimplifies the E8P codebook as "65536 E8
   lattice points selected to cover the most likely weight vectors." The actual
   construction exploits E8 symmetries via a half-integer lattice union structure,
   reducing lookup to a 2^8 codebook.

5. **SpQR venue**: Published at ICLR 2024, not just "2023."

6. **OBC/OBQ authorship**: Three authors (Frantar, Singh, Alistarh), not two.

### Uncertain Claims

1. **AWQ importance metric**: L2 norm vs. mean absolute magnitude per channel -- the
   document uses L2 norm, but the paper likely uses average magnitude.

2. **AWQ alpha ~ 0.5**: Not verified whether this specific default value is stated in
   the paper.

3. **GPTQ Cholesky description**: Correct in spirit but oversimplified. The claim about
   reading values directly from Cholesky rows without recomputation needs more nuance.

4. **AQLM 2-bit perplexity of 5.58**: May be incorrect -- one source suggests ~3.94 for
   Llama 2 70B, though this depends heavily on configuration.

### Numbers Requiring Verification

Every perplexity number in the document should be treated as approximate:

- All numbers in the EXL2 quality table (Section 4.5)
- All numbers in the QuIP# quality table (Section 7.5)
- All numbers in the AQLM quality table (Section 8.5)
- All numbers in the comparative table (Section 9.1)

The relative orderings and approximate magnitudes are plausible, but specific values may
differ by 0.1-1.0+ perplexity depending on the evaluation setup, calibration data,
context length, group size, and other hyperparameters.

### What the Document Gets Right

Despite the issues above, the document is strong on:

- Core mathematical formulations for AWQ, GPTQ, OBS/OBQ, and SpQR
- The GPTQ update rule derivation
- The conceptual descriptions of all methods
- Hessian connection and its role across methods
- Strengths/weaknesses analysis for each method
- The method decomposition table (Section 9.4)
- Overall narrative about the quantization landscape
- Design implications for MXQ
