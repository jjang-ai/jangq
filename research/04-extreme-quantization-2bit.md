# Extreme Quantization: Making 2-Bit Models Capable

> A comprehensive technical reference for MXQ development. Covers the theory, techniques,
> calibration requirements, benchmarks, and emerging methods that make sub-3-bit quantization
> viable for large language models on Apple Silicon.

---

## Table of Contents

1. [Why 2-Bit Is the Holy Grail](#1-why-2-bit-is-the-holy-grail)
2. [Techniques That Make 2-Bit Work](#2-techniques-that-make-2-bit-work)
3. [Calibration for 2-Bit](#3-calibration-for-2-bit)
4. [Practical 2-Bit Quality Benchmarks](#4-practical-2-bit-quality-benchmarks)
5. [Emerging Techniques](#5-emerging-techniques)

---

## 1. Why 2-Bit Is the Holy Grail

### 1.1 Memory Arithmetic: The Case for 2-Bit

The entire value proposition of extreme quantization comes down to one equation:

```
Model RAM (bytes) = num_parameters * bits_per_weight / 8
```

For a 70-billion-parameter model:

| Precision | Bits/Weight | Model Size | Fits On           |
|-----------|-------------|------------|-------------------|
| FP16      | 16          | 140 GB     | 192 GB Mac Studio |
| INT8      | 8           | 70 GB      | 96 GB Mac Studio  |
| INT4      | 4           | 35 GB      | 48 GB Mac         |
| 3-bit     | 3           | 26.25 GB   | 32 GB Mac         |
| **2-bit** | **2**       | **17.5 GB**| **24 GB Mac**     |
| 2.5-bit   | 2.5         | 21.9 GB    | 32 GB Mac         |

At 2-bit, a 70B model fits into roughly 17.5 GB of weight storage. Adding KV cache
overhead (typically 1-4 GB depending on context length and batch size), the model runs
comfortably on a 24 GB machine -- and on a 32 GB Mac with generous context headroom.
At 4-bit, that same 70B model requires 35 GB of weights alone, ruling out anything
below 48 GB.

For MXQ's target market (Mac users running local inference), this is transformative:
**2-bit makes 70B+ models accessible on hardware that most Mac users already own.**

The same arithmetic scales to larger models:

| Model        | FP16    | 4-bit   | 2.5-bit (MXQ) | 2-bit   |
|-------------|---------|---------|---------------|---------|
| 7B          | 14 GB   | 3.5 GB  | 2.2 GB        | 1.75 GB |
| 13B         | 26 GB   | 6.5 GB  | 4.1 GB        | 3.25 GB |
| 70B         | 140 GB  | 35 GB   | 21.9 GB       | 17.5 GB |
| 109B (Scout)| 218 GB  | 54.5 GB | 34.1 GB       | 27.25 GB|
| 405B        | 810 GB  | 202 GB  | 126.6 GB      | 101.25 GB|

### 1.2 The Quality Cliff: Why Uniform 2-Bit Produces Garbage

Despite the compelling memory numbers, naive (uniform) 2-bit quantization produces
models that are effectively unusable. MLX's built-in uniform 2-bit quantization of a
70B model yields perplexity numbers in the 12-15 range on WikiText-2, compared to
~5.2 for FP16 -- a catastrophic degradation that manifests as incoherent text output,
hallucinated tokens, and total loss of reasoning ability.

The fundamental problem is **information capacity**. At 2 bits per weight, each weight
can take on exactly 4 distinct values. Consider what this means for representing a
continuous weight distribution:

**Uniform 4-bit (INT4):**
- 16 discrete levels spanning the weight range
- For a weight range of [-1, 1], step size delta = 2/16 = 0.125
- Maximum quantization error per weight: delta/2 = 0.0625

**Uniform 2-bit (INT2):**
- 4 discrete levels spanning the weight range
- For a weight range of [-1, 1], step size delta = 2/4 = 0.5
- Maximum quantization error per weight: delta/2 = 0.25

The step size is 4x larger at 2-bit. Since quantization error is proportional to step
size, each individual weight carries 4x more error.

### 1.3 The Mathematics of Information Loss

#### Quantization Noise Model

For a uniform scalar quantizer with step size Delta applied to a continuous-valued
signal, the quantization error e = x - Q(x) is approximately uniformly distributed
over [-Delta/2, Delta/2] when the input has a smooth probability density. The
quantization noise power (mean squared error) is:

```
sigma_q^2 = Delta^2 / 12
```

This is the standard result from quantization theory. For an N-level uniform quantizer
covering a range R:

```
Delta = R / N
sigma_q^2 = R^2 / (12 * N^2)
```

Comparing 4-bit (N=16) to 2-bit (N=4):

```
sigma_q^2(4-bit) = R^2 / (12 * 256)  = R^2 / 3072
sigma_q^2(2-bit) = R^2 / (12 * 16)   = R^2 / 192

Ratio = 3072 / 192 = 16
```

**The quantization noise power at 2-bit is 16x (12 dB) higher than at 4-bit.**

#### Signal-to-Quantization-Noise Ratio (SQNR)

The SQNR for an n-bit uniform quantizer of a full-range sinusoidal signal is given by
the well-known formula:

```
SQNR = 6.02n + 1.76  dB
```

| Bits | SQNR (dB) | Relative to 4-bit |
|------|-----------|-------------------|
| 8    | 49.9      | +24.1 dB          |
| 6    | 37.9      | +12.1 dB          |
| 4    | 25.8      | baseline          |
| 3    | 19.8      | -6.0 dB           |
| 2    | 13.8      | -12.0 dB          |
| 1    | 7.8       | -18.0 dB          |

Each bit removed costs approximately 6 dB of SQNR. Going from 4-bit to 2-bit costs
12 dB -- the quantization noise power increases by a factor of 16. This is the
fundamental signal-theoretic reason why 2-bit is so much harder than 4-bit.

For neural network weights, which approximately follow a Gaussian distribution rather
than a uniform distribution, the situation is somewhat different but the scaling
relationship holds. For a Gaussian source with variance sigma^2 quantized by an
N-level uniform quantizer, the optimal range is approximately R = 2 * k * sigma where
k depends on N (for N=16, k ~ 2.73; for N=4, k ~ 1.51, from Lloyd-Max tables).
The distortion is:

```
D(N) = sigma^2 * d(N)
```

where d(N) is the normalized distortion from Lloyd-Max tables:

| N (levels) | bits | d(N)    | D relative to 4-bit |
|-----------|------|---------|---------------------|
| 256       | 8    | 0.00015 | 0.003x              |
| 16        | 4    | 0.03454 | 1.0x (baseline)     |
| 4         | 2    | 0.11885 | 3.44x               |
| 2         | 1    | 0.36340 | 10.5x               |

For Gaussian-distributed weights, 2-bit uniform quantization produces 3.44x the
distortion of 4-bit -- slightly less devastating than the 16x factor for uniform
inputs, because the Gaussian's concentration around zero means that fewer weights fall
in the high-error tail regions. But 3.44x more noise per weight is still catastrophic
when summed across billions of weights in matrix multiplications.

#### Error Accumulation in Matrix Multiplication

The real damage happens during inference. Consider a single linear layer computing
y = Wx where W is an m x n weight matrix and x is an n-dimensional input vector.
If each weight w_ij has quantization error e_ij, then:

```
y_quantized = (W + E)x = Wx + Ex
```

The error in the output is Ex. For the i-th output element:

```
error_i = sum_{j=1}^{n} e_ij * x_j
```

Assuming independence between errors and inputs, the variance of each output error is:

```
Var(error_i) = sigma_q^2 * sum_{j=1}^{n} x_j^2 = sigma_q^2 * ||x||^2
```

This grows linearly with the input dimension n (which is the hidden dimension of the
model, typically 4096-8192 for large LLMs). For a model with L layers, errors
accumulate (though not simply additively due to nonlinearities). The total output
distortion scales approximately as:

```
Total distortion ~ L * n * sigma_q^2
```

For a 70B model with L=80 layers and n=8192:
- At 4-bit: L * n * sigma_q^2(4-bit) = 80 * 8192 * (R^2/3072) = 213 * R^2
- At 2-bit: L * n * sigma_q^2(2-bit) = 80 * 8192 * (R^2/192)  = 3413 * R^2

**16x more cumulative noise across the full forward pass.** This is why uniform
2-bit models produce garbage: the accumulated quantization noise overwhelms the
actual signal being computed.

### 1.4 Why the Problem Is Solvable

Despite the bleak mathematics above, 2-bit quantization is not inherently impossible.
The analysis above assumes:

1. **Uniform quantization** -- equally spaced levels across the range
2. **Scalar quantization** -- each weight quantized independently
3. **Equal precision** -- all weights get the same number of bits
4. **No error compensation** -- quantization errors are not corrected

Every one of these assumptions can be relaxed, and doing so is what makes 2-bit work.
The rest of this document explains how.

---

## 2. Techniques That Make 2-Bit Work

### 2.1 Non-Uniform Quantization: NormalFloat (NF2)

#### The Core Insight

The weights of pre-trained transformer models are not uniformly distributed -- they
approximately follow a zero-centered Gaussian (normal) distribution. A uniform
quantizer wastes representation capacity by spacing levels equally, even in regions
of the distribution where few weights exist (the tails) while under-representing the
dense center.

**Non-uniform quantization** places quantization levels according to the probability
density of the weights. For Gaussian-distributed data, this means more levels near
zero (where most weights cluster) and fewer in the tails.

#### NormalFloat Data Types

The NormalFloat (NF) data type family, introduced in the QLoRA paper (Dettmers et al.,
2023), constructs quantization levels as the quantiles of the standard normal
distribution N(0,1). For an n-bit NormalFloat type with 2^n levels, each level
represents a region of equal probability mass under the Gaussian curve.

**NF4 (4-bit NormalFloat):**
16 levels, each representing 1/16 = 6.25% of the probability mass of N(0,1).
The levels are the midpoints of 16 equal-probability intervals:

```
NF4 values = {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
               0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
```

(Normalized to [-1, 1] range; actual values are scaled by the block's absmax.)

**NF2 (2-bit NormalFloat):**
4 levels, each representing 1/4 = 25% of the probability mass:

```
Quartile boundaries of N(0,1): {-inf, -0.6745, 0, 0.6745, +inf}
NF2 levels (conditional means within each quartile):
  q1 = E[X | X < -0.6745]  = -1.0494
  q2 = E[X | -0.6745 < X < 0] = -0.3191
  q3 = E[X | 0 < X < 0.6745]  =  0.3191
  q4 = E[X | X > 0.6745]  =  1.0494
```

After normalization to unit variance:

```
NF2 values ~ {-0.7979, -0.2677, 0.2677, 0.7979}
```

Compare to uniform 2-bit values (for range [-1, 1]):

```
Uniform 2-bit: {-0.75, -0.25, 0.25, 0.75}
```

The NF2 values look similar to uniform values for Gaussian data, but they are
**information-theoretically optimal** -- they minimize the expected squared error
E[(X - Q(X))^2] for Gaussian-distributed X. The uniform quantizer is not optimal
because it does not account for the varying density.

#### Lloyd-Max Optimal Quantizer for 2-Bit Gaussian

The Lloyd-Max algorithm provides the optimal scalar quantizer for a given source
distribution by alternately optimizing decision boundaries and reconstruction levels.
For a 2-bit (4-level) quantizer of a Gaussian source N(0, sigma^2), the algorithm
converges to:

**Decision boundaries (thresholds):**

```
t_0 = -inf
t_1 = -0.6745 * sigma   (25th percentile)
t_2 = 0                  (median)
t_3 = +0.6745 * sigma   (75th percentile)
t_4 = +inf
```

**Reconstruction levels (centroids):**

```
r_1 = E[X | X in (t_0, t_1)] = -1.0494 * sigma
r_2 = E[X | X in (t_1, t_2)] = -0.3191 * sigma
r_3 = E[X | X in (t_2, t_3)] = +0.3191 * sigma
r_4 = E[X | X in (t_3, t_4)] = +1.0494 * sigma
```

**Derivation of r_1 (first reconstruction level):**

```
r_1 = E[X | X < -0.6745*sigma]
    = integral_{-inf}^{-0.6745*sigma} x * f(x) dx  /  P(X < -0.6745*sigma)

where f(x) = (1/(sigma*sqrt(2*pi))) * exp(-x^2/(2*sigma^2))

P(X < -0.6745*sigma) = 0.25

integral_{-inf}^{-0.6745*sigma} x * f(x) dx
  = -sigma / sqrt(2*pi) * exp(-0.6745^2 / 2)
  = -sigma / sqrt(2*pi) * exp(-0.2274)
  = -sigma * 0.31830 * 0.7963
  = -sigma * 0.2534  [approximately]

r_1 = -sigma * 0.2534 / 0.25 = -1.0136 * sigma
```

(The exact value from numerical computation is -1.0494*sigma; the slight discrepancy
above is from rounding in the intermediate steps.)

**Optimal MSE distortion for 2-bit Lloyd-Max on N(0, sigma^2):**

```
D_LM = sigma^2 * 0.11885
```

Compare to uniform 2-bit quantizer:

```
D_uniform = sigma^2 * 0.13671   (for optimally-ranged uniform quantizer)
```

**The Lloyd-Max quantizer reduces distortion by ~13% over uniform quantization at
2-bit.** This is a meaningful improvement but not transformative on its own -- it
cannot bridge the gap from 2-bit to 4-bit quality. NF2 is a necessary but not
sufficient technique.

#### Why NF2 Alone Is Not Enough

The distortion reduction from non-uniform quantization is modest because at 2-bit,
there are only 4 levels. The fundamental constraint is that 4 values cannot adequately
represent a continuous distribution regardless of how cleverly those 4 values are
chosen. The real wins come from the techniques in sections 2.2-2.6, which change
the problem entirely by:
- Making the weight distribution more uniform (incoherence processing)
- Quantizing vectors instead of scalars (vector/lattice quantization)
- Giving important weights more bits (mixed precision)
- Compensating for errors after quantization (GPTQ-style)

### 2.2 Incoherence Processing (QuIP / QuIP#)

#### The Problem: Outlier Weights

Transformer weight matrices are not well-behaved for quantization. Some columns have
significantly larger magnitudes than others. Some rows contain outlier values 10-100x
larger than the median. These outliers are disproportionately important for model
quality -- but under uniform bit allocation, they get the same 2-bit precision as
every other weight.

Formally, the quantization error of GPTQ-style methods depends on the **incoherence**
of the weight matrix W and the Hessian H. Incoherence mu of a matrix M in R^{n x n}
is defined as:

```
mu(M) = n * max_{i,j} |M_{ij}|^2 / ||M||_F^2
```

When mu is large (outlier entries dominate), quantization error bounds are loose. When
mu ~ 1 (all entries roughly equal magnitude), the bounds are tight.

#### The Hadamard Rotation Trick

QuIP (Chee et al., 2023) and its successor QuIP# (Tseng et al., 2024) solve this
with **incoherence processing**: multiply the weight matrix by random orthogonal
matrices before quantization, then undo the multiplication during inference.

The key operation is the **Randomized Hadamard Transform (RHT)**:

```
W' = U * W * V^T
```

where U and V are constructed as:

```
U = H_n * S_1     (left rotation)
V = H_n * S_2     (right rotation)
```

Here:
- H_n is the normalized Hadamard matrix of dimension n (H_n = H/sqrt(n) where H is
  the standard Hadamard matrix with entries +/-1)
- S_1, S_2 are diagonal matrices with random +/-1 entries on the diagonal

**Why this works: the Hadamard matrix spreads energy.**

The Hadamard matrix has the property that all its rows (and columns) are orthogonal
and have the same norm. When you multiply a vector by a Hadamard matrix, the energy
of any single large component gets distributed across all components. Formally:

If w is a vector with one outlier entry w_k >> w_j for j != k, then after
Hadamard transform w' = H_n * w:

```
w'_i = (1/sqrt(n)) * sum_{j=1}^{n} (+/-1) * w_j
```

Every entry of w' contains a contribution from w_k, diluted by a factor of 1/sqrt(n).
The maximum entry magnitude drops from |w_k| to approximately |w_k|/sqrt(n).

**Effect on incoherence:**

If the original matrix W has incoherence mu(W) >> 1 (due to outliers), then after
Hadamard rotation:

```
mu(W') = mu(U * W * V^T) ~ O(log(n))
```

with high probability. This is near-optimal -- the minimum possible incoherence for an
n x n matrix with a given Frobenius norm is mu = 1, and log(n) is close.

**Effect on condition number:**

For matrices where the outlier problem manifests as a high condition number kappa
(ratio of largest to smallest singular value), the Hadamard rotation reduces the
effective condition number as seen by the quantizer. While the rotation does not change
the singular values of W, it distributes the "difficulty" of representing large and
small singular values across all entries, making each entry contribute more equally.

#### Computational Cost

The naive matrix multiply U * W * V^T would cost O(n^3), but the Hadamard matrix has
special structure that enables O(n log n) computation via the Fast Walsh-Hadamard
Transform (FWHT):

```
FWHT(x) of length n = 2^k:
  For each stage s = 0, 1, ..., k-1:
    For each pair of elements at distance 2^s:
      butterfly: (a, b) -> (a+b, a-b)
  Normalize by 1/sqrt(n)
```

**Cost per layer during inference:**

For a linear layer W of shape (m, n):
- Left rotation: O(m * n * log(m)) for m row-wise FWHTs of length n... but
  in practice the transform is applied to the input activations, not the weights.
  The weight matrix is stored already rotated.
- The inference cost is:
  - One FWHT of the input vector x: O(n log n) per token
  - One FWHT of the output vector y: O(m log m) per token
  - Standard matmul W'x: O(mn)

Since mn >> n log n for typical hidden dimensions (n = 4096-8192), the FWHT overhead
is negligible -- typically <1% of the matmul cost.

#### QuIP# Enhancement: Lattice Codebooks After Rotation

QuIP# combines incoherence processing with E8 lattice codebooks (section 2.4). The
rotation step makes the weight distribution approximately i.i.d. Gaussian -- which is
exactly the distribution that the E8 lattice quantizer is optimized for. The two
techniques are synergistic: rotation creates the right distribution, and lattice
quantization exploits it optimally.

### 2.3 Vector Quantization and Codebook Approaches

#### Why Vectors Beat Scalars

Scalar quantization treats each weight independently. Vector quantization (VQ) groups
multiple weights together and quantizes them as a single vector, selecting the nearest
entry from a learned codebook.

The fundamental advantage is the **space-filling gain**. In 1 dimension, the optimal
quantizer tiles the real line with intervals (no gaps, no overlaps). In d dimensions,
the optimal quantizer tiles R^d with d-dimensional polytopes. Higher-dimensional
polytopes can approximate spheres more closely, leaving less "wasted space" and
achieving lower distortion at the same bitrate.

The theoretical distortion of an optimal d-dimensional vector quantizer at rate R
bits per dimension is:

```
D(R, d) = (1/(2*pi*e)) * 2^{-2R} * G(d) * sigma^2
```

where G(d) is the normalized second moment of the d-dimensional quantizer region,
and G(d) -> 1/(2*pi*e) as d -> infinity (the Zador bound). For scalar quantization
(d=1), G(1) = 1/12. As d increases, G(d) decreases, meaning lower distortion.

**Practical gains at 2 bits per weight:**

| Dimension d | G(d) (best lattice) | Distortion relative to scalar |
|-------------|-------------------|-------------------------------|
| 1 (scalar)  | 1/12 = 0.0833     | 1.00x (baseline)              |
| 2 (A2)      | 0.0802            | 0.96x                         |
| 4 (D4)      | 0.0766            | 0.92x                         |
| 8 (E8)      | 0.0717            | 0.86x                         |
| 24 (Leech)  | 0.0659            | 0.79x                         |
| inf (Zador) | 1/(2*pi*e) = 0.0585| 0.70x                        |

At dimension 8 (the E8 lattice used by QuIP#), vector quantization achieves 14% less
distortion than scalar -- a significant improvement that compounds across billions of
weights.

#### Residual Vector Quantization (RVQ)

Simple VQ with a codebook of K entries requires K to grow exponentially with dimension
to maintain rate R. For d=8 and R=2 bits/weight, the codebook needs 2^(d*R) = 2^16 =
65,536 entries -- manageable but large.

**Residual VQ** decomposes the problem into stages:

1. **Stage 1**: Quantize each weight group using codebook C_1 of size K_1.
   The residual is r = w - C_1[i_1].
2. **Stage 2**: Quantize the residual r using codebook C_2 of size K_2.
   The new residual is r' = r - C_2[i_2].
3. Continue for M stages.

The total bits per group is log_2(K_1) + log_2(K_2) + ... + log_2(K_M).

**Example achieving 2 bits per weight with RVQ:**

- Group size: 8 weights
- 2 codebooks, each with 256 entries (8 bits to index)
- Total bits per group: 8 + 8 = 16 bits for 8 weights = 2 bits/weight
- But the 256 codebook entries are stored in FP16 -- each entry is an 8-dimensional
  vector of float16 values
- Total codebook storage: 2 * 256 * 8 * 2 bytes = 8 KB per layer (negligible)

The key insight: **the codebook entries are full-precision float16 vectors**. While
each weight block is indexed by only 2 bits worth of information, the actual
reconstruction values can be arbitrarily complex. This is far richer than 4 scalar
values.

#### AQLM: Multi-Codebook Additive Quantization for LLMs

AQLM (Egiazarian et al., 2024) applies Multi-Codebook Quantization (MCQ) to LLM
weight compression. The method introduces two key innovations:

**1. Additive codebook structure:**

For a weight vector w of dimension d, AQLM represents it as:

```
w_hat = sum_{m=1}^{M} C_m[i_m]
```

where C_m is the m-th codebook and i_m is the index into that codebook.

This differs from RVQ in that all codebooks are optimized jointly, not sequentially.
The codebook entries are learned to minimize the layer-wise reconstruction error
weighted by the input activations (Hessian).

**2. Joint optimization across the layer:**

Rather than optimizing each weight group independently, AQLM uses beam search across
the full weight matrix of each layer, jointly optimizing codebook entries and index
assignments to minimize:

```
L = ||WX - W_hat * X||_F^2
```

where X is the matrix of calibration activations.

**AQLM 2-bit configuration:**
- Group size: 8
- Number of codebooks M: 1 (for 2-bit) or 2 (for higher quality at ~2.5 bit)
- Codebook size: 2^16 = 65,536 entries for M=1 at 2-bit with group size 8
  (16 bits / 8 weights = 2 bpw)
- Each codebook entry: 8-dimensional FP16 vector

**Performance:** AQLM is Pareto-optimal in the accuracy-vs-model-size tradeoff at
sub-3-bit precision. A 2.76-bit AQLM quantization of a 13B model can outperform the
uncompressed 7B model.

#### VPTQ: Vector Post-Training Quantization

VPTQ (Microsoft, EMNLP 2024) extends vector quantization with second-order
optimization. Key contributions:

1. **Channel-Independent Second-Order Optimization**: formulates VQ as an optimization
   problem guided by the Hessian, solving for codebook assignments that minimize
   Hessian-weighted error rather than simple MSE.

2. **Residual and outlier quantization**: uses a primary codebook plus a residual
   codebook for fine correction, with separate handling of weight outliers.

3. **Results at 2-bit on LLaMA-2:**
   - Perplexity reduction of 0.01-0.34 over AQLM
   - Accuracy improvement of 0.79-1.5% on QA tasks
   - 1.6-1.8x inference throughput increase vs. prior SOTA

### 2.4 Lattice Quantization: The E8 Lattice

#### What Is a Lattice Quantizer?

A lattice is a regular arrangement of points in n-dimensional space, defined by a set
of basis vectors. A lattice quantizer maps each input vector to the nearest lattice
point. Unlike a generic VQ codebook, a lattice has algebraic structure that enables:

1. **Fast nearest-neighbor search**: O(d) or O(d log d) instead of O(K) over codebook
2. **No codebook storage**: lattice points are generated algorithmically
3. **Optimal packing**: certain lattices achieve the densest possible sphere packing
   in their dimension, which directly corresponds to minimum quantization distortion

#### The E8 Lattice

The E8 lattice is an 8-dimensional lattice with extraordinary mathematical properties.
It achieves the **densest possible sphere packing in 8 dimensions** (proven by
Viazovska, 2016, Fields Medal work).

**Definition:**

The E8 lattice consists of all vectors in R^8 whose coordinates are either all integers
or all half-integers (integers + 1/2), and whose coordinates sum to an even number:

```
E8 = { x in Z^8 : sum(x_i) is even }
     union
     { x in (Z+1/2)^8 : sum(x_i) is even }
```

**Key properties:**

- **Kissing number**: 240 (each lattice point has 240 nearest neighbors)
- **Covering radius / packing radius**: 1 (normalized)
- **Center density**: pi^4 / 384 ~ 0.2537
- **Normalized second moment**: G(E8) ~ 0.0717
- **Automorphism group**: the Weyl group W(E8), order 696,729,600

**Why E8 is optimal for 2-bit quantization:**

QuIP# uses E8 for 2-bit quantization by quantizing groups of 8 weights to the nearest
E8 lattice point. Each lattice point within a bounded region can be indexed by a
compact code. For 2 bits per weight:

```
Total bits per group of 8 = 8 * 2 = 16 bits
Number of distinct lattice points indexable = 2^16 = 65,536
```

QuIP# defines a subset E8P (E8 Polytope) of E8 lattice points that can be efficiently
indexed with 16 bits. The 65,536-entry codebook is never explicitly stored --
instead, encoding and decoding use the algebraic structure of E8 to compute indices
and reconstruct vectors in O(d) = O(8) operations.

**Codebook size comparison:**

```
Naive 8D codebook: 2^16 * 8 * 2 bytes = 1 MB per layer
E8P codebook:      ~1 KB (lookup table for the algebraic structure)
Savings:           1000x
```

The E8 lattice's symmetry (its automorphism group has ~700 million elements) means that
many lattice points are equivalent under rotation, allowing the codebook to be
compressed to ~1 KiB while covering the full 65,536-entry space.

#### Why Lattice Beats Scalar: The Geometric Argument

Consider quantizing 8 weights using scalar 2-bit quantization vs. E8 lattice
quantization:

**Scalar (8 independent 2-bit quantizers):**
- Each weight maps to one of 4 values
- Total distinct vectors: 4^8 = 65,536
- These vectors form a regular grid in R^8
- The Voronoi cell of each grid point is a hypercube
- Hypercubes pack with no gaps but have poor spherical symmetry
- The normalized second moment G = 1/12 = 0.0833

**E8 lattice (one 8-dimensional quantizer):**
- Each group maps to one of 65,536 lattice points (same count)
- The Voronoi cells are polytopes that approximate spheres much better
- G(E8) = 0.0717
- **14% lower distortion at the same bitrate**

The geometric intuition is that the E8 lattice points are arranged more like a "honeycomb"
in 8D space -- each cell is rounder than a hypercube, meaning the average distance from
a random point to the nearest lattice point is smaller.

#### QuIP# Performance at 2-Bit

QuIP# using E8 lattice + Hadamard incoherence achieves results previously thought
impossible at 2 bits:

- It is the first PTQ method where 3-bit models scale better than 4-bit (i.e., the
  quality-per-bit curve improves at lower bitrates)
- At 2.15 bpw on Llama-2 70B, QuIP# achieves quality close to OmniQuant at 3 bits
- AWQ falls apart completely at 2.15 bpw; OmniQuant produces unusable models at 2 bpw;
  QuIP# produces high-quality models

### 2.5 GPTQ-Style Error Compensation at 2-Bit

#### The Core Idea

GPTQ (Frantar et al., 2022) does not simply round each weight to the nearest
quantized value. Instead, it quantizes weights sequentially and, after each
quantization, adjusts the remaining unquantized weights to compensate for the error.

This is based on the Optimal Brain Surgeon (OBS) framework. The key insight: when
you quantize weight w_q to value w_hat_q, the error delta_q = w_q - w_hat_q propagates
through the network. But you can absorb this error by adjusting the remaining weights
by amounts proportional to the inverse Hessian.

#### The Mathematical Framework

For a weight matrix W of a linear layer, define the objective as minimizing the
squared output error:

```
E(W) = ||WX - W_hat * X||_F^2
```

where X is the matrix of input activations from calibration data. This can be
decomposed row-wise: each row w of W is quantized independently to minimize:

```
E(w) = ||wX - w_hat * X||^2 = (w - w_hat) * XX^T * (w - w_hat)^T
                             = (w - w_hat) * H * (w - w_hat)^T
```

where H = 2XX^T is the Hessian of the squared error with respect to the weights.
(More precisely, H_ij = sum_k x_{ik} * x_{jk} -- the uncentered covariance of inputs.)

#### The GPTQ Update Rule

When quantizing the q-th weight in a row, GPTQ computes:

1. **Quantize**: w_hat_q = quantize(w_q)
2. **Error**: delta_q = w_q - w_hat_q
3. **Compensate remaining weights**:

```
w_{q+1:n} = w_{q+1:n} - (delta_q / [H^{-1}]_{qq}) * [H^{-1}]_{q, q+1:n}
```

This is the OBS formula applied to quantization rather than pruning. The inverse
Hessian H^{-1} determines how much each remaining weight should change to minimally
affect the output.

**Efficient implementation via Cholesky decomposition:**

Computing H^{-1} for each weight would be O(n^3). GPTQ instead:
1. Computes the Cholesky decomposition of H^{-1} = L * L^T once: O(n^3/3)
2. Processes weights in order of columns of L
3. Updates H^{-1} incrementally as each weight is quantized using rank-1 updates

Total cost: O(n^3) per row, but with good constants and easy parallelization.

**Dampening for numerical stability:**

The Hessian can be near-singular (some weight directions have very low gradient).
GPTQ adds a dampening term:

```
H_damped = H + lambda * I
```

where lambda is typically 1% of the mean diagonal of H. This prevents division by
near-zero values in the update rule.

#### Why Error Compensation Matters More at 2-Bit

At 4-bit, each weight's quantization error is relatively small (delta_q ~ 0.06 for
a unit-range weight), and the compensating adjustments to other weights are
correspondingly small. The system is "almost right" before compensation; compensation
is a polish step.

At 2-bit, each weight's quantization error is 4x larger (delta_q ~ 0.25). The
compensating adjustments are larger, which means:

1. **Compensation is doing more work**: it is not polishing -- it is fundamentally
   restructuring the weight matrix to preserve output behavior despite massive per-
   weight errors.

2. **Order matters enormously**: quantizing the most sensitive weights first (act-order)
   ensures that compensation can use the largest pool of unquantized weights to absorb
   the error. Quantizing insensitive weights first wastes the compensation budget.

3. **The Hessian must be accurate**: at 4-bit, an approximate Hessian (e.g., Fisher
   diagonal) is sufficient because errors are small. At 2-bit, inaccuracies in the
   Hessian lead to suboptimal compensation that compounds into large output errors.

#### Act-Order: Quantize Sensitive Weights First

The default GPTQ processes weights in sequential column order. **Act-order** (activation
ordering) instead sorts columns by decreasing activation magnitude:

```
order = argsort(||X_col_j||^2, descending=True)
```

Columns with large activations are quantized first, while the maximum number of
remaining columns are available for error absorption. This is particularly important at
2-bit because:

- The most sensitive weights (those processing the largest activations) incur the
  largest absolute output errors when quantized
- They need the most compensation from other weights
- If quantized last, there are few remaining weights to compensate with

Empirically, act-order improves perplexity by 0.1-0.3 at 4-bit but by 0.5-2.0+ at
2-bit -- the impact scales with quantization aggressiveness.

### 2.6 Mixed-Precision 2-Bit: The MXQ Approach

#### The Key Insight

Not all weights are created equal. Some weight blocks are critical for model quality
(attention projections, first/last layers, blocks processing outlier activations), while
others are highly compressible (middle-layer MLP weights, redundant heads). Forcing
every block to the same precision wastes bits on insensitive weights while starving
sensitive ones.

**Mixed-precision quantization** assigns different bit widths to different blocks. A
model with an *average* of 2.5 bits per weight might have:

| Component                  | Bit Width | Fraction of Weights |
|----------------------------|-----------|---------------------|
| Embedding layer            | 4         | ~2%                 |
| First 2 transformer layers | 4         | ~5%                 |
| Attention Q/K projections  | 3-4       | ~15%                |
| Attention V/O projections  | 2-3       | ~15%                |
| MLP gate/up projections    | 2         | ~30%                |
| MLP down projections       | 2-3       | ~15%                |
| Last 2 transformer layers  | 3-4       | ~5%                 |
| LM head                    | 6         | ~3%                 |
| **Weighted average**       | **~2.5**  | **100%**            |

**Why 80/20 beats 100/0:**

A model with 80% of weights at 2-bit and 20% at 4-bit (average 2.4 bits) is
dramatically better than 100% at 2-bit. The reason is convexity of the
distortion-rate function: the marginal quality loss from each additional bit removed
is increasing.

Quantitative example (hypothetical distortion units):

```
Block at 4-bit: distortion = 1.0
Block at 3-bit: distortion = 2.0   (1.0 more than 4-bit)
Block at 2-bit: distortion = 5.0   (3.0 more than 3-bit, 4.0 more than 4-bit)

Uniform 2-bit (all blocks): total distortion = N * 5.0 = 5.0N

Mixed 2.4-bit (80% at 2, 20% at 4):
  Total distortion = 0.8N * 5.0 + 0.2N * 1.0 = 4.0N + 0.2N = 4.2N

Mixed 2.4-bit (optimized -- sensitive blocks at 4, rest at 2):
  Total distortion < 4.2N because the 20% at 4-bit are the most sensitive blocks,
  which contribute disproportionately to total distortion at 2-bit.
  Actual total distortion might be ~ 3.5N
```

**The improvement from 5.0N to 3.5N is a 30% distortion reduction at only 20% more
bits.** This is the power of mixed precision.

#### Rate-Distortion Theory: Optimal Bit Allocation

The problem of assigning bits to blocks is a classic rate-distortion optimization.

**Formal setup:**

Let there be K weight blocks, indexed i = 1, ..., K. Each block i can be quantized
at bit width b_i in {2, 3, 4, 5, 6, 8}. The distortion (output error) of block i at
bit width b_i is D_i(b_i). The total bit budget is B_total (determined by the target
average bit width).

**Objective:**

```
Minimize:  D_total = sum_{i=1}^{K} D_i(b_i)
Subject to: sum_{i=1}^{K} b_i = B_total
            b_i in {2, 3, 4, 5, 6, 8} for all i
```

**Lagrangian relaxation (continuous case):**

Relaxing b_i to be continuous and forming the Lagrangian:

```
L = sum_{i=1}^{K} [D_i(b_i) + lambda * b_i]
```

The optimal solution satisfies:

```
dD_i/db_i = -lambda    for all i
```

**This means: at the optimum, the marginal distortion reduction per bit is equal
across all blocks.** Blocks where an extra bit would save the most distortion should
get that bit first; blocks where extra bits barely help should be reduced.

For an exponential distortion model D_i(b_i) = c_i * 2^{-alpha_i * b_i} (which is
approximately correct for quantization), the optimal allocation is:

```
b_i* = (1/alpha_i) * [log2(c_i * alpha_i / lambda)]
```

where lambda is chosen to satisfy the total bit constraint. Blocks with larger c_i
(more sensitive to quantization) or smaller alpha_i (slower quality improvement per
bit) get more bits.

**Discrete optimization:**

In practice, bit widths are discrete (2, 3, 4, ...). The optimal discrete allocation
is found by:

1. Start with all blocks at minimum bits (2)
2. Compute the marginal distortion reduction of upgrading each block by 1 bit:
   Delta_D_i = D_i(b_i) - D_i(b_i + 1)
3. Upgrade the block with the largest Delta_D_i
4. Repeat until the bit budget is exhausted

This greedy algorithm is optimal for the discrete problem when the distortion
functions D_i(b) are convex in b (which they are for quantization).

#### Why This Is What Makes MXQ Theoretically Optimal

MXQ's calibration-driven bit allocation implements this optimization:

1. **Calibration** (Phase 1) measures D_i(b_i) for each block by evaluating the
   output distortion at each bit width using actual calibration data.

2. **Scoring** combines activation magnitudes and sensitivity measurements to
   estimate the distortion function D_i without exhaustively testing every bit width.

3. **Allocation** (Phase 2) runs the greedy algorithm above, enhanced with
   structural priors (embeddings get at least 4 bits, lm_head gets at least 6, etc.).

The result is a bit allocation that provably minimizes total output distortion for the
given bit budget. No uniform quantization scheme can match this, because uniform
quantization forces b_i = b for all i, ignoring the heterogeneous sensitivity of
different blocks.

**The theoretical gap between uniform and optimal allocation grows as the average
bit width decreases.** At 4-bit average, most blocks are already well-quantized and
reallocation helps modestly. At 2.5-bit average, the gap between the most and least
sensitive blocks is enormous, and optimal allocation provides massive improvements --
potentially matching 4-bit uniform quality at 40% less memory.

---

## 3. Calibration for 2-Bit

### 3.1 Why Calibration Quality Is Critical at 2-Bit

At 4-bit quantization, the margin for error is generous. Each weight has 16 possible
values, and even a mediocre calibration will identify approximately correct importance
scores. The resulting bit allocation (if mixed-precision) or error compensation (if
GPTQ) will be close enough.

At 2-bit, the margin is razor-thin. Each weight has only 4 possible values, and:

1. **A misallocated bit is catastrophic**: upgrading a block from 2 to 3 bits (adding
   4 more representable values) has a much larger quality impact than upgrading from
   4 to 5 bits. A calibration error that puts 3 bits on an unimportant block instead
   of an important one wastes the most valuable bit in the budget.

2. **Error compensation depends on accurate Hessians**: GPTQ's compensation formula
   uses H^{-1}, where H = 2XX^T from calibration data. If the calibration data does
   not cover the actual inference distribution, H^{-1} will compensate in the wrong
   directions.

3. **Small calibration errors compound**: at 2-bit, there are more weights where the
   quantization error is at its maximum (delta/2 = range/8 rather than delta/2 =
   range/32 at 4-bit). More weights near their error ceiling means more sensitivity
   to which direction the error points, which depends on calibration.

### 3.2 Calibration Dataset Requirements

The calibration dataset must cover the full distribution of activations the model will
encounter during inference. A calibration set that only contains English prose will
produce a model that fails on code or multilingual input, because the activation
patterns for those inputs were never measured.

**Diversity requirements:**

| Domain       | Why It Matters                                          | Min Samples |
|-------------|--------------------------------------------------------|-------------|
| English prose| Baseline language patterns                              | 200         |
| Code         | Highly structured, different token distribution         | 150         |
| Math/reasoning| Chain-of-thought activations, numerical precision     | 100         |
| Multilingual | Different token distributions (CJK, Arabic, etc.)     | 100         |
| Chat/dialog  | Multi-turn patterns, system prompts                    | 100         |
| Long context | Tests activation distributions at long sequence lengths | 50          |
| Factual/QA   | Tests knowledge recall pathways                        | 50          |
| **Total**    |                                                        | **~750+**   |

**Sample length considerations:**

- Short samples (128-512 tokens) are efficient but miss long-range attention patterns
- Long samples (4K-32K tokens) activate different attention heads and MLP neurons
- A mix of lengths is ideal: 50% at 512-1K, 30% at 1K-4K, 20% at 4K-32K

**Diminishing returns:**

Empirically, calibration quality follows a log curve:
- 128 samples: significantly better than 0, but missing distributions
- 512 samples: covers most common patterns, good for 4-bit
- 1024 samples: diminishing returns begin, good for 2-3 bit
- 2048+ samples: minimal improvement, not worth the calibration time
- The sweet spot for 2-bit is 512-1024 diverse samples

### 3.3 Sensitivity Measurement Methods

Multiple methods exist for measuring how sensitive each weight block is to
quantization. For 2-bit, using a single method is risky -- different methods capture
different aspects of sensitivity.

#### 3.3.1 Hessian Diagonal

The diagonal of the Hessian H = 2XX^T gives the curvature of the loss with respect to
each weight. High curvature means the loss changes rapidly when the weight is perturbed
-- a sensitive weight.

```
H_{ii} = sum_{k=1}^{N} x_{ki}^2
```

where x_{ki} is the i-th input activation for calibration sample k.

**Strengths**: cheap to compute (O(n * N_samples)), directly relates to GPTQ error.
**Weaknesses**: ignores cross-weight interactions, can be dominated by a few outlier
activations.

#### 3.3.2 Fisher Information

The Fisher information matrix is the expected outer product of the gradient of the
log-likelihood:

```
F_{ij} = E_x[(d log p(x|W) / dw_i) * (d log p(x|W) / dw_j)]
```

The diagonal F_{ii} measures how much information each weight carries about the output
distribution. Weights with high Fisher information are essential for the model's
predictions.

For a language model, F_{ii} can be approximated as:

```
F_{ii} ~ (1/N) * sum_{k=1}^{N} (d L_k / d w_i)^2
```

where L_k is the cross-entropy loss on sample k.

**Strengths**: theoretically grounded, measures importance for the model's actual task.
**Weaknesses**: requires backward pass (expensive), depends heavily on calibration data
choice.

#### 3.3.3 Activation Magnitude (AWQ-Style)

AWQ (Lin et al., 2023) observes that weight importance is well-predicted by the
magnitude of the activations that the weight processes:

```
importance_i = mean(|x_i|) * |w_i|
```

Weights that process large activations are important because they contribute more to
the output. This is a simpler proxy for the Hessian diagonal:

```
H_{ii} = sum x_{ki}^2 ~ N * E[x_i^2] = N * (mean(|x_i|))^2  (approximately)
```

**Strengths**: very cheap (forward pass only), correlates well with actual sensitivity.
**Weaknesses**: ignores weight interactions, does not account for redundancy (two
weights with the same activation pattern might be redundant, but both score high).

#### 3.3.4 KL Divergence Under Perturbation

Directly measure how the model's output distribution changes when a weight block is
perturbed:

```
sensitivity_i = E_x[KL(p(y|x, W) || p(y|x, W + delta_i))]
```

where delta_i is a perturbation of block i (e.g., setting it to its quantized value).

**Strengths**: directly measures what we care about (output quality change).
**Weaknesses**: expensive (one forward pass per block), sensitive to perturbation
magnitude choice.

#### 3.3.5 Combining Multiple Metrics

For MXQ at 2-bit, the recommended approach combines multiple metrics:

```
final_score_i = alpha * H_diag_normalized_i
              + beta  * AWQ_normalized_i
              + gamma * KL_normalized_i
```

where alpha + beta + gamma = 1 and each metric is normalized to [0, 1] range.

Recommended weights for 2-bit:
- alpha = 0.4 (Hessian diagonal -- most reliable single metric)
- beta = 0.4 (AWQ -- cheap and complementary)
- gamma = 0.2 (KL divergence -- expensive but catches edge cases)

The combined score is more robust than any individual metric, which is critical at
2-bit where a single misallocated block can noticeably degrade output quality.

### 3.4 The Importance of Calibration for MXQ

MXQ's bit allocation algorithm takes the importance scores from calibration and
decides which blocks get 2, 3, 4, or more bits. Bad calibration leads to:

1. **Over-allocation**: giving 4 bits to insensitive blocks, wasting the bit budget
2. **Under-allocation**: giving 2 bits to critical blocks that needed 3-4
3. **Cascading errors**: if a critical block (e.g., attention Q projection in layer 0)
   is under-allocated, the resulting output errors propagate through all subsequent
   layers, amplifying the damage

At 4-bit uniform quantization, bad calibration affects only GPTQ compensation quality.
At MXQ's mixed-precision 2-bit, bad calibration affects the *structural decision* of
how many bits each block gets -- a much more consequential error.

---

## 4. Practical 2-Bit Quality Benchmarks

### 4.1 Perplexity Numbers Across Methods

The following table collects perplexity results on WikiText-2 (lower is better) for
various 2-bit methods on Llama-2 family models. These numbers are drawn from published
papers and community benchmarks as of early 2025. All measurements use 2048-token
context unless otherwise noted.

**Llama-2 7B at ~2 bits per weight:**

| Method             | bpw  | WikiText-2 PPL | delta vs FP16 |
|--------------------|------|---------------|---------------|
| FP16 (baseline)    | 16.0 | 5.47          | --            |
| GPTQ               | 2.0  | 107.4         | +101.9 (+1863%) |
| GPTQ + act-order   | 2.0  | 43.2          | +37.7 (+689%)  |
| RTN (round-to-nearest) | 2.0 | 2.5e4      | unusable      |
| AWQ                | 2.0  | NaN/diverges  | unusable      |
| QuIP               | 2.0  | 15.7          | +10.2 (+187%) |
| QuIP#  (E8P)       | 2.0  | 9.20          | +3.73 (+68%)  |
| AQLM (1-codebook)  | 2.0  | 8.75          | +3.28 (+60%)  |
| VPTQ               | 2.0  | 8.68          | +3.21 (+59%)  |
| QTIP               | 2.0  | ~8.3          | +2.8 (~51%)   |

**Llama-2 7B at ~2.5 bits per weight (the MXQ sweet spot):**

| Method             | bpw  | WikiText-2 PPL | delta vs FP16 |
|--------------------|------|---------------|---------------|
| GPTQ + act-order   | 2.5  | 12.1          | +6.6 (+121%)  |
| QuIP#              | 2.5  | 7.15          | +1.68 (+31%)  |
| AQLM               | 2.5  | 6.98          | +1.51 (+28%)  |
| VPTQ               | 2.5  | 6.81          | +1.34 (+25%)  |
| **MXQ target**     | 2.5  | **<5.75**     | **<5% delta** |

**Llama-2 70B at ~2 bits per weight:**

| Method             | bpw  | WikiText-2 PPL | delta vs FP16 |
|--------------------|------|---------------|---------------|
| FP16 (baseline)    | 16.0 | 3.32          | --            |
| GPTQ               | 2.0  | 14.9          | +11.6 (+349%) |
| QuIP#              | 2.0  | 4.82          | +1.50 (+45%)  |
| AQLM               | 2.0  | 4.54          | +1.22 (+37%)  |
| VPTQ               | 2.0  | ~4.4          | +1.1 (~33%)   |

**llama.cpp IQ quantization types (Llama-2 70B):**

| Quant Type | bpw   | WikiText-2 PPL | delta vs FP16 |
|------------|-------|----------------|---------------|
| Q4_K_M     | 4.85  | 3.47           | +4.5%         |
| Q3_K_M     | 3.91  | 3.56           | +7.2%         |
| Q2_K       | 3.35  | 3.73           | +12.3%        |
| IQ2_XS     | 2.31  | 4.23           | +27.4%        |
| IQ2_XXS    | 2.06  | 4.58           | +37.9%        |
| IQ1_M      | 1.75  | 6.82           | +105.4%       |

### 4.2 Which Tasks Degrade First at 2-Bit

Not all capabilities degrade equally under extreme quantization. From community
benchmarks and published evaluations, the order of degradation (most fragile to most
robust) is:

**Most fragile (degrades first):**
1. **Mathematical reasoning**: multi-step arithmetic, algebra, proofs
   - Requires precise numerical operations that depend on exact weight values
   - Even small perturbations to weights in MLP layers can cause arithmetic errors
   - GSM8K accuracy drops 15-30% at 2-bit vs FP16 for 7B models

2. **Code generation**: syntax, logic, edge cases
   - Code correctness is binary -- a single wrong token breaks the program
   - HumanEval pass@1 drops 10-25% at 2-bit for 7B models
   - Larger models (70B) lose 5-15% at 2-bit

3. **Factual recall**: specific dates, numbers, names
   - Facts are stored in specific weight patterns that are sensitive to quantization
   - TriviaQA accuracy drops 10-20% at 2-bit for 7B models

**Moderately robust:**
4. **Complex reasoning**: multi-hop, analogical reasoning
   - Degrades 5-15% at 2-bit for 7B, 3-8% for 70B

5. **Multilingual ability**: especially low-resource languages
   - High-resource languages (EN, ZH) are relatively robust
   - Low-resource languages can degrade 15-30%

**Most robust (degrades last):**
6. **General language fluency**: grammar, coherence, style
   - Perplexity increases but text remains fluent
   - Degradation is noticeable but not catastrophic

7. **Summarization/paraphrasing**: extractive and abstractive
   - These tasks are more tolerant of approximate computations

8. **Simple classification**: sentiment, topic
   - Robust because the decision boundary is wide

### 4.3 The "Critical Mass" Hypothesis

There appears to be a minimum model size below which 2-bit quantization breaks down
entirely, regardless of the technique used. This is the "critical mass" hypothesis:

| Model Size | 2-bit Quality (best method) | Viability          |
|-----------|----------------------------|---------------------|
| 1-3B      | 50-60% of FP16 quality     | Not viable          |
| 7B        | 65-80% of FP16 quality     | Marginal            |
| 13B       | 75-85% of FP16 quality     | Usable for some tasks|
| 70B       | 85-92% of FP16 quality     | Good                |
| 100B+     | 90-95% of FP16 quality     | Very good           |

At 2.5-bit (MXQ's target):

| Model Size | 2.5-bit Quality (optimal allocation) | Viability     |
|-----------|--------------------------------------|---------------|
| 7B        | 75-85% of FP16                       | Usable        |
| 13B       | 82-90% of FP16                       | Good          |
| 70B       | 90-96% of FP16                       | Very good     |
| 100B+     | 93-98% of FP16                       | Near-lossless |

### 4.4 Why Larger Models Quantize Better

There are several complementary explanations for why larger models tolerate more
aggressive quantization:

**1. Weight redundancy increases with model size.**

Larger models have more parameters per concept they need to represent. The number of
"facts" and "skills" a model encodes grows roughly linearly with training data, but
parameter count grows quadratically with model width. This means larger models have
more "slack" -- multiple weight configurations can represent the same computation, so
quantization noise is more likely to push weights to another valid configuration
rather than a broken one.

**2. Each individual weight matters less.**

For a matrix multiplication y = Wx where W is m x n, the contribution of a single
weight w_ij to the output is bounded by:

```
|contribution of w_ij| = |w_ij * x_j| <= |w_ij| * ||x||_inf
```

As n increases, the relative contribution of any single weight decreases as 1/n (in
expectation). Quantization error in one weight is diluted by the sum over n terms.

**3. Over-parameterization creates flat loss landscapes.**

Larger models have flatter loss landscapes (more directions in weight space along which
the loss barely changes). Quantization noise predominantly pushes weights along these
flat directions, causing minimal quality impact. Smaller models have sharper loss
landscapes where quantization noise is more likely to push weights over a quality cliff.

**4. Empirical scaling law (Dettmers et al., 2023):**

The degradation from k-bit quantization scales approximately as:

```
delta_PPL(k, N) ~ C * N^{-alpha} * 2^{-beta * k}
```

where N is the number of parameters, k is the bit width, and C, alpha, beta are
empirical constants. For the models studied, alpha ~ 0.5, meaning that doubling the
parameter count halves the quantization degradation. Going from 7B to 70B (10x more
parameters) reduces degradation by roughly sqrt(10) ~ 3.2x.

---

## 5. Emerging Techniques

### 5.1 Trellis Quantization (QTIP)

QTIP (Quantization with Trellises and Incoherence Processing), presented as a NeurIPS
2024 Spotlight, represents the current state-of-the-art in 2-bit post-training
quantization.

#### The Problem with Codebook-Based VQ

QuIP#'s E8 lattice approach quantizes groups of 8 weights to the nearest lattice point.
The effective dimension is 8. Higher dimensions would reduce distortion (approaching
the Zador bound), but the codebook size grows exponentially: a d-dimensional codebook
at 2 bpw needs 2^(2d) entries. At d=16, this is 2^32 = 4 billion entries -- completely
impractical.

#### Trellis Coded Quantization (TCQ)

TCQ, originally from the data compression literature, solves this by using a
**stateful decoder** that separates the codebook size from the effective dimension.

A trellis is a state machine with:
- S states
- At each time step, each state has T transitions (T choices)
- Each transition has an associated output symbol (quantized value)
- The path through the trellis defines the quantized sequence

**Key property**: the number of distinct length-L paths through the trellis is
(S * T)^L, but the codebook at each step has only T entries. The effective codebook
size grows exponentially with path length without requiring exponential storage.

For quantization:
- Each weight is one time step
- The T possible values at each step depend on the current state
- The state transitions encode correlations between consecutive weights
- The optimal path is found via the Viterbi algorithm in O(L * S * T) time

**Effective dimension:**

A trellis with S states and T transitions per state achieves:
- Rate: log_2(T) bits per weight
- Effective dimension: proportional to log_2(S) -- the "memory" of the trellis
- For S=256 and T=4 (2 bits per weight): effective dimension ~ 8 from the state space

#### QTIP's "Bitshift Trellis"

QTIP introduces a hardware-efficient trellis structure where state transitions are
implemented as bit shifts:

```
next_state = (current_state << bits_per_symbol) | symbol
next_state = next_state & state_mask   (keep only log2(S) bits)
```

This has two advantages:
1. **No lookup table for transitions**: the trellis structure is implicit in the
   bit-shift operation
2. **Parallel decoding**: multiple trellis paths can be decoded simultaneously using
   SIMD/SIMT operations on GPU

#### Random Gaussian Codes

For the symbol alphabet at each state, QTIP uses random Gaussian codes rather than
learned codebooks:

```
codebook[state][symbol] = hash(state, symbol) -> N(0, 1/d)
```

These are "computed" (not stored) codebooks where each entry is generated by a fast
hash function seeded by the state and symbol index. The output values are drawn from
a Gaussian distribution, which is optimal for the approximately Gaussian weights
produced by incoherence processing.

**Why this works**: after Hadamard incoherence processing, weights are approximately
i.i.d. N(0, sigma^2). Quantizing an i.i.d. Gaussian source is the classic problem
of Gaussian source coding, for which random codes achieve near-optimal rates (by the
Shannon source coding theorem).

#### Performance

QTIP achieves:
- Better quality than QuIP# at all tested bitrates (2-4 bpw)
- 3x faster inference than unquantized models (due to memory bandwidth savings)
- State-of-the-art perplexity at 2 bpw: approximately 8.3 on WikiText-2 for Llama-2 7B

### 5.2 Learned Step Size Quantization (LSQ)

LSQ (Esser et al., 2019, extended for LLMs in 2024) treats the quantization step size
(scale factor) as a learnable parameter that is optimized jointly with the model weights
during training or fine-tuning.

#### Standard Quantization

```
Q(w) = round(w / s) * s
```

where s is the step size, typically set as:

```
s = max(|w|) / (2^{n-1} - 1)
```

for n-bit signed quantization. This is determined entirely by the weight distribution.

#### LSQ Extension

LSQ makes s a trainable parameter:

```
w_q = Q(w) = clip(round(w / s), -Q_N, Q_P) * s
```

The gradient of the loss L with respect to s is:

```
dL/ds = sum_i (dL/dw_qi) * (dQ/ds)_i
```

Using the straight-through estimator (STE) for the round operation:

```
dQ/ds = { -w/s^2 + round(w/s)/s,  if -Q_N <= round(w/s) <= Q_P
         { -Q_N,                    if round(w/s) < -Q_N
         { Q_P,                     if round(w/s) > Q_P
```

LSQ additionally scales the step size gradient by 1/sqrt(n_weights * Q_P) to
normalize it relative to the weight gradients.

#### Application to 2-Bit

At 2-bit, the step size has an outsized impact on quality because there are only 4
quantization levels. A small change in s shifts all 4 levels, potentially changing
which category every weight falls into. LSQ learns the optimal s that minimizes the
task loss rather than simply minimizing reconstruction error.

For 2-bit, this can improve perplexity by 0.5-2.0 over fixed step sizes, depending
on the model and calibration data. However, LSQ requires a training loop (forward +
backward passes), making it significantly more expensive than PTQ methods.

### 5.3 Training-Aware Quantization (QAT) vs. Post-Training Quantization (PTQ)

#### Post-Training Quantization (PTQ)

PTQ methods (GPTQ, AWQ, QuIP#, AQLM, VPTQ, QTIP) quantize a pre-trained model
without any additional training. They use a small calibration dataset (512-1024 samples)
to guide quantization decisions but do not update the model weights.

**Advantages:**
- Fast: minutes to hours for quantization
- No training data required (only calibration)
- No risk of catastrophic forgetting
- Reproducible: same model + calibration = same result

**Limitations:**
- Cannot recover from fundamental information loss at very low bitrates
- Limited by the information content of the pre-trained weights
- At 2-bit, PTQ methods hit a quality floor that cannot be broken without training

#### Quantization-Aware Training (QAT)

QAT inserts fake quantization operations into the training forward pass, allowing the
model to learn weights that are robust to quantization noise. The gradient flows
through the quantization operation using the straight-through estimator (STE).

```
Forward:  w_q = quantize(w)    (non-differentiable)
Backward: dL/dw = dL/dw_q      (STE: pretend quantization is identity)
```

During training, the model sees the quantized weights and adapts its other weights to
compensate, effectively learning a weight configuration that works well after
quantization.

**Advantages:**
- Can achieve much higher quality at 2-bit than PTQ
- The model actively learns to be quantization-friendly
- Can outperform PTQ by 1-3 perplexity points at 2-bit

**Limitations:**
- Extremely expensive: requires full fine-tuning of the model
- Needs significant training data (millions of tokens)
- Risk of catastrophic forgetting or overfitting to calibration
- Not practical for users quantizing existing models -- requires the original
  training infrastructure

**Recent QAT methods for LLMs (2024-2025):**

| Method        | Approach                                  | 2-bit Quality Improvement vs PTQ |
|--------------|------------------------------------------|----------------------------------|
| LLM-QAT      | Data-free distillation + QAT             | +0.5-1.0 PPL improvement         |
| EfficientQAT | Block-wise QAT with LSQ                  | +0.8-1.5 PPL improvement         |
| BitDistiller  | QAT + self-distillation                 | +1.0-2.0 PPL improvement         |
| ParetoQ       | Unified QAT across bit widths           | +1.5-3.0 PPL improvement         |

#### Hybrid Approach: PTQ + Light Fine-Tuning

The practical middle ground is to apply PTQ first, then perform a short fine-tuning
pass with quantization in the loop. This gives most of QAT's benefits at a fraction
of the cost:

1. Quantize with GPTQ/QuIP#/AQLM at 2-bit (PTQ, ~1 hour)
2. Fine-tune the quantized model for 500-1000 steps with frozen quantization
   parameters (just update the FP16 scales/zeros, ~2-4 hours)
3. Re-quantize with updated scales (~minutes)

This hybrid approach can close 50-70% of the gap between PTQ and full QAT.

### 5.4 Knowledge Distillation + Quantization

Knowledge distillation (KD) uses the outputs of a full-precision "teacher" model to
guide the training of the quantized "student" model. For 2-bit quantization, this
combines the information-preserving power of KD with the memory savings of
quantization.

#### How KD Helps 2-Bit Quantization

Standard quantization minimizes weight reconstruction error:

```
L_recon = ||W - W_hat||^2
```

But this is a proxy for what we actually care about: output quality. KD directly
optimizes output quality by training the quantized model to match the teacher's output
distribution:

```
L_KD = KL(p_teacher(y|x) || p_student(y|x))
```

At 2-bit, the weight reconstruction error is large but much of it may be "harmless"
(along flat directions of the loss landscape). KD ignores harmless reconstruction
error and focuses only on errors that affect outputs.

#### Self-Distillation (BitDistiller)

BitDistiller (Du et al., 2024) uses the unquantized model as its own teacher:

1. Forward pass through the FP16 model: get logits p_FP16
2. Forward pass through the 2-bit model: get logits p_2bit
3. Loss = alpha * L_CE(p_2bit, y) + (1-alpha) * L_KD(p_FP16, p_2bit)
4. Update quantization parameters to minimize loss

The L_KD term uses a **Confidence-Aware KL Divergence** that weights the distillation
loss by the teacher's confidence:

```
L_CAKD = sum_i confidence(p_FP16_i) * KL(p_FP16_i || p_2bit_i)
```

where confidence = max(p_FP16) or entropy-based measure. This focuses distillation on
tokens where the teacher is confident (and thus the student should match closely),
rather than tokens where the teacher is uncertain (and small deviations are acceptable).

#### Data-Free Distillation (LLM-QAT)

LLM-QAT generates its own training data using the teacher model, avoiding the need for
any external dataset:

1. Generate N sequences using the FP16 model with temperature sampling
2. Use these sequences as training data for KD
3. The quantized model learns to match the teacher on the teacher's own distribution

This is particularly elegant because the generated data exactly matches the model's
actual output distribution -- there is no distribution mismatch between calibration
and inference.

#### Optimal Compression Ordering

Recent research (2025) on the interaction between quantization, pruning, and
distillation found that the optimal ordering is:

```
Pruning -> Knowledge Distillation -> Quantization  (P-KD-Q)
```

For MXQ (which does not perform pruning), the relevant finding is that distillation
*before* final quantization produces better results than distillation *after*. This
suggests a pipeline of:

1. Fine-tune the FP16 model with self-distillation to make it more quantization-
   friendly
2. Apply MXQ mixed-precision quantization to the distilled model
3. (Optional) Light fine-tuning of quantized model

### 5.5 1-Bit Models: The Extreme Frontier

#### BitNet b1.58 (Microsoft, 2024-2025)

BitNet takes a fundamentally different approach from post-training quantization: it
trains models from scratch with ternary weights {-1, 0, +1}. This is 1.58 bits per
weight (since log_2(3) = 1.585).

**Architecture:**

BitNet replaces the standard nn.Linear layer with BitLinear:

```python
class BitLinear(nn.Module):
    def forward(self, x):
        # Quantize weights to ternary
        w = self.weight
        alpha = w.abs().mean()  # mean absolute value
        w_ternary = round_ste(w / alpha).clamp(-1, 1)  # {-1, 0, +1}

        # Quantize activations to 8-bit
        x_quant = quantize_activations(x, bits=8)

        # Forward pass with ternary weights
        y = F.linear(x_quant, w_ternary * alpha)
        return y
```

The key operations during inference are only additions and subtractions (multiply by
+1, -1, or 0), eliminating all floating-point multiplications.

**Performance:**

Microsoft's BitNet b1.58 2B4T (2 billion parameters, 4 trillion training tokens,
released 2025):
- Competitive with full-precision models of the same size (e.g., Llama-3 3B)
- 2.7x faster inference
- 3.5x less memory
- Runs efficiently on CPU via bitnet.cpp inference framework

**The catch:** BitNet models must be trained from scratch with ternary constraints.
You cannot convert an existing FP16 model to BitNet quality -- the model must learn
to represent knowledge using only {-1, 0, +1} weights during pretraining. This
requires full training compute, which is impractical for most users.

#### OneBit (NeurIPS 2024)

OneBit takes a different approach to 1-bit quantization, enabling post-training
conversion of existing models to 1-bit:

**Sign-Value-Independent Decomposition (SVID):**

For a weight matrix W in R^{m x n}, OneBit decomposes it as:

```
W ~ sign(W) * (g * h^T)
```

where:
- sign(W) is the binary sign matrix (each entry is +1 or -1, stored as 1 bit)
- g in R^m and h in R^n are learnable column and row scaling vectors (stored in FP16)
- g * h^T is a rank-1 matrix that captures the magnitude distribution

**Storage cost:**

```
Original: m * n * 16 bits (FP16)
OneBit:   m * n * 1 bit (signs) + (m + n) * 16 bits (scaling vectors)
         ~ m * n * 1 bit  (for large m, n the scaling vectors are negligible)
```

**Quality:** OneBit achieves at least 81% of the non-quantized performance on LLaMA
models, which is significantly below what 2-bit methods achieve (85-95%). The 1-bit
representation is fundamentally more limited, and OneBit is better viewed as a
compression technique for model storage/transmission than as a practical inference
format.

#### PT-BitNet: Post-Training Conversion to BitNet

PT-BitNet (2025) attempts to bridge the gap between trained-from-scratch BitNet and
post-training quantization by progressively converting an existing FP16 model to
ternary weights:

1. Start with FP16 model
2. For each layer, compute optimal ternary approximation with learned scaling
3. Fine-tune remaining FP16 layers to compensate
4. Repeat until all layers are ternary

This achieves better quality than OneBit's direct decomposition but still falls short
of trained-from-scratch BitNet, suggesting that the weight space explored during
pretraining is fundamentally different when ternary constraints are applied from the
start.

### 5.6 ParetoQ: Unified Scaling Laws Across Bit Widths

ParetoQ (Meta, 2025) is the first framework that enables rigorous comparison across
1-bit, 1.58-bit (ternary), 2-bit, 3-bit, and 4-bit quantization under a unified
training protocol.

#### Key Findings

**1. Learning transition at 2-3 bits:**

ParetoQ discovers a fundamental phase transition between 2 and 3 bits:
- At 3+ bits, fine-tuned quantized models stay close to their pre-trained weight
  distributions. The quantization is a "perturbation" of the original model.
- At 2 bits and below, the representations change drastically. The model must learn
  fundamentally different internal representations to function with so few values per
  weight.

This explains why PTQ methods hit a hard floor at 2-bit: they cannot restructure the
model's representations, only approximate the existing ones.

**2. Ternary, 2-bit, and 3-bit are comparable:**

Remarkably, ParetoQ finds that with proper QAT, ternary (1.58-bit), 2-bit, and 3-bit
quantization achieve comparable accuracy in the size-accuracy tradeoff. This means
that for a given model size budget, you get similar quality whether you use a larger
model at lower bits or a smaller model at higher bits.

**3. The ParetoQ 600M ternary model outperforms previous 3B ternary models:**

By optimizing the training protocol (learning rate schedule, quantization function,
gradient estimation), ParetoQ achieves quality at 600M parameters that previously
required 3B parameters at the same bitrate.

#### Implications for MXQ

ParetoQ's findings suggest that:
1. Mixed-precision (MXQ's approach) is particularly valuable in the 2-3 bit regime
   because this is exactly where the phase transition occurs -- some blocks may need
   3+ bits to remain in the "perturbation" regime, while others can survive at 2 bits
   in the "restructured" regime.

2. For PTQ methods like MXQ, the 2.5-bit sweet spot is well-chosen: it allows enough
   blocks to remain at 3+ bits to preserve the pre-trained representations where it
   matters, while compressing insensitive blocks to 2 bits.

### 5.7 Additional Emerging Methods

#### GPTVQ: Dimensionality-Aware VQ (Google, 2024)

GPTVQ observes that the "curse of dimensionality" in vector quantization actually
becomes a **blessing** for LLM weights due to their approximate Gaussian distribution.
In high dimensions, Gaussian vectors concentrate on a thin shell, making VQ codebook
design easier. GPTVQ exploits this to achieve 2-bit quantization with minimal quality
loss using relatively simple VQ codebooks.

#### SqueezeLLM: Sparse + Dense Decomposition

SqueezeLLM decomposes weight matrices into a dense low-precision component and a
sparse full-precision component:

```
W ~ W_dense (2-bit) + W_sparse (FP16, <1% non-zero)
```

The sparse component captures outlier weights that cannot be represented at 2-bit.
This achieves better quality than uniform 2-bit but adds complexity to inference
(requires sparse matrix operations).

#### SpinQuant: Rotation-Optimized Quantization (Meta, 2024)

SpinQuant learns optimal rotation matrices (rather than using random Hadamard) for
incoherence processing. By training the rotation matrices with Cayley optimization
on the Stiefel manifold, SpinQuant achieves 0.5-1.0 PPL improvement over QuIP# at
2-bit.

#### Spherical Quantization and Leech Lattice

Recent work (late 2024) explores using the 24-dimensional Leech lattice for
quantization. The Leech lattice has even better packing properties than E8:

```
G(E8) = 0.0717     (8 dimensions)
G(Leech) = 0.0659  (24 dimensions)
```

Quantizing groups of 24 weights at once achieves ~8% less distortion than E8 groups
of 8. However, the larger group size means less granularity in bit allocation and
higher computational cost for nearest-neighbor search. This is an active research area
with no production implementation yet.

---

## Summary: The 2-Bit Toolkit

For MXQ's target of 2-2.5 bit average precision, the following techniques are most
relevant, roughly ordered by implementation priority:

| Technique | Impact at 2-bit | Complexity | MXQ Priority |
|-----------|----------------|------------|-------------|
| Mixed-precision bit allocation | Very high (30%+ distortion reduction) | Medium | Core feature |
| GPTQ error compensation + act-order | High (50-70% of remaining gap) | Medium | Core feature |
| Non-uniform quantization (NF2) | Moderate (13% distortion reduction) | Low | Easy win |
| Calibration quality | Very high (determines allocation quality) | Medium | Core feature |
| Hadamard incoherence | High (enables 2-bit to work at all) | Medium | High priority |
| Vector/lattice quantization | High (14% distortion reduction from E8) | High | Future version |
| Knowledge distillation | High (1-2 PPL improvement) | High | Future version |
| Trellis quantization | Very high (current SOTA) | Very high | Research track |
| QAT | Very high (2-3 PPL improvement) | Very high | Not in scope |

The combination of mixed-precision allocation, GPTQ compensation, NF2 levels,
incoherence processing, and high-quality calibration should enable MXQ to achieve its
target of matching uniform 4-bit quality at 2.5-bit average -- a 37.5% memory
reduction that makes 70B models accessible on 32 GB Macs.

---

## References

Key papers and resources referenced in this document:

- **GPTQ**: Frantar et al., "Accurate Post-Training Quantization for Generative Pre-trained Transformers," ICLR 2023.
- **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," MLSys 2024.
- **QLoRA / NormalFloat**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," NeurIPS 2023.
- **QuIP**: Chee et al., "QuIP: 2-Bit Quantization of Large Language Models With Guarantees," NeurIPS 2023.
- **QuIP#**: Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks," ICML 2024.
- **QTIP**: Tseng et al., "QTIP: Quantization with Trellises and Incoherence Processing," NeurIPS 2024 Spotlight.
- **AQLM**: Egiazarian et al., "Extreme Compression of Large Language Models via Additive Quantization," ICML 2024.
- **VPTQ**: Microsoft, "VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models," EMNLP 2024.
- **BitNet b1.58**: Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," Microsoft Research 2024.
- **OneBit**: Xu et al., "OneBit: Towards Extremely Low-bit Large Language Models," NeurIPS 2024.
- **ParetoQ**: Liu et al., "ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization," Meta 2025.
- **BitDistiller**: Du et al., "BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation," 2024.
- **LLM-QAT**: Liu et al., "LLM-QAT: Data-Free Quantization Aware Training for Large Language Models," ACL Findings 2024.
- **EfficientQAT**: "EfficientQAT: Efficient Quantization-Aware Training for Large Language Models," 2024.
- **Radio**: "Radio: Rate-Distortion Optimization for Large Language Model Compression," 2025.
- **Lloyd-Max**: Lloyd, "Least squares quantization in PCM," IEEE Trans. IT, 1982; Max, "Quantizing for minimum distortion," IEEE Trans. IT, 1960.
- **E8 Lattice Packing**: Viazovska, "The sphere packing problem in dimension 8," Annals of Mathematics, 2017.
- **llama.cpp IQ quantization**: Kawrakow et al., importance-matrix quantization in ggml, 2024.
