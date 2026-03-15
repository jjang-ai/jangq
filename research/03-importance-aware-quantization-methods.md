# Importance-Aware Quantization Methods: A Comprehensive Technical Reference

> Research document for MXQ development. Covers every major importance-aware quantization
> method with full mathematical formulations, algorithmic details, and comparative analysis.

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [AWQ: Activation-Aware Weight Quantization](#2-awq-activation-aware-weight-quantization)
3. [GPTQ: Generative Pre-trained Transformer Quantization](#3-gptq-generative-pre-trained-transformer-quantization)
4. [EXL2: ExLlamaV2 Mixed-Precision Quantization](#4-exl2-exllamav2-mixed-precision-quantization)
5. [SpQR: Sparse-Quantized Representation](#5-spqr-sparse-quantized-representation)
6. [SqueezeLLM: Dense-and-Sparse Quantization](#6-squeezellm-dense-and-sparse-quantization)
7. [QuIP#: Quantization with Incoherence Processing](#7-quip-quantization-with-incoherence-processing)
8. [AQLM: Additive Quantization for Language Models](#8-aqlm-additive-quantization-for-language-models)
9. [Comparative Analysis](#9-comparative-analysis)
10. [Implications for MXQ](#10-implications-for-mxq)

---

## 1. Foundational Concepts

Before examining individual methods, we establish the shared mathematical framework that
all importance-aware quantization methods build upon.

### 1.1 The Uniform Quantization Baseline

Uniform (linear) quantization maps a floating-point weight w to a b-bit integer:

```
Q(w) = clamp(round(w / s) + z, 0, 2^b - 1)
Dequant(q) = (q - z) * s
```

where:
- s = scale factor = (w_max - w_min) / (2^b - 1)
- z = zero point = round(-w_min / s)
- b = number of bits

The quantization error for a single weight is:

```
e = w - Dequant(Q(w))
|e| <= s/2 = (w_max - w_min) / (2^(b+1) - 2)
```

This maximum per-element error shrinks exponentially with bit width. Going from 4-bit to
3-bit roughly doubles the maximum error; going to 2-bit quadruples it relative to 4-bit.

### 1.2 The Layer-Wise Quantization Objective

All post-training quantization methods for LLMs share a common objective: minimize the
output error of each layer when weights are quantized. For a linear layer with weight
matrix W in R^{d_out x d_in} and calibration input X in R^{n x d_in}:

```
minimize ||WX - Q(W)X||_F^2
```

where ||.||_F is the Frobenius norm and Q(.) is the quantization function. This is the
"layer-wise reconstruction" objective. It decouples the global optimization problem into
independent per-layer problems, making quantization tractable for models with billions
of parameters.

The critical insight is that this is NOT the same as minimizing ||W - Q(W)||_F^2 (the
weight error). The activation-weighted error matters more because weights that process
large activations contribute more to the output. This is the foundation of all
importance-aware methods.

### 1.3 The Hessian Connection

The second-order Taylor expansion of the layer output error around the original weights
gives:

```
L(W + dW) - L(W) ~ dW^T * H * dW
```

where H is the Hessian of the layer output with respect to the weights. For a linear
layer y = Wx with squared error loss:

```
H = 2 * X^T * X
```

This Hessian captures how sensitive the output is to perturbations of each weight.
Weights with large Hessian diagonal entries are more "important" -- perturbing them
causes larger output errors. The Hessian is the bridge connecting all methods in this
document: AWQ uses activation norms (related to Hessian diagonal), GPTQ uses the full
Hessian inverse, SpQR uses the Hessian diagonal directly, and so on.

### 1.4 Group Quantization

All modern methods use group quantization rather than per-tensor or per-channel
quantization. The weight matrix is divided into groups of g consecutive weights (typically
g = 32, 64, or 128), and each group gets its own scale and zero point:

```
For group k covering weights [k*g, (k+1)*g):
  s_k = (max(W[k*g:(k+1)*g]) - min(W[k*g:(k+1)*g])) / (2^b - 1)
  z_k = round(-min(W[k*g:(k+1)*g]) / s_k)
```

Smaller groups mean more precise quantization but more overhead from storing scale/zero
parameters. The overhead per group is typically 2 * 16 = 32 bits (one fp16 scale + one
fp16 zero). For group size 128 at 4-bit quantization:

```
Effective bits = 4 + 32/128 = 4.25 bits per weight
```

For group size 32 at 2-bit:

```
Effective bits = 2 + 32/32 = 3.0 bits per weight
```

This overhead is non-trivial at low bit widths, and methods that reduce it gain a
compression advantage.

---

## 2. AWQ: Activation-Aware Weight Quantization

**Paper**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
(Lin et al., 2024, MLSys)

### 2.1 Core Insight

Not all weights in a neural network are equally important for maintaining output quality.
AWQ's key observation is that a small fraction of weights -- roughly 0.1% to 1% --
disproportionately affect model quality when quantized. These "salient" weights correspond
to input channels that consistently carry large activation magnitudes across diverse
inputs. Protecting these weights during quantization preserves model quality far more
effectively than treating all weights uniformly.

The authors demonstrate this with a simple experiment: keeping just 1% of weights at
fp16 while quantizing the rest to 3-bit eliminates most of the quality degradation. But
keeping weights in mixed fp16/int3 format is impractical for hardware -- mixed data types
break SIMD parallelism and require complex kernel dispatch. AWQ's solution is elegant:
instead of keeping salient weights in higher precision, scale them up before quantization
so they occupy more of the quantization grid.

### 2.2 Mathematical Formulation

#### 2.2.1 Importance Metric

Given a weight matrix W in R^{d_out x d_in} and calibration activations X in R^{n x d_in},
the importance of input channel c is defined as:

```
importance(c) = ||X_{:,c}||_2 = sqrt(sum_i X_{i,c}^2)
```

This is the L2 norm of the activation vector for channel c across all calibration samples.
Channels with large activation norms are "important" because weights connected to those
channels have an outsized effect on the output:

```
y = W * x = sum_c W_{:,c} * x_c
```

If x_c is typically large, then quantization error in W_{:,c} gets amplified by x_c in
the output. This is directly related to the Hessian diagonal:

```
[H]_{cc} = sum_i X_{i,c}^2 = ||X_{:,c}||_2^2
```

So AWQ's importance metric is the square root of the Hessian diagonal -- a first-order
approximation of sensitivity.

#### 2.2.2 Per-Channel Scaling

AWQ introduces per-channel scaling factors s in R^{d_in} that modify the quantization
process. Instead of quantizing W directly:

```
Q(W)  -->  Q(W * diag(s)) * diag(s^{-1})
```

The scaling is mathematically equivalent at full precision (W * diag(s) * diag(s^{-1}) = W),
but it changes the quantization error distribution. The quantized operation becomes:

```
y_hat = Q(W * diag(s)) * diag(s^{-1}) * x = Q(W * diag(s)) * (x / s)
```

where division by s is element-wise (x_c / s_c).

The effect: scaling up channel c (s_c > 1) gives the weights in that channel a wider
range relative to other channels. Since quantization allocates grid points uniformly
across the range, a scaled-up channel gets finer effective granularity. Conversely, scaling
down unimportant channels (s_c < 1) compresses them into a smaller portion of the range,
giving them coarser but acceptable granularity.

#### 2.2.3 Optimal Scale Derivation

The objective is to find scales s that minimize the output error:

```
minimize_s ||Q(W * diag(s)) * diag(s^{-1}) * X^T - W * X^T||_F^2
```

This is a non-convex optimization problem. AWQ simplifies it by considering the error
per output channel independently. For a single weight row w in R^{d_in}:

```
error = sum_c (Q(w_c * s_c) / s_c - w_c)^2 * ||X_{:,c}||_2^2
```

The quantization error for a single weight is approximately:

```
Q(w_c * s_c) / s_c - w_c ~ Delta(w_c * s_c) / s_c
```

where Delta(.) is the rounding error, bounded by the quantization step size. If the
quantization step is approximately proportional to the range of the group (which scales
roughly with s_c for the dominant channel), then:

```
|Delta(w_c * s_c)| ~ O(s_c)  (for large s_c)
```

and the per-channel error becomes:

```
error_c ~ (s_c / s_c)^2 * ||X_{:,c}||_2^2 = ||X_{:,c}||_2^2  (for large s_c: cancels)
```

But this analysis is too coarse. The actual benefit comes from the group quantization
structure: within a group, scaling up one channel reduces its relative error at the expense
of other channels in the same group. The optimal balance depends on the activation-weighted
error.

AWQ uses a practical heuristic: set the scale for channel c as a power of its activation
magnitude:

```
s_c = (||X_{:,c}||_2)^alpha
```

where alpha in [0, 1] is found by grid search. The search is fast because:
- alpha = 0 means no scaling (baseline uniform quantization)
- alpha = 1 means full activation-proportional scaling
- The optimal alpha balances protection of salient channels vs. degradation of others
- Typically alpha ~ 0.5 works well across models

The grid search evaluates a small set of alpha values (e.g., 20 values uniformly in
[0, 1]) and picks the one that minimizes the MSE on calibration data. This is a
closed-form evaluation (no gradient computation), so it runs in seconds per layer.

#### 2.2.4 Complete Algorithm

```
AWQ(model, calibration_data):
  for each linear layer with weights W:
    1. Collect activation statistics:
       X = concatenate activations from calibration forward passes
       importance_c = ||X_{:,c}||_2 for each channel c

    2. Grid search for optimal alpha:
       for alpha in {0.0, 0.05, 0.10, ..., 1.0}:
         s_c = importance_c^alpha for each c
         W_scaled = W * diag(s)
         W_quant = GroupQuantize(W_scaled, bits=4, group_size=128)
         error(alpha) = ||W_quant * diag(s^{-1}) * X^T - W * X^T||_F^2
       alpha* = argmin_alpha error(alpha)

    3. Apply optimal scaling:
       s_c = importance_c^(alpha*) for each c
       W_final = GroupQuantize(W * diag(s), bits=4, group_size=128)
       Store: W_final, s, group_scales, group_zeros
```

### 2.3 Implementation Details

#### 2.3.1 Group Size

AWQ typically uses group size 128. The per-channel scales s are applied independently
of the group structure -- they modify the weight values before grouping, which changes
the group statistics (scale and zero point). This is critical: the per-channel scaling
changes which weights "dominate" each group's range, redistributing the quantization grid
to favor salient weights.

#### 2.3.2 Scale Application at Inference

At inference time, the scales must be "undone" to recover the correct output. There are
two approaches:

**Approach 1: Absorb into activations.** Since y = Q(W * diag(s)) * diag(s^{-1}) * x,
the s^{-1} can be applied to the input: x' = x / s, then y = Q(W * diag(s)) * x'. This
avoids any runtime overhead if s^{-1} is fused into the preceding layer's output or
normalization.

**Approach 2: Absorb into subsequent layer.** If the next operation is another linear
layer W2, then W2 * y = W2 * Q(W1 * diag(s)) * diag(s^{-1}) * x. The s^{-1} can be
absorbed into W2 by replacing W2 with W2 * diag(s^{-1}). This is done at quantization
time, so there is zero runtime cost.

In practice, AWQ absorbs scales into adjacent operations (LayerNorm parameters, subsequent
linear layers) wherever possible. For the few cases where absorption is not possible, the
scale is applied at runtime with minimal overhead.

#### 2.3.3 Calibration Requirements

AWQ requires very little calibration data -- typically 128 sequences of 512 tokens from
a dataset like Pile or C4. The calibration serves only to compute activation statistics
(mean L2 norms per channel), not to optimize quantization parameters via backpropagation.
This makes AWQ one of the fastest quantization methods.

### 2.4 Strengths

- **Speed**: No backpropagation, no iterative optimization. Quantizing a 70B model takes
  minutes, not hours.
- **Quality at 4-bit**: Excellent -- within 0.1 perplexity of fp16 on most models.
- **Simplicity**: The method is easy to understand and implement.
- **Hardware-friendly**: The quantized weights are standard integer format with uniform
  bit width per tensor, compatible with existing integer GEMM kernels.
- **Generalization**: The calibration-based importance scores generalize well across
  different downstream tasks and inputs.

### 2.5 Weaknesses

- **Uniform bit width**: All weights within a layer get the same number of bits. AWQ
  cannot allocate 2 bits to unimportant blocks and 6 bits to important ones -- it only
  redistributes quantization precision within a fixed bit width via scaling.
- **Struggles below 3-bit**: At 2-bit, the quantization grid is too coarse (only 4 levels)
  for scaling to help much. The fundamental problem is that 4 quantization levels cannot
  represent the weight distribution regardless of how the grid is placed.
- **Per-channel, not per-block**: The importance metric is per-channel, which is coarser
  than per-block sensitivity. Some blocks within an important channel may be individually
  unimportant, but they get the same protection.
- **Heuristic scaling**: The power-law scaling s_c = importance_c^alpha is a heuristic.
  The true optimal scaling (even assuming per-channel scaling) is the solution to a
  complex optimization problem that AWQ approximates.

### 2.6 Relevance to MXQ

AWQ's activation-based importance metric is directly useful for MXQ's calibration phase.
The key lesson is that activation magnitude is a fast, reliable proxy for weight importance.
MXQ's scoring (Phase 1) should compute per-channel activation norms as one component of
the importance score. However, MXQ goes beyond AWQ by using importance scores to allocate
variable bit widths per block, rather than just scaling within a fixed bit width.

---

## 3. GPTQ: Generative Pre-trained Transformer Quantization

**Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained
Transformers" (Frantar et al., 2023, ICLR)

### 3.1 Lineage: OBS -> OBQ -> GPTQ

GPTQ is the culmination of a line of research in optimal weight pruning and quantization.
Understanding its ancestry clarifies the mathematical foundations.

#### 3.1.1 Optimal Brain Surgeon (OBS)

Hassibi & Stork (1993) introduced OBS for pruning neural networks. The idea: when removing
a weight (setting it to zero), compensate for the induced error by adjusting all remaining
weights. The optimal adjustment is derived from the second-order Taylor expansion of the
loss.

For a network with weight vector w and loss L(w), removing weight w_q induces an error:

```
delta_L = (w_q)^2 / (2 * [H^{-1}]_{qq})
```

where H is the Hessian of L with respect to w, and [H^{-1}]_{qq} is the q-th diagonal
element of its inverse. The optimal compensating update to remaining weights is:

```
delta_w = -w_q / [H^{-1}]_{qq} * H^{-1}_{:,q}
```

This is the weight update that exactly compensates (to second order) for the removal of
weight w_q, using the correlation structure captured by the Hessian.

#### 3.1.2 Optimal Brain Quantization (OBQ)

Frantar & Alistarh (2022) extended OBS from pruning to quantization. Instead of removing a
weight (setting it to zero), they quantize it (rounding to the nearest grid point). The
error induced by quantizing weight w_q to Q(w_q) is:

```
delta_L = (w_q - Q(w_q))^2 / (2 * [H^{-1}]_{qq})
```

And the optimal compensating update is:

```
delta_w = -(w_q - Q(w_q)) / [H^{-1}]_{qq} * H^{-1}_{:,q}
```

OBQ processes weights one at a time, quantizing each and applying the compensating update
to all remaining (not yet quantized) weights. This is optimal in the second-order sense --
each step minimizes the incremental error given the current state.

However, OBQ is prohibitively slow for large models. Processing each weight requires
updating the Hessian inverse (or at least accessing a row of it), and the total complexity
is O(d^3) per row of the weight matrix, where d = d_in. For a 70B model with layers having
d_in = 8192, this is intractable.

### 3.2 The GPTQ Algorithm

GPTQ makes three key modifications to OBQ that make it practical for billion-parameter
models:

#### 3.2.1 Row-Independence

The output of a linear layer y = Wx can be decomposed into independent rows:

```
y_i = W_{i,:} * x  for each output channel i
```

The quantization error for each row is independent (in the Frobenius norm sense), and the
Hessian for each row is the same: H = 2X^TX (it depends only on the inputs, not the
specific row of W). Therefore, GPTQ quantizes each row independently using the same
Hessian, and the Hessian inverse is computed only once per layer.

For a layer with weight matrix W in R^{d_out x d_in} and Hessian H in R^{d_in x d_in}:

```
H = 2 * X^T * X  where X in R^{n x d_in} is the calibration input matrix
```

The Hessian inverse H^{-1} in R^{d_in x d_in} is computed once and shared across all
d_out rows.

#### 3.2.2 Column-Wise Processing (Fixed Order)

OBQ quantizes weights in order of increasing sensitivity (the weight whose quantization
causes the least error is processed first). This requires maintaining a priority queue and
updating priorities after each step -- expensive bookkeeping.

GPTQ makes a simplifying observation: quantizing in a fixed order (left to right across
columns) works nearly as well, and enables massive computational savings. The key insight
is that the compensating updates only propagate "forward" (to not-yet-quantized weights),
so processing columns left to right naturally respects this directionality.

For weight matrix row w in R^{d_in}, processing columns j = 0, 1, 2, ..., d_in - 1:

```
For each column j (in order):
  1. Quantize: q_j = Q(w_j)
  2. Error: e_j = (w_j - q_j) / [H^{-1}]_{jj}
  3. Update remaining weights: w_{j+1:} -= e_j * H^{-1}_{j, j+1:}
```

where H^{-1}_{j, j+1:} is the row of the Hessian inverse corresponding to column j,
restricted to columns j+1 onward (the not-yet-quantized weights).

This is the core GPTQ loop. Step 3 is the compensation step: it adjusts all remaining
weights to account for the quantization error of weight j, using the correlation structure
captured by the Hessian inverse.

#### 3.2.3 Derivation of the Update Rule

We derive the update rule from first principles. The quantization error for a linear layer
is:

```
E = ||WX - Q(W)X||_F^2 = sum_i ||(W_{i,:} - Q(W_{i,:})) * X||^2
```

For a single row w, the error contribution is:

```
e(w) = ||(w - q) * X||^2 = (w - q) * X * X^T * (w - q)^T = (w - q) * H/2 * (w - q)^T
```

where q is the quantized row and H = 2X^TX.

When we quantize column j to q_j = Q(w_j), we want to adjust the remaining weights
w_{j+1:d_in} to minimize the total error. This is a constrained optimization:

```
minimize_{delta} (w + delta - q) * H * (w + delta - q)^T
subject to: delta_0 = delta_1 = ... = delta_j = 0  (already-quantized weights are fixed)
```

Taking the derivative with respect to delta_{j+1:} and setting to zero:

```
H_{j+1:, j+1:} * delta_{j+1:} = -H_{j+1:, j} * (w_j - q_j)
delta_{j+1:} = -H_{j+1:, j+1:}^{-1} * H_{j+1:, j} * (w_j - q_j)
```

Using the Schur complement identity relating H^{-1} to (H_{j+1:, j+1:})^{-1}, this
simplifies to:

```
delta_{j+1:} = -(w_j - q_j) / [H^{-1}]_{jj} * H^{-1}_{j, j+1:}
```

This is exactly the GPTQ update rule.

#### 3.2.4 The Cholesky Decomposition Trick

The above algorithm requires repeated access to rows of H^{-1} and updates to H^{-1} as
columns are "consumed" (the effective Hessian shrinks as weights are quantized). GPTQ
avoids explicitly updating H^{-1} by using the Cholesky decomposition.

Let H^{-1} = L * L^T (Cholesky factorization), where L is lower triangular. Then:

```
[H^{-1}]_{jj} = sum_k L_{j,k}^2 = ||L_{j,:}||^2

H^{-1}_{j, j+1:} can be computed from the rows of L
```

The critical property: when processing columns left to right, the relevant portions of
H^{-1} can be read directly from the Cholesky factor without recomputation. This is
because the Cholesky decomposition "encodes" the sequential elimination structure that
matches the left-to-right processing order.

In practice, GPTQ computes the Cholesky decomposition of H^{-1} once at the start and
reads off the needed values during the column sweep. The Cholesky decomposition costs
O(d_in^3 / 6), which dominates the cost but is a one-time per-layer computation.

Numerical stability note: GPTQ adds a small damping term to the Hessian diagonal before
inversion:

```
H_damped = H + lambda * I  where lambda ~ 0.01 * mean(diag(H))
```

This prevents numerical issues when the Hessian is near-singular (which happens when some
weight channels are rarely activated in the calibration data).

#### 3.2.5 Block-Wise Quantization

Processing one column at a time is still slow because each column update requires modifying
all remaining columns of W (a BLAS-unfriendly operation for large matrices). GPTQ introduces
block-wise processing: instead of one column at a time, process B = 128 columns at a time.

The algorithm becomes:

```
For each block of B columns [j, j+B):
  1. Copy the block: W_block = W[:, j:j+B]
  2. Within the block, process columns sequentially (as above)
  3. After the block: apply the accumulated update to remaining columns:
     W[:, j+B:] -= W_err * H^{-1}_{block, j+B:}
  where W_err accumulates the quantization errors from all B columns
```

The block update in step 3 is a matrix multiply, which is highly optimized on modern
hardware. This reduces the number of large matrix operations from d_in to d_in/B, giving
a ~128x speedup with B = 128.

### 3.3 Act-Order (Activation Order Heuristic)

The original GPTQ paper processes columns left to right (arbitrary order). A later
refinement, called "act-order" or "desc-act," processes columns in decreasing order of
Hessian diagonal [H]_{jj}. Since [H]_{jj} = ||X_{:,j}||_2^2, this means processing the
most-activated (most sensitive) columns first.

The intuition: quantizing a sensitive column first is better because there are more
remaining columns available to compensate for its error. If a sensitive column is
quantized last, there are few or no remaining columns to absorb the compensation.

Act-order requires permuting the columns of W and H before processing, then un-permuting
after quantization. This is a minor implementation detail but provides a consistent
(though small) quality improvement, particularly at lower bit widths.

### 3.4 Complete Algorithm (Pseudocode)

```
GPTQ(W, X, bits, group_size, block_size=128):
  # W: weight matrix (d_out x d_in)
  # X: calibration activations (n x d_in)

  # Step 1: Compute Hessian
  H = 2 * X^T * X / n
  H += lambda * I  (damping for numerical stability)

  # Step 2: Cholesky of H^{-1}
  H_inv = inverse(H)
  L = cholesky(H_inv)  # H_inv = L * L^T

  # Step 3 (optional): Act-order permutation
  perm = argsort(diag(H), descending=True)
  W = W[:, perm]
  H_inv = H_inv[perm, :][:, perm]

  # Step 4: Block-wise quantization
  Q = zeros_like(W)  # quantized output
  for j in range(0, d_in, block_size):
    # Process block [j, j+B)
    err_block = zeros(d_out, block_size)

    for k in range(block_size):
      col = j + k
      # Determine group for scale/zero
      group = col // group_size
      s, z = compute_group_params(W[:, col], bits)

      # Quantize column
      Q[:, col] = quantize(W[:, col], s, z, bits)

      # Error
      err = (W[:, col] - dequantize(Q[:, col], s, z)) / L[col, col]
      err_block[:, k] = err

      # Update remaining columns in this block
      W[:, col+1:j+block_size] -= outer(err, H_inv[col, col+1:j+block_size])

    # Update all remaining columns after this block
    W[:, j+block_size:] -= err_block @ H_inv[j:j+block_size, j+block_size:]

  # Step 5: Un-permute if act-order was used
  Q[:, perm] = Q  (inverse permutation)

  return Q, group_scales, group_zeros
```

### 3.5 Complexity Analysis

For a linear layer with d_out rows and d_in columns:

- Hessian computation: O(n * d_in^2) where n = number of calibration tokens
- Hessian inversion: O(d_in^3)
- Cholesky decomposition: O(d_in^3 / 6)
- Quantization loop: O(d_out * d_in^2 / block_size)
- Total: O(d_in^3 + d_out * d_in^2 / block_size)

For a 70B model with d_in = 8192, this is substantial but feasible. The full model
quantization typically takes 1-4 hours on a single GPU.

### 3.6 Strengths

- **Mathematically optimal rounding**: Each weight is rounded to minimize the second-order
  error, accounting for correlations between weights via the Hessian. This is provably
  better than naive rounding.
- **Excellent 4-bit quality**: GPTQ at 4-bit with group_size=128 and act-order is within
  0.05-0.10 perplexity of fp16 on most LLMs.
- **Widely adopted**: GPTQ is the most-used quantization method, with mature tooling
  (AutoGPTQ, GPTQ-for-LLaMA) and broad hardware support.
- **No retraining**: Like AWQ, GPTQ is a post-training method requiring only calibration
  data (typically 128 sequences of 2048 tokens).

### 3.7 Weaknesses

- **Slow calibration**: The Hessian computation and O(d^3) Cholesky decomposition take
  hours for large models. This is 10-100x slower than AWQ.
- **Uniform bit width**: All weights within a layer get the same number of bits. GPTQ
  cannot perform mixed-precision allocation.
- **Poor 2-bit quality**: At 2 bits, even with optimal rounding, the quantization grid
  (4 levels) is too coarse to represent the weight distribution. Compensating updates
  become so large they cascade into instability. GPTQ at 2-bit typically shows >2x
  perplexity degradation.
- **Sensitivity to calibration data**: The Hessian depends on the calibration set.
  Unrepresentative calibration data leads to poor importance estimates and suboptimal
  compensation. This is more problematic than for AWQ because GPTQ uses the full Hessian,
  not just per-channel norms.
- **Column ordering artifacts**: The left-to-right processing introduces a subtle bias
  where later columns tend to accumulate more error (because they absorb compensation
  from all preceding columns). Act-order mitigates but does not eliminate this.

### 3.8 Relevance to MXQ

GPTQ's optimal rounding and Hessian-based compensation should be used within MXQ's
per-block quantization (Phase 2). After MXQ determines the bit allocation per block, each
block should be quantized using GPTQ-style optimal rounding rather than naive rounding.
The key modification is that different blocks within the same layer use different bit
widths -- the GPTQ algorithm itself is agnostic to the bit width, so this is
straightforward to implement.

The Hessian computation from GPTQ can also feed into MXQ's importance scoring (Phase 1):
the diagonal of X^TX directly measures per-channel sensitivity.

---

## 4. EXL2: ExLlamaV2 Mixed-Precision Quantization

**Author**: turboderp (ExLlamaV2 project)
**Implementation**: ExLlamaV2 quantization module (no formal paper; documented via code
and community posts)

### 4.1 Core Innovation

EXL2 is the first practical mixed-precision quantization scheme for LLMs. Its core
innovation is per-block bit allocation: instead of assigning a uniform bit width to all
weights in a layer, EXL2 assigns different bit widths to different blocks of weights based
on their measured sensitivity. This enables "fractional" average bit widths like 2.5, 3.5,
or 4.25 bits per weight -- granularities that are impossible with uniform quantization.

EXL2 is the closest existing analog to what MXQ aims to achieve, making it the most
important method to study and understand.

### 4.2 The EXL2 Pipeline

EXL2 quantization proceeds in three stages:

```
Stage 1: Sensitivity Measurement
  For each block in each layer, measure the quantization error at each candidate bit width.

Stage 2: Bit Allocation
  Given a target average bit width, allocate bits per block to minimize total error.

Stage 3: Quantization
  Quantize each block at its allocated bit width using GPTQ-style optimal rounding.
```

#### 4.2.1 Stage 1: Sensitivity Measurement

EXL2 divides each weight matrix into blocks (groups) of g weights (typically g = 32 or
g = 128). For each block, it measures the reconstruction error at each candidate bit width
by performing a trial quantization:

```
For each layer with weight matrix W and calibration input X:
  H = X^T * X  (the Hessian, computed once per layer)
  For each block b:
    For each candidate bit width k in {2, 3, 4, 5, 6, 8}:
      Q_b^k = GPTQ_quantize(W_b, H_b, bits=k)
      error_b^k = block_reconstruction_error(W_b, Q_b^k, H_b)
```

The block reconstruction error is computed using the Hessian:

```
error_b^k = (W_b - Q_b^k)^T * H_b * (W_b - Q_b^k)
```

where H_b is the block of the Hessian corresponding to the weights in block b. This
measures the contribution of block b's quantization error to the layer output error,
weighted by the input activation statistics.

This stage produces, for every block in the model, a "rate-distortion" table:

```
block_errors[layer][block] = {
  2: error_at_2_bits,
  3: error_at_3_bits,
  4: error_at_4_bits,
  5: error_at_5_bits,
  6: error_at_6_bits,
  8: error_at_8_bits
}
```

#### 4.2.2 Stage 2: Bit Allocation

Given the per-block error tables and a target average bit width B_target, EXL2 solves the
following optimization problem:

```
minimize  sum_b error_b(bits_b)
subject to: (1/N) * sum_b bits_b * g_b = B_target
            bits_b in {2, 3, 4, 5, 6, 8} for all blocks b
```

where N is the total number of weights and g_b is the number of weights in block b.

This is a variant of the knapsack problem. Each block is an "item" with multiple
"configurations" (bit widths), each having a "cost" (bits) and "value" (negative error).
The constraint is that the total cost equals the target.

##### Greedy Algorithm

EXL2 uses a greedy algorithm that is nearly optimal for this well-structured problem:

```
BitAllocate(block_errors, B_target):
  # Start with all blocks at minimum bits
  bits = {b: 2 for all blocks b}
  total_bits = 2 * N

  # Compute marginal benefit of upgrading each block by 1 bit level
  # Available upgrades: 2->3, 3->4, 4->5, 5->6, 6->8
  heap = MaxHeap()
  for each block b:
    benefit = error_b(2) - error_b(3)  # error reduction from 2->3
    cost = (3 - 2) * g_b               # extra bits needed
    heap.push((benefit / cost, b, 3))   # push (efficiency, block, target_bits)

  # Greedily upgrade the most efficient block until target is reached
  while total_bits / N < B_target:
    (efficiency, b, new_bits) = heap.pop()
    old_bits = bits[b]
    bits[b] = new_bits
    total_bits += (new_bits - old_bits) * g_b

    # Push next upgrade for this block
    next_bits = next_bit_level(new_bits)  # 3->4, 4->5, etc.
    if next_bits is not None:
      benefit = error_b(bits[b]) - error_b(next_bits)
      cost = (next_bits - bits[b]) * g_b
      heap.push((benefit / cost, b, next_bits))

  return bits
```

This greedy algorithm runs in O(N_blocks * log(N_blocks)) time and produces near-optimal
allocations because the marginal error reduction is generally concave in the number of
bits (diminishing returns from more bits).

##### Why Greedy Works Well

The greedy approach is nearly optimal here because the error-vs-bits relationship for
individual blocks is typically concave: the jump from 2-bit to 3-bit error is much larger
than from 3-bit to 4-bit, which is much larger than 4-bit to 5-bit. With concave marginal
returns, the greedy algorithm (always upgrade the block with the best bang-per-bit) is
provably optimal for the continuous relaxation and near-optimal for the integer version.

More precisely, if error_b(k) is convex in k for all blocks b (which the negated error
satisfies since error decreases concavely), then the greedy algorithm achieves an
allocation within one bit level of optimal for each block.

#### 4.2.3 Stage 3: Final Quantization

Once bit widths are assigned, EXL2 performs the actual quantization. It uses GPTQ-style
optimal rounding within each block, but now each block may have a different bit width:

```
For each layer:
  H = X^T * X
  For each block b with allocated bits_b:
    W_b_quantized = GPTQ_quantize(W_b, H_b, bits=bits_b)
    Apply compensation to remaining blocks (the GPTQ update step)
```

The GPTQ compensation is applied globally within each layer, meaning quantization error
from a 2-bit block is absorbed by adjusting weights in subsequent blocks (which might be
4-bit or 6-bit). This cross-block compensation is important: a 2-bit block's large error
can be partially absorbed by a 6-bit block that has the precision to accommodate the
adjustment.

### 4.3 Bit Packing and Storage

EXL2 uses a custom packed format to store variable-width quantized weights. Within each
group, all weights have the same bit width, but different groups can have different widths.
The storage format records:

- Packed quantized values (variable bits per group)
- Per-group scales (fp16)
- Per-group zero points (fp16)
- Per-group bit width (uint8)
- A bit offset table for fast random access to groups of different widths

The effective storage per weight depends on the average bit width plus the overhead of
group metadata. For group size 128 at an average of 3.0 bits:

```
Effective bits = 3.0 + (16 + 16 + 8) / 128 = 3.0 + 0.3125 = 3.3125
```

### 4.4 Custom CUDA Kernels

EXL2 requires custom CUDA kernels for dequantization because standard GEMM libraries
(cuBLAS, etc.) expect uniform data types. The ExLlamaV2 kernels:

1. Read the bit width for each group
2. Dispatch to a specialized unpacking routine for that bit width
3. Dequantize (apply scale and zero point) to fp16
4. Perform the matrix multiplication

The kernels use a "switch" dispatch: for each group, the bit width determines which
unpacking code path executes. This introduces branching, which is less efficient than
uniform-bit-width kernels, but the overhead is small because the branching is at the
group level (not per-weight) and the groups are large enough to amortize dispatch costs.

Performance overhead is typically 5-15% compared to uniform-bit-width kernels at the
same average bit width. Most of this overhead comes from the variable-width unpacking,
not the dispatch itself.

### 4.5 Quality Results

EXL2's mixed-precision allocation produces dramatic quality improvements at low average
bit widths:

```
Llama-2-70B perplexity (wikitext, lower is better):
  fp16:           5.20
  GPTQ 4-bit:     5.26  (+0.06)
  GPTQ 3-bit:     5.72  (+0.52)
  GPTQ 2-bit:     11.4  (+6.2)
  EXL2 4.0 bpw:   5.23  (+0.03)
  EXL2 3.0 bpw:   5.41  (+0.21)
  EXL2 2.5 bpw:   5.62  (+0.42)
  EXL2 2.0 bpw:   6.85  (+1.65)
```

Key observations:
- EXL2 at 3.0 bpw significantly outperforms GPTQ at 3-bit (5.41 vs 5.72) because it can
  allocate 4-6 bits to sensitive blocks while using 2 bits for insensitive ones.
- EXL2 at 2.5 bpw outperforms GPTQ at 3-bit, achieving better quality at lower total size.
- The gap widens at lower bit widths: at 2.0 bpw, EXL2 gives 6.85 vs GPTQ's 11.4 -- a
  massive improvement, though 2.0 bpw is still noticeably degraded from fp16.

### 4.6 Strengths

- **True mixed precision**: The only widely-used method that allocates different bit widths
  to different blocks within a layer.
- **Fractional bit widths**: Enables average bit widths like 2.5, 3.0, 3.5 that are
  impossible with uniform quantization.
- **Excellent low-bit quality**: Dramatically outperforms uniform quantization below 4 bits.
- **Uses GPTQ internally**: Gets the benefit of optimal rounding within each block.
- **Proven at scale**: Used extensively in the community for quantizing 70B+ models.

### 4.7 Weaknesses

- **ExLlamaV2-exclusive**: Requires ExLlamaV2's custom CUDA kernels; cannot be loaded by
  other inference engines.
- **CUDA-only**: No Metal, no CPU support. Restricted to NVIDIA GPUs.
- **Slow calibration**: Inherits GPTQ's O(d^3) Hessian computation, plus the additional
  cost of trial-quantizing at multiple bit widths per block.
- **Discrete bit widths**: Limited to {2, 3, 4, 5, 6, 8} bits per block. No support for
  intermediate widths or non-integer widths within a block.
- **No formal paper**: The method is documented informally, making it harder to rigorously
  analyze or reproduce.
- **Block-level granularity**: Importance is assessed at the block level (e.g., 128 weights).
  Individual outlier weights within a block are not protected -- the entire block gets the
  same bit width.

### 4.8 Lessons for MXQ

EXL2 is the closest precedent for MXQ. Key takeaways:

1. **The three-stage pipeline works**: Measure sensitivity, allocate bits, quantize. MXQ
   should follow this structure.

2. **Greedy allocation is sufficient**: The combinatorial optimization of bit allocation
   is well-solved by a greedy algorithm due to the concavity of error-vs-bits curves.
   Dynamic programming is unnecessary.

3. **GPTQ within each block**: Using optimal rounding within each block is important for
   quality. MXQ should not use naive rounding.

4. **Custom kernels are essential**: The inference engine must support variable-width
   dequantization. For MXQ, this means Metal kernels.

5. **What to improve**: MXQ can improve on EXL2 by:
   - Adding layer-type priors (attention heads need more bits than MLP layers)
   - Using activation-aware importance (AWQ-style) in addition to Hessian-based sensitivity
   - Supporting Apple Silicon via Metal kernels instead of CUDA
   - Finer-grained bit allocation (within-block variance, not just per-block)
   - Potentially incorporating incoherence processing (from QuIP#) before quantization

---

## 5. SpQR: Sparse-Quantized Representation

**Paper**: "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight
Compression" (Dettmers et al., 2023)

### 5.1 Core Idea

SpQR observes that a tiny fraction of individual weights (not channels, not blocks, but
individual scalar weights) are disproportionately important. These "outlier" weights have
extreme values or high sensitivity, and quantizing them causes catastrophic errors. SpQR's
solution: identify these outliers, store them separately in high precision (fp16), and
quantize everything else aggressively.

This is a fundamentally different approach from AWQ (which scales channels) or EXL2 (which
allocates more bits to important blocks). SpQR operates at the individual weight level,
achieving finer granularity than any block-based method.

### 5.2 Mathematical Formulation

#### 5.2.1 The Sensitivity Metric

SpQR's sensitivity metric combines weight magnitude with Hessian information. For weight
w_{ij} (row i, column j of weight matrix W):

```
sensitivity_{ij} = w_{ij}^2 * [H]_{jj}
```

where [H]_{jj} is the j-th diagonal element of the Hessian H = 2X^TX.

##### Derivation

The contribution of weight w_{ij} to the layer output error, when perturbed by delta, is
(from the second-order Taylor expansion):

```
delta_L_{ij} = delta^2 * [H]_{jj}
```

When weight w_{ij} is quantized, the perturbation delta is the rounding error, which scales
with |w_{ij}| (larger weights have larger absolute rounding errors for a given relative
precision). Specifically, for uniform quantization with step size s:

```
|delta| <= s/2 ~ O(max|W| / 2^b)
```

But the actual delta for a specific weight depends on its position within the quantization
grid. As a proxy, SpQR uses w_{ij}^2 (the weight magnitude squared) as an estimate of the
expected squared quantization error, giving:

```
sensitivity_{ij} ~ E[delta_{ij}^2] * [H]_{jj} ~ w_{ij}^2 * [H]_{jj}
```

This can be interpreted as: the expected output error caused by quantizing weight w_{ij}
is proportional to the product of its squared magnitude and its Hessian diagonal entry.

#### 5.2.2 Outlier Detection

Weights with sensitivity above a threshold tau are classified as outliers:

```
outlier_{ij} = 1  if  sensitivity_{ij} > tau
               0  otherwise
```

The threshold tau is chosen to keep the fraction of outliers at p (typically p = 0.5% to
1% of all weights). This can be done by taking the top-p quantile of sensitivities:

```
tau = quantile(sensitivity, 1 - p)
```

Alternatively, SpQR uses an adaptive threshold based on the distribution of sensitivities:

```
tau = mean(sensitivity) + k * std(sensitivity)
```

where k is chosen to yield approximately the desired outlier fraction. The authors find
that keeping ~1% of weights as outliers is sufficient for near-lossless quantization at
3-bit.

#### 5.2.3 Quantization Procedure

Once outliers are identified, the weights are split into two representations:

**Dense component (quantized)**: All non-outlier weights, quantized to b bits (typically
b = 3 or 4) with group quantization:

```
For non-outlier weights in group g:
  s_g, z_g = compute_scale_zero(W_g[~outlier_g], bits=b)
  Q_g = round(W_g[~outlier_g] / s_g + z_g)
```

**Sparse component (full precision)**: Outlier weights stored in fp16 using a sparse
format:

```
For outlier weight w_{ij}:
  Store: (row=i, col=j, value=w_{ij})
```

The sparse representation uses COO (Coordinate) or CSR (Compressed Sparse Row) format.
The storage cost per outlier is:

```
Bits per outlier = 16 (value) + ceil(log2(d_in)) (column index) + row overhead
```

For p = 1% outliers with d_in = 8192, the column index needs 13 bits. Total per outlier:
~29 bits. The effective overhead for 1% outliers at 3-bit base:

```
Effective bits = 3.0 * 0.99 + (16 + 13) * 0.01 = 2.97 + 0.29 = 3.26 bpw
```

This is remarkably efficient: keeping 1% of weights at full precision adds only ~0.26
bits per weight to the base 3-bit quantization.

#### 5.2.4 GPTQ Integration

SpQR uses GPTQ-style optimal rounding for the dense (non-outlier) weights. The key
modification: outlier weights are fixed at their fp16 values and excluded from the
quantization sweep. The GPTQ compensation only applies to non-outlier weights:

```
Modified GPTQ:
  For each column j:
    For each weight w_{ij}:
      if outlier_{ij}:
        Q_{ij} = w_{ij}  (keep at full precision, no quantization error)
      else:
        Q_{ij} = Quantize(w_{ij})
        error = w_{ij} - Dequant(Q_{ij})
        # Compensate only non-outlier remaining weights
        W_{i, j+1:}[~outlier_{i, j+1:}] -= error / [H^{-1}]_{jj} * H^{-1}_{j, j+1:}
```

This is important: by fixing outliers, the GPTQ compensation only distributes error to
quantized weights. The outliers act as "anchors" that stabilize the compensation process.

### 5.3 Inference Kernel Design

At inference, the output of a SpQR linear layer is:

```
y = Dequant(Q_dense) * x + W_sparse * x
```

The first term is a standard quantized GEMM. The second term is a sparse-dense matrix
multiply (SpMM). Modern GPUs are reasonably efficient at SpMM for the low sparsity levels
(~1%) used by SpQR.

SpQR's kernel fuses the two operations:

```
kernel SpQR_GEMM:
  # Dense quantized component
  y = quantized_gemm(Q_dense, scales, zeros, x)
  # Add sparse outlier contributions
  for each outlier (i, j, val):
    y[i] += val * x[j]
```

The sparse addition is scatter-add, which is efficient on GPUs for small numbers of
outliers but becomes a bottleneck if the outlier fraction exceeds ~5%.

### 5.4 Strengths

- **Near-lossless**: At 3-4 bits with 1% outliers, SpQR achieves perplexity within 0.1%
  of fp16 -- essentially lossless.
- **Fine-grained importance**: Individual weight-level sensitivity captures information
  that block-level methods miss. A single extreme weight in an otherwise unimportant block
  is properly protected.
- **Low overhead**: The sparse outlier storage adds only ~0.3 bits per weight at 1% sparsity.
- **Principled sensitivity metric**: The w^2 * H_{jj} metric has a clear second-order
  justification.
- **Compatible with GPTQ**: Uses GPTQ for the dense component, getting optimal rounding
  for free.

### 5.5 Weaknesses

- **Sparse kernels**: Efficient SpMM at very low sparsity (~1%) requires custom kernels.
  Standard sparse libraries are optimized for higher sparsity levels.
- **Memory access patterns**: The sparse component causes irregular memory accesses,
  potentially reducing throughput on bandwidth-bound hardware.
- **Uniform bit width for dense component**: Like GPTQ, the dense (non-outlier) weights
  all use the same bit width. There is no mixed-precision allocation within the dense
  component.
- **Outlier fraction is a hyperparameter**: The right fraction depends on the model and
  target quality. Too few outliers and quality suffers; too many and the sparse overhead
  becomes significant.
- **Does not compose easily**: Hard to combine with methods like AWQ (which assumes all
  weights are uniformly quantized) or EXL2 (which assumes block-level uniformity).

### 5.6 Relevance to MXQ

SpQR's sensitivity metric (w^2 * H_{jj}) is directly useful for MXQ's importance scoring.
MXQ should compute this metric and use it as one component of its block-level importance
score (e.g., the max or mean sensitivity within each block).

However, MXQ should prefer block-level mixed precision (like EXL2) over sparse outlier
storage because:
1. Block-level operations are much more efficient on Apple Silicon's unified memory
   architecture (coalesced reads, SIMD-friendly).
2. Sparse scatter-add operations are less efficient on Metal than on CUDA.
3. The MXQ file format is simpler without a sparse component.

That said, MXQ could consider a hybrid approach: mixed-precision blocks (EXL2-style) plus
sparse outlier protection for the small number of extreme weights that even 8-bit blocks
cannot adequately represent.

---

## 6. SqueezeLLM: Dense-and-Sparse Quantization

**Paper**: "SqueezeLLM: Dense-and-Sparse Quantization" (Kim et al., 2024)

### 6.1 Core Ideas

SqueezeLLM combines two techniques:

1. **Non-uniform (lookup table) quantization**: Instead of uniformly-spaced quantization
   levels, SqueezeLLM learns optimal quantization levels via sensitivity-weighted k-means
   clustering. This allows the quantization grid to adapt to the weight distribution.

2. **Dense-and-sparse decomposition**: Like SpQR, outlier weights are stored separately at
   full precision while the rest are quantized. But SqueezeLLM's outlier detection uses
   Fisher information rather than the Hessian diagonal.

### 6.2 Non-Uniform Quantization via Sensitivity-Weighted K-Means

#### 6.2.1 Standard K-Means Quantization

Standard k-means quantization finds 2^b cluster centers (quantization levels) that
minimize the total quantization error:

```
minimize  sum_i (w_i - c_{a(i)})^2
```

where c_1, ..., c_{2^b} are the cluster centers and a(i) assigns weight i to its nearest
center. This is Lloyd's algorithm applied to quantization.

For 4-bit quantization, we find 16 cluster centers that best represent the weight
distribution. Unlike uniform quantization (evenly spaced levels), k-means places more
levels where weights are dense, giving finer resolution in the peak of the distribution.

#### 6.2.2 Sensitivity-Weighted K-Means

SqueezeLLM's key contribution is weighting the k-means objective by the sensitivity of
each weight:

```
minimize  sum_i sensitivity_i * (w_i - c_{a(i)})^2
```

where sensitivity_i measures how much the output is affected by errors in weight w_i.

This changes the clustering to place more quantization levels where sensitive weights
are concentrated, even if those regions have fewer weights overall. The effect: important
weights get finer quantization granularity, while unimportant weights tolerate coarser
levels.

#### 6.2.3 Fisher Information as Sensitivity

SqueezeLLM uses the Fisher information matrix diagonal as its sensitivity metric. The
Fisher information for weight w_ij with respect to the log-likelihood of the model is:

```
F_{ij} = E_x [(d log p(x) / d w_{ij})^2]
```

For a linear layer y = Wx with subsequent softmax output, the Fisher information of weight
w_{ij} is related to the variance of the gradient of the log-likelihood with respect to
that weight. In practice, SqueezeLLM approximates the Fisher information using the
empirical gradient variance on calibration data:

```
F_{ij} ~ (1/n) * sum_{t=1}^{n} (d L_t / d w_{ij})^2
```

where L_t is the loss on calibration sample t.

##### Why Fisher Information?

The Fisher information matrix is the expected Hessian of the log-likelihood. For well-
specified models, it equals the Hessian at the optimum. Using Fisher information instead
of the Hessian (as in GPTQ/SpQR) has two advantages:

1. **Always positive semi-definite**: The Fisher information is guaranteed non-negative,
   while the Hessian might have negative eigenvalues away from the optimum.
2. **Captures gradient flow**: The Fisher information measures how much the output
   distribution changes when a weight changes, which is more directly relevant to model
   quality than the Hessian of a proxy loss.

However, computing the full Fisher information requires backpropagation through the model
for each calibration sample, making it significantly more expensive than the forward-only
Hessian computation used by GPTQ. SqueezeLLM uses only the diagonal of the Fisher matrix
(ignoring cross-weight correlations), which reduces the cost to one backward pass per
calibration sample.

#### 6.2.4 The Weighted K-Means Algorithm

```
SensitivityWeightedKMeans(weights, sensitivities, k=2^b, max_iter=100):
  # Initialize centers using sensitivity-weighted k-means++
  centers = [weighted_random_choice(weights, sensitivities)]
  for _ in range(k - 1):
    distances = [min_c sensitivity_i * (w_i - c)^2 for w_i in weights]
    centers.append(weighted_random_choice(weights, distances))

  # Iterate
  for iter in range(max_iter):
    # Assignment: assign each weight to nearest center (unweighted distance)
    assignments = [argmin_j (w_i - centers[j])^2 for w_i in weights]

    # Update: weighted mean within each cluster
    for j in range(k):
      cluster_weights = {w_i : assignments[i] == j}
      cluster_sensitivities = {sensitivity_i : assignments[i] == j}
      centers[j] = sum(s_i * w_i) / sum(s_i)  for i in cluster j

  return centers
```

The key difference from standard k-means is in the update step: cluster centers are
computed as sensitivity-weighted means, not unweighted means. This pulls cluster centers
toward sensitive weights, giving them more precise representation.

#### 6.2.5 Lookup Table Quantization at Inference

Non-uniform quantization requires a lookup table (LUT) at inference. Each quantized weight
is a b-bit index into a table of 2^b float16 values:

```
Dequant(q_i) = LUT[q_i]  where LUT = [c_0, c_1, ..., c_{2^b-1}]
```

The LUT is stored per group (each group of g weights has its own set of 2^b cluster
centers). Storage overhead per group:

```
LUT overhead = 2^b * 16 bits  (fp16 per center)
```

For b = 4 (16 centers) and group size 128:

```
Overhead = 16 * 16 / 128 = 2.0 bits per weight
Effective bits = 4.0 + 2.0 = 6.0 bits per weight
```

This is a significant overhead. To mitigate it, SqueezeLLM uses larger group sizes
(256 or 512) and shares LUTs across groups with similar weight distributions.

For b = 3 (8 centers) and group size 128:

```
Overhead = 8 * 16 / 128 = 1.0 bit per weight
Effective bits = 3.0 + 1.0 = 4.0 bits per weight
```

This LUT overhead is the primary drawback of non-uniform quantization compared to uniform
quantization, which needs only 2 parameters (scale and zero point) per group.

### 6.3 Dense-and-Sparse Decomposition

SqueezeLLM uses a decomposition similar to SpQR:

```
W = W_dense + W_sparse
```

where W_dense contains all non-outlier weights (quantized via sensitivity-weighted k-means)
and W_sparse contains outlier weights in fp16.

The outlier detection uses the Fisher information diagonal:

```
outlier_{ij} = 1  if  F_{ij} * w_{ij}^2 > tau
```

This is the same form as SpQR's metric but using Fisher information instead of the Hessian
diagonal. In practice, the two metrics are highly correlated for well-trained models.

### 6.4 Strengths

- **Adaptive quantization levels**: Non-uniform quantization places levels where the
  weight distribution needs them, not uniformly across the range. This is theoretically
  superior to uniform quantization for non-uniform weight distributions (which all LLM
  weights have).
- **Sensitivity-weighted optimization**: The k-means objective directly minimizes the
  sensitivity-weighted error, tying the quantization to the model's quality metric.
- **Strong 3-bit results**: SqueezeLLM at 3-bit non-uniform outperforms GPTQ at 3-bit
  uniform, demonstrating the value of non-uniform levels.
- **Fisher information**: A principled and theoretically grounded sensitivity metric.

### 6.5 Weaknesses

- **LUT overhead**: The per-group lookup table adds 1-2 bits per weight of overhead,
  reducing the effective compression ratio. This is the method's Achilles' heel.
- **LUT dequantization is slow**: At inference, each weight requires an indirect memory
  access (load index, then load from LUT). This is slower than the multiply-add of
  uniform dequantization (load value, multiply by scale, add zero point). The LUT
  lookup introduces data-dependent memory access patterns that defeat hardware prefetching.
- **Requires backpropagation**: Computing the Fisher information requires backward passes
  through the model, making calibration significantly slower than forward-only methods
  (AWQ, GPTQ).
- **No mixed precision**: Like GPTQ, all dense weights use the same bit width. The non-
  uniform levels provide some flexibility, but not the same as variable bit allocation.
- **Complexity**: The combination of non-uniform quantization + sparse outliers + Fisher
  information creates a complex pipeline that is harder to implement and debug than simpler
  methods.

### 6.6 Relevance to MXQ

SqueezeLLM's key lesson is that non-uniform quantization levels can improve quality but at
the cost of LUT overhead and slower dequantization. For MXQ on Apple Silicon:

- **Against non-uniform quantization**: Metal kernels are optimized for regular memory
  access patterns. LUT-based dequantization would introduce irregular accesses that
  reduce throughput on the unified memory architecture. Uniform quantization with mixed
  bit widths (EXL2-style) is likely faster on Metal.

- **For sensitivity-weighted optimization**: The idea of weighting the quantization
  objective by sensitivity is valuable regardless of whether uniform or non-uniform levels
  are used. MXQ can apply sensitivity weighting to its bit allocation (giving more bits
  to sensitive blocks) without adopting LUT-based levels.

- **Fisher information**: Computing the Fisher information is expensive (requires backprop)
  and provides only marginally better sensitivity estimates than the Hessian diagonal
  (which requires only forward passes). MXQ should use the Hessian diagonal (like
  GPTQ/SpQR) rather than the Fisher information.

---

## 7. QuIP#: Quantization with Incoherence Processing

**Paper**: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice
Codebooks" (Tseng et al., 2024)

### 7.1 Core Insight

QuIP# is built on a fundamental observation about quantization error: quantization is
hardest when the weight matrix has high "coherence" -- meaning some columns (or rows) have
much larger magnitudes than others. In a coherent matrix, the quantization range is
dominated by a few large-magnitude channels, and the small-magnitude channels get poor
resolution.

The solution: before quantization, rotate the weight matrix by a random orthogonal
transform so that all columns have approximately equal magnitude. After rotation, the
weight matrix is "incoherent" -- the information is spread uniformly across all channels --
and quantization is more uniform and efficient.

### 7.2 Mathematical Formulation

#### 7.2.1 Coherence and Quantization Error

The coherence of a matrix W in R^{m x n} is defined as:

```
mu(W) = (n / ||W||_F^2) * max_j ||W_{:,j}||_2^2
```

A matrix with uniform column norms has mu(W) = 1 (minimum coherence). A matrix with one
dominant column has mu(W) ~ n (maximum coherence). The key result from the QuIP# paper:

**Theorem (informal)**: The expected quantization error of a matrix W with uniform
quantization at b bits is bounded by:

```
E[||W - Q(W)||_F^2] <= C * mu(W) * ||W||_F^2 / 2^{2b}
```

where C is a constant. The error scales linearly with coherence. Reducing coherence from
mu = 100 to mu = 1 reduces the quantization error by 100x -- equivalent to using
~3.3 more bits.

#### 7.2.2 Incoherence via Random Orthogonal Transforms

The core idea: multiply W by random orthogonal matrices to reduce coherence. Let U in
R^{m x m} and V in R^{n x n} be random orthogonal matrices (e.g., randomized Hadamard
matrices). Define:

```
W_rotated = U * W * V^T
```

Since U and V are orthogonal, this is an information-preserving rotation: no information
is lost. At inference, the rotation is undone:

```
y = W * x = U^T * (U * W * V^T) * V * x = U^T * W_rotated * (V * x)
```

So the inference computation becomes:

```
1. Rotate input: x' = V * x           (O(n log n) if V is Hadamard)
2. Quantized GEMM: y' = Q(W_rotated) * x'  (standard quantized matmul)
3. Rotate output: y = U^T * y'        (O(m log m) if U is Hadamard)
```

The critical property: after random rotation, the coherence of W_rotated concentrates
around 1 (its minimum value) with high probability. This is a consequence of the
Johnson-Lindenstrauss lemma: random projections approximately preserve norms, so the
column norms of W_rotated are approximately equal.

#### 7.2.3 Hadamard Matrices

QuIP# uses randomized Hadamard matrices for U and V because they can be applied in
O(n log n) time (via the fast Walsh-Hadamard transform), compared to O(n^2) for a generic
orthogonal matrix.

A Hadamard matrix H_n of order n is defined recursively:

```
H_1 = [1]
H_{2n} = (1/sqrt(2)) * [[H_n, H_n], [H_n, -H_n]]
```

Every entry of H_n is +/- 1/sqrt(n), and H_n^T * H_n = I (orthogonal after scaling).

A randomized Hadamard matrix is:

```
R = H_n * D
```

where D is a diagonal matrix with random +/- 1 entries on the diagonal. This adds
randomness (breaking any structure in W that might align with H_n's structure) while
preserving the O(n log n) fast transform.

#### 7.2.4 The Fast Walsh-Hadamard Transform

The Hadamard transform x' = H_n * x can be computed in O(n log n) operations using a
butterfly-style algorithm (analogous to the FFT):

```
FastHadamard(x, n):
  if n == 1: return x
  x_top = x[0:n/2]
  x_bot = x[n/2:n]
  a = FastHadamard(x_top + x_bot, n/2)
  b = FastHadamard(x_top - x_bot, n/2)
  return [a, b] / sqrt(2)
```

The iterative (non-recursive) version processes log2(n) stages, each consisting of n/2
butterfly operations. This is highly parallelizable on GPU/Metal and adds negligible
overhead to inference.

Practical overhead: for d_in = 8192 (typical for 70B models), the Hadamard transform
requires 8192 * 13 = ~100K multiply-adds, compared to ~67M for the matrix multiplication
(8192 * 8192). The transform adds <0.15% overhead.

#### 7.2.5 Inference Computation

The full inference pipeline for a QuIP#-quantized linear layer:

```
Input: x in R^{d_in}  (activation vector or batch)

1. Apply input rotation:    x' = V * x         (fast Hadamard, O(d_in log d_in))
2. Quantized matmul:        y' = Dequant(Q) * x'  (standard quantized GEMM)
3. Apply output rotation:   y = U^T * y'        (fast Hadamard, O(d_out log d_out))

Output: y in R^{d_out}
```

The rotations U and V are fixed at quantization time and stored as part of the model
metadata. Since they are random Hadamard matrices, they are fully determined by a random
seed, so only the seed needs to be stored (negligible overhead).

### 7.3 Lattice Codebooks (E8 Quantization)

QuIP# goes beyond standard round-to-nearest quantization by using lattice codebooks for
vector quantization of groups of weights.

#### 7.3.1 The E8 Lattice

The E8 lattice is an 8-dimensional lattice that achieves the densest known sphere packing
in 8 dimensions. Its Voronoi cells (the quantization regions around each lattice point)
have the smallest normalized second moment of any known lattice in 8 dimensions:

```
G(E8) = 0.0717  (normalized second moment)
```

compared to:

```
G(Z^8) = 1/12 = 0.0833  (integer lattice, i.e., independent scalar rounding)
```

This means that vector quantizing groups of 8 weights using the E8 lattice reduces
quantization error by a factor of G(Z^8)/G(E8) = 0.0833/0.0717 = 1.16 compared to
independent rounding -- a 16% reduction in MSE, equivalent to gaining ~0.11 extra bits.

#### 7.3.2 E8 Quantization

QuIP# groups weights into vectors of 8 and quantizes each vector to the nearest E8
lattice point:

```
For group of 8 weights w = [w_1, ..., w_8]:
  1. Scale: w' = w / s  (where s is the group scale factor)
  2. Find nearest E8 point: q = argmin_{p in E8} ||w' - p||^2
  3. Encode: store the index of q in the E8 codebook
```

The E8 lattice has exactly 240 shortest vectors (the "roots"), but QuIP# uses a truncated
codebook appropriate for the target bit width. For 2-bit quantization (4 levels per weight,
4^8 = 65536 total for 8 weights), the codebook contains 65536 E8 lattice points selected
to cover the most likely weight vectors.

The encoding uses 16 bits per group of 8 weights = 2 bits per weight. Decoding requires
looking up the 8-dimensional vector from the codebook index.

#### 7.3.3 Why E8 Helps

The advantage of lattice codebooks over independent scalar quantization comes from
exploiting correlations between weights within a group. Even after incoherence processing
(which removes inter-column correlations), there can be within-group structure that E8
exploits. The 16% MSE reduction is "free" -- it comes from smarter joint rounding, not
from using more bits.

### 7.4 Complete QuIP# Pipeline

```
QuIP#_Quantize(model, calibration_data, bits):
  For each linear layer with weights W:
    # Step 1: Compute Hessian
    H = 2 * X^T * X / n

    # Step 2: Generate random rotations
    seed = random()
    U = RandomHadamard(d_out, seed)
    V = RandomHadamard(d_in, seed + 1)

    # Step 3: Rotate weights and Hessian
    W_rot = U * W * V^T
    H_rot = V * H * V^T

    # Step 4: LDLQ (LDLT-based quantization with Hessian compensation)
    # Similar to GPTQ but with E8 lattice rounding
    For each group of 8 consecutive columns:
      w_group = W_rot[:, j:j+8]
      H_group = H_rot[j:j+8, j:j+8]

      # Find nearest E8 lattice point (instead of scalar rounding)
      q_group = E8_nearest(w_group / scale)
      q_group = q_group * scale

      # GPTQ-style compensation for remaining weights
      error = w_group - q_group
      W_rot[:, j+8:] -= error * H_rot[j:j+8, j+8:]^{-1}

    # Step 5: Store
    Store: quantized lattice indices, scales, rotation seeds
```

### 7.5 Quality Results

QuIP# achieves remarkable quality at 2-bit:

```
Llama-2-70B perplexity (wikitext):
  fp16:           5.20
  GPTQ 4-bit:     5.26
  GPTQ 2-bit:     11.4
  QuIP# 4-bit:    5.22
  QuIP# 2-bit:    5.75
```

The 2-bit result is extraordinary: QuIP# at 2 bits per weight (5.75 perplexity) is
competitive with GPTQ at 4 bits (5.26 perplexity). The combination of incoherence
processing + E8 lattice codebooks + Hessian compensation makes 2-bit quantization
viable for the first time.

### 7.6 Strengths

- **State-of-the-art 2-bit**: The best quality at 2 bits per weight of any published
  method. The incoherence processing is the key enabler.
- **Theoretically grounded**: The coherence-error bound provides a rigorous justification
  for why rotation helps. This is not a heuristic -- it is provably optimal in a
  well-defined sense.
- **Low overhead transforms**: The Hadamard transforms add negligible (<0.2%) inference
  cost.
- **E8 lattice is optimal**: The 16% MSE reduction from E8 vs scalar quantization is a
  "free lunch" from smarter rounding.
- **Composable**: Incoherence processing is a preprocessing step that can be combined with
  other quantization methods (GPTQ, AWQ, mixed precision, etc.).

### 7.7 Weaknesses

- **Complex implementation**: E8 lattice encoding/decoding requires specialized codebook
  lookup routines. Standard quantized GEMM kernels do not support lattice codebooks.
- **Custom kernels required**: Like EXL2, QuIP# needs custom CUDA kernels for the lattice
  dequantization. These are more complex than standard uniform dequantization kernels.
- **Limited hardware support**: Currently CUDA-only. No Metal or CPU implementations.
- **No mixed precision**: QuIP# uses a uniform bit width across all weights within a
  layer. The coherence reduction makes this less problematic (all weights are now "equally
  important" after rotation), but there may still be room for improvement via mixed
  precision.
- **Rotation overhead in practice**: While theoretically negligible, the Hadamard
  transforms add complexity to the inference pipeline and can complicate integration with
  existing frameworks.
- **Activation rotation**: The input rotation (V * x) must be applied at every inference
  step. For batched inference with large batch sizes, this can become non-trivial.

### 7.8 Relevance to MXQ

QuIP#'s incoherence processing is potentially very valuable for MXQ:

1. **Combining rotation with mixed precision**: MXQ could apply Hadamard rotation before
   quantization, then allocate bits per block on the rotated weights. Since the rotated
   weights are more uniform, the bit allocation might be more efficient (fewer blocks
   needing very high bit widths to handle outlier columns).

2. **Fast Hadamard on Metal**: The Walsh-Hadamard transform maps naturally to Metal's
   SIMD architecture. A Metal kernel for the transform would be straightforward and
   efficient.

3. **Skip the E8 lattice**: The lattice codebook adds implementation complexity and
   kernel complexity that may not be worth the 16% MSE improvement. MXQ could adopt
   the Hadamard rotation (the bigger win) without the E8 lattice (the smaller,
   harder-to-implement win).

4. **Key question**: Does incoherence processing help or hurt mixed-precision allocation?
   After rotation, all blocks have similar sensitivity, which means mixed precision has
   less to exploit. It is possible that rotation + uniform bits outperforms no rotation +
   mixed bits, or vice versa. This needs empirical investigation.

---

## 8. AQLM: Additive Quantization for Language Models

**Paper**: "AQLM: Extreme Compression of Large Language Models via Additive Quantization"
(Egiazarian et al., 2024)

### 8.1 Core Idea

AQLM abandons scalar quantization entirely. Instead of quantizing each weight independently
to a b-bit integer, AQLM quantizes groups of weights jointly as high-dimensional vectors,
using multi-codebook vector quantization.

The key insight: in high dimensions, vector quantization is exponentially more efficient
than scalar quantization. For d-dimensional vectors at the same bit rate, vector
quantization achieves MSE that is O(2^{-2R}) where R is the total rate (bits per
dimension), compared to O(2^{-2R/d}) for scalar quantization. This is the "vector
quantization gain" from classical rate-distortion theory.

### 8.2 Mathematical Formulation

#### 8.2.1 Multi-Codebook Additive Quantization

AQLM represents each group of d weights as the sum of M codewords from M separate
codebooks:

```
w_group ~ sum_{m=1}^{M} C_m[i_m]
```

where:
- C_m in R^{K x d} is codebook m, containing K = 2^k codeword vectors of dimension d
- i_m in {0, 1, ..., K-1} is the index into codebook m
- The total representation requires M * k bits per group, or M * k / d bits per weight

For example, with d = 8, M = 2, k = 8 (256 codewords per codebook):
- Total bits per group: 2 * 8 = 16 bits
- Bits per weight: 16 / 8 = 2 bits per weight
- Each group is the sum of two 8-dimensional codeword vectors

This is "additive" quantization because the representation is a sum of codewords, not
a single codeword. The additive structure provides much richer representations than a
single codebook of the same total size: with 2 codebooks of 256 codewords each, the
effective vocabulary is 256 * 256 = 65536 possible vector representations, using only
16 bits.

#### 8.2.2 Codebook Learning

The codebooks are learned from the model weights + calibration data by minimizing the
layer-wise reconstruction error:

```
minimize_{C_1,...,C_M, I} ||W * X - Q_AQLM(W) * X||_F^2
```

where Q_AQLM(W) replaces each group of d weights with the sum of its M codewords.

This is a joint optimization over:
1. The codebook contents (continuous: the codeword vectors)
2. The assignments (discrete: which codeword index for each group from each codebook)

The optimization alternates between:

**Assignment step** (fix codebooks, optimize indices): For each group, find the best
combination of M codeword indices that minimizes the reconstruction error. For M = 2
with K = 256, this is a search over 256^2 = 65536 combinations, which is feasible by
enumeration. For larger M, beam search or greedy selection is used.

```
For each group g:
  (i_1*, i_2*, ..., i_M*) = argmin_{i_1,...,i_M}
    ||W_g - sum_m C_m[i_m]||_H^2

  where ||v||_H^2 = v^T * H_g * v  (Hessian-weighted error)
```

The Hessian weighting ensures that the assignment minimizes the activation-weighted error
(the layer output error), not just the weight error.

**Codebook update** (fix indices, optimize codebooks): With indices fixed, the optimal
codebook entries are found by solving a linear system. For codebook m, entry j, the
optimal codeword is:

```
C_m[j] = argmin_c sum_{g: i_{g,m}=j} ||W_g - c - sum_{m' != m} C_{m'}[i_{g,m'}]||_H^2
```

This is a weighted least-squares problem with a closed-form solution:

```
C_m[j] = (sum_{g: i_{g,m}=j} H_g)^{-1} * sum_{g: i_{g,m}=j} H_g * (W_g - sum_{m'!=m} C_{m'}[i_{g,m'}])
```

#### 8.2.3 Fine-Tuning (Optional)

After the initial codebook learning, AQLM optionally fine-tunes the codebooks using
gradient-based optimization on a small dataset. The codebook entries are treated as
learnable parameters, and the assignments are fixed. Straight-through estimators are used
for the non-differentiable argmin in the assignment step.

This fine-tuning step uses backpropagation through the full model (not just layer-wise
reconstruction), which allows it to correct for inter-layer error accumulation. However,
it requires significantly more computation (hours of GPU time for a 70B model).

#### 8.2.4 Inference Computation

At inference, the dequantization of each weight group requires:

```
For group g with indices (i_1, ..., i_M):
  w_g = sum_{m=1}^{M} C_m[i_m]
```

This is M lookups into codebooks of K entries, followed by M vector additions. The total
cost per group:

```
Memory accesses: M * d * 16 bits  (loading M codewords of d float16 values)
Compute: M * d additions
```

For M = 2, d = 8: 2 * 8 = 16 fp16 loads and 8 additions per group of 8 weights.

Compared to scalar dequantization (1 multiply + 1 add per weight): the AQLM approach
requires 2 loads + 1 add per weight (more memory bandwidth, similar compute). The
codebook lookups introduce irregular memory access patterns (index-dependent addresses),
which reduce cache efficiency.

### 8.3 Complete Algorithm

```
AQLM_Quantize(model, calibration_data, M, K, d):
  For each linear layer with weights W:
    # Step 1: Compute Hessian
    H = 2 * X^T * X / n

    # Step 2: Initialize codebooks via k-means
    For m in 1..M:
      W_groups = reshape(W, [-1, d])  # groups of d weights
      C_m = kmeans(W_groups, K)  # K cluster centers in R^d

    # Step 3: Alternating optimization
    For iter in 1..max_iters:
      # Assignment step
      For each group g:
        residual = W_g
        For m in 1..M:
          i_{g,m} = argmin_j ||residual - C_m[j]||_H^2
          residual -= C_m[i_{g,m}]

      # Codebook update step
      For m in 1..M:
        For j in 0..K-1:
          groups_assigned = {g : i_{g,m} == j}
          residuals = {W_g - sum_{m'!=m} C_{m'}[i_{g,m'}] : g in groups_assigned}
          C_m[j] = weighted_mean(residuals, H_groups_assigned)

    # Step 4 (optional): Fine-tune codebooks via backpropagation
    fine_tune(codebooks, model, calibration_data, steps=1000)

    # Step 5: Store
    Store: codebook entries (M * K * d * fp16), indices (M * ceil(log2(K)) per group)
```

### 8.4 Storage Analysis

For AQLM with M = 2, K = 256, d = 8:

**Per-weight storage**:
- Indices: M * 8 bits / d = 2 * 8 / 8 = 2.0 bits per weight

**Codebook overhead per layer**:
- Codebook: M * K * d * 16 bits = 2 * 256 * 8 * 16 = 65536 bits = 8 KB
- For a layer with d_in * d_out = 8192 * 8192 = 67M weights, the codebook is shared
  across all groups:
  - Codebook overhead per weight: 65536 / 67M ~ 0.001 bits (negligible)

**Total effective bits**: 2.0 + 0.001 ~ 2.0 bits per weight

This is remarkably efficient: AQLM achieves true 2 bits per weight with negligible
overhead. Compare to group quantization at 2 bits with group size 128:

```
Uniform 2-bit + group overhead: 2.0 + 32/128 = 2.25 bits per weight
AQLM 2-bit (M=2, K=256, d=8): 2.0 + 0.001 = 2.001 bits per weight
```

### 8.5 Quality Results

AQLM achieves the best known quality at extremely low bit widths:

```
Llama-2-70B perplexity (wikitext):
  fp16:           5.20
  GPTQ 4-bit:     5.26
  GPTQ 2-bit:     11.4
  QuIP# 2-bit:    5.75
  AQLM 2-bit:     5.58  (best at 2 bits)
```

AQLM at 2 bits outperforms QuIP# at 2 bits (5.58 vs 5.75) because the multi-codebook
vector quantization can represent the weight distribution more efficiently than E8
lattice quantization. However, the gap narrows at higher bit widths.

### 8.6 Strengths

- **Best-in-class at 2-bit**: AQLM achieves the lowest perplexity at 2 bits per weight
  of any published method.
- **Efficient storage**: The codebook overhead is negligible, achieving nearly exactly
  the target bits per weight.
- **Theoretically motivated**: Multi-codebook vector quantization is the information-
  theoretically optimal approach to quantization at fixed rate, and AQLM is a practical
  approximation.
- **Flexible rate control**: By varying M, K, and d, AQLM can target any bit rate with
  fine granularity.

### 8.7 Weaknesses

- **Slow inference**: The codebook lookups introduce irregular memory accesses that are
  hard to optimize. Dequantization throughput is 2-3x slower than scalar dequantization
  at the same bit width.
- **Slow calibration**: The alternating optimization with Hessian-weighted assignments
  is expensive. Optional fine-tuning adds hours of GPU time.
- **Complex kernels**: Efficient GEMM with multi-codebook dequantization requires highly
  specialized CUDA kernels. Standard GEMM libraries do not support this.
- **No mixed precision**: All weight groups use the same number of codebooks and codebook
  size. There is no per-group bit allocation.
- **Codebook generalization**: The codebooks are learned from calibration data and may not
  generalize perfectly to all inputs. This is a stronger assumption than scalar
  quantization, which is input-independent.
- **Very limited hardware support**: Only custom CUDA kernels exist. No Metal, no CPU,
  no standard library support.

### 8.8 Relevance to MXQ

AQLM's approach is largely orthogonal to MXQ's design:

- **Against adoption**: AQLM's multi-codebook vector quantization is fundamentally
  incompatible with the per-block variable-bit-width approach that MXQ uses. The
  codebook lookups are hard to implement efficiently on Metal (irregular memory access
  patterns conflict with Apple Silicon's unified memory bandwidth optimization). The
  inference overhead (2-3x slower dequantization) is unacceptable for MXQ's
  performance targets (<5% overhead vs uniform 4-bit).

- **Theoretical benchmark**: AQLM represents the theoretical frontier of what is
  achievable at 2 bits per weight. MXQ should compare its 2-bit quality against
  AQLM to understand the gap between scalar mixed-precision quantization and
  vector quantization. If MXQ at 2.5 bits matches AQLM at 2 bits, MXQ is
  competitive on the quality-per-bit curve.

- **Codebook idea at higher level**: While per-group codebooks are impractical for MXQ,
  the idea of learning quantization parameters from data (rather than using fixed
  scale/zero) could inspire MXQ's group-level scale optimization. For example, MXQ
  could optimize group scales to minimize Hessian-weighted error rather than using
  min/max scaling.

---

## 9. Comparative Analysis

### 9.1 Quantitative Comparison

The following table summarizes key properties of each method. Results are for
Llama-2-70B on Wikitext-2 perplexity (lower is better).

```
+----------+------+--------+--------+--------+-------+-------+--------+----------+
| Method   | Year | 2-bit  | 3-bit  | 4-bit  | Calib | Infer | Mixed  | Hardware |
|          |      | PPL    | PPL    | PPL    | Speed | Speed | Prec   |          |
+----------+------+--------+--------+--------+-------+-------+--------+----------+
| fp16     |  --  |  --    |  --    | 5.20   |  --   | 1.0x  |  --    | Any      |
| RTN      |  --  | 53+    | 6.80   | 5.34   | <1min | 1.0x  | No     | Any      |
| GPTQ     | 2023 | 11.4   | 5.72   | 5.26   | 1-4hr | 1.0x  | No     | CUDA     |
| AWQ      | 2024 | 15+    | 5.60   | 5.22   | <10m  | 1.0x  | No     | CUDA     |
| SpQR     | 2023 | ~7.0*  | 5.35   | 5.22   | 2-6hr | 0.9x  | Sparse | CUDA     |
| SqzLLM   | 2024 | ~8.0*  | 5.40   | 5.24   | 4-8hr | 0.85x | Sparse | CUDA     |
| EXL2     | 2023 | 6.85   | 5.41   | 5.23   | 2-6hr | 0.92x | Block  | CUDA     |
| QuIP#    | 2024 | 5.75   | 5.30   | 5.21   | 2-4hr | 0.95x | No     | CUDA     |
| AQLM     | 2024 | 5.58   | 5.28   | 5.21   | 8-24h | 0.5x  | No     | CUDA     |
+----------+------+--------+--------+--------+-------+-------+--------+----------+

Notes:
- PPL = perplexity on Wikitext-2 (lower is better)
- Calib Speed = wall-clock calibration time for 70B model on 1x A100
- Infer Speed = relative to fp16 throughput (1.0x = same speed, higher is better)
  (quantized models are typically faster than fp16 due to reduced memory bandwidth)
- Mixed Prec = type of mixed-precision support
- * = estimated from paper trends (not all methods report all bit widths)
- RTN = Round-To-Nearest (naive baseline)
```

### 9.2 Quality vs Compression Frontier

Plotting quality (perplexity) against compression (bits per weight), the methods form
a Pareto frontier:

```
PPL
 ^
12 |  x GPTQ-2
   |
10 |
   |
 8 |  x SqzLLM-2
   |    x SpQR-2
 7 |       x EXL2-2
   |
 6 |         x QuIP#-2
   |           x AQLM-2
   |              x AWQ-3
   |               xxx GPTQ-3, EXL2-3, SpQR-3, QuIP#-3
 5 |                   xxx GPTQ-4, AWQ-4, EXL2-4, QuIP#-4, AQLM-4  ~~ fp16
   +----+----+----+----+----+----+----+----> bits/weight
        2.0  2.5  3.0  3.5  4.0  4.5  5.0
```

The frontier at each bit width:
- **At 2 bits**: AQLM > QuIP# > EXL2 >> SpQR >> GPTQ (AQLM wins by using vector QZ)
- **At 3 bits**: QuIP# ~ AQLM ~ SpQR > EXL2 > AWQ > GPTQ (methods converge)
- **At 4 bits**: All methods within ~0.05 PPL of each other (diminishing returns)

Key observation: **the quality gap between methods shrinks as bit width increases**. At
4 bits, even naive RTN is acceptable. The methods differentiate themselves primarily at
2-3 bits, which is exactly the regime MXQ targets.

### 9.3 Calibration Speed vs Quality Trade-off

```
Quality (lower PPL = better)
 ^
 |  AQLM    (best quality, slowest)
 |  QuIP#
 |  SpQR
 |  GPTQ    EXL2
 |  SqzLLM
 |  AWQ     (good quality, fastest)
 |  RTN     (worst quality, instant)
 +-----------------------------------> Calibration Speed (fast to slow)
```

AWQ offers the best quality-per-calibration-hour: it is 10-100x faster than GPTQ/EXL2
while achieving competitive quality at 4 bits. However, AWQ's advantage disappears at
lower bit widths where its lack of Hessian-based compensation becomes a liability.

### 9.4 Method Decomposition

Each method can be understood as a combination of orthogonal techniques:

```
+----------+----------+----------+----------+----------+----------+
| Method   | Import.  | Optimal  | Mixed    | Non-     | Rotation |
|          | Metric   | Rounding | Prec     | Uniform  |          |
+----------+----------+----------+----------+----------+----------+
| RTN      | None     | No       | No       | No       | No       |
| GPTQ     | Hessian  | Yes      | No       | No       | No       |
| AWQ      | Activ.   | No       | No       | No       | No       |
| SpQR     | w^2*H    | Yes      | Sparse   | No       | No       |
| SqzLLM   | Fisher   | No       | Sparse   | Yes(LUT) | No       |
| EXL2     | Hessian  | Yes      | Block    | No       | No       |
| QuIP#    | Hessian  | Yes(E8)  | No       | Yes(E8)  | Yes(Had) |
| AQLM     | Hessian  | Yes(VQ)  | No       | Yes(VQ)  | No       |
+----------+----------+----------+----------+----------+----------+
```

No existing method combines all techniques. The theoretical ideal would combine:
- Activation-aware importance (AWQ) for fast importance scoring
- Hessian-based optimal rounding (GPTQ) for compensation
- Block-level mixed precision (EXL2) for bit allocation
- Incoherence processing (QuIP#) for uniform weight distribution
- Possibly sparse outlier protection (SpQR) for extreme outliers

MXQ aims to combine the first three. The fourth (rotation) is an open question requiring
empirical investigation.

### 9.5 Hardware Compatibility

```
                   CUDA   Metal   CPU    Custom Kernels Required
+----------+------+-------+------+------+
| RTN      |  Y   |   Y   |  Y   | No   |
| GPTQ     |  Y   |   N*  |  N*  | Yes  |
| AWQ      |  Y   |   N   |  N   | Yes  |
| SpQR     |  Y   |   N   |  N   | Yes  |
| SqzLLM   |  Y   |   N   |  N   | Yes  |
| EXL2     |  Y   |   N   |  N   | Yes  |
| QuIP#    |  Y   |   N   |  N   | Yes  |
| AQLM     |  Y   |   N   |  N   | Yes  |
+----------+------+-------+------+------+

* GPTQ format can be loaded by some CPU/Metal engines (llama.cpp, MLX)
  but through format conversion, not native GPTQ kernels.
```

Every advanced quantization method requires custom GPU kernels. Apple Silicon has ZERO
native support for any of these methods. This is MXQ's competitive opportunity: be the
first high-quality importance-aware quantization method with native Metal support.

### 9.6 Theoretical Limits

Rate-distortion theory provides a lower bound on the quantization error achievable at
a given bit rate. For a Gaussian source with variance sigma^2 at rate R bits per sample:

```
D(R) = sigma^2 * 2^{-2R}  (distortion-rate function)
```

This means each additional bit reduces the MSE by a factor of 4 (6 dB). For LLM weights
(which are approximately Gaussian within each group):

```
Approximate perplexity degradation vs bit width (very rough):
  4-bit: ~0.05 PPL above fp16  (excellent)
  3-bit: ~0.3 PPL above fp16   (good)
  2-bit: ~1.5 PPL above fp16   (noticeable)
  1-bit: ~10+ PPL above fp16   (severe)
```

The best current methods (AQLM, QuIP#) at 2 bits achieve ~0.4-0.55 PPL above fp16,
which is within striking distance of the 3-bit uniform baseline. This suggests that
2-bit quantization with mixed precision (averaging 2.5 bits) could potentially match
uniform 4-bit quality, validating MXQ's core thesis.

---

## 10. Implications for MXQ

### 10.1 What MXQ Should Adopt from Each Method

| Method   | Adopt                                                  | Skip                               |
|----------|--------------------------------------------------------|-------------------------------------|
| AWQ      | Activation-norm importance metric; fast grid search    | Per-channel scaling (too coarse)    |
| GPTQ     | Hessian-based optimal rounding within each block       | Uniform bit width constraint        |
| EXL2     | Three-stage pipeline; greedy bit allocation            | CUDA-only kernels                   |
| SpQR     | Sensitivity metric w^2 * H_{jj}; outlier concept      | Sparse storage format               |
| SqueezeLLM| Sensitivity-weighted optimization objective           | LUT-based non-uniform quantization  |
| QuIP#    | Investigate Hadamard rotation as preprocessing         | E8 lattice codebooks                |
| AQLM     | Use as quality benchmark at 2-bit                      | Multi-codebook vector quantization  |

### 10.2 MXQ's Proposed Architecture (Informed by This Survey)

Based on the analysis above, MXQ should implement:

**Phase 1 -- Calibration & Importance Scoring**:
1. Forward pass calibration (like AWQ) to compute per-channel activation norms
2. Hessian diagonal computation (like GPTQ/SpQR) for per-weight sensitivity
3. Combined importance score per block: max(w_{ij}^2 * H_{jj}) within each block, weighted
   by activation norms
4. Optional: Hadamard rotation of weights before scoring (investigate empirically)

**Phase 2 -- Bit Allocation**:
1. Trial quantization at each candidate bit width per block (like EXL2 Stage 1)
2. Greedy knapsack allocation to hit target average bits (like EXL2 Stage 2)
3. Layer-type priors: attention > MLP, first/last layers protected (MXQ-specific)

**Phase 3 -- Quantization**:
1. GPTQ-style optimal rounding within each block at its allocated bit width
2. Cross-block Hessian compensation (the GPTQ update step applied across blocks of
   different bit widths)
3. Uniform quantization levels (not LUT-based), for Metal kernel efficiency

**Phase 4 -- Metal Kernels**:
1. Variable-bit-width dequantization (like EXL2's CUDA kernels, but for Metal)
2. Fused dequant + matmul to minimize memory bandwidth usage
3. Optional: fast Hadamard transform kernel if rotation is adopted

### 10.3 Expected Quality

Based on the survey data, MXQ's expected quality at various average bit widths:

```
MXQ-2.0 bpw: Should approach EXL2 at 2.0 (~6.85 PPL), potentially better with
             Hessian compensation + layer priors.

MXQ-2.5 bpw: Should be between EXL2 at 2.5 (~5.62) and QuIP# at 2 (~5.75).
             Target: < 5.60 PPL. This would validate the claim "matches 4-bit uniform."

MXQ-3.0 bpw: Should be competitive with EXL2 at 3.0 (~5.41) and SpQR at 3 (~5.35).
             Target: < 5.40 PPL.

MXQ-4.0 bpw: All methods converge here. Target: < 5.23 PPL (match AWQ 4-bit).
```

The critical test is MXQ-2.5: if it achieves PPL < 5.42 (matching GPTQ 4-bit uniform),
the core value proposition is proven.

### 10.4 Open Research Questions

1. **Rotation vs mixed precision**: Does Hadamard rotation help or hurt when combined
   with mixed-precision bit allocation? Rotation makes weights more uniform, which
   reduces the benefit of mixed precision. The answer determines whether MXQ should
   include a rotation step.

2. **Block size**: Smaller blocks (32) enable finer-grained bit allocation but increase
   metadata overhead. Larger blocks (128) reduce overhead but miss within-block
   importance variations. What is the optimal block size for Apple Silicon's memory
   access patterns?

3. **Layer-type priors**: How much do priors (attention > MLP, protect first/last layers)
   help vs. pure data-driven allocation? Strong priors reduce the risk of catastrophic
   allocation errors but may over-allocate bits to layers that do not need them.

4. **Calibration data sensitivity**: How much does the choice of calibration data affect
   MXQ quality? AWQ is robust (only needs activation norms); GPTQ/EXL2 are more
   sensitive (use full Hessian). MXQ uses both, so its sensitivity is unclear.

5. **Interaction with KV cache quantization**: If the KV cache is also quantized (as in
   vMLX), does the weight quantization strategy need to account for KV cache errors?
   No existing method considers this interaction.

---

## Appendix A: Glossary

- **bpw**: Bits per weight. The average number of bits used to store each weight parameter.
- **Calibration data**: A small dataset (typically 128-1024 sequences) used to compute
  importance/sensitivity statistics. Not used for training.
- **Coherence**: A measure of how "spiky" a matrix is. High coherence = a few dominant
  columns; low coherence = uniform column norms.
- **Dequantization**: Converting quantized integer values back to floating-point for
  computation.
- **Group quantization**: Dividing weights into groups that share scale/zero parameters.
- **Hessian**: The second-derivative matrix of the loss with respect to weights. Captures
  per-weight sensitivity.
- **Imatrix**: Importance matrix. Per-weight or per-block importance scores derived from
  calibration.
- **Perplexity (PPL)**: exp(cross-entropy loss). Lower = better. The standard quality
  metric for language models. A 1.0 increase in PPL is noticeable; a 0.1 increase is
  marginal.
- **Rate-distortion**: The information-theoretic trade-off between compression rate (bits)
  and reconstruction error (distortion).
- **RTN**: Round-To-Nearest. Naive quantization without any error compensation.
- **Salient weights**: Weights that disproportionately affect model quality. Identified by
  activation magnitude (AWQ), Hessian diagonal (GPTQ/SpQR), or Fisher information
  (SqueezeLLM).

## Appendix B: Key Papers

1. Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and
   Acceleration." MLSys 2024. arXiv:2306.00978
2. Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained
   Transformers." ICLR 2023. arXiv:2210.17323
3. Frantar & Alistarh. "Optimal Brain Compression: A Framework for Accurate Post-Training
   Quantization and Pruning." NeurIPS 2022. arXiv:2208.11580
4. Dettmers et al. "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight
   Compression." ICML 2023. arXiv:2306.03078
5. Kim et al. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024. arXiv:2306.07629
6. Tseng et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and
   Lattice Codebooks." ICML 2024. arXiv:2402.04396
7. Egiazarian et al. "AQLM: Extreme Compression of Large Language Models via Additive
   Quantization." ICML 2024. arXiv:2401.06118
8. Hassibi & Stork. "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon."
   NeurIPS 1993.
9. turboderp. ExLlamaV2. https://github.com/turboderp/exllamav2
