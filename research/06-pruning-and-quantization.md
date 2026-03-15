# Pruning + Quantization: Combined Model Compression

> MXQ Research Document 06 -- Complete technical reference for implementing joint pruning and quantization in the MXQ format for Apple Silicon.

---

## Table of Contents

1. [Pruning Fundamentals](#1-pruning-fundamentals)
2. [Pruning Criteria -- Which Weights to Remove](#2-pruning-criteria--which-weights-to-remove)
3. [Why Pruning + Quantization is Better Than Either Alone](#3-why-pruning--quantization-is-better-than-either-alone)
4. [State-of-the-Art Combined Methods](#4-state-of-the-art-combined-methods)
5. [Implementing Pruning + Quantization for MXQ](#5-implementing-pruning--quantization-for-mxq)
6. [Practical Sparsity Levels for LLMs](#6-practical-sparsity-levels-for-llms)
7. [Hardware Considerations for Sparse+Quantized Inference](#7-hardware-considerations-for-sparsequantized-inference)

---

## 1. Pruning Fundamentals

### 1.1 What is Pruning

Pruning is the process of setting neural network weights to exactly zero. Unlike quantization, which reduces the precision of every weight, pruning eliminates weights entirely. The pruned weight contributes nothing to the computation -- its output is always zero regardless of the input activation.

Formally, given a weight matrix W of shape (d_out, d_in), pruning constructs a binary mask M of the same shape:

```
M_ij = 0  if weight W_ij is pruned
M_ij = 1  if weight W_ij is retained

W_pruned = W * M   (element-wise multiplication)
```

The sparsity ratio s is the fraction of weights that are zero:

```
s = (number of zeros in M) / (total elements in M)
s = 1 - (nnz(M) / numel(M))
```

A model with 50% sparsity has half its weights set to zero. The key question is always: which half?

### 1.2 Unstructured Pruning

In unstructured pruning, any individual weight in the matrix can independently be set to zero. There are no constraints on which weights are pruned -- the binary mask M can have zeros in any position. This provides maximum flexibility in choosing which weights to remove, and therefore achieves the best quality at any given sparsity level.

The cost of this flexibility is that the resulting sparse matrix has an irregular pattern of nonzeros. Standard dense matrix multiplication cannot skip the zero entries without explicit sparse matrix support.

#### 1.2.1 Sparse Matrix Representations

To actually benefit from sparsity (reduced memory and/or computation), the sparse matrix must be stored in a compressed format that avoids storing zeros explicitly. The major formats are:

**CSR (Compressed Sparse Row)**

CSR stores a matrix using three arrays:
- `values[]`: the nonzero values, in row-major order
- `col_indices[]`: the column index of each nonzero value
- `row_pointers[]`: for each row, the index into `values[]` where that row starts

For a matrix with nnz nonzero entries and m rows:
- Storage: nnz * (sizeof(value) + sizeof(int)) + (m + 1) * sizeof(int)
- Row access: O(1) to find row boundaries, then iterate nonzeros
- SpMV (sparse matrix-vector multiply): efficient because rows are contiguous
- SpMM (sparse matrix-matrix multiply): moderate efficiency

Example for a 4x4 matrix with 6 nonzeros:
```
Dense:          CSR:
[1 0 2 0]      values     = [1, 2, 3, 4, 5, 6]
[0 3 0 0]      col_indices = [0, 2, 1, 0, 3, 2]
[4 0 0 5]      row_pointers = [0, 2, 3, 5, 6]
[0 0 6 0]
```

**CSC (Compressed Sparse Column)**

CSC is the transpose of CSR -- it stores column pointers instead of row pointers. CSC is efficient for column-wise access and is preferred when the right-hand operand in SpMM is sparse.

- `values[]`: nonzero values in column-major order
- `row_indices[]`: row index of each nonzero
- `col_pointers[]`: index into values where each column starts

**Bitmap (Bitmask) Format**

A simple format where a bit array indicates which positions are nonzero:
- `bitmap[]`: 1 bit per weight position (1 = nonzero, 0 = zero)
- `values[]`: packed nonzero values only

Storage: numel / 8 bytes for bitmap + nnz * sizeof(value) bytes for values.

At 50% sparsity, bitmap overhead is 1 bit per weight position. For a weight matrix with 4096 x 4096 = 16M entries, the bitmap is 2 MB. The values array stores only the 8M nonzero entries.

Bitmap is particularly attractive for GPU implementations because:
- Constant-time lookup of whether any position is zero
- Bit-parallel operations can process 32 or 64 positions simultaneously
- No indirection -- the bitmap index directly maps to the dense position
- Well-suited for block-level masking (one bit per block)

**Block-Sparse (BSR/BCSR)**

Block-sparse format divides the matrix into fixed-size blocks (e.g., 32x32, 64x64, or 1x32 row-blocks) and stores only the nonzero blocks:

- `block_indices[]`: which blocks are nonzero
- `block_values[]`: the dense data for each nonzero block

This is the most hardware-friendly sparse format because:
- Each nonzero block is a small dense matrix -- standard GEMM works on it
- Memory access is coalesced within blocks
- No branch divergence within a block
- The block index is small (number of blocks << number of weights)

The tradeoff: block-sparse can only represent sparsity at the block granularity. If a block has even one important weight, the entire block must be retained. This means block-sparse typically achieves lower effective sparsity than element-wise unstructured pruning for the same quality target.

For LLM weight matrices with typical dimensions of 4096x4096 or larger, row-blocks of 32 or 64 elements are a natural fit. A "block" in this context means a contiguous group of weights within a single row (matching the quantization group size in MXQ).

#### 1.2.2 Hardware Support for Sparse Computation

**Apple Silicon: Sparse MatMul in Metal and Accelerate**

Apple's Accelerate framework provides the Sparse Solvers library (introduced at WWDC 2017) which supports sparse linear algebra operations. Key capabilities:

- `Sparse BLAS`: Sparse matrix-vector (SpMV) and sparse matrix-dense matrix (SpMM) multiplication via the standard BLAS-like interface
- Supported sparse formats: CSR, CSC, and COO (coordinate list)
- `BNNSNDArrayFullyConnectedSparsifySparseCOO()`: BNNS function that converts COO sparse tensors into an internal device-optimized sparse layout for use in fully connected (linear) layers
- BNNS fully connected layers support sparse weight matrices through this path

However, Apple's sparse support has significant limitations for LLM inference:

1. **No dedicated sparse hardware**: Unlike NVIDIA's Sparse Tensor Cores (Ampere A100 and later), Apple Silicon has no dedicated hardware unit that accelerates structured sparse computation at the matrix multiply level. The AMX (Apple Matrix coprocessor) units in Apple Silicon are designed for dense matrix multiplication.

2. **CPU-side sparse solvers**: The Accelerate Sparse Solvers library runs primarily on CPU (AMX), not GPU (Metal). This is suitable for scientific computing but not for LLM inference where GPU throughput matters.

3. **Metal Performance Shaders**: MPS provides `MPSMatrixMultiplication` and related operations, but these operate on dense matrices. There is no `MPSSparseMatrixMultiplication` equivalent in the public API as of 2025.

4. **MLX sparse support**: As of MLX 0.31 (early 2026), MLX does not expose a first-class sparse tensor type or sparse GEMM operation. Sparse computation in MLX must be implemented through custom Metal kernels or by using masking and dense operations.

**Practical implication for MXQ**: On Apple Silicon, the primary benefit of pruning is **memory bandwidth reduction** (fewer weights to load from unified memory), not computational speedup from sparse hardware. The Metal kernel can skip loading zero blocks entirely, achieving bandwidth savings proportional to the sparsity ratio. Actual FLOP savings require either block-sparse approaches (where entire matmul tiles are skipped) or a future Apple Silicon generation with sparse matmul acceleration.

### 1.3 Structured Pruning

Structured pruning removes entire structural components of the network -- complete rows, columns, attention heads, entire layers, or other architecturally meaningful units. The result is a network that is literally smaller: the weight matrices have fewer dimensions, and standard dense operations work at full efficiency on the smaller matrices.

No special sparse hardware or sparse matrix formats are needed. The pruned model is simply a smaller dense model.

#### 1.3.1 Types of Structured Pruning

**Channel Pruning (Row/Column Pruning)**

Remove entire output channels (rows of the weight matrix) or input channels (columns). For a linear layer `y = Wx + b`:

- Removing output channel i: delete row i from W and element i from b. The output dimension decreases by 1.
- Removing input channel j: delete column j from W. The input dimension decreases by 1 (which must be consistent with the previous layer's output dimension).

If we remove r rows from a (d_out, d_in) matrix, the resulting matrix is (d_out - r, d_in). The computation cost drops proportionally:

```
Original FLOPs:   2 * d_out * d_in
Pruned FLOPs:     2 * (d_out - r) * d_in
Speedup:          d_out / (d_out - r)
```

Channel pruning requires coordinating across connected layers -- removing an output channel from layer L means removing the corresponding input channel from layer L+1.

**Head Pruning (Attention Head Removal)**

In multi-head attention with h heads, each head has its own Q, K, V projection matrices of shape (d_model, d_head) and an output projection. Removing an attention head means:

1. Delete the corresponding d_head columns from W_Q, W_K, W_V
2. Delete the corresponding d_head rows from W_O
3. The model now has (h - 1) heads

This is particularly effective because:
- Many attention heads in deep transformers are redundant (studies show 20-40% of heads can be removed with minimal quality loss in models with 32+ heads)
- Each head removal saves 4 * d_model * d_head parameters
- Computation scales linearly with number of heads

**Layer Pruning (Entire Layer Removal)**

Remove entire transformer blocks. For a model with L layers:
- Skip layer i entirely: output of layer (i-1) feeds directly to layer (i+1)
- Requires the input and output dimensions to match (true for transformer blocks with residual connections)
- Each removed layer saves all parameters in that block: 2 attention matrices + 2-3 MLP matrices + layer norms

Layer pruning is aggressive but can be surprisingly effective:
- The middle layers of deep transformers are often the most redundant
- A 32-layer model can sometimes lose 4-8 middle layers with modest quality degradation
- NVIDIA's Minitron approach uses layer pruning followed by knowledge distillation to produce efficient smaller models

**Block Pruning (Sub-matrix Blocks)**

Remove fixed-size sub-matrices (blocks) from weight matrices. For example, divide a (4096, 4096) weight matrix into 32x32 blocks (16384 blocks total), and zero out entire blocks based on importance. This is a middle ground:
- More flexible than row/column pruning (can remove parts of channels)
- More hardware-friendly than element-wise unstructured pruning (blocks are dense tiles)
- Naturally aligns with GPU tiling strategies and quantization group sizes

### 1.4 Semi-Structured Pruning (N:M Sparsity)

Semi-structured sparsity constrains the pruning pattern: in every contiguous group of M weights, exactly N must be zero. The canonical example is 2:4 sparsity -- in every group of 4 consecutive weights, exactly 2 are zero, yielding 50% sparsity.

```
2:4 sparsity example (within a row):
Original weights: [0.5, -0.3, 0.8, -0.1, 0.2, 0.7, -0.4, 0.9]
2:4 pruned:        [0.5,  0,   0.8,  0,   0,   0.7,  0,   0.9]
                    ^^^^kept    ^^^^kept    ^^^^kept    ^^^^kept
                    (2 of 4)    (2 of 4)    (2 of 4)    (2 of 4)
```

The regularity of the pattern enables efficient hardware support:

**Compressed Storage**: Each group of 4 weights stores:
- 2 nonzero values (using the original data type)
- A 2-bit index indicating which 2 of the 4 positions are nonzero (there are C(4,2) = 6 possible patterns, fitting in 4 bits, but only 2 bits of index per value are needed)

The metadata overhead is small: 2 bits per original weight position for the index, which is 12.5% for 16-bit values and 25% for 8-bit values.

**Total storage for 2:4 at FP16**: 2 values * 16 bits + 4 bits metadata = 36 bits per 4 original values = 9 bits/value (vs 16 bits/value dense). This is a 44% storage reduction.

#### 1.4.1 NVIDIA Ampere Hardware Support for 2:4

NVIDIA's Ampere architecture (A100, 2020) introduced Sparse Tensor Cores with native hardware support for 2:4 structured sparsity:

- The Sparse Tensor Core loads only the nonzero values from the compressed representation
- Uses the metadata to index into the corresponding operand, pulling only the needed values
- Achieves up to 2x throughput compared to the equivalent dense Tensor Core operation
- Supported data types: FP16, BF16, INT8, INT4 (TF32 on A100)
- Available via cuSPARSELt library and PyTorch's `torch.sparse.to_sparse_semi_structured()`

Performance in practice:
- 2:4 sparse Llama-3.1-8B achieves up to 30% higher throughput and 20% lower latency vs dense, when combined with vLLM serving
- Combined with INT4 quantization via Sparse-Marlin kernels, speedups range from 1.2x to 3.0x depending on batch size and hardware

Other N:M patterns supported: 4:8 sparsity (also 50% sparse, but coarser grouping). NVIDIA's hardware accelerates 2:4 natively; 4:8 requires software handling.

#### 1.4.2 Apple Silicon and Semi-Structured Sparsity

As of early 2026, Apple Silicon does **not** have hardware-accelerated N:M structured sparsity support:

- The AMX (Apple Matrix coprocessor) units perform dense matrix multiplication. There is no sparse mode or sparse metadata path in the publicly documented AMX instruction set.
- The M5 GPU's Neural Accelerator (introduced 2025) is optimized for dense transformer operations (attention, GEMM, convolution) -- no sparse acceleration has been disclosed.
- Metal compute shaders can implement N:M sparsity in software, but without hardware support the benefit is limited to memory bandwidth savings, not computational throughput gains.

**BNNS sparse path**: Apple's BNNS (Basic Neural Network Subroutines) in the Accelerate framework does include `BNNSNDArrayFullyConnectedSparsifySparseCOO()`, which converts COO-format sparse weights into an internal layout optimized for BNNS fully connected layers. This suggests Apple has some internal sparse-aware execution path for neural network layers, but:
- It operates on CPU (AMX), not GPU
- The performance characteristics are not publicly documented
- It is unclear whether this provides actual computational speedup or just memory savings
- It is not accessible from Metal or MLX

**Implication for MXQ**: On Apple Silicon, the value of N:M sparsity is weaker than on NVIDIA hardware. Without dedicated sparse hardware, the overhead of managing the sparse metadata and performing indexed lookups may negate the benefit of reduced computation. The primary sparsity benefit on Apple Silicon remains memory bandwidth reduction through block-level skipping in the Metal kernel.

---

## 2. Pruning Criteria -- Which Weights to Remove

The quality of a pruned model depends critically on which weights are selected for removal. Different criteria lead to dramatically different results at the same sparsity level. This section covers the major approaches, from simplest to most sophisticated.

### 2.1 Magnitude Pruning

The simplest criterion: remove weights with the smallest absolute value.

```
Score(w_ij) = |w_ij|
Prune weights with lowest scores until desired sparsity is reached.
```

Intuition: small weights contribute less to the output. If |w_ij| is near zero, the product w_ij * x_j is small regardless of the input activation x_j, so removing it has little effect.

**Implementation**: Sort all weights by |w|, set the bottom s% to zero.

**Advantages**:
- Trivially simple to implement
- No calibration data needed
- Fast: O(n log n) for sorting, where n = total number of weights
- Surprisingly effective up to moderate sparsity levels (40-50%)

**Disadvantages**:
- Ignores activation magnitudes: a small weight on a huge activation channel may be more important than a large weight on a tiny activation channel
- Ignores weight interactions: removing weight w_ij may be harmless alone but catastrophic if w_ik (which partially compensates for w_ij) is also removed
- No error compensation: after pruning, the remaining weights are not adjusted
- Quality degrades rapidly beyond 50-60% sparsity

**Per-layer vs global thresholds**: Magnitude pruning can be applied with a global threshold (same |w| cutoff across all layers) or per-layer (each layer has its own threshold maintaining the same sparsity ratio). Per-layer is generally better because weight magnitude distributions vary significantly across layers.

### 2.2 Gradient-Based Pruning

Uses the product of weight magnitude and gradient magnitude as the importance score:

```
Score(w_ij) = |w_ij * g_ij|

where g_ij = dL/dw_ij is the gradient of the loss with respect to weight w_ij
```

Intuition: the gradient tells us how sensitive the loss is to changes in this weight. A weight with large |w * g| is both large and has a large impact on the loss -- removing it would cause the loss to increase significantly.

This comes from a first-order Taylor expansion of the loss change when weight w_ij is set to zero:

```
DeltaL approximately = -w_ij * g_ij + O(w_ij^2)
```

Weights where |w * g| is small can be safely removed because zeroing them causes only a small loss increase.

**Advantages over magnitude pruning**:
- Accounts for the loss landscape, not just weight magnitude
- Better at identifying weights that are numerically large but functionally unimportant

**Disadvantages**:
- Requires a backward pass to compute gradients (needs calibration data + labels, or a self-supervised loss)
- Gradients can be noisy -- need to average over multiple calibration samples
- First-order approximation may be inaccurate for large perturbations (setting a weight to zero is not a small perturbation)

### 2.3 Hessian-Based Pruning (OBS/OBD)

The gold standard of pruning criteria, using second-order information (the Hessian matrix) to predict the exact effect of removing each weight.

#### 2.3.1 Optimal Brain Damage (OBD) -- LeCun et al., 1990

OBD uses a diagonal approximation of the Hessian to estimate the increase in loss from removing each weight:

```
DeltaL_i approximately = (1/2) * w_i^2 * H_ii

where H_ii = d^2L / dw_i^2 is the i-th diagonal entry of the Hessian
```

Weights with small DeltaL_i are safe to prune. The key insight: it's not just the weight magnitude that matters, but the curvature of the loss surface in that direction. A large weight in a flat direction (small H_ii) can be safely removed, while a small weight in a steep direction (large H_ii) should be kept.

**Limitation**: OBD assumes the Hessian is diagonal, which is rarely true. Off-diagonal entries capture weight interactions, and ignoring them leads to suboptimal pruning decisions. In practice, Hessians for neural networks are strongly non-diagonal.

#### 2.3.2 Optimal Brain Surgeon (OBS) -- Hassibi & Stork, 1993

OBS uses the full inverse Hessian, correctly accounting for weight interactions:

```
Saliency of weight q:
  DeltaL_q = w_q^2 / (2 * [H^{-1}]_qq)

After pruning weight q, optimal adjustment to remaining weights:
  delta_w = -w_q / [H^{-1}]_qq * H^{-1} * e_q

where e_q is the unit vector in direction q
```

The critical advance over OBD: after identifying which weight to prune, OBS also computes the optimal adjustment to all remaining weights to compensate for the pruning error. This weight update is exact (to second order) and requires no retraining.

**The full algorithm**:
1. Compute the inverse Hessian H^{-1} (or update it incrementally)
2. For each weight q, compute saliency: s_q = w_q^2 / (2 * [H^{-1}]_qq)
3. Prune the weight with smallest s_q
4. Update remaining weights: w = w - (w_q / [H^{-1}]_qq) * column_q(H^{-1})
5. Update H^{-1} using the rank-1 update formula (matrix inversion lemma)
6. Repeat from step 2

**Computational cost**: O(d^2) per pruned weight for the weight update and Hessian update, where d is the number of weights. For a full model this is prohibitive, which is why modern methods (SparseGPT, GPTQ) apply OBS-like reasoning at the layer level with carefully optimized implementations.

### 2.4 Wanda (Weights AND Activations)

Wanda (Sun et al., 2023, ICLR 2024) combines weight magnitude with input activation norms:

```
Score(w_ij) = |w_ij| * ||X_j||_2

where:
  w_ij = weight at position (i, j) in the weight matrix
  X_j  = the j-th input feature across all calibration samples
  ||X_j||_2 = L2 norm of the j-th input channel across calibration data
```

Intuition: a weight matters if it is both large AND processes large activations. This is the same insight as AWQ (Activation-Aware Quantization), but applied to pruning instead of quantization:
- AWQ: protect important channels from quantization error by scaling
- Wanda: remove unimportant channels by pruning

**Key properties**:
- **One-shot**: no iterative optimization, no weight updates. Compute the metric, apply the mask, done.
- **Per-output pruning**: for each output neuron (row of W), independently select which input connections to prune. This is important because different output neurons may rely on different input channels.
- **Extremely fast**: computing ||X_j|| requires a single forward pass through calibration data. The pruning decision for each weight is a single multiplication and comparison. In practice, Wanda is ~300x faster than SparseGPT.
- **No retraining**: the pruned model is used as-is.

**Performance**: Wanda at 50% unstructured sparsity on LLaMA-65B and LLaMA-2-70B matches the zero-shot accuracy of the dense baseline. At 50% sparsity, Wanda performs comparably to the much more expensive SparseGPT, though SparseGPT pulls ahead at higher sparsity levels (60%+) where weight compensation becomes important.

**Connection to MXQ**: Wanda's importance metric |w| * ||x|| is essentially the same signal MXQ already uses for bit allocation (the AWQ-style activation-aware scoring). This means MXQ's existing calibration infrastructure can produce both:
1. Bit allocation decisions (how many bits per block) for quantization
2. Pruning decisions (which blocks to zero out) for sparsity

A unified importance score serves dual purposes.

### 2.5 SparseGPT

SparseGPT (Frantar & Alistarh, 2023, ICML 2023) is the state-of-the-art one-shot pruning method for large language models, directly applying OBS-like Hessian reasoning at massive scale.

**Core algorithm**:

SparseGPT solves the layer-wise pruning problem: for each linear layer with weight matrix W and calibration input X, find a sparse matrix W_hat that minimizes reconstruction error:

```
minimize ||WX - W_hat X||_F^2
subject to: W_hat has at most (1-s) * d_in nonzeros per row
```

This is exactly the same formulation as GPTQ uses for quantization, but the constraint is sparsity instead of quantization. The algorithm:

1. **Compute the Hessian**: H = 2 * X * X^T (the empirical Fisher / squared input correlations). This is the same Hessian used by GPTQ.

2. **Process columns left-to-right**: For each column j of W:
   a. Compute the pruning saliency for each row: s_ij = w_ij^2 / [H^{-1}]_jj
   b. For rows where weight j should be pruned (based on saliency ranking), set w_ij = 0
   c. For all remaining (unpruned) weights in columns j+1 to d_in, apply the OBS weight update to compensate for the pruning error:
      ```
      delta_w_i,j+1:d_in = -w_ij / [H^{-1}]_jj * [H^{-1}]_j,j+1:d_in
      ```
   d. Update the Hessian inverse using the Cholesky-based update from GPTQ

3. **Result**: A sparse weight matrix where remaining weights have been optimally adjusted to compensate for pruned weights.

**Key performance results** (from the paper):
- OPT-175B at 50% unstructured sparsity: perplexity increase of only 0.13 (from 8.34 to 8.47 on WikiText2)
- OPT-175B at 60% unstructured sparsity: perplexity increase of ~0.7
- Runs on a single A100 GPU in ~4.5 hours for 175B parameter models
- Compatible with 2:4 and 4:8 semi-structured sparsity patterns

**Connection to GPTQ**: SparseGPT and GPTQ share the same core Hessian computation and Cholesky-based column processing. They differ only in the perturbation applied to each column:
- GPTQ: quantize the weight (round to nearest quantization level) and compensate remaining weights
- SparseGPT: prune the weight (set to zero) and compensate remaining weights

This shared foundation means they can be combined in a single pass: for each column, first decide whether to prune (set to zero) or keep, then if kept, quantize to the target bit width. The compensation step handles both perturbation sources simultaneously.

---

## 3. Why Pruning + Quantization is Better Than Either Alone

### 3.1 The Redundancy Argument

Transformer weight matrices contain two fundamentally different types of redundancy:

1. **Weights that are unimportant** -- they contribute negligibly to the model's output regardless of precision. These should be **pruned** (set to exactly zero, not stored at all).

2. **Weights that are important but over-specified** -- they matter for model quality, but their exact value does not need 16-bit precision. These should be **quantized** (stored at reduced precision: 2, 3, 4, or more bits).

Pruning and quantization each address one type of redundancy. Applied alone, each technique must compromise:
- Pruning alone: retains all remaining weights at full precision (wasteful for weights that could be quantized)
- Quantization alone: quantizes every weight including ones that contribute nothing (wasteful bits on near-zero weights that could simply be pruned)

Combining them lets each technique handle the redundancy it is best suited for.

### 3.2 Compression Stacking

The compression ratios of pruning and quantization multiply:

```
Bits per original weight = (1 - sparsity) * bits_per_remaining_weight

Total model size = num_weights * (1 - sparsity) * bits_per_remaining_weight / 8 bytes
```

**Example calculations for a 70B parameter model (originally 70B * 16 bits = 140 GB)**:

| Technique | Sparsity | Bits/weight | Effective bits/weight | Model size |
|-----------|----------|-------------|----------------------|------------|
| Dense FP16 | 0% | 16 | 16.0 | 140 GB |
| Quantize only (4-bit) | 0% | 4 | 4.0 | 35 GB |
| Quantize only (2-bit) | 0% | 2 | 2.0 | 17.5 GB |
| Prune only (50%) | 50% | 16 | 8.0 | 70 GB |
| **Prune 50% + Q4** | **50%** | **4** | **2.0** | **17.5 GB** |
| **Prune 50% + Q3** | **50%** | **3** | **1.5** | **13.1 GB** |
| **Prune 50% + Q2** | **50%** | **2** | **1.0** | **8.75 GB** |
| Prune 30% + Q2.5 (MXQ) | 30% | 2.5 | 1.75 | 15.3 GB |

The combined approach achieves compression levels impossible with either technique alone. 50% pruning + 4-bit quantization achieves an effective 2 bits per weight -- the same storage as 2-bit quantization alone, but with dramatically better quality because the 2-bit quantization of useless weights is replaced by simply not storing them.

### 3.3 Quality Stacking

Beyond compression arithmetic, the quality benefits also stack favorably:

**Pruning budget goes to truly useless weights**: Pruning criteria (Wanda, SparseGPT) identify weights that contribute the least to model output. Removing these weights causes negligible quality loss. The pruning "budget" is spent on weights that don't matter.

**Quantization budget goes to precision allocation of remaining weights**: After pruning, the quantization step allocates bits to the remaining (important) weights. With fewer weights to quantize, each remaining weight can receive more attention in the bit allocation -- or the same total bit budget achieves lower effective bits per original weight.

**Each technique handles a different error mode**:
- Pruning error: output perturbation from removing entire weights (step function -- the weight goes from w to 0)
- Quantization error: output perturbation from rounding weights to discrete levels (bounded -- at most half a quantization step)
- These errors are approximately independent (they affect different weights), so they add in quadrature rather than linearly:
  ```
  Total_error approximately = sqrt(pruning_error^2 + quantization_error^2)
  ```
  This is less than pruning_error + quantization_error.

### 3.4 The Mathematics of Joint Optimization

Let W be a weight matrix of shape (d_out, d_in). We want to find:
- A pruning mask M in {0, 1}^{d_out x d_in} (which weights to keep)
- A bit allocation B in {2, 3, 4, 6, 8}^{d_out x d_in} (how many bits per kept weight)
- Quantized values Q (the quantized weight values)

The objective is to minimize reconstruction error subject to a total compression budget:

```
minimize  ||WX - (M * Q)X||_F^2

subject to:
  sum(M_ij * B_ij) <= C     (total bit budget constraint)
  Q_ij in Levels(B_ij)      (Q must be valid for its bit width)
  M_ij in {0, 1}            (binary pruning mask)
```

where C is the total allowed bits (derived from the target average bit width and sparsity).

This is a joint discrete optimization problem -- harder than either pruning or quantization alone because:
- The pruning mask M and bit allocation B interact: if a weight is pruned (M_ij = 0), its bit allocation B_ij is irrelevant (freed budget). If a weight is kept (M_ij = 1), it needs B_ij bits.
- The optimal quantization Q depends on M (after pruning, the remaining weights should be re-quantized to compensate)
- The problem has both discrete (M, B) and continuous (Q) variables

**Practical decomposition**: The full joint optimization is NP-hard. In practice, we decompose it into sequential steps with error compensation:

1. **Score all weights** using a unified importance metric (e.g., Wanda-style |w| * ||x|| or Hessian-based saliency)
2. **Prune** the lowest-scoring weights (set M_ij = 0)
3. **Compensate** remaining weights for pruning error (SparseGPT-style Hessian update)
4. **Re-score** remaining weights for bit allocation (importance may have changed after pruning + compensation)
5. **Allocate bits** to remaining weights based on re-scored importance
6. **Quantize** remaining weights at their allocated bit width with error compensation (GPTQ-style)

The key insight: step 3 (compensate after pruning) and step 4 (re-score) are critical. Skipping them -- simply pruning then quantizing without re-calibration -- produces worse results because:
- Pruning changes the weight distribution (the surviving weights may need different bit allocations)
- The compensation step in SparseGPT adjusts remaining weight values, which changes their optimal quantization

### 3.5 Empirical Evidence

From SparseGPT experiments on OPT-175B (WikiText2 perplexity):

| Configuration | Effective bits/param | Perplexity |
|--------------|---------------------|------------|
| Dense FP16 | 16.0 | 8.34 |
| GPTQ 4-bit | 4.0 | 8.45 |
| GPTQ 3-bit | 3.0 | 8.68 |
| GPTQ 2.5-bit | 2.5 | 8.94 |
| 50% sparse + 4-bit | 2.0 | 8.29 |
| 50% sparse + 3-bit | 1.5 | 8.60 |
| 2:4 sparse + 4-bit | 2.0 | 8.55 |
| 4:8 sparse + 4-bit | 2.0 | 8.85 |

Critical observation: **50% sparse + 4-bit (effective 2 bits/param) achieves 8.29 perplexity, which is better than GPTQ 2.5-bit (8.94) and even better than GPTQ 3-bit (8.68)**. The combined approach at 2 effective bits/param outperforms quantization-only at 3 effective bits/param. This is the fundamental argument for combining pruning and quantization.

---

## 4. State-of-the-Art Combined Methods

### 4.1 SparseGPT + GPTQ: Prune Then Quantize with Error Compensation

The most natural combination, since SparseGPT and GPTQ share the same algorithmic foundation.

**Algorithm (joint pass)**:

For each linear layer with weight matrix W and calibration input X:

1. Compute H = X * X^T (Hessian / input correlation matrix)
2. Compute Cholesky decomposition of H^{-1}
3. Process columns j = 1 to d_in:
   a. For each row i, compute pruning saliency: s_ij = w_ij^2 / [H^{-1}]_jj
   b. **Prune decision**: if s_ij is below the layer's pruning threshold (determined by target sparsity), set w_ij = 0
   c. **Quantize decision**: if weight is not pruned, round w_ij to nearest quantization level for its assigned bit width: q_ij = Quantize(w_ij, b_ij)
   d. **Error**: err_ij = w_ij - (pruned ? 0 : q_ij)
   e. **Compensate**: for all columns k > j, update: w_ik -= err_ij * [H^{-1}]_jk / [H^{-1}]_jj
4. Output: sparse + quantized weight matrix

The key advantage: error compensation handles both pruning errors and quantization errors in a unified framework. The remaining weights are adjusted to minimize the total reconstruction error from both compression sources.

**Performance**: On OPT-175B, this achieves:
- 50% sparse + 3-bit = 8.60 perplexity (effective 1.5 bits/param) -- better than GPTQ 2.5-bit at 8.94
- 50% sparse + 4-bit = 8.29 perplexity (effective 2 bits/param) -- better than dense FP16 on some models after careful compensation

### 4.2 Wanda + Quantization: Simple Pruning Metric + Standard Quantization

A simpler but effective combination: use Wanda for pruning (fast, no weight updates) then apply standard quantization to the remaining weights.

**Pipeline**:
1. Run one forward pass of calibration data to collect activation norms ||X_j|| per layer
2. Compute Wanda scores: Score(w_ij) = |w_ij| * ||X_j||
3. For each row, prune the lowest-scoring weights to reach target sparsity
4. Apply quantization (GPTQ, AWQ, or RTN) to the pruned model

**Advantages**:
- Much faster than SparseGPT + GPTQ (no Hessian computation or weight compensation during pruning)
- The same calibration data and activation statistics can be reused for quantization
- Suitable for models up to hundreds of billions of parameters

**Disadvantages**:
- No error compensation during pruning -- quality is slightly worse than SparseGPT at high sparsity
- The sequential approach (prune, then quantize independently) misses joint optimization opportunities
- At 50% sparsity, nearly matches SparseGPT; at 60%+, gap widens

**For MXQ**: This is the pragmatic starting point. MXQ already computes activation-aware importance scores (AWQ-style) for bit allocation. Extending this to also produce pruning decisions requires minimal additional infrastructure -- the same forward pass that calibrates quantization also calibrates pruning.

### 4.3 OWL (Outlier Weighed Layerwise Sparsity)

OWL (Yin et al., ICML 2024) addresses a crucial observation: not all layers should be pruned to the same sparsity ratio.

**Key insight**: Layers with more activation outliers (features with magnitudes significantly larger than others) are more sensitive to pruning and should receive lower sparsity ratios. Conversely, layers with uniform activation magnitudes can tolerate higher sparsity.

**Algorithm**:

1. For each layer l, compute the outlier ratio:
   ```
   OutlierRatio(l) = (number of features with magnitude > threshold) / (total features)

   where threshold = mean(|features|) + alpha * std(|features|)
   ```

2. Set layer-wise sparsity proportional to (1 - OutlierRatio):
   ```
   sparsity(l) = base_sparsity * (1 - beta * OutlierRatio(l)) / normalization_constant
   ```
   Layers with many outliers get lower sparsity (more weights retained).
   Layers with few outliers get higher sparsity (more weights pruned).

3. Within each layer, apply any pruning method (magnitude, Wanda, or SparseGPT) at the layer-specific sparsity ratio.

**Performance**: At 70% average sparsity, OWL + Wanda achieves 61.22 points lower perplexity than uniform-sparsity Wanda, and 6.80 points lower than uniform-sparsity SparseGPT. The non-uniform allocation is the "missing secret sauce" for high-sparsity pruning.

**Connection to MXQ**: OWL's per-layer sparsity allocation is directly analogous to MXQ's per-layer bit allocation. MXQ already assigns different bit widths to different layers based on importance. Extending this to also assign different sparsity ratios per layer is natural -- layers that are important get both more bits AND less pruning.

### 4.4 REAP (Router-weighted Expert Activation Pruning)

REAP (Cerebras Research, 2025) is a pruning method specifically designed for Sparse Mixture-of-Experts (SMoE) models. Unlike the weight-level pruning methods above, REAP operates at the **expert level** -- it prunes entire experts from MoE layers.

**Context**: MoE models like DeepSeek-V3, Qwen3-Coder-480B, and Kimi-K2 have many experts per MoE layer (e.g., 256 experts, of which 8 are activated per token). Most experts are rarely used and contribute little when they are.

**REAP's saliency score**:

```
Saliency(expert_e) = sum over tokens t of: gate_value(t, e) * ||output(t, e)||_2

where:
  gate_value(t, e)  = the router's weight for expert e on token t
  output(t, e)      = the expert's output vector on token t
```

This captures both selection frequency (how often the expert is chosen, reflected in gate values) and functional impact (how large the expert's output is when chosen).

**Results**: REAP achieves near-lossless compression on code generation tasks even when pruning 50% of experts from trillion-parameter models. Published pruned models include DeepSeek-V3.2, Qwen3-Coder-480B, and Kimi-K2 on HuggingFace under Cerebras.

**Relevance to MXQ's dense-model pruning**: REAP itself targets MoE expert pruning, which is a different problem than weight-level pruning in dense models. However, REAP's core principle -- combining routing/selection information with output magnitude for saliency -- is conceptually similar to Wanda's |w| * ||x|| metric. For MLXQ:

- **For MoE models**: REAP-style expert pruning is complementary to weight-level pruning + quantization. First prune unnecessary experts (structural compression), then apply MXQ (pruning + quantization) to the remaining experts' weights.
- **For dense models**: REAP's gating * activation-norm saliency concept can inspire per-block pruning metrics where the "gating" signal is replaced by the block's contribution to the layer output norm.

### 4.5 2:4 Structured Sparsity + Quantization

When hardware supports N:M sparsity, combining it with quantization gives both speed and size benefits:

**On NVIDIA (Ampere+)**:
- 2:4 sparsity provides up to 2x Tensor Core throughput
- Combined with INT4 quantization via Sparse-Marlin kernels: 1.2x-3.0x total inference speedup
- Storage: each weight occupies ~2.5 bits effective (4-bit quantized, 50% sparse, plus metadata)
- Example: Sparse-Llama-3.1-8B achieves 30% throughput improvement with 2:4 sparsity alone, further improved with quantization

**On Apple Silicon**:
- No hardware N:M acceleration, so the speed benefit is absent
- Memory benefit still applies: 50% fewer weights to load from memory
- For MXQ, block-level sparsity (entire quantization groups zeroed out) is more practical than element-wise N:M because:
  - The MXQ kernel already processes weight blocks (groups of 32 or 64)
  - A block-level bitmap (1 bit per block) has negligible overhead
  - Skipping an entire zero block saves one complete block-dequant + accumulate operation
  - No need for per-element sparse metadata within blocks

### 4.6 Knowledge Distillation + Pruning + Quantization: The Triple Compression Stack

The most aggressive compression strategy combines all three techniques:

**Triple-stack pipeline**:

1. **Prune** (structural or unstructured): Remove weights/heads/layers
   - Use SparseGPT or Wanda for weight-level pruning
   - Use importance scoring for head/layer pruning
   - Target: 30-50% parameter reduction

2. **Distill** (knowledge distillation): Use the original dense model as teacher to recover quality lost in pruning
   - The dense model generates soft labels (token probabilities) on a large corpus
   - The pruned model is fine-tuned to match the teacher's output distribution
   - Loss: reverse KL divergence (KLD) between teacher and student distributions
   - This step can recover most of the quality lost in pruning
   - NVIDIA's Minitron approach: prune Nemotron-4-15B to 8B parameters, then distill, achieving quality competitive with LLaMA-3-8B (trained from scratch at much higher cost)

3. **Quantize** (post-training): Apply MXQ-style mixed-precision quantization to the distilled model
   - The distilled model is already compact from pruning
   - Quantization further compresses the remaining weights
   - The combination achieves extreme compression ratios

**Compression ordering matters**: Research on compression ordering (ICLR 2026) establishes the **Progressive Intensity Hypothesis**: weaker perturbations should precede stronger ones. This gives the ordering:

```
Prune (weakest perturbation) -> Distill (recovery) -> Quantize (strongest perturbation)
```

Pruning is weaker because it removes redundant weights (small perturbation if done well). Quantization is stronger because it perturbs every remaining weight. Applying the weaker perturbation first gives the model more room to recover before the stronger perturbation is applied.

**Compression arithmetic for triple-stack**:

```
Example: 70B model
  After 50% pruning:     35B effective parameters
  After distillation:    quality recovered to near-original
  After MXQ-2.5 quant:   35B * 2.5 bits = 10.9 GB

Compared to:
  70B MXQ-2.5 (quant only): 70B * 2.5 bits = 21.9 GB
  70B at 4-bit (quant only): 70B * 4 bits   = 35.0 GB
```

The triple-stack achieves the same quality in roughly half the size of quantization alone. The catch is that distillation requires substantial compute (a teacher model inference pass over a large corpus + student fine-tuning), making it significantly more expensive than post-training methods.

**For MXQ**: The triple-stack is a future direction. The initial MXQ release should focus on post-training pruning + quantization (no distillation), which requires only calibration data and no fine-tuning. Distillation can be added as a quality-recovery step in a future version for users willing to invest the compute.

---

## 5. Implementing Pruning + Quantization for MXQ

### 5.1 Proposed Pipeline

The MXQ pruning + quantization pipeline extends the existing MXQ quantization pipeline with a pruning stage:

```
[Calibrate] -> [Score] -> [Prune] -> [Re-Score] -> [Allocate] -> [Quantize] -> [Pack .mxq]
               importance  zero-out   recalculate   bits/block    mixed-prec   sparse+quant
               scores      weights    importance     per block     per block    format
```

#### Step 1: Calibrate

Run the calibration dataset through the full-precision model, collecting:
- Per-channel input activation norms ||X_j|| for each linear layer
- Per-block output sensitivity (KL-divergence when block is perturbed)
- Optionally: Hessian diagonal approximation H_ii per layer

This is identical to the existing MXQ calibration step. No additional forward passes are needed -- the same calibration run produces both pruning and quantization metrics.

#### Step 2: Score (Unified Importance)

Compute a unified importance score for each weight block that serves both pruning and quantization decisions:

```
For each weight block b (group of 32 or 64 weights in a row):

  BlockImportance(b) = (1/|b|) * sum over (i,j) in b of: |w_ij| * ||X_j||

  This is the Wanda metric averaged over the block.
```

The distribution of BlockImportance across all blocks in the model determines both:
- Which blocks to prune (lowest importance blocks)
- How many bits to allocate to each surviving block (higher importance = more bits)

#### Step 3: Prune

Zero out entire blocks with importance below a threshold. The threshold is determined by the target sparsity level.

**Block-level pruning** (recommended for MXQ):
- Prune at the granularity of quantization groups (32 or 64 weights)
- If all weights in a block are unimportant, the entire block is zeroed
- This aligns with MXQ's block-based storage and Metal kernel processing

**Layer-wise sparsity allocation** (OWL-inspired):
- Not all layers get the same sparsity ratio
- Layers with more outliers / higher importance get lower sparsity
- Embedding and lm_head layers: no pruning (0% sparsity)
- First/last 2 transformer layers: reduced pruning (e.g., 20% max)
- Attention Q/K/V projections: moderate pruning (30-40% max)
- MLP layers (gate_proj, up_proj, down_proj): aggressive pruning (50-60%)

**Algorithm**:
```python
def prune_model(model, importance_scores, target_sparsity, layer_config):
    """
    Block-level pruning with OWL-inspired non-uniform layer sparsity.
    """
    # Step 1: Compute per-layer sparsity ratios (OWL-style)
    layer_sparsities = compute_owl_sparsities(
        importance_scores,
        target_sparsity,
        layer_config  # min/max sparsity per layer type
    )

    # Step 2: For each layer, prune lowest-importance blocks
    pruning_mask = {}
    for layer_name, weights in model.items():
        s = layer_sparsities[layer_name]
        scores = importance_scores[layer_name]  # one score per block

        # Find threshold: bottom s% of blocks get pruned
        threshold = np.percentile(scores, s * 100)

        # Create block mask (1 = keep, 0 = prune)
        block_mask = (scores >= threshold).astype(np.uint8)
        pruning_mask[layer_name] = block_mask

        # Zero out pruned blocks
        for block_idx in range(len(block_mask)):
            if block_mask[block_idx] == 0:
                start = block_idx * block_size
                end = start + block_size
                weights[start:end] = 0.0

    return model, pruning_mask
```

#### Step 4: Re-Score (Post-Pruning Recalibration)

After pruning, re-run the calibration to compute updated importance scores for the remaining (unpruned) weight blocks. This is important because:

- Pruning changes the activation patterns: removing weights alters the information flow, so activation magnitudes change
- Weights that were moderately important before pruning may become critical after their neighbors are removed
- The optimal bit allocation for the pruned model differs from the optimal allocation for the dense model

**Cost**: One additional forward pass through the calibration data with the pruned model. This doubles the calibration time but significantly improves quantization quality.

**When to skip**: For light pruning (less than 20% sparsity), the activation changes are small and re-scoring can be skipped without meaningful quality loss. For aggressive pruning (40%+), re-scoring is essential.

#### Step 5: Allocate Bits

Using the post-pruning importance scores, allocate bit widths to each surviving (non-pruned) block. This is the standard MXQ bit allocation algorithm, but operating only on the subset of blocks that survived pruning.

```
Available bit widths: {2, 3, 4, 5, 6, 8}
Target: average bit width across surviving blocks = target_bits

Algorithm:
  1. Initialize all surviving blocks at minimum bits (2)
  2. Sort surviving blocks by importance (ascending)
  3. While average_bits < target:
       Upgrade the most important under-allocated block by 1 bit
  4. Apply layer-type priors (attention Q/K >= 4-bit, lm_head >= 6-bit)
```

Note: The total bit budget is computed over surviving blocks only. If 30% of blocks are pruned and the target is 2.5 bits average, the budget is:

```
Total bits = num_surviving_blocks * block_size * 2.5
Effective bits per original weight = 0.70 * 2.5 = 1.75 bits
```

#### Step 6: Quantize

Quantize each surviving block at its allocated bit width using the existing MXQ quantization (GPTQ-style with error compensation). Pruned blocks are skipped entirely.

For the Hessian-based compensation step, the Hessian is computed from the pruned model's activations (using the post-pruning calibration data from step 4). This ensures the compensation accounts for the changed activation patterns.

#### Step 7: Pack

Store the sparse + quantized representation in the `.mxq` format.

### 5.2 Sparse Storage in .mxq Format

The MXQ format must store both the pruning mask (which blocks are zero) and the quantized data (for non-zero blocks). The proposed format extension:

**Per-tensor storage**:

```
Standard MXQ tensors (existing):
  layers.N.self_attn.q_proj.weight           -> packed quantized data
  layers.N.self_attn.q_proj.weight_scales    -> per-block scale factors
  layers.N.self_attn.q_proj.weight_zeros     -> per-block zero points
  layers.N.self_attn.q_proj.weight_bits      -> per-block bit widths

New tensor for sparsity:
  layers.N.self_attn.q_proj.weight_sparse_mask -> bitmap (1 bit per block)
```

**Bitmap format**:

The `weight_sparse_mask` tensor is a uint8 array where each bit indicates whether the corresponding block is present (1) or pruned (0):

```
mask_bytes = ceil(num_blocks / 8)

Bit i of byte j corresponds to block (j * 8 + i):
  bit = 1 -> block is stored in the quantized data
  bit = 0 -> block is all-zero (not stored)
```

For a weight matrix of shape (4096, 4096) with block_size = 64:
- num_blocks = 4096 * 4096 / 64 = 262,144 blocks
- mask_bytes = 262,144 / 8 = 32,768 bytes = 32 KB

The mask overhead is negligible: 32 KB for a tensor that is typically 4-16 MB when quantized.

**Packed quantized data**:

Only non-zero blocks are stored in the quantized weight tensor. The blocks are stored contiguously in the order they appear (skipping zero blocks):

```
Quantized data layout:
  [block_0_data][block_3_data][block_4_data][block_7_data]...
   (block 1, 2, 5, 6 are pruned -- not stored)

The mask tells the reader which dense block index each stored block corresponds to.
```

**Reading the data (Metal kernel pseudo-code)**:

```metal
kernel void mxq_sparse_dequant(
    device const uint8_t* sparse_mask,
    device const uint8_t* packed_data,
    device const half* scales,
    device const half* zeros,
    device const uint8_t* bits,
    device half* output,
    uint block_idx [[thread_position_in_grid]]
) {
    // Check if this block is pruned
    uint byte_idx = block_idx / 8;
    uint bit_idx = block_idx % 8;
    bool is_present = (sparse_mask[byte_idx] >> bit_idx) & 1;

    if (!is_present) {
        // Block is pruned -- output zeros
        for (int i = 0; i < BLOCK_SIZE; i++) {
            output[block_idx * BLOCK_SIZE + i] = 0.0h;
        }
        return;
    }

    // Compute the data offset: count set bits before this position
    uint data_block_idx = popcount_before(sparse_mask, block_idx);

    // Dequantize from packed data at data_block_idx
    uint bit_width = bits[data_block_idx];
    half scale = scales[data_block_idx];
    half zero = zeros[data_block_idx];

    // ... standard MXQ dequantization of packed_data at data_block_idx ...
}
```

The `popcount_before` function counts the number of set bits in the mask before position `block_idx`, which gives the index into the packed data array. This can be computed efficiently using hardware popcount instructions and prefix sums.

**Alternative: dense storage with zero blocks**

A simpler approach stores all blocks (including zero blocks) but marks them with bit_width = 0:

```
weight_bits[block_idx] = 0  means "this block is all-zero, skip it"
weight_bits[block_idx] > 0  means "dequantize this block normally"
```

Advantages:
- No packed/sparse data layout -- all blocks have fixed positions
- No popcount_before computation
- The kernel simply checks `if (bits[block_idx] == 0) { output zeros; return; }`

Disadvantages:
- Zero blocks still occupy space in the quantized data array (though they could be stored as a single zero byte each)
- Does not achieve full memory savings from pruning -- the quantized data array is the same size as without pruning

For MXQ v1, the simpler approach (bit_width = 0 sentinel) is recommended. The packed bitmap approach can be added in v2 when memory savings from pruning are critical.

### 5.3 Metal Kernel Integration

The MXQ Metal dequant kernel needs minimal modification to support sparsity:

**For the simple approach (bit_width = 0 sentinel)**:

```metal
// In mxq_dequant_matmul kernel:
uint bits = weight_bits[block_idx];

if (bits == 0) {
    // Pruned block: skip entirely
    // No contribution to the output accumulator
    // (This block's output is zero, so matmul contribution is zero)
    return;  // or continue to next block
}

// ... existing dequant + matmul code for bits = {2, 3, 4, 5, 6, 8} ...
```

This is the most efficient approach because:
- The check is a single comparison (bits == 0)
- Pruned blocks cause early exit -- no memory loads for their data, scales, or zeros
- The branch is coherent within thread groups if blocks are pruned in clusters (likely, since unimportant blocks tend to be adjacent)

**Memory bandwidth savings**:

At 30% block sparsity:
- 30% of blocks trigger early return (no data loaded)
- Effective memory bandwidth usage drops by ~30%
- For a memory-bandwidth-bound operation (which LLM inference is), this translates to approximately 30% speedup

At 50% block sparsity:
- Half of all blocks skip their data load
- Approximately 50% bandwidth savings -> ~50% speedup (theoretical maximum)
- In practice, the speedup is less due to thread group stalls and non-uniform pruning

### 5.4 Interaction Effects

**Pruning changes the optimal quantization**:

After pruning, fewer weights remain, and each remaining weight carries more responsibility for the model's output. This means:
- The per-weight quantization error tolerance is lower (each weight matters more)
- Higher bit widths may be needed for some blocks that were previously allocated low bits
- The overall bit allocation should be re-optimized based on post-pruning importance

**Should we re-calibrate after pruning? Yes.**

Re-running a forward pass through calibration data after pruning serves two purposes:
1. Updated activation norms for Wanda-style importance scoring
2. Updated Hessian for GPTQ-style error compensation during quantization

The cost is one additional forward pass (typically a few minutes). The quality improvement is significant, especially at high sparsity levels.

**Joint optimization vs sequential (prune-then-quantize)**:

| Approach | Quality | Speed | Complexity |
|----------|---------|-------|------------|
| Sequential (prune, then quantize independently) | Good | Fast | Low |
| Sequential with re-calibration | Better | Moderate | Low |
| Joint (SparseGPT+GPTQ single pass) | Best | Slow | High |

Recommendation for MXQ v1: **Sequential with re-calibration**. It achieves near-optimal quality with manageable complexity. The joint approach can be added in a future version for maximum quality.

### 5.5 Updated mxq_config.json

The MXQ configuration file extends to include sparsity information:

```json
{
  "format": "mxq",
  "format_version": "1.1",
  "quantization": {
    "method": "mxq-importance-sparse",
    "target_bits": 2.5,
    "actual_bits": 2.52,
    "block_size": 64,
    "sparsity": {
      "enabled": true,
      "method": "wanda-block",
      "target_sparsity": 0.30,
      "actual_sparsity": 0.298,
      "effective_bits_per_weight": 1.76,
      "layer_sparsity_allocation": "owl",
      "recalibrated_after_pruning": true
    },
    "calibration_dataset": "mxq-calib-v1",
    "scoring_method": "awq+sensitivity"
  },
  "layer_allocation": {
    "embed_tokens": {"bits": 4, "sparsity": 0.0},
    "lm_head": {"bits": 6, "sparsity": 0.0},
    "layers.0-1": {"avg_bits": 4.2, "avg_sparsity": 0.15},
    "layers.2-29": {"avg_bits": 2.3, "avg_sparsity": 0.35},
    "layers.30-31": {"avg_bits": 4.0, "avg_sparsity": 0.15},
    "attention.q_proj": {"avg_bits": 3.8, "avg_sparsity": 0.20},
    "mlp.gate_proj": {"avg_bits": 2.1, "avg_sparsity": 0.45}
  },
  "quality_metrics": {
    "perplexity_bf16": 5.21,
    "perplexity_mxq_sparse": 5.35,
    "perplexity_mxq_dense": 5.38,
    "perplexity_uniform_4bit": 5.42
  }
}
```

### 5.6 CLI Interface Extension

```bash
# Prune + quantize in one step
mxq quantize \
  --model mlx-community/Qwen3.5-72B-bf16 \
  --imatrix ./imatrix.safetensors \
  --bits 2.5 \
  --sparsity 0.30 \
  --sparsity-method wanda \
  --sparsity-allocation owl \
  --recalibrate \
  --output ./Qwen3.5-72B-MXQ-S30-2.5bit

# Naming convention: S30 = 30% sparse
# Effective bits: 0.70 * 2.5 = 1.75 bits per original weight
```

---

## 6. Practical Sparsity Levels for LLMs

### 6.1 Achievable Sparsity Without Quality Loss

The maximum sparsity level varies by model size, pruning method, and the definition of "quality loss." The following summarizes empirical findings across the literature:

**Unstructured sparsity (element-wise)**:

| Sparsity | Quality impact | Method | Notes |
|----------|---------------|--------|-------|
| 30% | Negligible | Magnitude | Even simple pruning works at low sparsity |
| 40% | Negligible | Wanda/SparseGPT | No measurable perplexity increase on most models |
| 50% | Minimal | SparseGPT | <0.5 PPL increase on 7B+ models; <0.2 on 70B+ |
| 50% | Minimal | Wanda | Comparable to SparseGPT at this level |
| 60% | Slight | SparseGPT | 0.5-2.0 PPL increase, model-dependent |
| 60% | Moderate | Wanda | Without weight compensation, gap vs SparseGPT widens |
| 70% | Significant | SparseGPT + OWL | OWL's non-uniform allocation critical at this level |
| 70% | Severe | Wanda/magnitude | Perplexity degrades 10-60+ points without OWL |
| 80%+ | Usually unacceptable | Any method | Requires fine-tuning/distillation to recover quality |

**Block-level sparsity** (zeroing entire blocks of 32 or 64 weights):

Block sparsity is more constrained than element-wise, so achievable sparsity is lower for the same quality:

| Block sparsity | Equivalent element sparsity needed | Quality |
|---------------|----------------------------------|---------|
| 20% block | ~30% element-wise | Negligible loss |
| 30% block | ~40-45% element-wise | Minimal loss |
| 40% block | ~55-60% element-wise | Slight degradation |
| 50% block | ~65-70% element-wise | Significant degradation |

The "equivalent element sparsity" column shows what element-wise sparsity level gives similar quality. Block pruning is less precise because it cannot selectively keep individual important weights within an otherwise unimportant block.

**2:4 structured sparsity** (hardware-supported on NVIDIA):

| Model | Dense PPL | 2:4 Sparse PPL | Increase |
|-------|----------|----------------|----------|
| OPT-175B | 8.34 | ~8.55 | +0.21 |
| LLaMA-2-70B | ~3.32 | ~3.55 | +0.23 |
| LLaMA-7B | ~5.68 | ~6.15 | +0.47 |

2:4 sparsity is fixed at exactly 50% and achieves surprisingly good quality due to the fine-grained nature of the constraint (choosing 2 of 4 is highly flexible at the element level).

### 6.2 Layer-Wise Sparsity Tolerance

Not all layers tolerate pruning equally. General patterns observed across LLaMA, OPT, Qwen, and similar architectures:

**Attention layers (self_attn.q_proj, k_proj, v_proj, o_proj)**:
- Typically **less prunable** than MLP layers
- These projections directly handle the attention mechanism, which is critical for contextual understanding
- Recommended maximum sparsity: 30-40%
- Q and K projections are especially sensitive (they compute attention scores)
- V and O projections are slightly more tolerant

**MLP layers (mlp.gate_proj, up_proj, down_proj)**:
- **More prunable** -- contain more redundancy
- The MLP transforms are heavily overparameterized in most transformer architectures
- Recommended maximum sparsity: 40-60%
- gate_proj and up_proj are slightly more prunable than down_proj
- down_proj's output directly enters the residual stream, making errors more impactful

**Embedding layer (embed_tokens)**:
- **Not pruned** -- each row corresponds to a vocabulary token
- Pruning rows would eliminate the model's ability to process certain tokens
- Pruning columns would reduce embedding dimension, requiring changes throughout the model
- This layer should remain at 0% sparsity

**Language model head (lm_head)**:
- **Not pruned** -- each row produces the logit for a vocabulary token
- Pruning would eliminate the model's ability to output certain tokens
- Often shares weights with embed_tokens (tied embeddings), so pruning either affects both
- This layer should remain at 0% sparsity

**Layer position effects**:
- **First 1-2 transformer layers**: Less prunable. They process raw token embeddings and establish initial representations. Errors here propagate through all subsequent layers.
- **Middle layers (layers 2 through L-2)**: Most prunable. These perform incremental refinement and contain the most redundancy.
- **Last 1-2 transformer layers**: Moderately prunable, but less than middle layers. They produce the final representation used for next-token prediction.

**OWL-style non-uniform allocation** (recommended):

```python
def compute_layer_sparsity(model, target_sparsity=0.30):
    """
    Allocate per-layer sparsity based on outlier ratios (OWL).
    """
    layer_sparsities = {}

    for name, layer in model.layers.items():
        if 'embed' in name or 'lm_head' in name:
            layer_sparsities[name] = 0.0  # Never prune
            continue

        # Compute outlier ratio for this layer
        outlier_ratio = compute_outlier_ratio(layer)

        # Base sparsity modified by outlier ratio
        # More outliers -> lower sparsity (more weight retention)
        base = target_sparsity

        if 'q_proj' in name or 'k_proj' in name:
            base *= 0.7  # Reduce sparsity for Q/K
        elif 'v_proj' in name or 'o_proj' in name:
            base *= 0.8  # Slightly reduce for V/O
        elif 'gate_proj' in name or 'up_proj' in name:
            base *= 1.3  # Increase sparsity for MLP up/gate
        elif 'down_proj' in name:
            base *= 1.1  # Slightly increase for MLP down

        # Adjust by outlier ratio (OWL)
        adjusted = base * (1.0 - outlier_ratio)

        # Apply position-based adjustment
        layer_idx = extract_layer_index(name)
        total_layers = model.num_layers
        if layer_idx < 2 or layer_idx >= total_layers - 2:
            adjusted *= 0.5  # Halve sparsity for first/last layers

        layer_sparsities[name] = min(adjusted, 0.6)  # Cap at 60%

    # Normalize so overall average matches target_sparsity
    layer_sparsities = normalize_to_target(
        layer_sparsities, target_sparsity, model
    )

    return layer_sparsities
```

### 6.3 Interaction with Model Size

Larger models tolerate more pruning due to greater redundancy:

**Empirical observations**:

| Model size | Max sparsity (negligible loss) | Max sparsity (acceptable loss) |
|-----------|-------------------------------|-------------------------------|
| 1-3B | 30% | 40% |
| 7-13B | 40% | 50% |
| 30-70B | 50% | 60% |
| 100B+ | 50-60% | 70% |

**Why larger models prune better**:
- Overparameterization: larger models have more parameters than needed to represent the learned function
- Redundant heads: in a 64-head model, many heads learn similar attention patterns; in an 8-head model, each head is more unique
- Wider MLP: 4096->11008->4096 MLP has massive redundancy in the hidden layer; smaller models have proportionally smaller hidden layers
- More layers: a 80-layer model has more redundant layers than a 32-layer model

**Compression implication**:

A 70B model at 50% sparsity (35B effective parameters) does NOT perform like a 35B model trained from scratch. It performs significantly better because:
- The 70B model's 50% surviving weights were selected to maximize information retention
- These weights benefited from the full 70B model's training, which explored a much richer optimization landscape
- The pruned model retains the 70B architecture (many layers, many heads) with a sparse connectivity pattern that a 35B dense model cannot replicate

Conversely, a pruned 70B at 50% sparsity typically matches or slightly exceeds a well-trained 35B dense model, at the same parameter count but with a different computational profile (sparse matmuls vs. smaller dense matmuls).

---

## 7. Hardware Considerations for Sparse+Quantized Inference

### 7.1 Sparse MatMul on Apple Silicon

Apple Silicon's compute capabilities for sparse operations as of 2025-2026:

**Accelerate Framework (CPU / AMX)**:
- `Sparse BLAS`: SpMV and SpMM via `sparse_matrix_vector_multiply_*` and related functions
- Supports CSR, CSC, and COO sparse formats
- Runs on the AMX (Apple Matrix coprocessor) units within the CPU cores
- Performance: good for scientific computing, but not the right path for LLM inference (CPU-bound, not GPU-bound)

**BNNS (CPU Neural Network Subroutines)**:
- `BNNSNDArrayFullyConnectedSparsifySparseCOO()`: converts COO sparse weights to internal format
- BNNS fully connected layers can use this sparse format
- CPU-only, not accessible from Metal shaders
- Performance characteristics not publicly benchmarked

**Metal (GPU)**:
- `MPSMatrixMultiplication`: dense matmul only, no sparse variant in the public API
- Custom Metal compute kernels: can implement sparse operations manually
- No hardware sparse matrix acceleration (no Sparse Tensor Cores equivalent)
- The GPU excels at dense, regular memory access patterns

**MLX**:
- No sparse tensor type as of v0.31
- `mx.fast` operations are dense
- Custom Metal kernels (via `mx.fast.metal_kernel()`) can implement block-sparse patterns
- The unified memory architecture means CPU sparse solvers and GPU dense matmul share the same memory -- data does not need to be copied

### 7.2 Block-Sparse vs Element-Wise Sparse on Metal

For MXQ on Apple Silicon, the choice between block-sparse and element-wise sparse has dramatic performance implications:

**Element-wise sparse** (individual zero weights):

```
Performance characteristics:
- Memory access: Irregular. Each nonzero weight is at a different offset.
  Must use gather/scatter operations or indexed loads.
- Cache utilization: Poor. Sparse indices cause cache line partial fills.
- Branch divergence: High. Each thread may have a different sparsity pattern,
  causing SIMD lanes to diverge.
- Metadata overhead: CSR indices or bitmap must be loaded and decoded.
- Practical speedup: Negligible or even slower than dense at <80% sparsity
  on GPU hardware (consistent with Flash-LLM finding that traditional sparse
  kernels need >95% sparsity to beat cuBLAS on NVIDIA GPUs).
```

**Block-sparse** (entire blocks of 32-64 weights are zero or nonzero):

```
Performance characteristics:
- Memory access: Regular within blocks. Each block is a dense tile loaded
  contiguously. Only the block-level index is irregular.
- Cache utilization: Good. Each block fills cache lines completely.
- Branch divergence: Low if block size >= SIMD width (32 on Apple GPU).
  All threads in a SIMD group process the same block -- either all skip
  (zero block) or all compute (nonzero block).
- Metadata overhead: Minimal. 1 bit per block for the mask. For 64-weight
  blocks in a 4096x4096 matrix: 32 KB bitmap.
- Practical speedup: Proportional to sparsity for memory-bound operations.
  At 50% block sparsity, ~40-45% fewer memory loads, translating to
  ~30-40% speedup on memory-bandwidth-bound operations.
```

**Recommendation for MXQ**: Block-sparse at the quantization group granularity (32 or 64 weights per block). This naturally aligns with:
- MXQ's per-block bit allocation (each block already has its own scale, zero, and bit width)
- Metal thread group processing (32 threads per SIMD group matches 32-weight blocks)
- The dequantization kernel (which already processes one block per iteration)

### 7.3 The Overhead of Checking Sparsity Masks in the Kernel

Every block in the MXQ kernel must check the sparsity mask to decide whether to skip or compute. The overhead of this check depends on the implementation:

**Bit_width == 0 sentinel check (simplest)**:

```metal
uint bits = weight_bits[block_idx];
if (bits == 0) return;  // or continue
```

Cost: One uint8 load + one comparison + conditional branch.
Overhead: ~2-5 nanoseconds per block. For a 4096x4096 matrix with 64-weight blocks (262K blocks), total overhead is ~0.5-1.3 ms. This is negligible compared to the typical 10-50 ms for a full matrix multiply.

**Bitmap check**:

```metal
uint byte_idx = block_idx / 8;
uint bit_idx = block_idx % 8;
bool present = (sparse_mask[byte_idx] >> bit_idx) & 1;
if (!present) return;
```

Cost: One uint8 load + shift + AND + comparison + conditional branch.
Overhead: ~3-8 nanoseconds per block. Slightly more than the sentinel check, but the mask is only 32 KB (fits in L1 cache after first access).

**Bitmap with packed data (popcount index)**:

```metal
uint data_idx = popcount_before(sparse_mask, block_idx);
```

Cost: This requires counting set bits in all mask bytes before the current position. Can be done with:
- Hardware popcount on each byte: O(block_idx / 8) operations
- Prefix sum table (precomputed): O(1) lookup + O(1) popcount of partial byte

With a precomputed prefix sum table (one entry per 256 blocks = ~1 KB), the index computation is ~5-10 nanoseconds.

**Overall assessment**: Sparsity mask overhead is negligible (< 1% of kernel time) for all approaches. The sentinel check (bit_width == 0) is simplest and fastest, and should be the default for MXQ v1.

### 7.4 When Sparsity Helps vs Hurts Performance

**When sparsity helps (reduces inference time)**:

1. **Memory bandwidth savings**: The primary benefit on Apple Silicon. LLM inference (especially at batch size 1, the autoregressive decoding case) is memory-bandwidth-bound. Each token generation requires loading all model weights from memory. Skipping zero blocks reduces the total data loaded:
   ```
   Bandwidth savings = sparsity * weight_data_size
   Token generation speedup approximately = 1 / (1 - sparsity * weight_fraction)

   where weight_fraction = fraction of bandwidth used for weights (vs KV cache, etc.)
   For batch=1 autoregressive: weight_fraction approximately = 0.85-0.95
   ```

2. **Compute savings with block-sparse**: Entire matmul tiles can be skipped. If a block of weights is zero, the corresponding partial dot product is zero and need not be computed. This saves both bandwidth and ALU cycles.

3. **Cache pressure reduction**: Fewer weights to load means less cache pollution. The remaining weights and activation data have more effective cache capacity.

**When sparsity hurts (increases inference time)**:

1. **Irregular memory access** (element-wise sparsity): Sparse indices cause non-contiguous memory loads. On GPU architectures optimized for coalesced access, irregular loads waste bandwidth and cache lines.

2. **Branch divergence** (element-wise sparsity): SIMD groups (32 threads on Apple GPU) must execute the same instruction. If some threads encounter zero weights and others encounter nonzero weights, the group must execute both paths, negating the speedup.

3. **Metadata overhead**: Loading and processing sparsity metadata (CSR indices, bitmaps, compressed indices) consumes bandwidth and compute that partly offsets the savings from skipping zero weights.

4. **Kernel launch overhead**: If sparsity requires a fundamentally different kernel (sparse vs dense), the overhead of kernel dispatch and pipeline state changes may dominate for small operations.

5. **Load imbalance**: If sparsity is non-uniform across rows/columns, some thread groups have more work than others, leading to underutilization of GPU resources.

**Break-even analysis for Apple Silicon with block-sparse MXQ**:

```
Assumptions:
  - M4 Pro memory bandwidth: ~273 GB/s
  - 70B model at MXQ-2.5: 22 GB weight data
  - Time to load all weights: 22 GB / 273 GB/s = 80.6 ms
  - Sparsity mask check overhead: ~0.5 ms (negligible)

At 30% block sparsity:
  - Weight data reduced to: 22 * 0.70 = 15.4 GB
  - Load time: 15.4 / 273 = 56.4 ms
  - Speedup: 80.6 / 56.4 = 1.43x (30% faster)
  - Quality: negligible loss with Wanda + OWL

At 50% block sparsity:
  - Weight data reduced to: 22 * 0.50 = 11 GB
  - Load time: 11 / 273 = 40.3 ms
  - Speedup: 80.6 / 40.3 = 2.0x (theoretical maximum)
  - Quality: slight degradation, model-dependent

At 10% block sparsity:
  - Weight data reduced to: 22 * 0.90 = 19.8 GB
  - Load time: 19.8 / 273 = 72.5 ms
  - Speedup: 80.6 / 72.5 = 1.11x (11% faster)
  - Quality: zero loss for any model
```

Even 10% block sparsity is worth implementing because:
- The quality cost is zero (pruning the truly useless blocks)
- The 11% speedup is free
- The implementation cost is minimal (one conditional check in the kernel)

There is no meaningful break-even point for block-sparse on Apple Silicon -- any amount of block sparsity provides proportional memory bandwidth savings with negligible overhead. This is because block-sparse avoids the irregular memory access patterns that cause problems with element-wise sparsity.

### 7.5 Block Sparsity Design Recommendations for MXQ

Based on the hardware analysis, these are the recommended design choices for MXQ v1:

**Block size: 64 weights** (matching MXQ quantization group size)

- Aligns with MXQ's existing per-block scale/zero/bits metadata
- Two SIMD groups (2 * 32 = 64) process one block
- Efficient cache line usage (64 weights at 2.5 bits = 20 bytes, fits in one cache line)

**Sparsity representation: bit_width = 0 sentinel**

- No separate bitmap tensor needed
- The existing weight_bits tensor already has one entry per block
- Set bits[block_idx] = 0 for pruned blocks
- Kernel checks bits == 0 as the first operation
- Migration path: future versions can add packed bitmap for maximum compression

**Data layout: dense with zero blocks**

- All block positions are allocated in the quantized weight tensor
- Pruned blocks store no actual quantized data (or store a single zero byte as placeholder)
- No popcount/prefix-sum index computation needed
- Simpler implementation, slightly more memory usage, significantly simpler kernel

**Target sparsity: 20-30% for quality-first profiles (MXQ-2.5, MXQ-3)**

- At 20-30% block sparsity, quality impact is negligible for all tested models
- Memory savings: 20-30% fewer weights to load
- Effective bits per weight: 0.70 * 2.5 = 1.75 (for MXQ-2.5 at 30% sparsity)
- This achieves MXQ-2.5 quality at MXQ-1.75 effective size

**Target sparsity: 40-50% for compression-first profiles (MXQ-2, MXQ-1.5)**

- Higher sparsity for users who need maximum compression
- Quality will degrade noticeably but may be acceptable for specific use cases
- Effective bits: 0.50 * 2.0 = 1.0 bits per weight at 50% sparsity + 2-bit
- Only recommended for 70B+ models where overparameterization provides headroom

**Sparsity allocation: OWL-inspired non-uniform**

- Per-layer sparsity ratios based on outlier analysis
- embed_tokens and lm_head: 0% (never pruned)
- Attention projections: 60-70% of base sparsity
- MLP layers: 110-130% of base sparsity
- First/last 2 layers: 50% of base sparsity

---

## Appendix A: Summary of All Pruning Methods

| Method | Year | Type | Needs Calibration | Weight Update | Speed | Quality at 50% |
|--------|------|------|-------------------|---------------|-------|----------------|
| Magnitude | 1990 | Unstructured | No | No | Very fast | Poor |
| OBD | 1990 | Unstructured | Yes (Hessian) | No | Slow | Moderate |
| OBS | 1993 | Unstructured | Yes (Hessian) | Yes (optimal) | Very slow | Good |
| SparseGPT | 2023 | Unstructured/N:M | Yes (128 samples) | Yes (Hessian) | Moderate | Excellent |
| Wanda | 2023 | Unstructured/N:M | Yes (128 samples) | No | Fast (300x vs SparseGPT) | Very good |
| OWL | 2024 | Layer allocation | Yes (outlier analysis) | Depends on base method | Fast (wrapper) | Excellent at 70% |
| REAP | 2025 | Expert-level (MoE) | Yes (router + activation) | No | Fast | Excellent (MoE) |

## Appendix B: Effective Compression Ratios

Reference table for combined pruning + quantization compression:

| Block Sparsity | Quant Bits | Effective Bits/Weight | Compression vs FP16 | 70B Model Size |
|---------------|-----------|----------------------|---------------------|---------------|
| 0% | 4.0 | 4.00 | 4.0x | 35.0 GB |
| 0% | 3.0 | 3.00 | 5.3x | 26.3 GB |
| 0% | 2.5 | 2.50 | 6.4x | 21.9 GB |
| 0% | 2.0 | 2.00 | 8.0x | 17.5 GB |
| 20% | 3.0 | 2.40 | 6.7x | 21.0 GB |
| 20% | 2.5 | 2.00 | 8.0x | 17.5 GB |
| 30% | 2.5 | 1.75 | 9.1x | 15.3 GB |
| 30% | 2.0 | 1.40 | 11.4x | 12.3 GB |
| 40% | 2.5 | 1.50 | 10.7x | 13.1 GB |
| 50% | 2.5 | 1.25 | 12.8x | 10.9 GB |
| 50% | 2.0 | 1.00 | 16.0x | 8.75 GB |
| 50% | 4.0 | 2.00 | 8.0x | 17.5 GB |

Note: Actual model sizes will be slightly larger due to metadata (scales, zeros, bit widths, sparsity masks, embedding layer, lm_head at higher precision).

## Appendix C: Decision Framework for MXQ Users

```
Question: How should I choose sparsity and bit width for my model?

If RAM is not a constraint:
  -> MXQ-4bit, 0% sparsity (best quality, ~35 GB for 70B)

If you need 70B on 32 GB Mac:
  -> MXQ-2.5bit, 0% sparsity (21.9 GB, matches 4-bit quality)

If you need 70B on 24 GB Mac:
  -> MXQ-2.5bit, 20% sparsity (17.5 GB, near-identical to 4-bit)
  or MXQ-3bit, 30% sparsity (18.4 GB, near-identical to 4-bit)

If you need 70B on 16 GB Mac:
  -> MXQ-2.5bit, 40% sparsity (13.1 GB, slight quality loss)
  or MXQ-2bit, 30% sparsity (12.3 GB, moderate quality loss)

If you need maximum compression (research/experimental):
  -> MXQ-2bit, 50% sparsity (8.75 GB, noticeable quality loss)
  -> Only recommended for 70B+ models
```

## Appendix D: Comparison with Competing Approaches

| Approach | Platform | Effective Bits | Sparse HW Accel | Block-Sparse | Quality |
|----------|----------|---------------|-----------------|-------------|---------|
| MXQ sparse (proposed) | Apple Silicon | 1.5-2.5 | No (bandwidth only) | Yes (block) | Good-Excellent |
| Sparse-Marlin (NVIDIA) | CUDA (Ampere+) | 2.0 | Yes (2:4 Tensor Core) | 2:4 structured | Good |
| SparseGPT+GPTQ | CUDA | 1.5-2.0 | Optional | Unstructured | Excellent |
| GGUF Q2_K sparse | CPU | ~2.5 | No | No | Mediocre |
| EXL2 + sparse | CUDA | variable | No | No | Good |

MXQ's advantage on Apple Silicon: block-sparse at quantization-group granularity provides consistent memory bandwidth savings without requiring sparse hardware support. The unified memory architecture ensures that bandwidth savings translate directly to inference speedup, unlike discrete GPU systems where PCIe transfer and GPU memory bandwidth are separate bottlenecks.
