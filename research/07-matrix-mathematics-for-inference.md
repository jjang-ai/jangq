# 07 -- Matrix Mathematics for LLM Inference

> Mathematical foundations for understanding, implementing, and optimizing MXQ quantization on Apple Silicon.

---

## Table of Contents

1. [Linear Algebra Fundamentals for Transformers](#1-linear-algebra-fundamentals-for-transformers)
2. [Transformer Architecture Math](#2-transformer-architecture-math)
3. [Numerical Precision and Error Accumulation](#3-numerical-precision-and-error-accumulation)
4. [The Hessian Matrix and Its Role in Quantization](#4-the-hessian-matrix-and-its-role-in-quantization)
5. [Singular Value Decomposition Perspective](#5-singular-value-decomposition-perspective)
6. [Weight Distribution Statistics in Transformers](#6-weight-distribution-statistics-in-transformers)
7. [Optimal Bit Allocation Theory](#7-optimal-bit-allocation-theory-rate-distortion)
8. [Practical Matmul Performance on Apple Silicon](#8-practical-matmul-performance-on-apple-silicon)

---

## 1. Linear Algebra Fundamentals for Transformers

### 1.1 Matrix Multiplication: The Core Operation

Every linear layer in a transformer is a matrix multiplication. This single operation dominates the computational cost of LLM inference. Understanding it precisely is non-negotiable for building MXQ.

**Definition.** Given matrices A of shape (n x k) and B of shape (k x m), their product C = A x B is a matrix of shape (n x m) where each element is defined by:

```
C_ij = sum over p from 1 to k of: A_ip * B_pj
```

Written concretely for a single element:

```
C_ij = A_i1 * B_1j + A_i2 * B_2j + ... + A_ik * B_kj
```

Each element of C requires k multiplications and (k-1) additions, for a total of k multiply-add operations. The full matrix C has n * m elements, giving a total operation count of:

```
Total multiply-adds = n * m * k
Total FLOPs = 2 * n * m * k    (counting multiplies and adds separately)
```

**Computational complexity**: O(n * m * k) multiply-add operations.

### 1.2 Matmul in Transformer Linear Layers

In a transformer, every linear layer computes:

```
y = x * W + b
```

where:
- x is the input activation tensor
- W is the weight matrix (learned parameters, stored on disk)
- b is the bias vector (often absent in modern transformers)
- y is the output activation tensor

The shapes in practice, for a model like Llama-3-70B:

```
x: (batch_size, seq_len, hidden_dim)    e.g., (1, 1, 8192)
W: (hidden_dim, output_dim)             e.g., (8192, 28672)
y: (batch_size, seq_len, output_dim)    e.g., (1, 1, 28672)
```

For inference with batch_size=1 and seq_len=1 (autoregressive token generation), x is effectively a row vector of dimension hidden_dim. The matmul reduces to a matrix-vector product:

```
y[j] = sum over i from 1 to hidden_dim of: x[i] * W[i][j]
```

This is the operation that happens hundreds of times per generated token.

### 1.3 Concrete Dimension Examples

The following dimensions are representative of a 70B-class model (Llama-3-70B, Qwen-2.5-72B):

| Component | Shape of W | Multiply-adds per matmul |
|-----------|-----------|--------------------------|
| Q projection | (8192, 8192) | 67,108,864 (67M) |
| K projection | (8192, 1024) | 8,388,608 (8.4M) |
| V projection | (8192, 1024) | 8,388,608 (8.4M) |
| O projection | (8192, 8192) | 67,108,864 (67M) |
| MLP gate_proj | (8192, 28672) | 234,881,024 (235M) |
| MLP up_proj | (8192, 28672) | 234,881,024 (235M) |
| MLP down_proj | (28672, 8192) | 234,881,024 (235M) |

Note: K and V projections in 70B models often use grouped-query attention (GQA) with fewer heads, hence the smaller output dimension. The exact K/V dimensions depend on the number of KV heads. With 8 KV heads and head_dim=128: output_dim = 8 * 128 = 1024.

**Worked example -- single MLP gate projection:**

```
x shape: (1, 1, 8192)     -- one token, one batch
W shape: (8192, 28672)     -- gate_proj weights

Multiply-adds: 1 * 1 * 8192 * 28672 = 234,881,024 ~ 235 million

At fp16: each multiply-add is roughly 1 FLOP on the ALU
Total FLOPs: ~470 million (counting multiply and add separately)
```

This single matmul processes 235 million weight values. The weight matrix W occupies:
- In fp16: 8192 * 28672 * 2 bytes = 469,762,048 bytes ~ 448 MB
- In 4-bit: 8192 * 28672 * 0.5 bytes = 117,440,512 bytes ~ 112 MB
- In 2-bit: 8192 * 28672 * 0.25 bytes = 58,720,256 bytes ~ 56 MB

### 1.4 Matrix-Vector vs. Matrix-Matrix Products

During LLM inference, two regimes exist:

**Prefill (prompt processing):** The entire input prompt is processed in parallel. This is a matrix-matrix product:

```
x: (1, seq_len, hidden_dim)  e.g., (1, 2048, 8192)
W: (hidden_dim, output_dim)  e.g., (8192, 28672)
y: (1, seq_len, output_dim)  e.g., (1, 2048, 28672)

Multiply-adds: seq_len * hidden_dim * output_dim
             = 2048 * 8192 * 28672
             = 481,036,337,152 ~ 481 billion
```

This is compute-bound: the GPU has enough data to keep its ALUs busy.

**Decode (token generation):** Each new token is generated one at a time. This is a matrix-vector product:

```
x: (1, 1, hidden_dim)       e.g., (1, 1, 8192)
W: (hidden_dim, output_dim) e.g., (8192, 28672)
y: (1, 1, output_dim)       e.g., (1, 1, 28672)

Multiply-adds: 1 * hidden_dim * output_dim
             = 8192 * 28672
             = 234,881,024 ~ 235 million
```

This is memory-bandwidth-bound: the GPU must read the entire weight matrix W from memory but performs relatively few operations per weight value read. The ratio of compute to memory access (arithmetic intensity) is low. This is the regime where quantization matters most.

### 1.5 The Batched Case

With continuous batching (serving multiple users), decode becomes a thin matrix-matrix product:

```
x: (batch_size, 1, hidden_dim)  e.g., (8, 1, 8192)
W: (hidden_dim, output_dim)     e.g., (8192, 28672)
y: (batch_size, 1, output_dim)  e.g., (8, 1, 28672)

Multiply-adds: batch_size * hidden_dim * output_dim
             = 8 * 8192 * 28672
             = 1,879,048,192 ~ 1.88 billion
```

The weight matrix W is read once from memory but used for 8 different inputs. This increases arithmetic intensity by 8x, moving closer to compute-bound territory. However, on single-user desktop inference (the primary MXQ use case), batch_size=1 is the dominant regime.

---

## 2. Transformer Architecture Math

### 2.1 Multi-Head Attention: 4 Weight Matmuls

A single multi-head attention (MHA) layer performs the following sequence of operations. Let x be the input tensor of shape (batch, seq_len, d_model) where d_model is the hidden dimension (e.g., 8192).

**Step 1: Compute Q, K, V projections (3 matmuls)**

```
Q = x * W_Q    shape: (batch, seq_len, d_model) x (d_model, d_model) -> (batch, seq_len, d_model)
K = x * W_K    shape: (batch, seq_len, d_model) x (d_model, d_kv)   -> (batch, seq_len, d_kv)
V = x * W_V    shape: (batch, seq_len, d_model) x (d_model, d_kv)   -> (batch, seq_len, d_kv)
```

With grouped-query attention (GQA), d_kv = n_kv_heads * head_dim, which is smaller than d_model = n_heads * head_dim. For Llama-3-70B: n_heads = 64, n_kv_heads = 8, head_dim = 128, so d_model = 8192 and d_kv = 1024.

**Step 2: Reshape into heads and compute attention scores (1 activation matmul)**

Q and K are reshaped into separate heads:

```
Q: (batch, n_heads, seq_len, head_dim)
K: (batch, n_kv_heads, seq_len, head_dim)
```

With GQA, each KV head is shared across n_heads/n_kv_heads query heads (8 in Llama-3-70B).

Attention scores:

```
S = (Q * K^T) / sqrt(head_dim)
```

Shape of S: (batch, n_heads, seq_len, seq_len)

This is a matmul on activations (not weights), so it is not subject to weight quantization. However, it is O(seq_len^2) in memory and compute, which is why KV caching exists.

**Step 3: Apply softmax and compute attention output (1 activation matmul)**

```
A = softmax(S, dim=-1)      shape: (batch, n_heads, seq_len, seq_len)
O = A * V                    shape: (batch, n_heads, seq_len, head_dim)
```

Again, this matmul operates on activations, not on stored weights.

**Step 4: Output projection (1 weight matmul)**

```
O_reshaped: (batch, seq_len, d_model)     -- concatenate heads
out = O_reshaped * W_O                     -- shape: (d_model, d_model)
```

**Summary of weight matmuls per attention layer:**

| Operation | Weight matrix shape | Multiply-adds (per token) |
|-----------|-------------------|---------------------------|
| Q projection | (d_model, d_model) = (8192, 8192) | 67.1M |
| K projection | (d_model, d_kv) = (8192, 1024) | 8.4M |
| V projection | (d_model, d_kv) = (8192, 1024) | 8.4M |
| O projection | (d_model, d_model) = (8192, 8192) | 67.1M |
| **Total** | | **151.0M** |

Total weight parameters in attention: 8192*8192 + 8192*1024 + 8192*1024 + 8192*8192 = 67.1M + 8.4M + 8.4M + 67.1M = 151.0M parameters.

### 2.2 MLP / Feed-Forward Network: 3 Weight Matmuls

Modern transformers (Llama, Qwen, Mistral, Gemma) use the SwiGLU variant of the feed-forward network, which replaces the traditional two-layer MLP with a gated architecture.

**Traditional MLP (older models):**

```
hidden = ReLU(x * W_1 + b_1)    -- up projection
output = hidden * W_2 + b_2     -- down projection
```

Two matmuls. W_1 shape: (d_model, d_ff), W_2 shape: (d_ff, d_model), where d_ff is typically 4 * d_model.

**SwiGLU MLP (modern models):**

```
gate = x * W_gate               -- gate projection
up   = x * W_up                 -- up projection
hidden = SiLU(gate) * up        -- element-wise gating (Hadamard product)
output = hidden * W_down        -- down projection
```

Three matmuls. The SiLU activation (also called Swish) is:

```
SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
```

The element-wise product gate * up is a Hadamard product, not a matmul -- it is cheap (O(n)) and operates only on activations.

**Dimensions for 70B model:**

| Operation | Weight matrix shape | Multiply-adds (per token) |
|-----------|-------------------|---------------------------|
| Gate projection | (d_model, d_ff) = (8192, 28672) | 234.9M |
| Up projection | (d_model, d_ff) = (8192, 28672) | 234.9M |
| Down projection | (d_ff, d_model) = (28672, 8192) | 234.9M |
| **Total** | | **704.6M** |

Total weight parameters in MLP: 3 * 8192 * 28672 = 704,643,072 ~ 704.6M parameters.

Note: d_ff for SwiGLU models is typically set to (8/3) * d_model, rounded to the nearest multiple of 256 for hardware alignment. For d_model=8192: (8/3) * 8192 = 21845, rounded up to a convenient size. In Llama-3-70B, d_ff = 28672 (which is 3.5 * d_model, a common choice).

### 2.3 Per Transformer Block: 7 Weight Matmuls

Each transformer block consists of one attention layer and one MLP layer, preceded by RMSNorm layers (which have no matmuls -- they are element-wise operations with a small learned scale vector).

```
Block structure:
  h = x + Attention(RMSNorm(x))     -- 4 weight matmuls
  out = h + MLP(RMSNorm(h))         -- 3 weight matmuls
                                     -- Total: 7 weight matmuls per block
```

Weight parameters per block: 151.0M (attention) + 704.6M (MLP) = 855.6M ~ 856M parameters.

### 2.4 Full Model: 560 Weight Matmuls

A 70B model like Llama-3-70B has 80 transformer layers.

```
Total weight matmuls per forward pass = 7 * 80 = 560

Total weight parameters in transformer blocks: 856M * 80 = 68.5 billion
Plus embedding layer: vocab_size * d_model = 128256 * 8192 = 1.05 billion
Plus lm_head: d_model * vocab_size = 8192 * 128256 = 1.05 billion
(often tied, so counted once)

Grand total: ~69.5 billion parameters ~ "70B"
```

**Worked example -- total compute per generated token (70B model):**

```
Per-block multiply-adds: 151.0M (attn) + 704.6M (MLP) = 855.6M
All blocks: 855.6M * 80 = 68,448,000,000 ~ 68.4 billion multiply-adds
lm_head: 8192 * 128256 = 1,050,738,688 ~ 1.05 billion multiply-adds
Total: ~69.5 billion multiply-adds per token
Total FLOPs: ~139 billion (counting multiply and add separately)
```

At fp16 on M4 Max (38 TFLOPS fp16): theoretical minimum time per token = 139e9 / 38e12 = 3.66 ms. But this assumes the GPU is compute-bound and fully utilized. In practice, at batch_size=1, the bottleneck is memory bandwidth, not compute. This is the core reason quantization matters.

### 2.5 Why Quantized Matmul Performance Is Everything

From the numbers above:

- Generating one token requires reading all 69.5 billion weight parameters from memory
- At fp16 (2 bytes/param): 139 GB must be read per token
- At 4-bit (0.5 bytes/param): 34.75 GB must be read per token
- At 2.5-bit MXQ (0.3125 bytes/param): 21.7 GB must be read per token
- At 2-bit (0.25 bytes/param): 17.4 GB must be read per token

On M4 Max with 546 GB/s memory bandwidth:

```
Theoretical tokens/sec at fp16:  546 / 139 = 3.9 tok/s
Theoretical tokens/sec at 4-bit: 546 / 34.75 = 15.7 tok/s
Theoretical tokens/sec at MXQ-2.5: 546 / 21.7 = 25.2 tok/s
Theoretical tokens/sec at 2-bit: 546 / 17.4 = 31.4 tok/s
```

The relationship is nearly linear: halving the bit width roughly doubles the generation speed. This is the entire reason MXQ exists. If we can make 2.5-bit quality match 4-bit quality, we get 60% faster generation and 37% less RAM usage for free.

---

## 3. Numerical Precision and Error Accumulation

### 3.1 IEEE 754 Floating Point Formats

Understanding floating point representation is essential because quantization is fundamentally about trading numerical precision for memory savings. Every floating point number is stored as:

```
value = (-1)^sign * 2^(exponent - bias) * (1 + mantissa/2^mantissa_bits)
```

The three formats relevant to LLM inference:

**float32 (FP32):**
```
Layout: [1 sign][8 exponent][23 mantissa] = 32 bits
Exponent bias: 127
Range: +/- 3.4 x 10^38
Precision: ~7.2 decimal digits (2^23 = 8,388,608 significand values)
Smallest normal: 1.18 x 10^-38
Machine epsilon: 2^-23 ~ 1.19 x 10^-7
```

**float16 (FP16):**
```
Layout: [1 sign][5 exponent][10 mantissa] = 16 bits
Exponent bias: 15
Range: +/- 65504
Precision: ~3.3 decimal digits (2^10 = 1024 significand values)
Smallest normal: 6.1 x 10^-5
Machine epsilon: 2^-10 ~ 9.77 x 10^-4
```

**bfloat16 (BF16):**
```
Layout: [1 sign][8 exponent][7 mantissa] = 16 bits
Exponent bias: 127
Range: +/- 3.4 x 10^38  (same as float32!)
Precision: ~2.4 decimal digits (2^7 = 128 significand values)
Smallest normal: 1.18 x 10^-38
Machine epsilon: 2^-7 ~ 7.81 x 10^-3
```

**Comparison table:**

| Property | FP32 | FP16 | BF16 |
|----------|------|------|------|
| Total bits | 32 | 16 | 16 |
| Exponent bits | 8 | 5 | 8 |
| Mantissa bits | 23 | 10 | 7 |
| Dynamic range | 10^38 | 65504 | 10^38 |
| Decimal precision | 7.2 | 3.3 | 2.4 |
| Machine epsilon | 1.19e-7 | 9.77e-4 | 7.81e-3 |

### 3.2 Why BF16 for Training, FP16 for Inference

**BF16 is preferred for training** because training involves gradients and loss values that can span an enormous dynamic range. Gradient magnitudes can vary by factors of 10^10 or more across layers and training stages. BF16's 8-bit exponent matches FP32's range, preventing overflow/underflow without requiring loss scaling. The sacrifice in precision (2.4 vs 3.3 decimal digits compared to FP16) is acceptable during training because stochastic gradient descent is inherently noisy.

**FP16 is preferred for inference** because:

1. During inference, all values are in a narrower, predictable range (no gradients).
2. Activations in trained models rarely exceed +/- 100, well within FP16's +/- 65504 range.
3. FP16 provides 3.3 decimal digits of precision versus BF16's 2.4, meaning less rounding error per operation.
4. For quantized inference, weights are dequantized to FP16 before computation. The extra precision reduces the total error chain.
5. Apple Silicon's Neural Engine and GPU are optimized for FP16 operations.

**Worked example -- precision difference:**

```
Value to represent: 3.14159265

FP32: 3.1415927   (error = 4 x 10^-8)
FP16: 3.140625    (error = 9.7 x 10^-4)
BF16: 3.125       (error = 1.6 x 10^-2)

For a weight value of 0.0537:
FP32: 0.05370000  (error ~ 0)
FP16: 0.05371094  (error = 1.1 x 10^-5)
BF16: 0.05468750  (error = 9.9 x 10^-4)  -- 18x worse than FP16
```

In a deep network, these per-value errors accumulate through hundreds of matmuls. The difference between FP16 and BF16 precision is significant for inference quality.

### 3.3 Error Accumulation in Deep Networks

When a quantized model runs inference, each layer introduces quantization error. The key question is: how does this error grow through N layers?

**Setup.** Let f_l denote the computation of layer l (a matmul plus nonlinearity). The full model computes:

```
y = f_N(f_{N-1}(...f_2(f_1(x))))
```

Let f_l_hat denote the quantized version of layer l. The quantized model computes:

```
y_hat = f_N_hat(f_{N-1}_hat(...f_2_hat(f_1_hat(x))))
```

The error at each layer has two components:

1. **Local quantization error**: the error introduced by using quantized weights in this layer
2. **Propagated error**: the error in the input (from all previous layers' errors) amplified by this layer

**Additive error model (best case).** If each layer introduces an independent error e_l and the layer functions have bounded Lipschitz constant L_l <= 1 (nonexpansive):

```
Total error ~ e_1 + e_2 + ... + e_N = sum(e_l)

||y - y_hat|| <= sum over l from 1 to N of: e_l
```

Growth rate: O(N). For 80 layers, total error is at most 80 times the per-layer error.

**Multiplicative error model (worst case).** If each layer amplifies the input error by a factor L_l > 1 (expansive -- which happens when weight norms are large):

```
||y - y_hat|| <= e_N + L_N * e_{N-1} + L_N * L_{N-1} * e_{N-2} + ...
             <= sum over l from 1 to N of: e_l * product over j from l+1 to N of: L_j
```

If all L_j = c > 1 (constant expansion factor):

```
||y - y_hat|| <= e * (c^N - 1) / (c - 1) ~ e * c^N
```

Growth rate: O(c^N). This is exponential. Even c = 1.01 with N = 80 gives c^80 ~ 2.2, meaning total error is 2.2 times what the additive model predicts. For c = 1.1, c^80 ~ 2000 -- catastrophic.

**Reality falls between these models.** Residual connections (the skip connections in transformers) act as error dampers:

```
Block output: out = x + f(x)
```

The identity path x passes the input through unchanged, while f(x) is the layer's transformation. If f introduces an error e, the block output error is also e (not amplified). This is why residual connections are essential for deep networks and why transformers with 80+ layers can work at all.

However, the layer norm (RMSNorm) that precedes each sub-block can amplify errors in certain directions. The net effect for a well-trained transformer is roughly:

```
Effective error growth ~ O(N * polylog(N))
```

Subexponential but superlinear. For 80 layers, expect total error to be roughly 100-200 times the per-layer quantization error. This is manageable if the per-layer error is small enough.

### 3.4 Why First and Last Layers Need More Precision

**First layers (embedding + first 2-3 transformer blocks):**

- The embedding layer converts discrete tokens into continuous vectors. Any error here propagates through all 80 subsequent layers.
- Using the multiplicative error model: error in layer 1 is amplified by L_2 * L_3 * ... * L_80. Even with residual connections, this is the maximum possible amplification chain.
- Quantization error in the embedding layer directly distorts the model's "vocabulary" of concepts. Unlike intermediate layers where errors can be partially corrected by subsequent layers, embedding errors are foundational.

**Last layers (last 2-3 transformer blocks + lm_head):**

- The lm_head maps from hidden_dim to vocab_size (e.g., 8192 -> 128256). It produces logits that are converted to probabilities via softmax.
- Softmax is exponentially sensitive to small changes in logits: P(token_i) = exp(z_i) / sum(exp(z_j)). A small error in z_i translates to a multiplicatively larger error in P(token_i).

**Worked example -- softmax sensitivity:**

```
True logits: z = [5.0, 4.0, 3.0, 2.0]
True probs:  P = [0.6439, 0.2369, 0.0871, 0.0321]

Quantization error of +0.5 on the top logit:
z_hat = [5.5, 4.0, 3.0, 2.0]
P_hat = [0.7054, 0.1573, 0.0579, 0.0213]

P(token_1) changed from 0.6439 to 0.7054 -- a 9.6% relative error
P(token_2) changed from 0.2369 to 0.1573 -- a 33.6% relative error!
```

A 10% error in one logit caused a 34% error in a probability. This is why lm_head weights are critical and MXQ assigns them a 6-bit minimum.

**Error impact by position (qualitative):**

```
Layer position:   1   2   3   ...  40  ...  78  79  80  lm_head
Error impact:    EXTREME HIGH  MED  ...  LOW  ...  MED  HIGH EXTREME EXTREME
Recommended bits: 6   5   4        2-3       3    4    5     6
```

This matches MXQ's design: first/last layer protection with +1 bit bonus, embedding at 4-bit minimum, lm_head at 6-bit minimum.

### 3.5 Mixed-Precision Accumulation

When performing a quantized matmul, the standard practice is:

```
1. Dequantize weights from N-bit integer to fp16 (fast, done per block)
2. Multiply fp16 activation * fp16 dequantized weight -> fp16 product
3. Accumulate the products in fp32 (to prevent precision loss in sums)
4. After accumulation, cast result back to fp16 for the next layer
```

**Why fp32 accumulation matters:**

Consider accumulating 8192 products (one output element of a matmul with hidden_dim=8192). Each product is roughly in the range [-0.01, +0.01] (weights and activations are small). The sum is roughly in [-1, +1].

In fp16 with machine epsilon 9.77e-4:
```
After accumulating k terms, the rounding error is approximately:
error ~ k * epsilon * |average_term| ~ 8192 * 9.77e-4 * 0.01 = 0.080

This is 8% of a typical sum value of 1.0 -- unacceptable.
```

In fp32 with machine epsilon 1.19e-7:
```
error ~ 8192 * 1.19e-7 * 0.01 = 9.75e-6

This is 0.001% of a typical sum value -- negligible.
```

The fp32 accumulation costs almost nothing on modern hardware (Apple Silicon has fp32 ALUs) but eliminates a major source of numerical error.

**In the MXQ pipeline:**
```
Quantized weight (2-8 bit integer)
  -> Dequantize: multiply by scale, add zero_point -> fp16
  -> Multiply with fp16 activation -> fp16 product
  -> Accumulate in fp32 register
  -> After full dot product: cast to fp16 for output
```

The dequantization step (integer -> fp16) is exact for the quantized values (no rounding needed since the fp16 format can represent all values in the dequantized range). The precision loss comes from the quantization itself (reducing continuous weights to discrete levels), not from the dequantization arithmetic.

### 3.6 Kahan Summation

For numerically critical reductions (such as computing the Hessian diagonal or calibration statistics), standard floating-point summation accumulates rounding error proportional to N (the number of terms). Kahan compensated summation reduces this to O(1) rounding error.

**Standard summation error:**
```
S = a_1 + a_2 + ... + a_N
Rounding error: O(N * epsilon * max|a_i|)
```

**Kahan compensated summation:**
```
S = 0
c = 0                          // compensation for lost low-order bits
for each a_i:
    y = a_i - c                // compensate
    t = S + y                  // add (some bits may be lost)
    c = (t - S) - y            // recover what was lost
    S = t
Rounding error: O(epsilon * max|a_i|)    // independent of N!
```

**Why this matters for MLXQ:**

During calibration, we accumulate activation statistics across potentially thousands of samples. For a channel with 8192 values accumulated over 1000 samples, the standard fp32 sum has error ~ 8,192,000 * 1.19e-7 ~ 0.97. Using Kahan summation reduces this to ~ 1.19e-7, which is negligible. The cost is one extra addition and three extra subtractions per accumulation step -- roughly 4x the arithmetic cost, but summation is never the bottleneck (matmuls are).

In practice, using fp64 accumulators is an alternative that achieves similar accuracy for N < 10^15, but fp64 arithmetic is slow on Apple Silicon GPUs (which lack native fp64 ALUs). Kahan summation in fp32 is the pragmatic choice.

---

## 4. The Hessian Matrix and Its Role in Quantization

### 4.1 Definition and Meaning

The Hessian matrix captures how the loss function curves around the current weights. It is the matrix of second partial derivatives of the loss L with respect to the model weights W:

```
H_ij = d^2 L / (dW_i * dW_j)
```

For a weight vector W of length d (e.g., all weights in one linear layer), H is a d x d matrix.

**What the Hessian tells us:**

- **Diagonal entries H_ii**: The curvature of the loss with respect to weight i. Large H_ii means the loss is very sensitive to changes in weight i -- this weight is "important" and should be quantized carefully (more bits).
- **Off-diagonal entries H_ij**: The interaction between weights i and j. Nonzero H_ij means that the optimal quantization of weight i depends on how weight j is quantized (they are coupled).
- **Eigenvalues of H**: The principal curvatures. Large eigenvalues indicate directions in weight space where the loss changes rapidly. Small eigenvalues indicate "flat" directions where weights can be perturbed freely.
- **H^{-1}** (inverse Hessian): Indicates how much we can perturb each weight without significantly increasing the loss. Large (H^{-1})_ii means weight i has a flat loss landscape and can tolerate large quantization errors.

### 4.2 Hessian for a Linear Layer

For a single linear layer y = xW (ignoring bias), with squared loss L = ||y - y_target||^2, the Hessian takes a particularly clean form.

Consider the layer y = xW where x is the input matrix of shape (N, d_in) (N samples, each of dimension d_in) and W is the weight matrix of shape (d_in, d_out).

Flattening W into a vector w of length d_in * d_out, the Hessian of the squared loss with respect to w is:

```
H = 2 * (X^T * X) kron I_{d_out}
```

where kron denotes the Kronecker product and X is the input data matrix. More practically, the Hessian decomposes into d_out independent blocks, each of size d_in x d_in:

```
H_j = 2 * X^T * X    for each output column j
```

This is the Gram matrix of the inputs, scaled by 2. The (i,k) entry is:

```
(H_j)_ik = 2 * sum over n from 1 to N of: x_n_i * x_n_k
```

This is proportional to the empirical covariance of the inputs. High-covariance input features contribute more to the Hessian, meaning the weights connected to those features are more sensitive.

**Worked example -- tiny linear layer:**

```
Layer: y = xW where W is 3x2 (3 input features, 2 outputs)

Input data X (4 samples):
X = [[1, 0, 2],
     [0, 1, 1],
     [1, 1, 0],
     [2, 0, 1]]

Gram matrix: X^T * X =
[[1+0+1+4, 0+0+1+0, 2+0+0+2],
 [0+0+1+0, 0+1+1+0, 0+1+0+0],
 [2+0+0+2, 0+1+0+0, 4+1+0+1]]
=
[[6, 1, 4],
 [1, 2, 1],
 [4, 1, 6]]

Hessian (for each output column): H = 2 * X^T * X =
[[12, 2, 8],
 [2,  4, 2],
 [8,  2, 12]]

Diagonal entries: H_11=12, H_22=4, H_33=12
Interpretation: Weights connected to features 1 and 3 are 3x more sensitive
than weights connected to feature 2. Feature 2 weights can tolerate
more quantization error.
```

### 4.3 Computing the Hessian Efficiently

**The full Hessian is impractical for large layers.** For a linear layer with d_in=8192 and d_out=28672, the weight vector has length 8192 * 28672 = 234,881,024. The full Hessian would be a 234M x 234M matrix -- about 220 petabytes in fp32. This is clearly impossible.

**Approach 1: Diagonal approximation**

Only compute H_ii for each weight i. This requires computing the squared input activations averaged over calibration data:

```
H_ii ~ (2/N) * sum over n from 1 to N of: x_n_i^2
```

where i indexes the input feature connected to weight i. This is simply the mean squared activation for each input channel.

Cost: O(N * d_in) -- trivially cheap.
Storage: O(d_in) per layer.

Limitation: ignores all weight interactions (off-diagonal terms). This is the approach AWQ implicitly uses.

**Approach 2: Block-diagonal (per output column)**

Compute the d_in x d_in Gram matrix X^T * X, then use it to quantize each output column independently. This is what GPTQ does.

Cost: O(N * d_in^2) to compute X^T * X.
For d_in = 8192, N = 128 calibration samples:
128 * 8192^2 = 8.6 billion multiplies ~ 8.6 GFLOPS.
This takes under a second on modern hardware.

Storage: O(d_in^2) = 67M entries ~ 256 MB per layer in fp32.
Manageable.

**Approach 3: Empirical Fisher approximation**

The Fisher information matrix F approximates the Hessian using first derivatives only:

```
F = E[g * g^T]    where g = dL/dW (gradient vector)
```

For a single sample, this is the outer product of the gradient with itself. Averaged over N samples:

```
F ~ (1/N) * sum over n from 1 to N of: g_n * g_n^T
```

The empirical Fisher is always positive semi-definite (unlike the true Hessian, which can have negative eigenvalues away from a minimum). It equals the true Hessian at the optimum for certain loss functions (cross-entropy with the true distribution).

Cost: Similar to the Gram matrix approach.
Advantage: Does not require computing second derivatives.
Disadvantage: Only an approximation; can differ significantly from the true Hessian.

### 4.4 How GPTQ Uses the Hessian

GPTQ (Generative Pre-Trained Transformer Quantization) is the most mathematically rigorous quantization method. It directly uses the inverse Hessian to optimally compensate for quantization errors.

**The GPTQ algorithm for one output column:**

Given a weight vector w (one column of W, length d_in) and the Hessian H = 2 * X^T * X for this column:

```
1. Compute Cholesky decomposition: H^{-1} = L * L^T (lower triangular)
2. For i = 1 to d_in:
   a. Quantize weight i: q_i = Quantize(w_i)     -- round to nearest quantization level
   b. Compute error: delta_i = w_i - q_i
   c. Compensate remaining weights:
      For j = i+1 to d_in:
        w_j = w_j - delta_i * (H^{-1})_ij / (H^{-1})_ii
```

**The key formula explained:**

```
w_j -= delta_i * (H^{-1})_ij / (H^{-1})_ii
```

This says: when we quantize weight i and introduce error delta_i, we should adjust all remaining weights j to compensate. The adjustment is proportional to:
- delta_i: the quantization error (larger error -> larger adjustment)
- (H^{-1})_ij: how weights i and j interact in the loss landscape
- 1/(H^{-1})_ii: inverse of how tolerant weight i is to perturbation

Intuitively: if weights i and j are correlated in their effect on the loss (large (H^{-1})_ij), then an error in i should be compensated by adjusting j. The Cholesky decomposition makes this computation efficient because we only need to look at the lower-triangular factor.

**Why Cholesky?** The Cholesky decomposition H^{-1} = L * L^T factorizes the inverse Hessian into a lower-triangular matrix L. This is useful because:
1. We process weights in order (i = 1, 2, ..., d_in)
2. Row i of L gives us all the compensation coefficients for weight i
3. The computation is O(d_in^2) total, not O(d_in^3)

**GPTQ cost:**
- Cholesky decomposition: O(d_in^3) -- done once per output column
- For d_in = 8192: 8192^3 ~ 550 billion operations
- Per output column processing: O(d_in^2) -- done d_out times
- Total per layer: O(d_in^3 + d_in^2 * d_out) ~ O(d_in^3) since d_out ~ d_in

For a 70B model with 80 layers:
- Each layer has ~4 large weight matrices
- Per matrix: ~550 billion operations for Cholesky + ~550 billion for quantization
- Total: 80 * 4 * 1.1 trillion ~ 352 trillion operations
- At 38 TFLOPS (M4 Max): ~9200 seconds ~ 2.5 hours

This is why GPTQ quantization is slow but produces the best quality at a given bit width.

### 4.5 How AWQ Approximates Without the Full Hessian

AWQ (Activation-Aware Weight Quantization) takes a much faster approach by observing that the diagonal of the Hessian is approximately proportional to the squared activation magnitudes:

```
H_ii ~ (2/N) * sum_n x_n_i^2 = 2 * E[x_i^2]
```

AWQ defines importance as:

```
importance_i = mean(|x_i|)     -- average absolute activation for channel i
```

This is a proxy for sqrt(H_ii) -- channels with large activations are connected to sensitive weights.

**AWQ's approach:**

Instead of compensating for quantization errors (like GPTQ), AWQ pre-scales the weights:

```
1. Compute per-channel importance: s_i = mean(|x_i|)
2. Find optimal scaling: alpha = argmin over alpha of: ||Q(W * diag(s^alpha)) * diag(s^{-alpha}) - W||
3. Scale weights: W_scaled = W * diag(s^alpha)
4. Quantize: W_q = Quantize(W_scaled)
5. At inference: y = x * diag(s^{-alpha}) * dequantize(W_q) * diag(s^alpha)
   (the scaling is absorbed into the computation)
```

The optimal alpha is found by grid search over [0, 1], typically alpha ~ 0.5.

**Why this works:** Multiplying a weight column by s_i^alpha makes it larger, reducing its relative quantization error. The corresponding activation is divided by s_i^alpha to compensate. Channels with large activations (high importance) get their weights scaled up more, reducing their quantization error at the cost of slightly increasing error on less important channels.

**AWQ cost:**
- Compute mean activations: O(N * d_in) per layer -- trivial
- Grid search over alpha: ~20 iterations of quantize + evaluate -- fast
- Total for 70B model: minutes, not hours

**AWQ vs. GPTQ tradeoffs:**

| Property | GPTQ | AWQ |
|----------|------|-----|
| Quality at 4-bit | Best | ~1% worse |
| Quality at 3-bit | Best | ~2-3% worse |
| Quality at 2-bit | Best | Significantly worse |
| Speed | Hours | Minutes |
| Uses off-diagonal Hessian | Yes | No |
| Compensates errors | Yes (exact) | No (pre-scaling only) |

**MXQ's approach** combines both: AWQ-style activation importance for fast per-block scoring (to decide bit allocation), with optional GPTQ-style compensation within each block (to minimize error at the allocated bit width).

---

## 5. Singular Value Decomposition Perspective

### 5.1 SVD of Weight Matrices

Every weight matrix W of shape (m x n) can be decomposed as:

```
W = U * Sigma * V^T
```

where:
- U is an m x m orthogonal matrix (columns are left singular vectors)
- Sigma is an m x n diagonal matrix (entries are singular values, sigma_1 >= sigma_2 >= ... >= 0)
- V^T is an n x n orthogonal matrix (rows are right singular vectors)

The singular values sigma_i represent the "importance" of each component of the matrix. The matrix can be written as a sum of rank-1 matrices:

```
W = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T
```

where r = rank(W) = number of nonzero singular values.

### 5.2 Low-Rank Approximation

The Eckart-Young-Mirsky theorem states that the best rank-k approximation (in Frobenius norm) is obtained by keeping only the top k singular values:

```
W_k = sigma_1 * u_1 * v_1^T + ... + sigma_k * u_k * v_k^T
```

The approximation error is:

```
||W - W_k||_F = sqrt(sigma_{k+1}^2 + sigma_{k+2}^2 + ... + sigma_r^2)
```

The fraction of "energy" (squared Frobenius norm) captured by the top k components is:

```
Energy_k = (sigma_1^2 + ... + sigma_k^2) / (sigma_1^2 + ... + sigma_r^2)
```

**Worked example -- singular value distribution of a transformer weight matrix:**

In a typical 70B model attention Q projection (8192 x 8192), the singular values follow a roughly exponential decay:

```
Rank    Singular value    Cumulative energy
1       ~15.0             ~0.5%
10      ~8.0              ~3%
100     ~3.5              ~15%
500     ~1.2              ~45%
1000    ~0.6              ~65%
2000    ~0.3              ~82%
4000    ~0.1              ~93%
8192    ~0.01             100%
```

This shows that many dimensions carry relatively little information. Roughly 50% of the "energy" is in the top 1000 singular values (12% of the rank).

### 5.3 Relationship to Quantization

SVD and quantization are complementary but distinct forms of compression:

**SVD compression:** Replace W (m x n) with U_k (m x k) and V_k (k x n). Storage goes from m*n to k*(m+n). For k < mn/(m+n), this saves space.

**Quantization:** Keep W's structure but represent each element with fewer bits. Storage goes from m*n*b_original to m*n*b_quantized.

The connection is that **weights corresponding to small singular values are less important for the model's output and can tolerate more quantization error**. This has several implications:

1. Weight blocks that align with small singular values can be safely quantized to fewer bits.
2. The "importance" of a weight block for quantization purposes is related to the singular values of the matrix region it belongs to.
3. After quantization, the effective rank of the matrix is reduced -- extreme quantization (2-bit) can only represent a limited number of distinct weight patterns, implicitly performing a form of rank reduction.

### 5.4 Why SVD Is Not Used Directly for Compression

Despite its theoretical optimality for low-rank approximation, SVD is not practical for direct LLM compression because:

1. **Computational cost:** Computing the full SVD of an (8192 x 28672) matrix requires O(min(m,n)^2 * max(m,n)) operations = O(8192^2 * 28672) ~ 1.9 trillion operations per matrix. For 320+ matrices in a 70B model, this is ~600 trillion operations.

2. **Inference overhead:** Using the factored representation W_k = U_k * V_k requires two matmuls instead of one during inference. For a rank-k approximation, the compute cost is k*(m+n) multiply-adds instead of m*n. For useful compression ratios (k < mn/(m+n)), this is cheaper in memory but comparable or worse in compute.

3. **Suboptimal for transformers:** Transformer weights are not well-approximated by low-rank matrices at useful compression ratios. To achieve 4x compression (equivalent to 4-bit quantization), we need k ~ m/8, but the top m/8 singular values typically capture only 60-80% of the energy. Quantization to 4-bit achieves better quality because it preserves the full-rank structure.

4. **Not compatible with efficient GPU execution:** Quantized matmul (dequantize + multiply + accumulate) is a single fused kernel. SVD requires two separate matmuls, each with its own memory access pattern, kernel launch overhead, and activation memory allocation.

### 5.5 LoRA Connection

LoRA (Low-Rank Adaptation) adds a low-rank update to frozen pretrained weights:

```
W_new = W_frozen + delta_W = W_frozen + A * B
```

where A is (d_in, r) and B is (r, d_out), with r << min(d_in, d_out).

This is directly an SVD-like decomposition of the weight update delta_W. LoRA works because fine-tuning updates are typically low-rank -- the model only needs to adjust a few "directions" in weight space to learn a new task.

**LoRA + quantization (QLoRA):** The frozen weights W_frozen are quantized (e.g., to 4-bit NF4), while the LoRA adapters A, B are kept in fp16/bf16. During forward pass:

```
y = x * (dequantize(W_frozen_quantized) + A * B)
  = x * dequantize(W_frozen_quantized) + (x * A) * B
```

This is complementary to MLXQ: a model could be quantized with MXQ for efficient inference, and then LoRA-adapted for specific tasks with the LoRA weights stored separately at full precision.

---

## 6. Weight Distribution Statistics in Transformers

### 6.1 Empirical Observations

Understanding the actual statistical distribution of weights in trained transformers is critical for designing quantization schemes. All quantization methods implicitly assume something about the weight distribution.

**Key observation 1: Approximately Gaussian.** Most weight matrices in trained transformers have elements that are approximately normally distributed with zero mean:

```
W_ij ~ N(0, sigma^2)
```

where sigma varies by layer and component. For a 70B model:

```
Component                  Typical sigma     Typical range (99.7%)
Embedding                  0.01 - 0.02       [-0.06, 0.06]
Attention Q/K              0.005 - 0.01      [-0.03, 0.03]
Attention V/O              0.005 - 0.015     [-0.045, 0.045]
MLP gate/up                0.005 - 0.02      [-0.06, 0.06]
MLP down                   0.005 - 0.015     [-0.045, 0.045]
lm_head                    0.01 - 0.03       [-0.09, 0.09]
```

**Key observation 2: NOT exactly Gaussian.** The tails of the distribution are heavier than a Gaussian (leptokurtic). This means:
- More weights near zero than a Gaussian predicts (tall, narrow peak)
- More extreme values than a Gaussian predicts (heavy tails)
- The excess kurtosis is typically 1-5 (Gaussian has kurtosis 3, excess kurtosis 0)

This has a direct impact on quantization: uniform quantization (equal spacing of quantization levels) wastes bits on the sparse tails, while the dense center needs finer granularity.

**Key observation 3: Outlier channels.** Some channels (rows or columns of weight matrices) have weights that are 10-100x larger than the average. These "outlier channels" are:
- Consistent across samples (always the same channels)
- Critical for model quality (removing them degrades output severely)
- More prevalent in larger models (the "emergent outlier" phenomenon)
- Concentrated in certain layers (especially MLP gate/up projections and attention V projections)

### 6.2 Layer-by-Layer Distribution Differences

Different components of a transformer have qualitatively different weight distributions, which has direct implications for how MXQ should allocate bits.

**Embedding layer:**
```
Shape: (vocab_size, d_model) = (128256, 8192)
Distribution: Not continuous. Many rows are near-zero (rare tokens), others have
large magnitudes (common tokens). The distribution has multiple modes.
Kurtosis: High (10-20). Many near-zero values interspersed with significant values.
Quantization impact: Errors in common-token embeddings are catastrophic.
                     Errors in rare-token embeddings are usually harmless.
Recommended: 4-bit minimum. Ideally 6-bit for common tokens, 2-bit for rare tokens.
```

**Attention Q/K projections:**
```
Shape: (d_model, d_model) = (8192, 8192) for Q; (8192, d_kv) for K
Distribution: Relatively uniform, moderate range, close to Gaussian.
Kurtosis: Low (3-5). Close to normal distribution.
Outliers: Rare and small in magnitude.
Quantization impact: Q and K interact to produce attention scores.
                     Errors in Q/K cause attention to attend to wrong positions.
                     Relative errors between Q and K matter more than absolute errors.
Recommended: 3-4 bits for Q (larger matrix, more total parameters).
             3-4 bits for K (smaller with GQA, but sensitive due to attention interaction).
```

**Attention V/O projections:**
```
Shape: (d_model, d_kv) for V; (d_model, d_model) for O
Distribution: More variation than Q/K. Some outlier channels.
Kurtosis: Moderate (5-8).
Outliers: Present in V projection, especially in deeper layers.
Quantization impact: V determines what information is read from each position.
                     O is the output projection -- errors here directly affect the
                     residual stream.
Recommended: 3-4 bits for V. 3-4 bits for O.
```

**MLP gate/up projections:**
```
Shape: (d_model, d_ff) = (8192, 28672)
Distribution: Largest weight magnitudes in the model. Most outliers.
              The gating mechanism creates a bimodal activation pattern
              that puts unusual demands on the weight distribution.
Kurtosis: High (8-15). Heavy tails, many small values near zero.
Outliers: 0.1-1% of channels can have weights 10-100x the median.
          These are the most problematic for quantization.
Quantization impact: The gate projection controls information flow through the MLP.
                     Errors in gate weights cause wrong features to be amplified/suppressed.
                     However, these are also the largest matrices (most parameters),
                     so they dominate the total model size.
Recommended: 2-3 bits for most blocks. 4-6 bits for outlier blocks.
             This is where MXQ's mixed precision has the most impact.
```

**MLP down projection:**
```
Shape: (d_ff, d_model) = (28672, 8192)
Distribution: Moderate. Fewer outliers than gate/up.
Kurtosis: Moderate (5-8).
Quantization impact: Projects back to residual stream. Errors here add to the
                     residual which persists through all subsequent layers.
Recommended: 2-3 bits for most blocks. 3-4 bits for outlier blocks.
```

**lm_head (output projection):**
```
Shape: (d_model, vocab_size) = (8192, 128256)
Distribution: Similar to embedding (often weight-tied with embedding).
              Discrete-like for rare tokens, significant for common tokens.
Kurtosis: High.
Quantization impact: EXTREME. Maps directly to logits.
                     Softmax exponentially amplifies errors (see Section 3.4).
Recommended: 6-bit minimum. 8-bit for critical columns.
```

### 6.3 Kurtosis and Skewness: Implications for Quantization

**Kurtosis** measures how heavy-tailed a distribution is. For a random variable X with mean mu and standard deviation sigma:

```
kurtosis = E[(X - mu)^4] / sigma^4
excess_kurtosis = kurtosis - 3     (normal distribution has excess kurtosis = 0)
```

High kurtosis (leptokurtic, excess > 0) means:
- More probability mass in the tails and center, less in the "shoulders"
- Outliers are more common than in a Gaussian distribution
- Uniform quantization is suboptimal because the quantization levels should be denser near zero and sparser in the tails

**Skewness** measures asymmetry:

```
skewness = E[(X - mu)^3] / sigma^3
```

Most transformer weight distributions have near-zero skewness (symmetric around zero), but some layers show slight positive or negative skewness due to learned biases in the representation.

**Impact on quantization design:**

For a distribution with excess kurtosis K_e > 0, the information-theoretically optimal quantizer (Lloyd-Max quantizer) allocates quantization levels non-uniformly:

```
For K_e = 0 (Gaussian): levels are approximately equally spaced in [-3sigma, 3sigma]
For K_e = 3 (typical transformer weights): more levels near 0, fewer in tails
For K_e = 10 (outlier-heavy layers): many levels near 0, a few widely-spaced levels for tails
```

This is one reason why NormalFloat (NF4) quantization (used in QLoRA) outperforms standard uniform integer quantization: NF4 places quantization levels at the quantiles of a normal distribution, effectively accounting for the approximately Gaussian shape.

MXQ handles kurtosis differently: instead of non-uniform quantization levels, it uses uniform levels within each block but allocates more bits to blocks that contain outliers (the blocks responsible for the heavy tails). This is simpler to implement in hardware (uniform dequantization per block) while still adapting to the distribution shape.

### 6.4 The Emergent Outlier Phenomenon

One of the most important empirical discoveries in LLM quantization is that larger models develop extreme outlier features. First documented by Dettmers et al. (LLM.int8()), this phenomenon has the following characteristics:

**Scale with model size:**
```
Model size       Max outlier magnitude / median weight     % of channels with outliers
125M             ~5x                                       ~0.01%
1.3B             ~10x                                      ~0.05%
6.7B             ~20x                                      ~0.1%
13B              ~40x                                      ~0.3%
30B              ~60x                                      ~0.5%
65B+             ~100x                                     ~1%
```

**Consistency:** The outlier channels are the same regardless of the input. They appear to be structural features of the trained model, not data-dependent.

**Spatial pattern:** Outliers tend to appear in the same channel indices across multiple layers, suggesting they correspond to specific "features" that the model has learned to use with large magnitudes.

**Impact on quantization:** If a block of 64 weights contains even one outlier that is 100x the median, the quantization scale for that block must accommodate the outlier, leaving only a fraction of the quantization levels for the remaining 63 normal-magnitude weights. This is the primary mechanism by which uniform quantization fails at low bit widths.

**Example:**

```
Block of 64 weights (before quantization):
[0.01, -0.02, 0.015, ..., 0.008, -0.012, ..., 3.5, ..., -0.01]

With 2-bit quantization (4 levels):
Scale = max(|block|) / 1.5 = 3.5 / 1.5 = 2.333
Levels: {-3.5, -1.167, 1.167, 3.5}

The outlier (3.5) is perfectly represented.
But all 63 other weights (range [-0.02, 0.02]) are rounded to the nearest
level: either -1.167 or 1.167 -- completely wrong!

Mean quantization error for non-outlier weights: ~1.17
Mean weight magnitude: ~0.012
Relative error: ~97x -- the quantized values are 100x wrong.
```

MXQ addresses this by giving the block containing the outlier more bits (e.g., 6-bit = 64 levels), allowing both the outlier and the normal weights to be represented accurately, while other blocks without outliers can use 2 bits safely.

---

## 7. Optimal Bit Allocation Theory (Rate-Distortion)

### 7.1 Rate-Distortion Function

Rate-distortion theory, from Shannon's information theory, provides the theoretical foundation for optimal quantization. It answers: given a source with known statistics, what is the minimum number of bits needed to achieve a given distortion?

**Definition.** For a continuous source X with probability density f(x), the rate-distortion function R(D) is:

```
R(D) = min over p(x_hat|x) of: I(X; X_hat)

subject to: E[d(X, X_hat)] <= D
```

where I(X; X_hat) is the mutual information between the source X and its reconstruction X_hat, and d(x, x_hat) is the distortion measure (typically squared error: d(x, x_hat) = (x - x_hat)^2).

### 7.2 Gaussian Source

For a Gaussian source X ~ N(0, sigma^2) with mean squared error distortion, the rate-distortion function has a closed-form solution:

```
R(D) = max(0, 0.5 * log2(sigma^2 / D))
```

Equivalently, for a given rate R (bits per sample):

```
D(R) = sigma^2 * 2^(-2R)
```

**Interpretation:** Each additional bit reduces the distortion by a factor of 4 (6 dB in signal-to-noise ratio).

**Worked example:**

```
Source: X ~ N(0, 1), so sigma^2 = 1

At 1 bit/sample: D = 1 * 2^(-2) = 0.25    (SNR = 1/0.25 = 4 = 6 dB)
At 2 bits/sample: D = 1 * 2^(-4) = 0.0625  (SNR = 16 = 12 dB)
At 3 bits/sample: D = 1 * 2^(-6) = 0.0156  (SNR = 64 = 18 dB)
At 4 bits/sample: D = 1 * 2^(-8) = 0.0039  (SNR = 256 = 24 dB)

Source: X ~ N(0, 0.01), so sigma^2 = 0.01

At 2 bits/sample: D = 0.01 * 2^(-4) = 6.25e-4   (SNR = 16 = 12 dB)
At 4 bits/sample: D = 0.01 * 2^(-8) = 3.91e-5   (SNR = 256 = 24 dB)

The distortion is proportional to sigma^2 -- sources with smaller variance
need fewer bits for the same SNR.
```

### 7.3 Mixed-Precision Allocation Across Blocks

Now consider N blocks (weight groups), each with variance sigma_i^2, and a total bit budget B. We want to allocate b_i bits to block i such that:

```
sum over i from 1 to N of: b_i = B     (total budget constraint)
b_i >= 0 for all i                       (non-negativity)
```

and the total distortion is minimized:

```
Total distortion D = sum over i from 1 to N of: sigma_i^2 * 2^(-2 * b_i)
```

**Optimal allocation (Lagrange multiplier method):**

Set up the Lagrangian:

```
L = sum_i sigma_i^2 * 2^(-2*b_i) + lambda * (sum_i b_i - B)
```

Take derivative with respect to b_i and set to zero:

```
dL/db_i = -2 * ln(2) * sigma_i^2 * 2^(-2*b_i) + lambda = 0
sigma_i^2 * 2^(-2*b_i) = lambda / (2 * ln(2))     -- same for all i
```

This gives:

```
2^(-2*b_i) = C / sigma_i^2     (C is a constant)
b_i = 0.5 * log2(sigma_i^2 / C)
```

Using the budget constraint sum_i b_i = B:

```
sum_i 0.5 * log2(sigma_i^2 / C) = B
0.5 * log2(product_i sigma_i^2 / C^N) = B
0.5 * (sum_i log2(sigma_i^2) - N * log2(C)) = B
```

Solving for C and substituting back:

```
b_i = B/N + 0.5 * log2(sigma_i^2 / G^2)
```

where G^2 is the geometric mean of all variances:

```
G^2 = (product over i from 1 to N of: sigma_i^2)^{1/N}
```

**The optimal bit allocation formula:**

```
b_i = B/N + 0.5 * log2(sigma_i^2 / (product_j sigma_j^2)^{1/N})
```

**Interpretation:**
- B/N is the average allocation (if all blocks were equal, everyone gets the average)
- The second term adjusts for variance differences
- Blocks with variance above the geometric mean get MORE bits (positive adjustment)
- Blocks with variance below the geometric mean get FEWER bits (negative adjustment)
- Each doubling of variance adds 0.5 bits

### 7.4 Reverse Water-Filling

The optimal allocation can yield b_i < 0 for blocks with very small variance, which is not physically meaningful (we cannot use negative bits). The solution is "reverse water-filling":

```
Algorithm: Reverse Water-Filling
1. Compute optimal b_i for all blocks using the formula above
2. If any b_i < b_min (minimum bit width, e.g., 2):
   a. Set those blocks to b_min
   b. Remove them from the optimization
   c. Reduce the remaining budget: B' = B - count(clamped) * b_min
   d. Re-solve for the remaining blocks with budget B'
   e. Repeat until all b_i >= b_min
```

**Worked example -- 5 blocks, budget = 15 bits (average 3 bits/block):**

```
Block variances: sigma^2 = [0.01, 0.04, 0.01, 0.16, 0.01]

Geometric mean: G^2 = (0.01 * 0.04 * 0.01 * 0.16 * 0.01)^{1/5}
                    = (6.4e-10)^{0.2}
                    = 0.0229

Optimal allocation (unconstrained):
b_1 = 3 + 0.5 * log2(0.01 / 0.0229) = 3 + 0.5 * (-1.196) = 3 - 0.598 = 2.40
b_2 = 3 + 0.5 * log2(0.04 / 0.0229) = 3 + 0.5 * (0.804)  = 3 + 0.402 = 3.40
b_3 = 3 + 0.5 * log2(0.01 / 0.0229) = 3 + 0.5 * (-1.196) = 3 - 0.598 = 2.40
b_4 = 3 + 0.5 * log2(0.16 / 0.0229) = 3 + 0.5 * (2.804)  = 3 + 1.402 = 4.40
b_5 = 3 + 0.5 * log2(0.01 / 0.0229) = 3 + 0.5 * (-1.196) = 3 - 0.598 = 2.40

Check: 2.40 + 3.40 + 2.40 + 4.40 + 2.40 = 15.0 (correct)

With minimum 2 bits: all b_i >= 2.0, so no clamping needed.

Result: Block 4 (highest variance) gets 4.4 bits.
        Blocks 1, 3, 5 (lowest variance) get 2.4 bits.
        Block 2 (moderate variance) gets 3.4 bits.
```

Since practical bit widths are integers (or at least from a discrete set like {2, 3, 4, 5, 6, 8}), the real allocation rounds to the nearest available bit width and then adjusts the budget iteratively. MXQ's allocate.py implements this rounding with a greedy algorithm that iteratively upgrades the most-underserved block.

### 7.5 This Is the Theoretical Foundation for MXQ

MXQ's bit allocation algorithm is a practical implementation of this rate-distortion theory:

1. **Calibration** (Phase 1) measures the "variance" of each block -- but instead of simple statistical variance, it uses activation-weighted importance (a better proxy for distortion than raw variance).

2. **Bit allocation** (Phase 2) solves the constrained optimization problem: minimize total distortion subject to a total bit budget.

3. **The formula** is essentially the rate-distortion optimal allocation, but with importance scores replacing variances:

```
MXQ allocation: b_i ~ B/N + 0.5 * log2(importance_i / geometric_mean(importance))
```

### 7.6 Sensitivity-Weighted Distortion

When the Hessian (sensitivity) information is available, the distortion function changes from raw squared error to sensitivity-weighted squared error:

```
Weighted distortion: D_weighted = sum_i h_i * sigma_i^2 * 2^(-2*b_i)
```

where h_i is the sensitivity (Hessian diagonal) for block i. The optimal allocation becomes:

```
b_i = B/N + 0.5 * log2(h_i * sigma_i^2 / (product_j (h_j * sigma_j^2))^{1/N})
```

**Interpretation:** Blocks get more bits when:
- Their weights have high variance (large sigma_i^2) -- more values to represent
- Their weights are sensitive (large h_i) -- errors cost more
- The product h_i * sigma_i^2 combines both factors

**Worked example -- sensitivity matters more than variance:**

```
Block A: sigma^2 = 0.01, h = 100 (small variance, very sensitive)
Block B: sigma^2 = 0.10, h = 1   (large variance, insensitive)

Weighted importance:
Block A: h * sigma^2 = 100 * 0.01 = 1.0
Block B: h * sigma^2 = 1 * 0.10 = 0.1

Despite having 10x less variance, Block A gets more bits because
its 100x sensitivity makes it 10x more important overall.

With B/N = 3 bits average:
G^2 = (1.0 * 0.1)^{0.5} = 0.316
b_A = 3 + 0.5 * log2(1.0 / 0.316) = 3 + 0.5 * 1.66 = 3.83
b_B = 3 + 0.5 * log2(0.1 / 0.316) = 3 + 0.5 * (-1.66) = 2.17

Block A: 3.83 bits (rounds to 4)
Block B: 2.17 bits (rounds to 2)
```

This is precisely the scenario in MLXQ: MLP gate projections have low variance but high sensitivity (through the gating mechanism), while MLP up projections have higher variance but lower sensitivity. The sensitivity-weighted allocation gives gate projections more bits than raw variance alone would suggest.

---

## 8. Practical Matmul Performance on Apple Silicon

### 8.1 The Roofline Model

The roofline model characterizes the maximum achievable performance of a computation based on two hardware limits:

```
Performance = min(Peak_Compute, Arithmetic_Intensity * Peak_Bandwidth)
```

where:
- **Peak_Compute**: maximum operations per second (e.g., 38 TFLOPS for M4 Max fp16)
- **Peak_Bandwidth**: maximum bytes per second from memory (e.g., 546 GB/s for M4 Max)
- **Arithmetic_Intensity (AI)**: operations per byte of memory accessed (ops/byte)

The **ridge point** is the arithmetic intensity where the two limits are equal:

```
Ridge_AI = Peak_Compute / Peak_Bandwidth
```

Below the ridge point, the computation is **memory-bandwidth-bound** (performance limited by how fast data can be read). Above, it is **compute-bound** (limited by how fast the ALU can process data).

**Apple Silicon ridge points:**

| Chip | Peak FP16 (TFLOPS) | Bandwidth (GB/s) | Ridge AI (ops/byte) |
|------|-------|-----------|----------|
| M1 Max | 10.4 | 400 | 26 |
| M2 Ultra | 27.2 | 800 | 34 |
| M3 Max | 14.2 | 400 | 36 |
| M4 | 3.3 | 120 | 28 |
| M4 Pro | 8.7 | 273 | 32 |
| M4 Max | 38.0 | 546 | 70 |
| M4 Ultra (est) | 76.0 | 1092 | 70 |

### 8.2 Arithmetic Intensity of Quantized Matmul

For a matrix-vector product y = x * W (decode phase, batch=1):

```
Operations: 2 * d_in * d_out   (multiply-add counts as 2 ops for FLOP counting)

Memory reads:
  - Weight matrix W: d_in * d_out * bytes_per_weight
  - Input vector x:  d_in * 2 bytes (fp16)
  - Scales/zeros:    (d_in * d_out / block_size) * 4 bytes (2 fp16 values per block)

Memory writes:
  - Output vector y: d_out * 2 bytes (fp16)

For large matrices (d_in, d_out >> block_size), the dominant term is the weight read.
```

Arithmetic intensity:

```
AI = 2 * d_in * d_out / (d_in * d_out * bytes_per_weight + d_in * 2 + d_out * 2 + overhead)
```

For large dimensions, d_in * 2 and d_out * 2 are negligible compared to d_in * d_out * bytes_per_weight:

```
AI ~ 2 / bytes_per_weight
```

**Arithmetic intensity by quantization format:**

| Format | Bytes per weight | AI (ops/byte) | Regime on M4 Max |
|--------|-----------------|--------------|------------------|
| FP32 | 4.0 | 0.5 | Memory-bound (0.5 << 70) |
| FP16/BF16 | 2.0 | 1.0 | Memory-bound (1.0 << 70) |
| 8-bit | 1.0 | 2.0 | Memory-bound |
| 4-bit | 0.5 | 4.0 | Memory-bound |
| 3-bit | 0.375 | 5.3 | Memory-bound |
| MXQ-2.5 | 0.3125 | 6.4 | Memory-bound |
| 2-bit | 0.25 | 8.0 | Memory-bound |
| 1-bit (binary) | 0.125 | 16.0 | Memory-bound (approaching ridge) |

**Critical insight:** Even at 2-bit quantization, the arithmetic intensity (8 ops/byte) is far below the ridge point of M4 Max (70 ops/byte). This means:

1. **Decode-phase inference is always memory-bandwidth-bound** on Apple Silicon at batch=1, regardless of quantization format.
2. The dequantization overhead (converting integers to fp16) adds compute but not memory access, so it does not change the fundamental bottleneck.
3. **Reducing bit width directly translates to proportionally faster inference** because the only thing that matters is how quickly we can read weights from memory.

### 8.3 Tokens Per Second: The Key Formula

For autoregressive generation at batch_size=1, the time to generate one token is dominated by reading all weight parameters from memory:

```
time_per_token = total_weight_bytes / memory_bandwidth
tokens_per_second = memory_bandwidth / total_weight_bytes
```

This ignores attention computation (KV cache lookups), which is small for moderate sequence lengths, and overhead (kernel launch, synchronization), which is a few percent.

**Total weight bytes:**

```
total_weight_bytes = total_parameters * bits_per_weight / 8
```

For a 70B model:

| Format | Bits | Total weight bytes | Theoretical tok/s (M4 Max) |
|--------|------|-------------------|---------------------------|
| FP16 | 16 | 139 GB | 3.9 |
| 8-bit | 8 | 69.5 GB | 7.9 |
| 4-bit | 4 | 34.75 GB | 15.7 |
| 3-bit | 3 | 26.1 GB | 20.9 |
| MXQ-2.5 | 2.5 | 21.7 GB | 25.2 |
| 2-bit | 2 | 17.4 GB | 31.4 |

**This is the entire economic argument for MXQ.** At 2.5-bit average with 4-bit quality, MXQ delivers 60% more tokens per second than uniform 4-bit quantization.

### 8.4 Worked Example: M4 Max with 70B Model

**Setup:**
```
Chip: M4 Max (16-core GPU, 40-core GPU variant)
Memory bandwidth: 546 GB/s
FP16 compute: 38 TFLOPS
Memory: 128 GB unified

Model: Llama-3-70B (69.5B parameters)
```

**Case 1: 4-bit uniform quantization**
```
Weight storage: 69.5e9 * 0.5 bytes = 34.75 GB
Quantization metadata (scales/zeros): ~0.5 GB (at block_size=64, fp16 scale+zero)
Total: ~35.25 GB
Fits in memory: Yes (128 GB)
Remaining for KV cache: 128 - 35.25 = 92.75 GB

Time per token: 35.25 GB / 546 GB/s = 64.6 ms
Tokens per second: 15.5 tok/s

Effective compute utilization:
  Operations per token: 139 GFLOP
  Time: 64.6 ms
  Achieved compute: 139 / 0.0646 = 2.15 TFLOPS
  Utilization: 2.15 / 38 = 5.7%
  (Very low -- all that GPU compute is idle, waiting for memory)
```

**Case 2: MXQ-2.5bit**
```
Weight storage: 69.5e9 * 0.3125 bytes = 21.72 GB
Quantization metadata: ~1.0 GB (more metadata due to variable bit widths per block)
  Per-block: scale (fp16, 2B) + zero (fp16, 2B) + bit_width (uint8, 1B) = 5B per block
  Blocks: 69.5e9 / 64 ~ 1.086e9 blocks * 5B = 5.43 GB
  Wait -- this is too much. At block_size=64, metadata is 5B per 64 weights.
  That is 5/64 = 0.078 bytes per weight = 0.625 bits per weight overhead.
  Need to amortize with larger blocks or compressed bit_width maps.

  In practice (block_size=64, shared bit_width per larger group):
  Metadata: ~1.5 GB
Total: ~23.2 GB
Fits in memory: Yes (128 GB)
Remaining for KV cache: 128 - 23.2 = 104.8 GB

Time per token: 23.2 GB / 546 GB/s = 42.5 ms
Tokens per second: 23.5 tok/s

Speedup vs 4-bit: 23.5 / 15.5 = 1.52x (52% faster)
RAM savings: 35.25 - 23.2 = 12.05 GB saved (34% less)
```

**Case 3: MXQ-2.5bit on M4 Max 36GB (32GB usable)**
```
Can the model fit at all?
  Model weights + metadata: 23.2 GB
  OS and system: ~4 GB
  KV cache needed: ~2 GB (for 4K context)
  Total: ~29.2 GB

  Fits in 36 GB? Yes, barely.
  This is the key use case: 70B model on a 36GB Mac that cannot run 4-bit (35.25 GB model alone).
```

### 8.5 The Prefill Regime

During prefill, the same weights are used for all tokens in the prompt simultaneously. This changes the arithmetic intensity:

```
AI_prefill = 2 * batch * seq_len * d_in * d_out / (d_in * d_out * bytes_per_weight + batch * seq_len * d_in * 2)
```

For seq_len = 2048, the weight read is amortized across 2048 tokens:

```
AI_prefill ~ 2 * seq_len / bytes_per_weight (when seq_len * d_in >> d_in * d_out / bytes_per_weight ... this doesn't hold)
```

More precisely:

```
AI_prefill = 2 * seq_len / (bytes_per_weight + 2 * seq_len / d_out)
```

For seq_len = 2048, bytes_per_weight = 0.5 (4-bit), d_out = 8192:

```
AI_prefill = 2 * 2048 / (0.5 + 2 * 2048 / 8192)
           = 4096 / (0.5 + 0.5)
           = 4096 / 1.0
           = 4096 ops/byte
```

This is far above the ridge point (70 ops/byte for M4 Max). Prefill is compute-bound, not memory-bound. Therefore, quantization does not significantly speed up prefill -- it only reduces memory usage.

This is why quantization primarily affects generation speed (decode), not prompt processing speed (prefill).

### 8.6 Quantization Metadata Overhead

An often-overlooked cost in mixed-precision quantization is the per-block metadata. Each block of weights needs:

```
Per block metadata:
  - scale: 1 fp16 value = 2 bytes
  - zero_point: 1 fp16 value = 2 bytes
  - bit_width: 1 uint8 value = 1 byte (MXQ-specific, not needed in uniform quantization)
  Total: 5 bytes per block
```

At block_size = 64:
```
Metadata overhead = 5 bytes / 64 weights = 0.078 bytes/weight = 0.625 bits/weight
```

At block_size = 32:
```
Metadata overhead = 5 bytes / 32 weights = 0.156 bytes/weight = 1.25 bits/weight
```

At block_size = 128:
```
Metadata overhead = 5 bytes / 128 weights = 0.039 bytes/weight = 0.3125 bits/weight
```

**Impact on effective bit width:**

```
For MXQ-2.5 with block_size=64:
  Effective bits = 2.5 (weight data) + 0.625 (metadata) = 3.125 bits/weight

For MXQ-2.5 with block_size=128:
  Effective bits = 2.5 (weight data) + 0.3125 (metadata) = 2.8125 bits/weight
```

This means the practical bit width is higher than the nominal bit width. Larger block sizes reduce overhead but also reduce the granularity of the quantization (fewer blocks means less adaptive allocation). The MXQ plan specifies block_size=64, which is a reasonable tradeoff: 0.625 bits overhead is significant but manageable at the 2.5-bit target.

**Optimization: compress the bit_width map.** Since bit_widths come from a small set {2, 3, 4, 5, 6, 8}, they can be encoded in 3 bits instead of 8, or run-length encoded if adjacent blocks tend to have the same bit width. This reduces the per-block overhead from 5 bytes to 4.375 bytes (saving 0.1 bits/weight at block_size=64) -- a minor improvement.

A more impactful optimization is to assign bit widths at a coarser granularity (e.g., groups of 4 or 8 blocks share a bit width) while keeping the scale/zero per block. This reduces the bit_width overhead to nearly zero at the cost of some allocation flexibility.

### 8.7 Dequantization Compute Overhead

Reading the weights from memory is the bottleneck, but the dequantization itself has a nonzero compute cost:

```
Per weight dequantization:
  1. Read packed integer value (bit extraction): 2-3 integer ops (shift, mask)
  2. Cast to fp16: 1 op
  3. Multiply by scale: 1 fp16 multiply
  4. Add zero point: 1 fp16 add
  Total: ~5-6 ops per weight
```

For MXQ with variable bit widths, there is additional overhead:
```
  5. Read bit_width for this block: 1 memory read (cached per block)
  6. Compute bit offset: 1-2 integer ops
  Total: ~7-8 ops per weight
```

Compare to the matmul itself: 2 ops per weight (1 multiply + 1 add for the dot product). The dequantization adds ~3-4x the raw matmul compute, but since we are memory-bound (not compute-bound), this does not affect throughput as long as the dequantization can be done while waiting for the next memory read.

**Can the GPU dequantize fast enough?**

```
M4 Max: 546 GB/s bandwidth, 38 TFLOPS compute
At 2.5-bit (0.3125 bytes/weight):
  Weights read per second: 546e9 / 0.3125 = 1.747e12 weights/sec
  Dequant ops needed: 1.747e12 * 8 = 14.0 TFLOPS (for 8 ops/weight)
  Available compute: 38 TFLOPS
  Headroom: 38 / 14 = 2.7x

At 2-bit (0.25 bytes/weight):
  Weights read per second: 546e9 / 0.25 = 2.184e12 weights/sec
  Dequant ops needed: 2.184e12 * 8 = 17.5 TFLOPS
  Available compute: 38 TFLOPS
  Headroom: 38 / 17.5 = 2.17x
```

Even at 2-bit, the GPU has 2.17x more compute than needed for dequantization. The dequantization is comfortably within the compute budget. This confirms that the MXQ Metal kernels will not create a computational bottleneck, even with the additional variable-bitwidth logic.

However, the headroom shrinks at lower bit widths, and a poorly-optimized dequantization kernel could become the bottleneck. The kernel must:
- Coalesce memory reads (all threads in a SIMD group read adjacent memory)
- Minimize divergence (avoid per-thread branching on bit_width within a SIMD group)
- Use the SIMD shuffle/broadcast for sharing block metadata (scale, zero, bit_width) across threads

### 8.8 Summary: The Numbers That Matter

```
For a 70B model on M4 Max (546 GB/s):

Format         Weight size   tok/s (theory)   RAM needed   Quality
-----------    -----------   ---------------  ----------   --------
FP16           139.0 GB      3.9              139+ GB      Baseline
8-bit          69.5 GB       7.9              70+ GB       ~Lossless
4-bit          34.75 GB      15.7             35+ GB       Good
MXQ-3.0        26.1 GB       20.9             27+ GB       Near 4-bit
MXQ-2.5        21.7 GB       25.2             23+ GB       Matches 4-bit
2-bit uniform  17.4 GB       31.4             18+ GB       Garbage
MXQ-2.0        17.4 GB       31.4             18+ GB       Usable*

*MXQ-2.0 is a stretch target. Quality at 2.0 average bits depends heavily
on the model and how well the importance-weighted allocation works.
MXQ-2.5 is the sweet spot.

Key relationship:
  tokens_per_second ~ memory_bandwidth / (parameters * bits / 8)
  Halving the bits roughly doubles the speed.
  This is why MXQ exists.
```

---

## Appendix A: Notation Reference

| Symbol | Meaning |
|--------|---------|
| W | Weight matrix of a linear layer |
| x | Input activation tensor |
| y | Output activation tensor |
| d_model | Hidden dimension of the transformer (e.g., 8192) |
| d_ff | Feed-forward intermediate dimension (e.g., 28672) |
| d_kv | Key/value dimension = n_kv_heads * head_dim |
| n_heads | Number of attention heads |
| n_kv_heads | Number of key/value heads (GQA) |
| head_dim | Dimension per attention head (e.g., 128) |
| N | Number of transformer layers |
| Q, K, V | Query, Key, Value matrices in attention |
| H | Hessian matrix: d^2L/dW^2 |
| L | Loss function |
| sigma^2 | Variance of a weight distribution |
| b_i | Bit allocation for block i |
| B | Total bit budget |
| R(D) | Rate-distortion function |
| AI | Arithmetic intensity (ops/byte) |
| kron | Kronecker product |

## Appendix B: Key Formulas Quick Reference

**Matrix multiplication complexity:**
```
C = A x B where A is (n x k), B is (k x m)
Cost: n * m * k multiply-adds = 2*n*m*k FLOPs
```

**Rate-distortion for Gaussian source:**
```
R(D) = 0.5 * log2(sigma^2 / D)
D(R) = sigma^2 * 2^(-2R)
```

**Optimal bit allocation across blocks:**
```
b_i = B/N + 0.5 * log2(sigma_i^2 / (product_j sigma_j^2)^{1/N})
```

**Sensitivity-weighted allocation:**
```
b_i = B/N + 0.5 * log2(h_i * sigma_i^2 / (product_j h_j * sigma_j^2)^{1/N})
```

**GPTQ weight compensation:**
```
w_j -= delta_i * (H^{-1})_ij / (H^{-1})_ii
```

**Tokens per second (batch=1):**
```
tok/s = memory_bandwidth / (total_parameters * bits_per_weight / 8)
```

**Arithmetic intensity (decode, batch=1):**
```
AI ~ 2 / bytes_per_weight
```

**Error accumulation (additive model):**
```
||y - y_hat|| <= sum_{l=1}^{N} e_l
```

**Error accumulation (multiplicative model):**
```
||y - y_hat|| <= sum_{l=1}^{N} e_l * product_{j=l+1}^{N} L_j
```

---

*This document provides the mathematical foundation for all phases of MXQ implementation. The formulas and analysis here directly inform the design of the calibration engine (Phase 1), bit allocation algorithm (Phase 2), and Metal dequantization kernels (Phase 4). Every design decision in MXQ -- from block sizes to minimum bit widths to layer-type priors -- can be traced back to the mathematics described above.*
