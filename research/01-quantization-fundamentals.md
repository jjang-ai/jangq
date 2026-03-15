# Quantization Fundamentals

> MXQ Research Document 01 -- Mathematical foundations for mixed-precision quantization.
> All formulations in this document apply directly to the MXQ pipeline: calibrate, score, allocate, quantize, pack.

---

## 1. What Is Quantization Mathematically

Quantization is the process of mapping a continuous or high-precision set of values to a smaller, discrete set. In the context of neural network weight compression, we map floating-point weights (typically float16 or bfloat16, with 65,536 distinct values) down to a small integer set (e.g., 16 values for 4-bit).

### 1.1 The Core Mapping Function

The fundamental quantization operation has two phases: **quantize** (compress) and **dequantize** (reconstruct).

**Quantize** (float to integer):

$$q = \text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil + z,\; 0,\; 2^N - 1\right)$$

**Dequantize** (integer back to float):

$$\hat{w} = s \cdot (q - z)$$

Where:
- $w$ is the original float weight
- $q$ is the quantized integer representation
- $s$ is the scale factor (a positive float)
- $z$ is the zero-point (an integer offset)
- $N$ is the bit width
- $\lfloor \cdot \rceil$ denotes rounding to the nearest integer
- $\text{clamp}(x, a, b) = \min(\max(x, a), b)$

The combined quantize-then-dequantize operation is what we call $Q(w)$:

$$Q(w) = s \cdot \left(\text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil + z,\; 0,\; 2^N - 1\right) - z\right)$$

The key insight: $Q(w) \neq w$ in general. The difference $w - Q(w)$ is the **quantization error**, and the entire art of quantization is managing this error so the model still works.

### 1.2 Symmetric Quantization

In symmetric quantization, the zero-point is fixed at zero ($z = 0$), and the quantization grid is centered at the origin. The representable range is symmetric around zero: $[-\alpha, +\alpha]$.

**Formulas:**

$$s = \frac{\alpha}{2^{N-1} - 1}$$

$$q = \text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil,\; -2^{N-1},\; 2^{N-1} - 1\right)$$

$$\hat{w} = s \cdot q$$

Where $\alpha = \max(|w_i|)$ over all weights in the quantization group.

For $N = 4$: the integer range is $[-8, +7]$, giving 16 representable values. The scale is $s = \alpha / 7$.

**When to use symmetric quantization:**
- Weights with distributions centered near zero (which is the common case for transformer weights)
- When you want simpler dequantization (no zero-point subtraction, saving one operation per weight)
- When kernel throughput matters more than precision (MXQ's Metal kernels benefit from the simpler math)

**Limitation:** If the weight distribution is not symmetric around zero, one side of the range is wasted. For example, if weights range from $[-0.1, +1.0]$, symmetric quantization uses $\alpha = 1.0$ and allocates half the grid to $[-1.0, 0]$, but only a tiny sliver of that range is actually occupied.

### 1.3 Asymmetric Quantization

Asymmetric quantization uses a non-zero zero-point to shift the grid so it covers the actual $[\min(w), \max(w)]$ range exactly.

**Formulas:**

$$s = \frac{w_{\max} - w_{\min}}{2^N - 1}$$

$$z = \left\lfloor -\frac{w_{\min}}{s} \right\rceil$$

$$q = \text{clamp}\!\left(\left\lfloor \frac{w}{s} \right\rceil + z,\; 0,\; 2^N - 1\right)$$

$$\hat{w} = s \cdot (q - z)$$

For $N = 4$: the integer range is $[0, 15]$, giving 16 representable values.

**When to use asymmetric quantization:**
- Activations (which are often ReLU-gated and thus non-negative or skewed)
- Weights with significantly asymmetric distributions
- When you can afford the extra storage for the zero-point per block/group

**Worked example:** Suppose $w \in [-0.1, +0.5]$ with $N = 4$.
- Scale: $s = (0.5 - (-0.1)) / (16 - 1) = 0.6 / 15 = 0.04$
- Zero-point: $z = \lfloor -(-0.1) / 0.04 \rceil = \lfloor 2.5 \rceil = 3$ (note: rounding)
- The value $w = 0.0$ maps to $q = \lfloor 0/0.04 \rceil + 3 = 3$
- The value $w = 0.5$ maps to $q = \lfloor 0.5/0.04 \rceil + 3 = 13 + 3 = 16$, clamped to $15$
- Reconstructed: $\hat{w} = 0.04 \times (3 - 3) = 0.0$ (exact for zero)
- Reconstructed: $\hat{w} = 0.04 \times (15 - 3) = 0.48$ (error of $0.02$ for the max value)

### 1.4 Granularity: Per-Tensor vs Per-Channel vs Per-Group vs Per-Block

The choice of which weights share a single $(s, z)$ pair is called **quantization granularity**. Finer granularity means better accuracy but more metadata overhead.

**Per-tensor quantization:**
One scale and zero-point for the entire weight matrix $W \in \mathbb{R}^{m \times n}$.

$$s = \frac{\max_{i,j}(|W_{ij}|)}{2^{N-1} - 1}$$

Total overhead: 1 scale + 1 zero-point for all $m \times n$ weights. Effectively zero overhead. But if different rows or columns have wildly different magnitudes, the scale is dominated by the largest value and everything else gets crushed into a tiny fraction of the grid.

**Per-channel (per-row) quantization:**
One $(s_i, z_i)$ per output channel (row of the weight matrix). For $W \in \mathbb{R}^{m \times n}$:

$$s_i = \frac{\max_j(|W_{ij}|)}{2^{N-1} - 1}, \quad i = 1, \ldots, m$$

Total overhead: $m$ scales + $m$ zero-points. For a typical $4096 \times 4096$ matrix, that is 4096 pairs -- negligible relative to the 16M weights.

**Per-group quantization:**
Weights within each row are divided into groups of $g$ consecutive elements. Each group gets its own $(s, z)$.

For a row of $n$ weights with group size $g$, there are $\lceil n/g \rceil$ groups per row and $m \cdot \lceil n/g \rceil$ groups total.

$$s_{i,k} = \frac{\max_{j \in \text{group}_k}(|W_{ij}|)}{2^{N-1} - 1}$$

Common group sizes: $g = 32, 64, 128$.

**Per-block quantization (what MXQ uses):**
A generalization of per-group where blocks are contiguous chunks of the flattened weight tensor, not necessarily aligned to row boundaries. MXQ uses block sizes of 32 or 64.

For block size $B$ and a weight tensor with $P$ total parameters:
- Number of blocks: $\lceil P / B \rceil$
- Each block stores: $s_k$ (float16, 2 bytes) + $z_k$ (float16 or int, 2 bytes)

The key tradeoff: smaller blocks mean more accurate quantization but more metadata overhead.

### 1.5 Uniform vs Non-Uniform Quantization

**Uniform quantization** places grid points at equally spaced intervals:

$$\text{grid} = \{s \cdot (q - z) : q \in \{0, 1, \ldots, 2^N - 1\}\}$$

The spacing between adjacent representable values is constant: $\Delta = s$.

**Non-uniform quantization** allows the grid points to be placed at arbitrary positions, chosen to match the actual weight distribution. The grid becomes a **codebook** $C = \{c_0, c_1, \ldots, c_{2^N - 1}\}$, and quantization maps each weight to its nearest codebook entry:

$$Q(w) = \underset{c \in C}{\arg\min}\; |w - c|$$

Non-uniform quantization can achieve lower error for the same bit width because it concentrates representable values where the weight density is highest. The downside is:
1. The codebook itself must be stored (overhead: $2^N$ float values per group)
2. Dequantization requires a lookup table instead of a simple multiply-add
3. Lookup tables are slower on GPU hardware (especially Metal) than arithmetic

In practice, MXQ uses uniform quantization with per-block scaling because the Metal dequant kernels need to be fast, and the per-block granularity compensates for much of the accuracy loss versus non-uniform schemes.

### 1.6 The Quantization Error

The most natural error metric is the squared Frobenius norm:

$$E = \|W - Q(W)\|_F^2 = \sum_{i,j} (W_{ij} - Q(W_{ij}))^2$$

For a single weight with uniform quantization, the maximum error is $\Delta/2$ where $\Delta = s$ is the step size. Assuming weights are uniformly distributed within each quantization bin, the expected squared error per weight is:

$$\mathbb{E}[(w - Q(w))^2] = \frac{\Delta^2}{12}$$

This is the variance of a uniform distribution on $[-\Delta/2, +\Delta/2]$.

**Why minimizing $\|W - Q(W)\|_F^2$ is not always optimal:**

The layer output is $Y = XW$ (ignoring bias). The actual quantity we care about is the **output error**:

$$E_{\text{output}} = \|XW - XQ(W)\|_F^2 = \|X(W - Q(W))\|_F^2$$

Expanding:

$$E_{\text{output}} = \text{tr}\!\left((W - Q(W))^T X^T X (W - Q(W))\right)$$

The matrix $H = X^T X$ is the **Hessian** (or more precisely, the second-moment matrix of activations). This tells us that weight errors in directions that are heavily activated by $X$ matter much more than errors in rarely-activated directions.

This is exactly why importance-aware quantization (MXQ, AWQ, GPTQ) outperforms naive round-to-nearest: they weight the quantization error by the Hessian, allocating more precision to weights that the model actually uses heavily during inference.

The true objective is:

$$\min_{Q} \; \mathbb{E}_X\!\left[\|X(W - Q(W))\|_F^2\right] = \min_{Q} \; \text{tr}\!\left((W - Q(W))^T H (W - Q(W))\right)$$

where $H = \mathbb{E}[X^T X]$ is estimated from calibration data.

---

## 2. Bit-Width Math

### 2.1 Representable Values

An $N$-bit integer can represent exactly $2^N$ distinct values:

| Bit Width ($N$) | Distinct Values ($2^N$) | Unsigned Range | Signed Range |
|:-:|:-:|:-:|:-:|
| 1 | 2 | $[0, 1]$ | $[-1, 0]$ |
| 2 | 4 | $[0, 3]$ | $[-2, 1]$ |
| 3 | 8 | $[0, 7]$ | $[-4, 3]$ |
| 4 | 16 | $[0, 15]$ | $[-8, 7]$ |
| 5 | 32 | $[0, 31]$ | $[-16, 15]$ |
| 6 | 64 | $[0, 63]$ | $[-32, 31]$ |
| 8 | 256 | $[0, 255]$ | $[-128, 127]$ |

### 2.2 Dynamic Range and Step Size

For a weight range $R = w_{\max} - w_{\min}$, the quantization step size (resolution) is:

$$\Delta = \frac{R}{2^N - 1}$$

| Bit Width | $2^N - 1$ | $\Delta$ (for $R = 1.0$) | Relative Precision |
|:-:|:-:|:-:|:-:|
| 2 | 3 | 0.3333 | 1x (baseline) |
| 3 | 7 | 0.1429 | 2.3x better |
| 4 | 15 | 0.0667 | 5.0x better |
| 5 | 31 | 0.0323 | 10.3x better |
| 6 | 63 | 0.0159 | 21.0x better |
| 8 | 255 | 0.00392 | 85.0x better |

Each additional bit roughly doubles the precision (halves $\Delta$), since $\Delta_{N+1} / \Delta_N = (2^N - 1)/(2^{N+1} - 1) \approx 1/2$ for large $N$.

### 2.3 Why 2-Bit Uniform Quantization Destroys Model Quality

With 2-bit quantization, every weight in a block must map to one of exactly 4 values. Consider a typical transformer weight distribution, which is approximately Gaussian with zero mean:

$$w \sim \mathcal{N}(0, \sigma^2)$$

With symmetric 2-bit quantization, the representable values are: $\{-3s, -s, +s, +3s\}$ where $s = \sigma \cdot k / 3$ for some clipping factor $k$.

**The information loss is catastrophic:**

1. **Coarse binning**: 99.7% of a Gaussian falls within $\pm 3\sigma$. Dividing this into 4 bins means each bin covers a range of $\approx 1.5\sigma$. Two weights that differ by $1.4\sigma$ get mapped to the same integer and become indistinguishable.

2. **Signal-to-quantization-noise ratio (SQNR)**: For $N$-bit uniform quantization of a Gaussian source:

$$\text{SQNR}(N) \approx 6.02N + 4.35 - 10\log_{10}\!\left(\frac{12 \cdot \text{loading factor}}{(2^N - 1)^2}\right) \text{ dB}$$

Approximate values:
- 2-bit: SQNR $\approx$ 13.6 dB (signal is only ~23x the noise power)
- 4-bit: SQNR $\approx$ 25.8 dB (signal is ~380x the noise power)
- 8-bit: SQNR $\approx$ 50.2 dB (signal is ~105,000x the noise power)

Going from 4-bit to 2-bit loses about 12 dB of SQNR -- the quantization noise power increases by a factor of ~16.

3. **Layer-by-layer error accumulation**: A transformer with $L$ layers compounds the error. If each layer introduces a multiplicative noise factor of $(1 + \epsilon_l)$, the total distortion after $L$ layers is approximately:

$$\text{Total distortion} \approx \prod_{l=1}^{L} (1 + \epsilon_l) \approx \exp\!\left(\sum_{l=1}^{L} \epsilon_l\right)$$

With uniform 2-bit, $\epsilon_l$ is large enough that for $L = 80$ (a 70B model), the output distribution diverges from the original. This is why uniform 2-bit produces incoherent text.

4. **Why MXQ makes 2-bit viable**: MXQ does not apply 2-bit uniformly. It applies 2-bit only to the least important blocks (where $\epsilon_l$ is small because the weights barely affect the output), while giving 4-6+ bits to the critical blocks. The average may be 2.5 bits, but the distribution of bits is matched to the distribution of importance.

### 2.4 Quantization Error as a Function of Bit Width

For uniform quantization of a signal with range $R$:

$$\Delta = \frac{R}{2^N - 1}$$

The mean squared quantization error (assuming uniform distribution within each bin):

$$\text{MSE} = \frac{\Delta^2}{12} = \frac{R^2}{12(2^N - 1)^2}$$

For large $N$, $(2^N - 1)^2 \approx 2^{2N}$, so:

$$\text{MSE} \propto \frac{R^2}{2^{2N}} = R^2 \cdot 4^{-N}$$

**Each additional bit reduces MSE by a factor of 4 (6 dB).**

This is the fundamental tradeoff. Going from 4-bit to 3-bit increases MSE by 4x. Going from 3-bit to 2-bit increases it by another 4x. The relationship is exponential, which is why there is a quality cliff as you reduce bit width.

For a Gaussian weight distribution $\mathcal{N}(0, \sigma^2)$, the optimal clipping range and MSE are slightly different because tails extend beyond the clipping boundaries. With optimal clipping at $\pm k\sigma$:

$$\text{MSE}_{\text{Gaussian}} = \sigma^2 \cdot f(N, k)$$

where $f(N, k)$ is minimized by choosing:
- $N = 2$: optimal clip at $k \approx 1.71$, MSE $\approx 0.1175\sigma^2$
- $N = 3$: optimal clip at $k \approx 2.15$, MSE $\approx 0.03454\sigma^2$
- $N = 4$: optimal clip at $k \approx 2.51$, MSE $\approx 0.009497\sigma^2$
- $N = 8$: optimal clip at $k \approx 3.29$, MSE $\approx 0.0000226\sigma^2$

### 2.5 Bit Packing

When weights are quantized to $N < 8$ bits, multiple quantized values are packed into a single byte (or larger word) to avoid wasting storage.

**Values per byte:**

| Bit Width | Values per Byte | Values per 32-bit Word | Wasted Bits per Byte |
|:-:|:-:|:-:|:-:|
| 2 | 4 | 16 | 0 |
| 3 | 2 (6 bits used) | 10 (30 bits used) | 2 per byte |
| 4 | 2 | 8 | 0 |
| 5 | 1 (5 bits used) | 6 (30 bits used) | 3 per byte |
| 6 | 1 (6 bits used) | 5 (30 bits used) | 2 per byte |
| 8 | 1 | 4 | 0 |

Bit widths that do not evenly divide the storage word (3-bit, 5-bit, 6-bit) require more careful packing. There are two strategies:

**Strategy 1: Wasteful byte-aligned packing.** Pad each value to fill the byte. Simple but wastes bits.

**Strategy 2: Tight bit-stream packing.** Pack values contiguously in a bit stream, allowing values to cross byte boundaries.

**Tight packing formulas:**

To store $K$ values of $N$ bits each:
- Total bits required: $K \times N$
- Total bytes required: $\lceil K \times N / 8 \rceil$

To extract value $i$ from the packed stream:
```
bit_offset = i * N
byte_offset = bit_offset / 8          (integer division)
bit_shift   = bit_offset % 8          (remainder)
mask        = (1 << N) - 1
value       = (read_u16(data + byte_offset) >> bit_shift) & mask
```

We read a 16-bit (or 32-bit) word starting at `byte_offset` to handle the case where the value straddles a byte boundary. The `bit_shift` aligns the value, and the `mask` isolates the $N$ relevant bits.

**Worked example: packing four 3-bit values into 2 bytes.**

Values: $[5, 3, 7, 2]$ in binary: $[101, 011, 111, 010]$

Tight packing (LSB first within each byte):
```
Byte 0: bits 0-7:   101 011 11   (values 0, 1, and lower 2 bits of value 2)
Byte 1: bits 8-11:  1 010 ----   (upper 1 bit of value 2, then value 3)
```

Total: 12 bits used out of 16 bits (2 bytes). Four bits wasted.

Extraction of value 2 ($i = 2$, $N = 3$):
```
bit_offset = 2 * 3 = 6
byte_offset = 6 / 8 = 0
bit_shift = 6 % 8 = 6
mask = (1 << 3) - 1 = 7 = 0b111
read_u16(byte 0) = byte1 << 8 | byte0 = 0b00001010_11101111 (example)
(read_u16 >> 6) & 7 = extract the 3 bits starting at position 6 = 111 = 7
```

**MXQ packing approach for Metal kernels:**

For GPU efficiency, MXQ uses 32-bit word-aligned packing. Each block of $B$ weights at $N$ bits is packed into $\lceil B \times N / 32 \rceil$ 32-bit words. This guarantees that block boundaries align to word boundaries, enabling coalesced memory reads on the GPU.

For block size $B = 64$:

| Bit Width | Bits per Block | 32-bit Words per Block | Bytes per Block |
|:-:|:-:|:-:|:-:|
| 2 | 128 | 4 | 16 |
| 3 | 192 | 6 | 24 |
| 4 | 256 | 8 | 32 |
| 5 | 320 | 10 | 40 |
| 6 | 384 | 12 | 48 |
| 8 | 512 | 16 | 64 |

---

## 3. Scale and Zero-Point Computation

The scale factor and zero-point determine how the quantization grid maps onto the actual weight values. Poor choices waste representable values on empty regions of the distribution; optimal choices minimize the reconstruction error.

### 3.1 Absmax Scaling (Symmetric)

The simplest approach: set the scale so the largest magnitude weight maps to the largest representable integer.

$$s = \frac{\max_i(|w_i|)}{2^{N-1} - 1}$$

For $N = 4$: $s = \max(|w_i|) / 7$.

**Properties:**
- Zero-point is always 0
- The value $w = 0$ maps exactly to $q = 0$ (no error at zero)
- Symmetric: the negative range $[-\alpha, 0]$ has the same resolution as $[0, +\alpha]$
- A single outlier weight can inflate $\alpha$ and ruin precision for all other weights

**Outlier mitigation**: Clip at a percentile instead of the absolute max:

$$\alpha = \text{percentile}(|w_i|, p), \quad p \in [99.0, 99.99]$$

Weights beyond $\pm\alpha$ are clamped, introducing clipping error for outliers but reducing $\Delta$ for the bulk of the distribution.

### 3.2 Min-Max Scaling (Asymmetric)

Uses the actual range of the weight distribution:

$$s = \frac{w_{\max} - w_{\min}}{2^N - 1}$$

$$z = \left\lfloor -\frac{w_{\min}}{s} \right\rceil$$

**Worked example:** Weights in a block range from $-0.3$ to $+0.8$, $N = 4$.

$$s = \frac{0.8 - (-0.3)}{15} = \frac{1.1}{15} = 0.07\overline{3}$$

$$z = \left\lfloor \frac{0.3}{0.07\overline{3}} \right\rceil = \lfloor 4.09 \rceil = 4$$

Grid points: $\{s \cdot (q - 4) : q = 0, 1, \ldots, 15\} = \{-0.293, -0.220, \ldots, +0.807\}$

Compare with absmax: $s = 0.8/7 = 0.1143$, giving step size 0.1143 vs 0.0733. Min-max gives 56% better resolution because it does not waste grid points on the empty region $[-0.8, -0.3]$.

### 3.3 Per-Block Scaling with Block Sizes

MXQ quantizes weights in contiguous blocks of $B$ weights, each with its own scale and zero-point.

**Block sizes and their properties:**

| Block Size ($B$) | Scale Params per 4096x4096 Layer | Overhead (fp16 scale + fp16 zero) | Adaptability |
|:-:|:-:|:-:|:-:|
| 32 | 524,288 | 2.0 MB | Excellent -- captures local variation |
| 64 | 262,144 | 1.0 MB | Good -- standard MXQ default |
| 128 | 131,072 | 0.5 MB | Moderate -- misses fine-grained variation |
| 256 | 65,536 | 0.25 MB | Poor -- too coarse for mixed-precision |

**The tradeoff:** Smaller blocks mean each block's weights are more homogeneous, so the quantization grid fits them better. But smaller blocks mean more scale/zero parameters, which adds to the model size.

For a weight tensor with $P$ parameters and block size $B$:
- Number of blocks: $n_b = \lceil P / B \rceil$
- Metadata per block: $s$ (float16 = 16 bits) + $z$ (float16 = 16 bits) = 32 bits
- Total metadata: $32 \cdot n_b$ bits

**Metadata as fraction of weight data:**

$$\text{overhead ratio} = \frac{32}{N \cdot B}$$

For $N = 2, B = 64$: overhead = $32 / (2 \times 64) = 25\%$ of the weight data size.
For $N = 4, B = 64$: overhead = $32 / (4 \times 64) = 12.5\%$.
For $N = 2, B = 32$: overhead = $32 / (2 \times 32) = 50\%$ -- this is large, which is why MXQ defaults to $B = 64$.

### 3.4 Super-Blocks (Scales of Scales)

GGUF (the format used by llama.cpp) introduces a two-level hierarchy to reduce metadata overhead. A **super-block** groups multiple blocks together, and stores the block scales in a lower precision relative to a single super-block scale.

**GGUF Q2_K structure (as a concrete example):**

For a super-block of 256 weights divided into 16 sub-blocks of 16 weights each:
- Super-block scale: 1 x float16 (2 bytes) -- the "scale of scales"
- Super-block min: 1 x float16 (2 bytes) -- the "min of mins"
- Sub-block scales: 16 x 4-bit (8 bytes) -- each sub-block's scale, quantized relative to the super-block scale
- Sub-block mins: 16 x 4-bit (8 bytes) -- each sub-block's min value
- Quantized weights: 256 x 2-bit (64 bytes)

Total per super-block: $2 + 2 + 8 + 8 + 64 = 84$ bytes for 256 weights.
Effective bits per weight: $84 \times 8 / 256 = 2.625$ bits/weight.

**The math of the two-level scheme:**

Let $S$ be the super-block scale and $s_k$ be the $k$-th sub-block scale (stored as a 4-bit integer $d_k$):

$$s_k = S \cdot \frac{d_k}{15}$$

Dequantization of weight $w_{k,j}$ in sub-block $k$:

$$\hat{w}_{k,j} = s_k \cdot q_{k,j} + m_k$$

where $m_k = M \cdot (d_k^{(\min)} / 15)$ is the sub-block minimum, similarly quantized.

The advantage is that sub-block scales are stored in 4 bits instead of 16, reducing overhead by 4x at the sub-block level. The cost is an additional quantization error in the scale itself.

**MXQ's approach:** MXQ stores per-block scales in full float16 (no super-blocks in v1.0). This is acceptable because MXQ uses block size 64, which already provides a good overhead ratio. Super-blocks may be introduced in a future version to squeeze more out of the 2-bit regime.

### 3.5 Optimal Scale Factor via MSE Minimization

Absmax and min-max compute the scale analytically from the data range. But the MSE-optimal scale may differ because:
1. The weight distribution is not uniform within the range
2. Clipping some outliers can reduce overall MSE
3. The rounding behavior depends nonlinearly on the scale

**Grid search approach:**

For a block of weights $\{w_i\}$ and a candidate scale $s$, the MSE is:

$$\text{MSE}(s) = \frac{1}{B} \sum_{i=1}^{B} \left(w_i - s \cdot \text{clamp}\!\left(\left\lfloor \frac{w_i}{s} \right\rceil, q_{\min}, q_{\max}\right)\right)^2$$

This is a non-smooth, piecewise-quadratic function of $s$. We can find its minimum by evaluating $\text{MSE}(s)$ over a grid:

$$s^* = \underset{s \in \mathcal{S}}{\arg\min}\; \text{MSE}(s)$$

where $\mathcal{S} = \{s_{\text{absmax}} \cdot r : r \in \{0.80, 0.81, \ldots, 1.00\}\}$ is a set of candidate scales ranging from 80% to 100% of the absmax scale (the 80% lower bound corresponds to clipping ~0.1% of weights for a Gaussian).

**Analytical approach (for Gaussian weights):**

For $w \sim \mathcal{N}(0, \sigma^2)$ with symmetric $N$-bit quantization clipping at $\pm c$:

$$\text{MSE}(c) = \underbrace{2\sigma^2 \!\int_{c}^{\infty}\! (w - c)^2 \phi(w/\sigma) \, dw}_{\text{clipping error}} + \underbrace{\frac{c^2}{3(2^{N-1} - 1)^2} \left(1 - 2\Phi(-c/\sigma)\right)}_{\text{granularity error}}$$

where $\phi$ is the standard normal PDF and $\Phi$ is its CDF. The optimal $c$ balances clipping error (too small $c$ clips too many weights) against granularity error (too large $c$ spreads the grid too thin).

The optimal ratio $c^*/\sigma$ depends only on $N$:

| $N$ | Optimal $c/\sigma$ | Fraction Clipped |
|:-:|:-:|:-:|
| 2 | 1.71 | 8.7% |
| 3 | 2.15 | 3.2% |
| 4 | 2.51 | 1.2% |
| 8 | 3.29 | 0.10% |

At 2-bit, the optimal strategy clips nearly 9% of weights -- those extreme values would otherwise inflate the scale and waste resolution on the interior.

---

## 4. Round-to-Nearest vs Optimal Rounding

### 4.1 RTN (Round to Nearest)

The simplest rounding strategy: each weight is independently rounded to the nearest quantization grid point.

$$q_i = \left\lfloor \frac{w_i}{s} \right\rceil$$

where $\lfloor x \rceil = \lfloor x + 0.5 \rfloor$ denotes rounding to the nearest integer.

**Properties:**
- Minimizes per-weight error: each $|w_i - Q(w_i)| \leq \Delta/2$
- Fast: $O(P)$ time for $P$ weights, no data dependencies
- Ignores correlations: the rounding direction for $w_i$ is chosen independently of all other weights

**Why it is suboptimal:** RTN minimizes $\sum_i (w_i - \hat{w}_i)^2$ but not $\|X(W - \hat{W})\|^2$. Two weights that individually incur small rounding errors may both round in the same direction, causing their errors to reinforce rather than cancel in the layer output.

### 4.2 GPTQ's Optimal Rounding

GPTQ (Frantar et al., 2022) quantizes weights column-by-column (or row-by-row), using the Hessian $H = X^T X$ to determine how rounding errors propagate.

**The key idea:** After quantizing weight $w_i$, the quantization error $\delta_i = w_i - \hat{w}_i$ is compensated by adjusting the remaining unquantized weights. This is a form of error feedback or sigma-delta modulation applied to weight matrices.

**Algorithm (simplified):**

For a weight matrix $W$ with columns $w_1, w_2, \ldots, w_n$, and Hessian $H = X^T X$:

1. Compute $H^{-1}$ (or its Cholesky factor)
2. For $i = 1$ to $n$:
   a. Quantize: $\hat{w}_i = Q(w_i)$ (round to nearest grid point)
   b. Compute error: $\delta_i = w_i - \hat{w}_i$
   c. Update remaining weights: for all $j > i$:

   $$w_j \leftarrow w_j - \frac{\delta_i \cdot [H^{-1}]_{ij}}{[H^{-1}]_{ii}}$$

The update rule distributes the quantization error of $w_i$ across the unquantized weights $w_j$ in proportion to how correlated they are (as measured by the Hessian). Weights that are highly correlated with $w_i$ receive a larger correction.

**Mathematical justification:**

GPTQ solves the optimization problem:

$$\min_{\hat{W}} \; \text{tr}\!\left((W - \hat{W})^T H (W - \hat{W})\right) \quad \text{s.t.} \; \hat{w}_i \in \text{grid}$$

The greedy column-by-column approach with Hessian-based error compensation gives an approximate solution. The correction step is derived from the optimality condition: given that $w_i$ has been rounded (with error $\delta_i$), the optimal adjustment to the remaining weights is the one that minimizes the total Hessian-weighted error.

**Why this matters for MLXQ:** MXQ can use GPTQ-style optimal rounding within each block during the quantization phase. The Hessian is estimated from the calibration data collected in Phase 1. This is especially valuable for 2-bit and 3-bit blocks where the rounding error per weight is large.

### 4.3 AdaRound (Learning the Rounding)

AdaRound (Nagel et al., 2020) formulates the rounding decision as a continuous optimization problem.

For each weight $w_i$, instead of always rounding to the nearest grid point, AdaRound introduces a continuous variable $v_i \in [0, 1]$ that interpolates between rounding down ($\lfloor w_i/s \rfloor$) and rounding up ($\lceil w_i/s \rceil$):

$$\hat{w}_i = s \cdot \left(\lfloor w_i/s \rfloor + h(v_i)\right)$$

where $h(v) = \text{clamp}(\sigma(v) \cdot (\zeta - \gamma) + \gamma, 0, 1)$ is a stretched sigmoid function ($\sigma$ is the logistic sigmoid, $\zeta = 1.1$, $\gamma = -0.1$).

The variables $\{v_i\}$ are optimized via gradient descent to minimize the layer-wise reconstruction error:

$$\min_{\{v_i\}} \; \|XW - X\hat{W}\|_F^2 + \lambda \sum_i \beta \cdot h(v_i) \cdot (1 - h(v_i))$$

The regularization term $h(v_i)(1 - h(v_i))$ encourages $v_i$ to converge to either 0 (round down) or 1 (round up), not remain at intermediate values.

After optimization, the final rounding decision is:

$$\hat{w}_i = s \cdot \left(\lfloor w_i/s \rfloor + \mathbb{1}[h(v_i) \geq 0.5]\right)$$

**Advantage over GPTQ:** AdaRound optimizes all rounding decisions jointly (within each layer) rather than greedily column-by-column. This can find better solutions but is more computationally expensive.

### 4.4 Error Cascading: Why a Single Rounding Decision Matters

Consider a toy example: a 2-bit block of 4 weights multiplied by an activation vector.

Weights: $w = [0.15, 0.45, 0.75, -0.30]$
Scale: $s = 0.3$ (symmetric, grid = $\{-0.9, -0.3, +0.3, +0.9\}$)

RTN quantization: $\hat{w} = [0.3, 0.3, 0.9, -0.3]$
Errors: $e = [-0.15, +0.15, -0.15, 0.0]$

Now consider activation $x = [1, 1, 1, 1]$:
- True output: $xw = 0.15 + 0.45 + 0.75 - 0.30 = 1.05$
- RTN output: $x\hat{w} = 0.3 + 0.3 + 0.9 - 0.3 = 1.2$
- Error: $0.15$ (14.3% relative error)

Alternative rounding (round $w_1$ up instead of down): $\hat{w}' = [0.3, 0.3, 0.9, -0.3]$ -- same result here since 0.15 is equidistant. But if $w_1 = 0.14$:
- RTN rounds to $0.0$ (if available) or $0.3$. With this grid, rounds to $0.3$ (error = $-0.16$).
- Optimal: might round to $0.3$ as well, but compensate by adjusting $w_2$'s rounding.

In a real network, a single rounding error of $0.15$ is small. But this error feeds into the next layer's activations, where it gets multiplied by that layer's weights, potentially amplified, and passed forward. After 80 layers, small coherent biases in rounding accumulate.

**The mathematical bound:** For a feed-forward network with $L$ layers, each with weight matrices $W_l$ and quantized versions $\hat{W}_l$, the output error satisfies:

$$\|y - \hat{y}\| \leq \sum_{l=1}^{L} \|W_l - \hat{W}_l\| \cdot \prod_{k=l+1}^{L} \|\hat{W}_k\| \cdot \|x\|$$

The product term $\prod_{k=l+1}^{L} \|\hat{W}_k\|$ means early-layer errors are amplified by the norms of all subsequent layers. This is why MXQ gives extra bits to early layers (the "first/last layer protection" policy in the bit allocation algorithm).

---

## 5. Information Theory Perspective

### 5.1 Rate-Distortion Theory Applied to Weight Quantization

Rate-distortion theory provides the fundamental limit on how well a source can be compressed for a given level of distortion. For weight quantization:

- **Rate** $R$: the number of bits per weight (the average bit width)
- **Distortion** $D$: the mean squared error $\mathbb{E}[(W - \hat{W})^2]$

The **rate-distortion function** $R(D)$ gives the minimum number of bits per weight needed to achieve distortion at most $D$.

For a Gaussian source $W \sim \mathcal{N}(0, \sigma^2)$ with squared-error distortion:

$$R(D) = \begin{cases} \frac{1}{2} \log_2 \frac{\sigma^2}{D} & \text{if } D \leq \sigma^2 \\ 0 & \text{if } D > \sigma^2 \end{cases}$$

Equivalently, the minimum distortion achievable at rate $R$ bits/weight is:

$$D(R) = \sigma^2 \cdot 2^{-2R}$$

**Interpretation:** Each additional bit per weight reduces the MSE by a factor of $2^2 = 4$ (6.02 dB). This matches the empirical observation from Section 2.4.

**Practical gap:** Real quantization schemes (uniform with per-block scaling) do not achieve the rate-distortion bound. The gap arises from:
1. Using a uniform grid instead of a distribution-matched grid
2. Finite block sizes (the rate-distortion function assumes optimal coding over the entire source)
3. Using fixed-width integers (cannot assign fractional bits to individual weights)

The gap is typically 1-3 dB for well-designed quantization schemes, meaning real 4-bit achieves quality comparable to what the rate-distortion bound predicts for ~3.5 bits.

### 5.2 Shannon Entropy of Weight Distributions

The Shannon entropy of a continuous source with PDF $p(w)$ is:

$$h(W) = -\int_{-\infty}^{\infty} p(w) \log_2 p(w) \, dw$$

For a Gaussian $\mathcal{N}(0, \sigma^2)$:

$$h(W) = \frac{1}{2} \log_2(2\pi e \sigma^2)$$

This tells us the "intrinsic information content" per weight. For $\sigma = 0.01$ (typical for a transformer weight):

$$h(W) = \frac{1}{2} \log_2(2\pi e \cdot 10^{-4}) = \frac{1}{2} \log_2(1.71 \times 10^{-3}) \approx \frac{1}{2}(-9.19) = -4.60 \text{ bits}$$

The negative value (differential entropy can be negative) does not directly tell us the number of bits needed, but the entropy difference between a high-precision representation and a quantized one quantifies the information loss.

For quantized weights with $2^N$ levels, the discrete entropy is at most $N$ bits (achieved when all levels are equally likely). In practice, the entropy of quantized weights is less than $N$ because the distribution is not uniform across levels -- weights near zero are more common than weights near the extremes.

**Typical discrete entropy of quantized transformer weights:**

| Bit Width | Max Entropy | Typical Entropy | Compression Headroom |
|:-:|:-:|:-:|:-:|
| 2 | 2.0 bits | 1.7 bits | 15% |
| 3 | 3.0 bits | 2.4 bits | 20% |
| 4 | 4.0 bits | 3.2 bits | 20% |
| 8 | 8.0 bits | 5.5 bits | 31% |

The "compression headroom" means that entropy coding (e.g., arithmetic coding) could further reduce size. GGUF exploits this partially through its K-quant packing schemes.

### 5.3 Why Transformer Weights Are Approximately Gaussian

Empirically, the weight distributions of trained transformers are well-approximated by Gaussians (or mixtures of Gaussians). This arises from several factors:

1. **Initialization:** Weights are typically initialized from $\mathcal{N}(0, \sigma_{\text{init}}^2)$ with $\sigma_{\text{init}} \propto 1/\sqrt{d}$.

2. **Central limit effect:** Each weight receives gradient updates from many training examples. By the central limit theorem, the sum of many small independent updates converges to a Gaussian.

3. **Regularization:** Weight decay (L2 regularization) pulls weights toward zero, reinforcing the zero-mean Gaussian shape.

4. **Empirical validation:** Measuring the kurtosis of trained transformer weight tensors typically gives values between 3 and 6 (a Gaussian has kurtosis 3). Values above 3 indicate heavier tails than Gaussian -- these are the outlier weights that cause problems for low-bit quantization.

**What the Gaussian approximation means for quantization:**

- Symmetric quantization is well-suited (the distribution is symmetric around zero)
- The optimal clipping ratio depends only on the bit width and the variance (see Section 3.5)
- Non-uniform quantization with more grid points near zero and fewer in the tails would be optimal
- The rate-distortion function for Gaussian sources gives a tight theoretical bound

**Where the Gaussian approximation breaks down:**
- Outlier channels: some channels in attention layers have a few weights with $|w| > 10\sigma$, giving the distribution heavy tails
- Bimodal layers: some MLP layers develop bimodal weight distributions after training
- Embedding layers: token embeddings often have non-Gaussian, heavy-tailed distributions

MXQ handles these deviations through per-block scaling (each block adapts to its local distribution) and importance-aware bit allocation (outlier-heavy blocks can get more bits).

### 5.4 Optimal Codebook Design: The Lloyd-Max Quantizer

For non-uniform quantization, the optimal placement of quantization levels is given by the Lloyd-Max algorithm (the 1-D case of k-means clustering).

Given a source distribution $p(w)$ and $K = 2^N$ quantization levels, the Lloyd-Max quantizer finds levels $\{c_0, c_1, \ldots, c_{K-1}\}$ and decision boundaries $\{b_0, b_1, \ldots, b_K\}$ (with $b_0 = -\infty, b_K = +\infty$) that minimize:

$$D = \sum_{k=0}^{K-1} \int_{b_k}^{b_{k+1}} (w - c_k)^2 \, p(w) \, dw$$

**Optimality conditions (necessary):**

1. **Nearest-neighbor rule:** Decision boundaries are midpoints between adjacent levels:
   $$b_k = \frac{c_{k-1} + c_k}{2}, \quad k = 1, \ldots, K-1$$

2. **Centroid condition:** Each level is the centroid of its Voronoi region:
   $$c_k = \frac{\int_{b_k}^{b_{k+1}} w \, p(w) \, dw}{\int_{b_k}^{b_{k+1}} p(w) \, dw} = \mathbb{E}[W \mid b_k \leq W < b_{k+1}]$$

**Lloyd-Max algorithm:**
1. Initialize levels (e.g., uniformly spaced)
2. Repeat until convergence:
   a. Compute decision boundaries using rule 1
   b. Update levels using rule 2

For a Gaussian source with $K = 4$ (2-bit), the optimal Lloyd-Max levels are approximately:

$$c \in \{-1.510\sigma, -0.4528\sigma, +0.4528\sigma, +1.510\sigma\}$$

Compare with uniform 2-bit symmetric: $c \in \{-3s, -s, +s, +3s\}$ which, with optimal clipping at $1.71\sigma$, gives $c \in \{-1.71\sigma, -0.57\sigma, +0.57\sigma, +1.71\sigma\}$.

The Lloyd-Max quantizer achieves MSE of $0.1175\sigma^2$ for 2-bit Gaussian, while uniform achieves $0.1188\sigma^2$ with optimal clipping -- only 1.1% worse. At 2-bit, uniform is nearly as good as non-uniform for Gaussian sources. The gap widens for non-Gaussian (heavy-tailed) distributions.

**Relevance to MLXQ:** MXQ uses uniform quantization for Metal kernel efficiency, but the small optimality gap for Gaussian weights means this is not a significant sacrifice. The per-block scaling is more impactful than non-uniform grids.

### 5.5 Why Mixed-Precision Is Information-Theoretically Optimal

Consider a model with $L$ weight tensors, each with its own variance $\sigma_l^2$ and importance $\rho_l$ (how much layer $l$ contributes to model quality). The total bit budget is $B_{\text{total}} = \sum_l n_l \cdot R_l$ where $n_l$ is the number of weights in layer $l$ and $R_l$ is the bit width for layer $l$.

The total distortion, weighted by importance:

$$D_{\text{total}} = \sum_l \rho_l \cdot D_l(R_l) = \sum_l \rho_l \cdot \sigma_l^2 \cdot 2^{-2R_l}$$

**Optimal bit allocation (Lagrangian method):**

Minimize $D_{\text{total}}$ subject to $\sum_l n_l R_l = B_{\text{total}}$:

$$\mathcal{L} = \sum_l \rho_l \sigma_l^2 \cdot 2^{-2R_l} + \lambda \left(\sum_l n_l R_l - B_{\text{total}}\right)$$

Taking the derivative with respect to $R_l$ and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial R_l} = -2 \ln 2 \cdot \rho_l \sigma_l^2 \cdot 2^{-2R_l} + \lambda n_l = 0$$

$$2^{-2R_l} = \frac{\lambda n_l}{2 \ln 2 \cdot \rho_l \sigma_l^2}$$

$$R_l = \frac{1}{2} \log_2 \frac{2 \ln 2 \cdot \rho_l \sigma_l^2}{\lambda n_l}$$

This is the **reverse water-filling** solution. The optimal bit allocation:
- **Allocates more bits to layers with larger $\rho_l \sigma_l^2 / n_l$** -- layers that are important AND have high variance AND are small get the most bits per weight
- **Allocates fewer bits to layers with small $\rho_l \sigma_l^2$** -- unimportant, low-variance layers can be aggressively compressed

**In simpler terms:** bits should flow to where the entropy is highest and where errors are most costly.

**This is exactly the MXQ strategy.** The calibration phase estimates $\rho_l$ (importance) and the weight statistics give $\sigma_l^2$ (variance). The bit allocation phase then assigns bit widths per block to approximate the optimal reverse water-filling solution, subject to the constraint that bit widths must be integers in $\{2, 3, 4, 5, 6, 8\}$.

**Why uniform quantization (same bits everywhere) is suboptimal:**

Uniform allocation sets $R_l = R$ for all $l$. The total distortion is:

$$D_{\text{uniform}} = 2^{-2R} \sum_l \rho_l \sigma_l^2$$

The mixed-precision distortion is:

$$D_{\text{mixed}} = \sum_l \rho_l \sigma_l^2 \cdot 2^{-2R_l^*}$$

By the convexity of $2^{-2R}$ and Jensen's inequality applied in reverse, $D_{\text{mixed}} \leq D_{\text{uniform}}$ whenever the $\rho_l \sigma_l^2$ values are not all identical -- that is, whenever different parts of the model have different importance, which is always the case in practice.

**Quantifying the gain:** For a model where $\rho_l \sigma_l^2$ varies by a factor of 100 across layers (typical for large transformers), mixed-precision quantization at average 2.5 bits can match uniform quantization at ~4 bits. This is the core theoretical justification for MXQ.

---

## 6. Block-Wise Quantization Deep Dive

### 6.1 Block Size Selection: Overhead vs Accuracy

The block size $B$ controls the granularity of quantization. Each block of $B$ consecutive weights shares a single scale factor $s$ and zero-point $z$.

**Accuracy model:** Within a block, the quantization step size is $\Delta_k = s_k / 1$ (for symmetric) where $s_k$ depends on the local weight distribution. The block-level MSE is:

$$\text{MSE}_k = \frac{1}{B} \sum_{i \in \text{block}_k} (w_i - \hat{w}_i)^2$$

Smaller blocks have more homogeneous weight distributions (lower local variance), so $s_k$ is smaller and $\Delta_k$ is smaller, giving lower MSE.

Modeling weights within a block as $\mathcal{N}(0, \sigma_k^2)$ where $\sigma_k$ is the local standard deviation:

$$\text{MSE}_k \propto \frac{\sigma_k^2}{(2^N - 1)^2}$$

As $B$ decreases, $\sigma_k$ decreases (less variation within each block), so MSE decreases. Empirically, for transformer weights:

$$\sigma_k(B) \approx \sigma_{\text{global}} \cdot \left(\frac{B}{P}\right)^{0.1}$$

The improvement diminishes as blocks get smaller -- there's a law of diminishing returns.

**Overhead model:** Each block requires metadata:

| Component | Size | Required? |
|-----------|------|-----------|
| Scale ($s$) | 2 bytes (float16) | Always |
| Zero-point ($z$) | 2 bytes (float16) or 1 byte (int8) | Only for asymmetric |
| Bit-width indicator | 1 byte (uint8) | Only for mixed-precision (MXQ) |

For MXQ with asymmetric quantization, each block requires 5 bytes of metadata. For symmetric quantization, 3 bytes (scale + bit-width).

**The overhead formulas:**

Total model size in bytes:

$$S_{\text{total}} = \underbrace{\sum_{k} B_k \cdot \frac{N_k}{8}}_{\text{weight data}} + \underbrace{n_b \cdot M}_{\text{metadata}}$$

where $B_k$ is the block size (typically constant = $B$), $N_k$ is the bit width for block $k$, $n_b = P/B$ is the number of blocks, and $M$ is the metadata bytes per block.

For uniform bit width $N$ with block size $B$ and metadata $M$ bytes per block:

$$S_{\text{total}} = P \cdot \frac{N}{8} + \frac{P}{B} \cdot M$$

$$\frac{S_{\text{total}}}{P} = \frac{N}{8} + \frac{M}{B} \quad \text{bytes per weight}$$

$$\text{bits}_{\text{eff}} = N + \frac{8M}{B} \quad \text{effective bits per weight}$$

### 6.2 Effective Bits Per Weight Including Overhead

This is the critical formula for comparing quantization schemes. The "advertised" bit width $N$ does not include metadata overhead. The effective bit width does.

**Formula:**

$$\text{bits}_{\text{eff}} = N + \frac{\text{overhead bits per block}}{B}$$

**MXQ with symmetric quantization (scale only, fp16):**

Overhead per block: scale (16 bits) + bit-width (8 bits) = 24 bits.

| $N$ | $B = 32$ | $B = 64$ | $B = 128$ |
|:-:|:-:|:-:|:-:|
| 2 | $2 + 24/32 = 2.75$ | $2 + 24/64 = 2.375$ | $2 + 24/128 = 2.1875$ |
| 3 | $3 + 24/32 = 3.75$ | $3 + 24/64 = 3.375$ | $3 + 24/128 = 3.1875$ |
| 4 | $4 + 24/32 = 4.75$ | $4 + 24/64 = 4.375$ | $4 + 24/128 = 4.1875$ |

**MXQ with asymmetric quantization (scale + zero-point, both fp16):**

Overhead per block: scale (16 bits) + zero (16 bits) + bit-width (8 bits) = 40 bits.

| $N$ | $B = 32$ | $B = 64$ | $B = 128$ |
|:-:|:-:|:-:|:-:|
| 2 | $2 + 40/32 = 3.25$ | $2 + 40/64 = 2.625$ | $2 + 40/128 = 2.3125$ |
| 3 | $3 + 40/32 = 4.25$ | $3 + 40/64 = 3.625$ | $3 + 40/128 = 3.3125$ |
| 4 | $4 + 40/32 = 5.25$ | $4 + 40/64 = 4.625$ | $4 + 40/128 = 4.3125$ |

At $N = 2$ with $B = 32$ and asymmetric quantization, the effective bit width is 3.25 -- the metadata overhead is 62.5% of the weight data. This is why MXQ defaults to $B = 64$.

### 6.3 Worked Example: MXQ-2.5 for a 70B Model

**Model specifications (Qwen3.5-72B-like):**
- Total parameters: $P = 72 \times 10^9$ = 72 billion
- Full precision (bf16): $72 \times 10^9 \times 2$ bytes = 144 GB

**MXQ-2.5 target: average 2.5 effective bits per weight.**

Using block size $B = 64$ with symmetric quantization (24 bits overhead per block):

The bit allocation from the MXQ plan assigns:
- Embeddings and lm_head (~1B params): 4-6 bits
- Attention Q/K/V (~15B params): 3-4 bits
- MLP gate/up/down (~45B params): 2-3 bits
- First/last 2 layers (~3B params): 4 bits
- Remaining layers (~8B params): 2-3 bits

**Size calculation (approximate):**

| Component | Params | Avg Bits | Eff Bits ($B=64$) | Size (GB) |
|-----------|--------|----------|-------------------|-----------|
| Embeddings | 0.6B | 5.0 | 5.375 | 0.40 |
| lm_head | 0.6B | 6.0 | 6.375 | 0.48 |
| Attn (first/last) | 3.0B | 4.0 | 4.375 | 1.64 |
| Attn (middle) | 12.0B | 3.5 | 3.875 | 5.81 |
| MLP (middle) | 45.0B | 2.1 | 2.475 | 13.92 |
| Other | 10.8B | 2.5 | 2.875 | 3.88 |
| **Total** | **72.0B** | **2.50** | **2.87** | **26.13** |

Note: the effective bits (2.87) are higher than the target (2.5) because of metadata overhead. The actual target must account for this:

$$N_{\text{target}} = \text{bits}_{\text{eff, target}} - \frac{8M}{B} = 2.5 - \frac{24}{64} = 2.125 \text{ average raw bits}$$

So to achieve 2.5 effective bits per weight, MXQ needs to average ~2.1 raw bits (before metadata). This means the MLP blocks (which dominate parameter count) must be overwhelmingly 2-bit.

**Comparison with alternatives:**

| Format | 72B Size | Effective bits/weight |
|--------|----------|-----------------------|
| bf16 | 144 GB | 16.0 |
| MLX 4-bit uniform ($B=64$) | ~40 GB | 4.5 |
| GGUF Q4_K_M | ~42 GB | ~4.7 |
| MLX 2-bit uniform ($B=64$) | ~22 GB | 2.5 |
| GGUF Q2_K | ~27 GB | ~2.6 |
| **MXQ-2.5** | **~26 GB** | **~2.9** |

MXQ-2.5 is slightly larger than uniform 2-bit (due to mixed-precision overhead), but the quality difference is massive: MXQ-2.5 targets perplexity within 5% of uniform 4-bit, while uniform 2-bit is unusable.

### 6.4 Block Alignment and GPU Efficiency

On Metal (Apple GPU), memory reads are most efficient when aligned to 16-byte (128-bit) boundaries. MXQ's packing scheme aligns blocks to 32-bit word boundaries as a minimum, and ideally to 128-bit boundaries for vectorized reads.

For block size $B = 64$ at $N = 2$:
- Block data size: $64 \times 2 / 8 = 16$ bytes = 128 bits (exactly one 128-bit read)
- Scale: 2 bytes (part of a separate scale buffer, read separately)

For $N = 3$:
- Block data size: $64 \times 3 / 8 = 24$ bytes = 192 bits
- Not aligned to 128 bits, but aligned to 32 bits (6 x 32-bit words)
- Requires two 128-bit reads (with 64 bits of the second read unused)

For $N = 4$:
- Block data size: $64 \times 4 / 8 = 32$ bytes = 256 bits (two 128-bit reads)

**Mixed-precision complication:** In MXQ, adjacent blocks may have different bit widths. The byte offset of block $k$ depends on the bit widths of all preceding blocks:

$$\text{byte\_offset}(k) = \sum_{j=0}^{k-1} \left\lceil \frac{B \cdot N_j}{8} \right\rceil$$

This variable-offset indexing prevents simple stride-based access. MXQ addresses this by precomputing a block offset table (a small array of $n_b$ uint32 values giving the byte offset of each block). The Metal kernel loads this table into shared memory for fast lookup.

**Alternative: fixed-stride with padding.** Allocate the maximum block size ($B \times 8 / 8 = B$ bytes per block, assuming max 8-bit) for every block, regardless of actual bit width. This wastes space but enables stride-based access:

$$\text{byte\_offset}(k) = k \times B$$

The space overhead for a 2.5-bit average model: each block is allocated $B = 64$ bytes but uses only $64 \times 2.5 / 8 = 20$ bytes on average. The waste is $44 / 64 = 69\%$, which is unacceptable.

MXQ uses the offset table approach, which adds $n_b \times 4$ bytes (about 0.2% overhead for a 70B model) but eliminates all padding waste.

### 6.5 Block Quantization Error Analysis

For a block of $B$ weights drawn from $\mathcal{N}(\mu_k, \sigma_k^2)$, quantized to $N$ bits with symmetric absmax scaling:

**Scale factor:**

$$s_k = \frac{\max_{i \in \text{block}_k}(|w_i|)}{2^{N-1} - 1}$$

The expected maximum of $B$ i.i.d. samples from $\mathcal{N}(0, \sigma^2)$ is approximately:

$$\mathbb{E}[\max_i |w_i|] \approx \sigma \sqrt{2 \ln(2B)}$$

For $B = 64$: $\mathbb{E}[\max_i |w_i|] \approx \sigma \sqrt{2 \ln 128} = \sigma \sqrt{9.70} \approx 3.11\sigma$.

So the expected scale is:

$$\mathbb{E}[s_k] \approx \frac{3.11\sigma}{2^{N-1} - 1}$$

And the expected step size:

$$\mathbb{E}[\Delta_k] = \mathbb{E}[s_k] \approx \frac{3.11\sigma}{2^{N-1} - 1}$$

**Expected MSE per weight within a block:**

$$\text{MSE}_{\text{block}} \approx \frac{\Delta_k^2}{12} = \frac{(3.11\sigma)^2}{12(2^{N-1} - 1)^2} = \frac{9.67\sigma^2}{12(2^{N-1} - 1)^2} = \frac{0.806\sigma^2}{(2^{N-1} - 1)^2}$$

| $N$ | $(2^{N-1} - 1)^2$ | MSE / $\sigma^2$ | Relative to 4-bit |
|:-:|:-:|:-:|:-:|
| 2 | 1 | 0.806 | 36.6x worse |
| 3 | 9 | 0.0896 | 4.1x worse |
| 4 | 49 | 0.0164 | 1.0x (baseline) |
| 5 | 225 | 0.00358 | 4.6x better |
| 6 | 961 | 0.000839 | 19.6x better |
| 8 | 16129 | 0.0000500 | 328x better |

This table quantifies the MXQ tradeoff: giving a block 2 bits instead of 4 increases its MSE by 37x, but if that block has importance $\rho$ that is 37x lower than average, the contribution to total output error is the same. MXQ's calibration identifies which blocks can tolerate 37x more MSE without degrading the model.

### 6.6 Summary of Key Formulas for MXQ Implementation

For quick reference, the essential formulas for the MXQ quantizer:

**Quantize (per-block, symmetric):**
$$s_k = \frac{\max_{i \in \text{block}_k}(|w_i|)}{2^{N_k - 1} - 1}$$
$$q_i = \text{clamp}\!\left(\left\lfloor \frac{w_i}{s_k} \right\rceil,\; -(2^{N_k - 1}),\; 2^{N_k - 1} - 1\right)$$

**Dequantize (Metal kernel):**
$$\hat{w}_i = s_k \cdot q_i$$

**Effective bits per weight:**
$$\text{bits}_{\text{eff}} = \bar{N} + \frac{8M}{B}$$

where $\bar{N} = \frac{1}{P}\sum_k B \cdot N_k$ is the average raw bit width and $M$ is metadata bytes per block.

**Bit allocation objective:**
$$\min_{\{N_k\}} \sum_k \rho_k \cdot \sigma_k^2 \cdot (2^{N_k - 1} - 1)^{-2} \quad \text{s.t.} \quad \frac{1}{n_b}\sum_k N_k = \bar{N}_{\text{target}}$$

**Block offset table:**
$$\text{offset}(k) = \sum_{j=0}^{k-1} \left\lceil \frac{B \cdot N_j}{8} \right\rceil$$

**MSE per block:**
$$\text{MSE}_k \approx \frac{0.806 \cdot \sigma_k^2}{(2^{N_k - 1} - 1)^2}$$

---

## References

- Frantar, E., et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." arXiv:2210.17323.
- Nagel, M., et al. (2020). "Up or Down? Adaptive Rounding for Post-Training Quantization." ICML 2020. arXiv:2004.10568.
- Lin, J., et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978.
- Dettmers, T., et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022. arXiv:2208.07339.
- Frantar, E. & Alistarh, D. (2023). "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." arXiv:2307.13304.
- GGML/GGUF quantization type specifications. https://github.com/ggerganov/ggml
- Lloyd, S. (1982). "Least Squares Quantization in PCM." IEEE Transactions on Information Theory.
- Max, J. (1960). "Quantizing for Minimum Distortion." IRE Transactions on Information Theory.
- Cover, T. & Thomas, J. (2006). "Elements of Information Theory." Wiley. (Rate-distortion theory, Chapter 10.)
