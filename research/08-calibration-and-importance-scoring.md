# Calibration and Importance Scoring for Quantization

## MXQ Research Document 08

This document provides the complete technical foundation for MXQ Phase 1: the calibration engine and importance scoring pipeline. Everything in Phase 2 (bit allocation and quantization) and Phase 3 (the MXQ file format) depends on the quality of the importance matrix produced here. A bad importance matrix means bad quantization, no matter how sophisticated the bit allocation algorithm is.

---

## 1. What Calibration Is and Why It Matters

### The Core Problem

A large language model like Qwen3.5-72B has approximately 72 billion weights. When we quantize from bfloat16 (16 bits per weight) down to an average of 2.5 bits per weight, we are throwing away roughly 84% of the information stored in those weights. The question is: which 84%?

**Uniform quantization** treats every weight identically. Every weight in every layer gets the same number of bits. This is what MLX's built-in quantization does. At 4 bits per weight, uniform quantization works well enough because 4 bits can represent 16 distinct values per block, which is sufficient to approximate most weight distributions. But at 2-3 bits, uniform quantization falls apart. With only 4-8 distinct values per block, the approximation error becomes catastrophic for weights that the model relies on heavily during inference.

**Importance-aware quantization** (what MXQ does) recognizes that not all weights contribute equally to the model's output. Some weights process activations that are consistently large during inference. Some weights sit on paths that are critical for the model's predictions. Others are nearly dead -- they process near-zero activations or contribute negligibly to the output. The key insight: we can give fewer bits to the unimportant weights and more bits to the important ones, achieving the same average bit width as uniform quantization but with dramatically lower effective error.

**Calibration** is the process of determining which weights are important. We run representative data through the full-precision model and observe how each weight interacts with real activations. This produces an importance matrix: a score for every weight block in the model that tells us how many bits it deserves.

### The Calibration Paradox

There is a fundamental bootstrapping problem: to determine how to compress the model, we need the full-precision model. The full-precision model is the thing we are trying to compress because it does not fit in memory (or at least, we want it to fit in less memory). This means calibration must happen on a machine with enough resources to run the full model at full precision, even though the end goal is to run the quantized model on a smaller machine.

For MXQ, this means:
- **Calibration** happens on a machine with enough unified memory to load the bf16/f16 model (e.g., a Mac Studio with 192GB for a 72B model, or an A100 cluster)
- **Inference** happens on the target machine (e.g., a MacBook with 32GB)
- The importance matrix bridges the gap: computed once on the big machine, used to produce a model that runs on the small machine

This is analogous to how EXL2 calibration works: you need a GPU with enough VRAM to run the full model during calibration, but the resulting quantized model runs on a smaller GPU.

### Why Calibration Quality is Everything

At 4-bit uniform quantization, calibration barely matters. The quantization error is small enough that even a naive round-to-nearest approach produces good results. But as we push below 3 bits average, the importance matrix becomes the single most important factor determining model quality. Consider the following scenario for a 70B model at 2.5-bit average:

- Total weight blocks: approximately 2.2 million (with block size 64)
- Target average: 2.5 bits per block
- Available bit widths: 2, 3, 4, 5, 6, 8

If we allocate bits randomly, the model produces garbage. If we allocate bits based on a high-quality importance matrix, the model matches 4-bit uniform quality. The difference between "garbage" and "matches 4-bit" is entirely determined by the quality of the importance scores.

The importance matrix is, in a very real sense, the intellectual property of MXQ. The calibration dataset composition, the scoring algorithm, and the aggregation method together form the "secret sauce" that determines whether MXQ-2.5 bit produces a usable model or not.

---

## 2. Calibration Datasets

### Requirements for a Good Calibration Dataset

The calibration dataset must satisfy several competing constraints:

**Representativeness.** The dataset must cover the distribution of inputs the model will encounter during inference. If the calibration data is all English prose but the model will be used for code generation, the importance scores will be wrong for code-related weights. Activations that fire strongly during code processing but not during prose will be scored as unimportant, and those weights will be aggressively quantized, destroying code quality.

**Diversity.** The dataset must exercise as many of the model's capabilities as possible. A general-purpose LLM handles text, code, math, conversation, translation, reasoning, and more. Each of these tasks activates different subsets of the model's weights. The calibration dataset must touch all of them.

**Sufficient size.** Importance scores based on too few samples will be noisy -- high variance due to random fluctuations in which tokens happen to appear. The scores must be stable: running calibration twice on different random samples from the same distribution should produce nearly identical importance matrices. Empirically, 128 samples is the minimum for stable scores, 256-512 is good, and 1024 is diminishing returns for most models.

**Not too large.** Each calibration sample requires a full forward pass through the model at full precision. For a 72B model, each forward pass takes significant time and memory. Running 10,000 samples provides negligible benefit over 1,000 samples but costs 10x the compute. There is a clear knee in the stability-vs-compute curve, usually around 256-512 samples.

**No contamination with evaluation data.** The calibration dataset must not overlap with the evaluation dataset used to measure quantization quality. If we calibrate on Wikitext-2 and then evaluate perplexity on Wikitext-2, we are overfitting the importance scores to the evaluation benchmark. The model may score well on Wikitext-2 but poorly on real-world use.

### Common Calibration Datasets

**C4 (Colossal Clean Crawled Corpus).** Derived from Common Crawl, filtered for quality. Pros: large, diverse web text, widely used as a calibration standard (GPTQ and AWQ both use C4 subsets). Cons: English-only, no code, no conversation format, no math notation. Good as a baseline but insufficient for a general-purpose LLM.

**Wikitext-2.** Wikipedia articles, clean and well-structured. Pros: standard benchmark, easy to obtain, good for reproducibility. Cons: very narrow distribution (encyclopedia-style prose), poor representation of code/chat/reasoning. Not recommended as a sole calibration dataset. Useful as an evaluation dataset.

**RedPajama.** A diverse collection of web text, books, code, and academic papers. Pros: broad coverage, multiple domains. Cons: not specifically designed for calibration, requires filtering and curation.

**Pile subsets.** The Pile is a diverse corpus with named subsets (PubMed, ArXiv, GitHub, StackExchange, etc.). Pros: can select specific subsets to control the mix. Cons: quality varies by subset.

**Custom curated (MXQ approach).** For MXQ, we build a custom calibration dataset rather than relying on a single existing corpus. This gives us precise control over the distribution.

### MXQ Calibration Dataset Composition

For a general-purpose LLM, the MXQ calibration dataset (`mxq-calib-v1`) targets the following composition:

```
Category breakdown (~1000 samples total):

General text (30%, ~300 samples):
  - Web articles, news, blog posts: 100 samples
  - Books/literature excerpts: 50 samples
  - Academic/scientific text: 50 samples
  - Wikipedia articles: 50 samples
  - Technical documentation: 50 samples

Code (20%, ~200 samples):
  - Python: 60 samples
  - JavaScript/TypeScript: 40 samples
  - C/C++/Rust: 30 samples
  - Java/Go/Swift: 30 samples
  - Shell/SQL/config files: 20 samples
  - Mixed code + natural language (docstrings, comments): 20 samples

Conversation (20%, ~200 samples):
  - Multi-turn chat (user/assistant format): 80 samples
  - System prompt + instruction following: 40 samples
  - Tool calling / function calling: 40 samples
  - Roleplay / creative writing prompts: 20 samples
  - Adversarial / edge case prompts: 20 samples

Reasoning (15%, ~150 samples):
  - Math (arithmetic, algebra, calculus): 50 samples
  - Logic puzzles and deduction: 30 samples
  - Chain-of-thought / step-by-step: 30 samples
  - Coding problems with solutions: 20 samples
  - Scientific reasoning: 20 samples

Multilingual (15%, ~150 samples):
  - Chinese (Simplified + Traditional): 40 samples
  - Japanese: 30 samples
  - Korean: 20 samples
  - Spanish: 20 samples
  - French: 15 samples
  - German: 15 samples
  - Arabic: 10 samples
```

This composition is tuned for MXQ's target use case: general-purpose LLMs running on Apple Silicon via vMLX Engine. The heavy code allocation reflects the fact that code generation is disproportionately sensitive to quantization (code requires precise token prediction; a single wrong token breaks syntax). The conversation allocation reflects that most vMLX users will be running chat-format inference.

### Calibration Sequence Length

The sequence length of calibration samples affects which activation patterns are captured:

**Short sequences (512 tokens or fewer).** Capture local patterns: token embeddings, short-range attention, basic feed-forward activation patterns. Miss long-range dependencies, attention head specialization, and context-dependent behavior. Not recommended.

**Medium sequences (2048-4096 tokens).** The standard range. Captures most activation patterns relevant for inference. This is the range where most LLM interactions fall. 2048 tokens is the sweet spot for computational cost vs. coverage.

**Long sequences (8192-32768 tokens).** Necessary for models with large context windows (32K, 128K). Long-context models develop specific activation patterns for position-dependent processing (RoPE frequency patterns at different positions, attention sink patterns, context compression in middle layers). If the model will be used for long-context tasks, at least 10% of calibration samples should be at maximum context length.

**Very long sequences (32K+).** Only needed for models specifically designed for very long context (128K+ models). The computational cost is high and the marginal benefit is small unless long-context use is the primary use case.

For MXQ, the standard configuration is:

```
Sequence length distribution:
  70% of samples: 2048 tokens
  20% of samples: 4096 tokens
  10% of samples: 8192+ tokens (for long-context models)
```

### How llama.cpp Generates imatrix

For reference, llama.cpp's imatrix tool works as follows:

```bash
# Generate importance matrix from calibration text
./llama-imatrix -m model-f16.gguf -f calibration.txt -o imatrix.dat

# Parameters:
#   -m model.gguf       : the full-precision model
#   -f calibration.txt  : plain text calibration data
#   -o imatrix.dat      : output importance matrix
#   --chunks N          : number of chunks to process (default: all)
#   -c CONTEXT_SIZE     : context size for processing
#   -b BATCH_SIZE       : batch size for processing
```

The llama.cpp imatrix stores, for each weight tensor, a float array of per-column importance values. The importance is computed as the sum of squared input activations for each column (channel), accumulated across all calibration tokens. This is equivalent to the diagonal of the Hessian approximation X^T X, which we discuss in detail in Section 4.

MXQ uses a more sophisticated approach than llama.cpp's imatrix:
1. We collect richer statistics (not just squared activation sums)
2. We compute per-block scores (not just per-column)
3. We combine multiple scoring methods (not just activation magnitude)
4. We store the result in safetensors format with full metadata

---

## 3. Activation Collection -- The Forward Pass

### Architecture of the Activation Collection System

The activation collector hooks into every `nn.Linear` layer in the model and records statistics about the activations that flow through each layer during the calibration forward passes. The key constraint is memory: for a 72B model, we cannot store raw activations for every layer across all calibration samples. We must aggregate online, computing running statistics that converge to stable values as more samples are processed.

### Hooking into Linear Layers

In MLX (and PyTorch, for reference), we can register forward hooks on modules. A forward hook is a callback that is invoked every time a module's `forward()` method is called, with access to both the input and output tensors.

```python
import mlx.core as mx
import mlx.nn as nn

class ActivationCollector:
    """
    Collects per-channel activation statistics from every Linear layer
    in the model during calibration forward passes.
    """

    def __init__(self, model, block_size=64):
        self.block_size = block_size
        self.stats = {}  # layer_name -> ChannelStats
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model, prefix=""):
        """Recursively register hooks on all Linear layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_name = f"{prefix}.{name}" if prefix else name
                self.stats[layer_name] = ChannelStats(
                    n_channels=module.weight.shape[1],  # input dimension
                    block_size=self.block_size
                )
                # In MLX, we wrap the module's __call__ method
                original_call = module.__call__
                def hook_fn(self_module, *args, _name=layer_name,
                           _orig=original_call, **kwargs):
                    result = _orig(*args, **kwargs)
                    x = args[0]  # input activations
                    self.stats[_name].update(x)
                    return result
                module.__call__ = hook_fn.__get__(module)
```

In practice, the hook mechanism varies between MLX and PyTorch. In PyTorch, `register_forward_hook()` is a first-class API. In MLX, we typically need to override the module's call method or use a wrapper pattern. The important thing is that we intercept the input tensor `x` before it is multiplied by the weight matrix.

### What to Record Per Layer

For each `nn.Linear` layer with weight shape `(output_dim, input_dim)`, the input activation `x` has shape `(batch_size, seq_len, input_dim)`. We flatten batch and sequence dimensions to get a 2D tensor of shape `(n_tokens, input_dim)` where `n_tokens = batch_size * seq_len`.

For each channel `c` in `[0, input_dim)`, we collect:

**Sum of squared activations (required for Hessian diagonal):**

```
S2[c] = sum over all tokens t of x[t, c]^2
```

This is the most important statistic. It directly gives us the diagonal of the Hessian approximation H = X^T X. It measures how much each input channel is used across all calibration tokens. Channels with high S2 values are frequently activated with large magnitudes.

**Sum of activations (for computing mean):**

```
S1[c] = sum over all tokens t of x[t, c]
```

Needed to compute the per-channel mean, which is used for zero-point adjustment during quantization.

**Sum of absolute activations (for L1 norm):**

```
SA[c] = sum over all tokens t of |x[t, c]|
```

Alternative to L2 norm. Less sensitive to outlier activations but also less informative about the importance of extreme values.

**Maximum absolute activation (for range estimation):**

```
Mabs[c] = max over all tokens t of |x[t, c]|
```

Needed for determining the dynamic range of each channel. Channels with extremely large maximum activations may need special handling (outlier channels).

**Token count:**

```
N = total number of tokens processed across all calibration samples
```

Needed to compute averages from running sums.

### Running Statistics with Welford's Algorithm

For numerical stability when accumulating statistics across potentially millions of tokens, we use Welford's online algorithm for computing mean and variance. This avoids the catastrophic cancellation that occurs when computing variance as `E[X^2] - E[X]^2` with floating-point arithmetic.

```python
class ChannelStats:
    """
    Online statistics accumulator for per-channel activation data.
    Uses Welford's algorithm for numerically stable variance computation.
    """

    def __init__(self, n_channels, block_size=64):
        self.n_channels = n_channels
        self.block_size = block_size
        self.n_tokens = 0

        # Running sums (for fast computation)
        self.sum_sq = mx.zeros(n_channels, dtype=mx.float32)  # S2[c]
        self.sum_abs = mx.zeros(n_channels, dtype=mx.float32)  # SA[c]
        self.max_abs = mx.zeros(n_channels, dtype=mx.float32)  # Mabs[c]

        # Welford's running statistics (for numerically stable variance)
        self.welford_mean = mx.zeros(n_channels, dtype=mx.float32)
        self.welford_m2 = mx.zeros(n_channels, dtype=mx.float32)

    def update(self, x):
        """
        Update statistics with a new batch of activations.

        Args:
            x: activation tensor of shape (batch, seq_len, n_channels)
               or (n_tokens, n_channels)
        """
        # Flatten to (n_tokens, n_channels)
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])

        # Cast to float32 for numerical stability
        x = x.astype(mx.float32)

        n_new = x.shape[0]

        # Update sum of squares: S2[c] += sum_t x[t,c]^2
        self.sum_sq += mx.sum(x * x, axis=0)

        # Update sum of absolute values: SA[c] += sum_t |x[t,c]|
        self.sum_abs += mx.sum(mx.abs(x), axis=0)

        # Update max absolute value: Mabs[c] = max(Mabs[c], max_t |x[t,c]|)
        batch_max = mx.max(mx.abs(x), axis=0)
        self.max_abs = mx.maximum(self.max_abs, batch_max)

        # Welford's online algorithm for mean and variance
        # Process the batch mean (batch Welford update)
        batch_mean = mx.mean(x, axis=0)
        batch_var = mx.var(x, axis=0)

        n_old = self.n_tokens
        n_total = n_old + n_new

        delta = batch_mean - self.welford_mean
        self.welford_mean = (
            self.welford_mean * (n_old / max(n_total, 1))
            + batch_mean * (n_new / max(n_total, 1))
        )
        self.welford_m2 += (
            batch_var * n_new
            + delta * delta * (n_old * n_new / max(n_total, 1))
        )

        self.n_tokens = n_total

    @property
    def channel_variance(self):
        """Per-channel variance of activations."""
        if self.n_tokens < 2:
            return mx.zeros(self.n_channels, dtype=mx.float32)
        return self.welford_m2 / (self.n_tokens - 1)

    @property
    def channel_mean(self):
        """Per-channel mean of activations."""
        return self.welford_mean

    @property
    def channel_l2_norm(self):
        """Per-channel L2 norm (RMS of activations)."""
        if self.n_tokens == 0:
            return mx.zeros(self.n_channels, dtype=mx.float32)
        return mx.sqrt(self.sum_sq / self.n_tokens)

    @property
    def channel_l1_norm(self):
        """Per-channel L1 norm (mean absolute activation)."""
        if self.n_tokens == 0:
            return mx.zeros(self.n_channels, dtype=mx.float32)
        return self.sum_abs / self.n_tokens

    @property
    def hessian_diagonal(self):
        """
        Diagonal of the Hessian approximation H = X^T X.
        H[c,c] = sum_t x[t,c]^2 = S2[c]
        """
        return self.sum_sq
```

### Memory Management During Collection

For a 72B model with ~100 Linear layers, each with input dimension up to 8192 channels, the statistics require:

```
Per layer: 5 float32 arrays * 8192 channels * 4 bytes = 163 KB
100 layers: ~16 MB total

This is negligible compared to the model size (~144 GB at bf16).
```

The critical memory constraint is not the statistics but the activations during the forward pass. A single forward pass of a 72B model at 2048 sequence length requires storing intermediate activations across all layers. With MLX's lazy computation and memory-mapped weights on Apple Silicon unified memory, this is manageable but must be carefully orchestrated:

1. Process one calibration sample at a time (batch size 1)
2. Let MLX handle computation lazily, freeing intermediate activations as they are consumed
3. Force materialization of statistics after each sample to prevent graph explosion
4. Use `mx.eval()` strategically to control peak memory

```python
def run_calibration(model, dataset, collector, max_samples=512):
    """
    Run calibration forward passes and collect activation statistics.

    Args:
        model: the full-precision MLX model
        dataset: list of tokenized calibration samples
        collector: ActivationCollector instance
        max_samples: maximum number of samples to process
    """
    for i, sample in enumerate(dataset[:max_samples]):
        if i >= max_samples:
            break

        # Tokenize and prepare input
        tokens = mx.array(sample["input_ids"]).reshape(1, -1)

        # Forward pass (activations are captured by hooks)
        logits = model(tokens)

        # Force materialization to free the computation graph
        mx.eval(logits)

        # Force materialization of all statistics tensors
        for stats in collector.stats.values():
            mx.eval(
                stats.sum_sq, stats.sum_abs, stats.max_abs,
                stats.welford_mean, stats.welford_m2
            )

        # Log progress
        if (i + 1) % 10 == 0:
            tokens_so_far = collector.stats[
                list(collector.stats.keys())[0]
            ].n_tokens
            print(
                f"Calibration: {i+1}/{max_samples} samples, "
                f"{tokens_so_far} tokens processed"
            )

    total_tokens = collector.stats[
        list(collector.stats.keys())[0]
    ].n_tokens
    print(f"Calibration complete. {total_tokens} total tokens processed.")
```

### Handling Outlier Activations

LLMs frequently exhibit outlier channels: specific activation channels that have magnitudes 10-100x larger than the average channel. These outliers were first documented in the LLM.int8() paper (Dettmers et al., 2022) and are now well-understood to be a systematic property of transformer models, not random noise.

Outlier channels are critically important for model function. They act as "information highways" that carry high-magnitude signals through the residual stream. If we naively include outlier channels in our importance scoring without special handling, two problems arise:

1. **Score distortion.** A single outlier channel can dominate the importance score for an entire layer, causing all non-outlier channels to be rated as unimportant by comparison.

2. **Quantization damage.** Outlier channels need the most bits because their large dynamic range is hardest to represent with few quantization levels.

MXQ handles outliers in two ways:

**Detection:** A channel is classified as an outlier if its maximum absolute activation exceeds 6 standard deviations of the channel-wise maximum distribution for that layer.

```python
def detect_outlier_channels(stats):
    """
    Identify outlier activation channels.

    Returns: boolean mask of shape (n_channels,) where True = outlier
    """
    max_vals = stats.max_abs
    mean_max = mx.mean(max_vals)
    std_max = mx.std(max_vals)
    threshold = mean_max + 6.0 * std_max
    return max_vals > threshold
```

**Protection:** Outlier channels receive a guaranteed minimum bit width (4 bits) regardless of the overall bit budget. The weight blocks that process outlier channels are marked as "protected" in the importance matrix.

---

## 4. Importance Scoring Methods -- Detailed Mathematics

The importance score for a weight block determines how many bits it will receive during quantization. Higher importance means more bits, lower importance means fewer bits. This section covers five scoring methods in full mathematical detail, including their derivations, computational costs, and quality tradeoffs.

### Notation

Let:
- `W` be a weight matrix of shape `(d_out, d_in)` for a single Linear layer
- `X` be the matrix of all calibration activations for this layer, shape `(T, d_in)` where `T` is the total number of tokens across all calibration samples
- `Y = XW^T` be the output activations, shape `(T, d_out)`
- `B` be a block of weights within `W`, consisting of `block_size` consecutive weights (e.g., 64 weights)
- `L(W)` be the model's loss function on the calibration data
- `W_ij` denote the weight at row `i`, column `j` of `W`

### 4a. Activation Magnitude Scoring (AWQ-style)

**Origin:** Activation-Aware Weight Quantization (AWQ) by Lin et al. (2023).

**Core idea:** A weight is important if it is large AND it processes large activations. A large weight connected to a dead (zero-activation) channel is not important. A small weight connected to a huge-activation channel is not important. Both the weight magnitude and the activation magnitude must be large for the weight to matter.

**Per-weight importance:**

```
importance(W_ij) = s_j * |W_ij|

where:
  s_j = (1/T) * sum_{t=1}^{T} |X_{t,j}|

  s_j is the mean absolute activation for channel j
  |W_ij| is the absolute value of weight W at position (i,j)
```

The mean absolute activation `s_j` is the L1 channel norm from our statistics collector (`channel_l1_norm`). Using the L2 norm (RMS) is an alternative:

```
importance_L2(W_ij) = rms_j * |W_ij|

where:
  rms_j = sqrt((1/T) * sum_{t=1}^{T} X_{t,j}^2)
```

**Per-block importance:** To get a single score for a block of weights, we average the per-weight importance scores within the block.

```
importance(block_k) = (1/|B_k|) * sum_{(i,j) in B_k} importance(W_ij)
```

where `B_k` is the set of weight indices in block `k` and `|B_k|` is the block size.

**Block boundaries:** For a weight matrix of shape `(d_out, d_in)` with block size `g`, the blocks are defined along the input dimension. Block `k` for output row `i` contains weights `W[i, k*g : (k+1)*g]`. The total number of blocks per row is `ceil(d_in / g)`, and the total number of blocks for the entire matrix is `d_out * ceil(d_in / g)`.

```python
def compute_awq_importance(weight, channel_stats, block_size=64):
    """
    Compute AWQ-style activation-magnitude importance scores.

    Args:
        weight: weight matrix, shape (d_out, d_in)
        channel_stats: ChannelStats instance with calibration data
        block_size: number of weights per block

    Returns:
        importance: per-block importance scores,
                    shape (d_out, n_blocks_per_row)
    """
    d_out, d_in = weight.shape
    n_blocks = (d_in + block_size - 1) // block_size

    # Per-channel activation norm (L1 or L2)
    s = channel_stats.channel_l1_norm  # shape (d_in,)

    # Per-weight importance: s[j] * |W[i,j]|
    weight_abs = mx.abs(weight.astype(mx.float32))  # (d_out, d_in)
    per_weight = s[None, :] * weight_abs  # broadcast: (d_out, d_in)

    # Pad d_in to multiple of block_size
    if d_in % block_size != 0:
        pad_size = block_size - (d_in % block_size)
        per_weight = mx.pad(per_weight, [(0, 0), (0, pad_size)])

    # Reshape to (d_out, n_blocks, block_size) and average over block
    per_weight = per_weight.reshape(d_out, n_blocks, block_size)
    block_importance = mx.mean(per_weight, axis=2)  # (d_out, n_blocks)

    return block_importance
```

**Computational cost:**

```
Forward passes: C (number of calibration samples, typically 256-1024)
Scoring computation: O(d_out * d_in) per layer (a single element-wise multiply)
Total: O(C * model_forward_pass) + O(total_weights)
```

This is fast. The scoring computation itself is negligible compared to the calibration forward passes.

**Quality assessment:** AWQ-style scoring is the workhorse of importance-aware quantization. It captures roughly 85-90% of the information that more expensive methods (Hessian, Fisher) provide, at a fraction of the cost. For MXQ, this is the primary scoring method.

### 4b. Hessian Diagonal Scoring

**Origin:** Optimal Brain Damage (LeCun et al., 1989), Optimal Brain Surgeon (Hasselmo et al., 1993), and more recently, GPTQ (Frantar et al., 2023) and SqueezeLLM (Kim et al., 2023).

**Core idea:** The importance of a weight is determined by how much the loss function changes when that weight is perturbed. The Hessian matrix of the loss with respect to the weights captures this sensitivity. The diagonal of the Hessian gives a per-weight measure of curvature: how sharply the loss changes when a single weight is modified.

**Derivation:** Consider perturbing a single weight `W_ij` by a small amount `delta`. The Taylor expansion of the loss is:

```
L(W + delta * e_ij) = L(W) + delta * (dL/dW_ij) + (1/2) * delta^2 * H_ij,ij + O(delta^3)

where:
  e_ij is the unit vector in the direction of W_ij
  dL/dW_ij is the gradient (zero at a local minimum)
  H_ij,ij is the (ij,ij) diagonal element of the Hessian
```

At a local minimum (or near one, as pretrained models approximately are), the gradient term vanishes, leaving:

```
delta_L approximately equals (1/2) * delta^2 * H_ij,ij
```

If we quantize `W_ij` (replacing it with a nearby quantized value), the perturbation `delta` is the quantization error. The change in loss is proportional to:

```
delta_L proportional to (quantization_error)^2 * H_ij,ij
```

So the importance of a weight for quantization is:

```
importance(W_ij) = H_ij,ij * W_ij^2
```

The `W_ij^2` factor comes from the fact that the expected quantization error is proportional to the weight magnitude (larger weights have larger absolute rounding errors in most quantization schemes).

**Computing the Hessian diagonal:** For a linear layer `Y = XW^T`, the Hessian of the squared error loss with respect to `W` has a convenient form. Specifically, the diagonal elements of the Hessian with respect to column `j` of `W` are:

```
H_jj = sum_{t=1}^{T} X_{t,j}^2 = S2[j]
```

This is exactly the sum of squared activations per channel, which we already collect in our statistics (`sum_sq`). The full Hessian is `H = X^T X` (a `d_in x d_in` matrix), and its diagonal is the vector of per-channel squared activation sums.

**Per-weight Hessian importance:**

```
importance_hessian(W_ij) = W_ij^2 * H_jj = W_ij^2 * sum_{t=1}^{T} X_{t,j}^2
```

**Per-block Hessian importance:**

```
importance_hessian(block_k) = (1/|B_k|) * sum_{(i,j) in B_k} W_ij^2 * H_jj
```

```python
def compute_hessian_importance(weight, channel_stats, block_size=64):
    """
    Compute Hessian-diagonal importance scores.

    Args:
        weight: weight matrix, shape (d_out, d_in)
        channel_stats: ChannelStats instance
        block_size: number of weights per block

    Returns:
        importance: per-block scores, shape (d_out, n_blocks)
    """
    d_out, d_in = weight.shape
    n_blocks = (d_in + block_size - 1) // block_size

    # Hessian diagonal: H_jj = sum of squared activations for channel j
    h_diag = channel_stats.hessian_diagonal  # shape (d_in,), = sum_sq

    # Per-weight importance: W_ij^2 * H_jj
    w_sq = weight.astype(mx.float32) ** 2  # (d_out, d_in)
    per_weight = w_sq * h_diag[None, :]  # (d_out, d_in)

    # Pad and reshape to blocks
    if d_in % block_size != 0:
        pad_size = block_size - (d_in % block_size)
        per_weight = mx.pad(per_weight, [(0, 0), (0, pad_size)])

    per_weight = per_weight.reshape(d_out, n_blocks, block_size)
    block_importance = mx.mean(per_weight, axis=2)  # (d_out, n_blocks)

    return block_importance
```

**Relationship to AWQ scoring:** Hessian scoring and AWQ scoring are closely related. AWQ uses `s_j * |W_ij|` while Hessian uses `H_jj * W_ij^2`. Since `H_jj = sum(X_j^2)` and `s_j = mean(|X_j|)`, we can write:

```
AWQ:     s_j * |W_ij|  = mean(|X_j|) * |W_ij|
Hessian: H_jj * W_ij^2 = T * mean(X_j^2) * W_ij^2
```

The key differences:
1. Hessian uses squared weights vs. absolute weights -- Hessian penalizes large weights more aggressively
2. Hessian uses mean squared activations (L2) vs. mean absolute activations (L1) -- Hessian is more sensitive to outlier activations
3. Hessian has a stronger theoretical foundation (derived from loss curvature)

**Computational cost:** Same as AWQ -- the Hessian diagonal is computed from the same statistics we already collect. The scoring computation is `O(d_out * d_in)` per layer.

**Quality assessment:** Hessian diagonal scoring is moderately better than AWQ scoring, especially for layers with outlier channels or unusual weight distributions. The improvement is roughly 2-5% in terms of quantization error reduction at the same average bit width. Given that it costs the same to compute, there is no reason not to use it alongside AWQ scoring.

### 4c. Fisher Information Scoring

**Origin:** Fisher information pruning and quantization literature, used in various forms in adaptive quantization methods.

**Core idea:** The Fisher information measures how much information each weight carries about the model's predictions. Weights with high Fisher information are those where the model's output is most sensitive to their precise values -- these are the weights we should preserve with more bits.

**Definition:** The Fisher information for weight `W_ij` is:

```
F_ij = E_x [ (d log P(y|x; W) / dW_ij)^2 ]

where the expectation is over the data distribution x, y
```

In practice, we approximate this with the calibration data:

```
F_ij = (1/T) * sum_{t=1}^{T} (dL_t / dW_ij)^2

where L_t is the loss for token t
```

This is the average squared gradient of the loss with respect to each weight. Weights that have large gradients (on average) are those where small changes cause large loss changes.

**Connection to the Hessian:** Under certain conditions (near a local minimum, well-specified model), the Fisher information matrix equals the Hessian. In practice, they differ because:
1. The model is not exactly at a local minimum
2. The calibration data may not match the training distribution
3. The Fisher uses squared gradients while the Hessian uses second derivatives

**Computation:**

```python
def compute_fisher_importance(model, calibration_data, block_size=64):
    """
    Compute Fisher information importance scores.
    WARNING: Requires backward passes -- much more expensive than
    AWQ/Hessian.

    Args:
        model: full-precision model (must support gradient computation)
        calibration_data: list of tokenized samples
        block_size: weights per block

    Returns:
        dict mapping layer_name -> per-block importance scores
    """
    fisher_scores = {}

    for sample in calibration_data:
        tokens = mx.array(sample["input_ids"]).reshape(1, -1)

        # Forward pass with gradient tracking
        def loss_fn(model_params):
            logits = model(tokens)
            # Cross-entropy loss
            targets = tokens[:, 1:]  # shifted targets
            logits = logits[:, :-1, :]  # aligned logits
            loss = mx.mean(
                nn.losses.cross_entropy(logits, targets)
            )
            return loss

        # Compute gradients
        loss, grads = nn.value_and_grad(model, loss_fn)(model)

        # Accumulate squared gradients per weight
        for name, grad in grads.items():
            if name not in fisher_scores:
                fisher_scores[name] = mx.zeros_like(
                    grad, dtype=mx.float32
                )
            fisher_scores[name] += grad.astype(mx.float32) ** 2

    # Average over samples
    n_samples = len(calibration_data)
    for name in fisher_scores:
        fisher_scores[name] /= n_samples

    # Convert to per-block scores
    block_scores = {}
    for name, per_weight_fisher in fisher_scores.items():
        d_out, d_in = per_weight_fisher.shape
        n_blocks = (d_in + block_size - 1) // block_size

        if d_in % block_size != 0:
            pad_size = block_size - (d_in % block_size)
            per_weight_fisher = mx.pad(
                per_weight_fisher, [(0, 0), (0, pad_size)]
            )

        per_weight_fisher = per_weight_fisher.reshape(
            d_out, n_blocks, block_size
        )
        block_scores[name] = mx.mean(per_weight_fisher, axis=2)

    return block_scores
```

**Computational cost:**

```
Forward passes: C (calibration samples)
Backward passes: C (one per forward pass)
Total: O(C * (model_forward + model_backward))
     = O(C * 3 * model_forward)  [backward is roughly 2x forward]
     = 3x the cost of AWQ/Hessian calibration
```

For a 72B model with 512 calibration samples, this means 3x the calibration time. On Apple Silicon with unified memory, the backward pass also requires storing more intermediate activations (for gradient computation), potentially doubling peak memory usage.

**Quality assessment:** Fisher information is theoretically the most principled scoring method. In practice, it provides roughly 5-10% better quantization error reduction compared to Hessian diagonal scoring at low bit widths (2-3 bits). However, the 3x compute cost and 2x memory cost make it impractical for routine use. Fisher scoring is most valuable as a reference: use it to validate that cheaper methods (AWQ, Hessian) are producing reasonable rankings, then use the cheaper methods in production.

### 4d. Sensitivity Analysis (Perturbation-Based)

**Origin:** Direct measurement approach used in various neural network compression methods.

**Core idea:** Instead of estimating importance from proxy metrics (activations, Hessian, gradients), directly measure the effect of quantizing each block. For each block, quantize it to the target bit width while keeping all other blocks at full precision, and measure how much the model's output changes. The blocks where quantization causes the most output change are the most important.

**Algorithm:**

```
For each weight block B_k in the model:
    1. Save original weights: W_orig = W[B_k]
    2. Quantize just this block: W_quant = quantize(W[B_k], target_bits)
    3. Replace W[B_k] with W_quant
    4. Run all calibration samples through the model
    5. Compute KL divergence between original and perturbed outputs:
       score(B_k) = (1/T) * sum_t KL(P_orig(.|x_t) || P_perturbed(.|x_t))
    6. Restore original weights: W[B_k] = W_orig
```

**KL divergence computation:**

```
KL(P || Q) = sum_v P(v) * log(P(v) / Q(v))

where the sum is over the vocabulary V
```

For each token position, the model outputs a probability distribution over the vocabulary. The KL divergence measures how much this distribution changes when a single block is quantized. Large KL divergence means the block is important.

**Practical optimization -- block-group sensitivity:** Computing per-block sensitivity for millions of blocks is prohibitively expensive. Instead, compute sensitivity at the level of block groups:

```
Block group = all blocks in one layer's weight matrix for one row group

For a weight matrix of shape (d_out, d_in) with block_size 64:
  n_blocks = d_out * ceil(d_in / 64)
  n_block_groups = ceil(d_out / 128) * ceil(d_in / 64)

Quantize an entire block group at once, measure KL divergence.
Then distribute the group's score across its constituent blocks
proportionally to their AWQ/Hessian scores.
```

This reduces the number of forward passes from `n_blocks * C` to `n_block_groups * C`, which is typically 100-1000x fewer.

```python
def compute_sensitivity_importance(
    model, calibration_data, target_bits=3,
    group_size=128, block_size=64
):
    """
    Compute perturbation-based sensitivity scores.

    This is the most expensive scoring method but provides ground truth.
    Use for validation, not routine calibration.

    Args:
        model: full-precision model
        calibration_data: list of tokenized samples
        target_bits: bit width for perturbation quantization
        group_size: number of output rows per block group
        block_size: weights per block (input dimension)

    Returns:
        dict mapping layer_name -> sensitivity scores per block group
    """
    # First, collect reference outputs
    reference_logits = []
    for sample in calibration_data:
        tokens = mx.array(sample["input_ids"]).reshape(1, -1)
        logits = model(tokens)
        reference_logits.append(logits)

    sensitivity_scores = {}

    for layer_name, layer_module in model.named_modules():
        if not isinstance(layer_module, nn.Linear):
            continue

        weight = layer_module.weight
        d_out, d_in = weight.shape
        n_output_groups = (d_out + group_size - 1) // group_size
        n_input_blocks = (d_in + block_size - 1) // block_size

        layer_scores = mx.zeros(
            (n_output_groups, n_input_blocks), dtype=mx.float32
        )

        for og in range(n_output_groups):
            for ib in range(n_input_blocks):
                # Define block group boundaries
                row_start = og * group_size
                row_end = min((og + 1) * group_size, d_out)
                col_start = ib * block_size
                col_end = min((ib + 1) * block_size, d_in)

                # Save original weights
                original = weight[
                    row_start:row_end, col_start:col_end
                ].copy()

                # Quantize this block group
                quantized = uniform_quantize(original, target_bits)
                weight[
                    row_start:row_end, col_start:col_end
                ] = quantized

                # Measure KL divergence
                total_kl = 0.0
                for idx, sample in enumerate(calibration_data):
                    tokens = mx.array(
                        sample["input_ids"]
                    ).reshape(1, -1)
                    perturbed_logits = model(tokens)

                    # KL divergence
                    ref_probs = mx.softmax(
                        reference_logits[idx], axis=-1
                    )
                    pert_probs = mx.softmax(
                        perturbed_logits, axis=-1
                    )
                    kl = mx.sum(
                        ref_probs * (
                            mx.log(ref_probs + 1e-10)
                            - mx.log(pert_probs + 1e-10)
                        ),
                        axis=-1
                    )
                    total_kl += mx.mean(kl).item()

                layer_scores[og, ib] = (
                    total_kl / len(calibration_data)
                )

                # Restore original weights
                weight[
                    row_start:row_end, col_start:col_end
                ] = original

        sensitivity_scores[layer_name] = layer_scores

    return sensitivity_scores
```

**Computational cost:**

```
Forward passes per block group: C (calibration samples)
Number of block groups per layer: ceil(d_out/group_size) * ceil(d_in/block_size)
Number of layers: L

Total forward passes = C * sum_layers(n_block_groups_per_layer)

For a 70B model (L=80, d_out=8192, d_in=8192, group_size=128, block_size=64):
  Block groups per layer: 64 * 128 = 8,192
  Total block groups: 80 * 8,192 = 655,360
  Forward passes: 655,360 * 256 = ~168 million forward passes

This is completely impractical for routine use.
```

**Practical use in MXQ:** Sensitivity analysis is used only for validation:
1. Run on a small model (1B-7B parameters) to validate that AWQ/Hessian scoring produces the same ranking as sensitivity analysis
2. Spot-check specific layers where AWQ/Hessian scores seem suspicious
3. Never run on the full model as part of the calibration pipeline

**Quality assessment:** Sensitivity analysis is the gold standard. It directly measures what we care about (effect of quantization on model output). All other methods are approximations of what sensitivity analysis measures. If AWQ/Hessian scoring agrees with sensitivity analysis on the small model, we can trust AWQ/Hessian scoring on the large model.

### 4e. Combined Scoring

MXQ uses a combined scoring approach that takes the best of each method while keeping computational cost manageable.

**Primary approach: weighted combination of AWQ and Hessian.**

```
score(block_k) = alpha * awq_score(block_k) + beta * hessian_score(block_k)
```

The scores must be normalized before combining because they have different scales:

```
normalized_awq(block_k) = awq_score(block_k) / mean(awq_scores_for_layer)
normalized_hessian(block_k) = hessian_score(block_k) / mean(hessian_scores_for_layer)

score(block_k) = alpha * normalized_awq(block_k) + beta * normalized_hessian(block_k)
```

Default weights: `alpha = 0.5, beta = 0.5` (equal weighting). These can be tuned on a validation set.

**Why not just use one method?** AWQ and Hessian scoring capture slightly different aspects of importance:
- AWQ captures the "traffic volume" through each weight: how much activation flows through it
- Hessian captures the "sensitivity" of each weight: how much the loss changes if it is perturbed

In most cases, these agree. But they diverge for specific weight patterns:

- **High activation, small weight:** AWQ rates this as moderately important (large activation * small weight). Hessian rates it as less important (small weight^2 * large Hessian). The Hessian is usually right here: small weights can be quantized more aggressively.

- **Low activation, large weight:** AWQ rates this as moderately important. Hessian rates it as less important (large weight^2 but small Hessian because low activation). This is where AWQ is often right: large weights, even if rarely activated, may be important for rare but critical inputs (e.g., rare tokens, edge cases).

By combining both, we get a more robust score that is less likely to make catastrophic errors on any particular block.

**Outlier protection layer:**

After computing the combined score, apply outlier channel protection:

```python
def compute_combined_importance(weight, channel_stats, block_size=64,
                                 alpha=0.5, beta=0.5):
    """
    Compute combined AWQ + Hessian importance with outlier protection.
    """
    # Compute individual scores
    awq = compute_awq_importance(weight, channel_stats, block_size)
    hessian = compute_hessian_importance(
        weight, channel_stats, block_size
    )

    # Normalize per-layer (so scores are comparable across methods)
    awq_norm = awq / (mx.mean(awq) + 1e-10)
    hessian_norm = hessian / (mx.mean(hessian) + 1e-10)

    # Combine
    combined = alpha * awq_norm + beta * hessian_norm

    # Outlier protection: boost blocks that process outlier channels
    outlier_mask = detect_outlier_channels(channel_stats)
    n_blocks = combined.shape[1]
    d_in = weight.shape[1]

    for b in range(n_blocks):
        col_start = b * block_size
        col_end = min((b + 1) * block_size, d_in)
        if mx.any(outlier_mask[col_start:col_end]):
            # Boost importance to guarantee minimum bit allocation
            combined[:, b] = mx.maximum(
                combined[:, b],
                mx.max(combined) * 0.9  # Ensure top-10% importance
            )

    return combined
```

**Alternative: Hessian-corrected AWQ.** Instead of a linear combination, use AWQ as the base score and apply Hessian as a correction factor:

```
score(block_k) = awq_score(block_k) * (1 + gamma * hessian_correction(block_k))

where:
  hessian_correction(block_k) = (hessian_score(block_k) / mean(hessian)) - 1

  gamma = correction strength (default 0.3)
```

This approach says: "Start with the AWQ ranking, but adjust it upward for blocks where the Hessian says the loss is extra-sensitive, and downward for blocks where the Hessian says the loss is insensitive." The advantage is that AWQ provides the overall ranking (which is usually correct) while Hessian provides targeted corrections for edge cases.

**The practical recommendation for MXQ:**

```
For calibration speed and quality:
  1. Always compute AWQ scores (free -- uses activation statistics from forward passes)
  2. Always compute Hessian diagonal scores (free -- uses the same statistics)
  3. Combine with equal weights (alpha=0.5, beta=0.5)
  4. Apply outlier protection
  5. Use Fisher/sensitivity only for validation on small models

This gives 90-95% of the quality of the most expensive methods (Fisher, sensitivity)
at roughly 0% additional cost over basic AWQ.
```

---

## 5. Importance Matrix (imatrix) Format

### What the Importance Matrix Contains

The importance matrix is the complete output of the calibration pipeline. It contains everything needed by the bit allocation algorithm (Phase 2) to decide how many bits each weight block gets. The importance matrix must be:

1. **Complete:** A score for every weight block in every layer
2. **Self-describing:** Contains metadata about how it was produced
3. **Reproducible:** Includes enough information to regenerate it from the same model and data
4. **Efficient:** Fast to load and small relative to the model itself

### Content Structure

For a model with `L` transformer layers, each containing multiple Linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), plus an embedding layer and language model head, the importance matrix stores:

```
For each Linear layer in the model:
  - layer_name: string (e.g., "model.layers.15.self_attn.q_proj")
  - importance_scores: float32 tensor of shape (d_out, n_blocks_per_row)
  - channel_stats:
    - sum_sq: float32 tensor of shape (d_in,)      -- Hessian diagonal
    - l1_norm: float32 tensor of shape (d_in,)      -- mean |activation|
    - l2_norm: float32 tensor of shape (d_in,)      -- RMS activation
    - max_abs: float32 tensor of shape (d_in,)       -- max |activation|
    - mean: float32 tensor of shape (d_in,)          -- channel mean
    - variance: float32 tensor of shape (d_in,)      -- channel variance
  - outlier_channels: bool tensor of shape (d_in,)    -- outlier flags
  - weight_stats:
    - mean: float32 scalar                            -- mean of weight values
    - std: float32 scalar                             -- std of weight values
    - max_abs: float32 scalar                         -- max |weight|

Metadata:
  - format_version: "1.0"
  - scoring_method: "awq+hessian"  (or "awq", "hessian", "fisher", etc.)
  - scoring_weights: {"alpha": 0.5, "beta": 0.5}
  - block_size: 64
  - calibration_dataset: "mxq-calib-v1"
  - calibration_samples: 512
  - calibration_tokens: 1048576  (total tokens processed)
  - calibration_seq_length: 2048
  - dataset_hash: "sha256:abc123..."  (for reproducibility)
  - source_model: "Qwen/Qwen3.5-72B"
  - source_dtype: "bfloat16"
  - timestamp: "2026-03-14T10:30:00Z"
  - mxq_version: "0.1.0"
```

### File Format: Safetensors

MXQ stores the importance matrix in safetensors format. Safetensors provides:

1. **Memory-mapped loading:** Can load individual tensors without reading the entire file
2. **Metadata header:** JSON metadata stored in the file header
3. **Type safety:** Explicit dtype for each tensor
4. **Security:** No arbitrary code execution (unlike pickle-based formats)
5. **Ecosystem compatibility:** Used by HuggingFace, MLX, and most modern ML frameworks

```python
from safetensors.numpy import save_file, load_file
import numpy as np
import json

def save_imatrix(filepath, importance_data, metadata):
    """
    Save importance matrix to safetensors format.

    Args:
        filepath: output path (e.g., "imatrix.safetensors")
        importance_data: dict of {layer_name: {
            "importance": np.ndarray,
            "sum_sq": np.ndarray,
            "l1_norm": np.ndarray,
            "l2_norm": np.ndarray,
            "max_abs": np.ndarray,
            "mean": np.ndarray,
            "variance": np.ndarray,
            "outlier_channels": np.ndarray (uint8, 0 or 1)
        }}
        metadata: dict of calibration metadata
    """
    tensors = {}

    for layer_name, data in importance_data.items():
        # Store importance scores
        tensors[f"{layer_name}.importance"] = data[
            "importance"
        ].astype(np.float32)

        # Store channel statistics
        for stat_name in [
            "sum_sq", "l1_norm", "l2_norm",
            "max_abs", "mean", "variance"
        ]:
            tensors[f"{layer_name}.{stat_name}"] = data[
                stat_name
            ].astype(np.float32)

        tensors[f"{layer_name}.outlier_channels"] = data[
            "outlier_channels"
        ].astype(np.uint8)

    # Safetensors metadata must be {str: str}
    meta_str = {
        k: json.dumps(v) if not isinstance(v, str) else v
        for k, v in metadata.items()
    }

    save_file(tensors, filepath, metadata=meta_str)


def load_imatrix(filepath):
    """
    Load importance matrix from safetensors.

    Returns:
        importance_data: dict of layer data
        metadata: dict of calibration metadata
    """
    from safetensors import safe_open

    tensors = {}
    metadata = {}

    with safe_open(filepath, framework="numpy") as f:
        metadata_raw = f.metadata()
        # Parse JSON-encoded metadata values
        for k, v in metadata_raw.items():
            try:
                metadata[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                metadata[k] = v

        # Load all tensors
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Organize by layer
    importance_data = {}
    for key, tensor in tensors.items():
        parts = key.rsplit(".", 1)
        layer_name = parts[0]
        field_name = parts[1]

        if layer_name not in importance_data:
            importance_data[layer_name] = {}
        importance_data[layer_name][field_name] = tensor

    return importance_data, metadata
```

### File Size Estimates

For a 72B model (80 layers, each with 7 Linear sublayers):

```
Per Linear layer:
  importance scores: d_out * n_blocks * 4 bytes
  For q_proj (8192 x 8192, block_size 64):
    importance: 8192 * 128 * 4 = 4 MB
    channel_stats: 8192 * 6 * 4 = 192 KB
    outlier_channels: 8192 * 1 = 8 KB
    Total per sublayer: ~4.2 MB

Per transformer layer:
  7 sublayers * 4.2 MB = ~29 MB

Total for 80 transformer layers:
  80 * 29 MB = ~2.3 GB

Plus embedding and lm_head:
  ~50 MB

Grand total: ~2.4 GB
```

This is significant (1.6% of the bf16 model size) but acceptable. The importance matrix is computed once and reused for all quantization profiles (MXQ-2, MXQ-2.5, MXQ-3, etc.).

**Optimization: store only block-level scores.** If we omit the per-channel statistics (which are only needed for debugging and re-scoring) and store only the per-block importance scores, the file size drops dramatically:

```
Per Linear layer (scores only):
  importance: 8192 * 128 * 4 = 4 MB

Total: 80 * 7 * 4 MB + 50 MB = ~2.3 GB (still dominated by scores)
```

The scores themselves are the bulk. To reduce further, we can quantize the importance scores themselves to float16 (cutting in half to ~1.2 GB) or to 8-bit integers (cutting to ~600 MB). Since the scores are only used for ranking (not for precise computation), even 8-bit precision is sufficient.

### Comparison with llama.cpp imatrix Format

The llama.cpp imatrix format is simpler:

```
llama.cpp imatrix binary format:
  - Header: n_entries (uint32)
  - For each tensor:
    - tensor_name: null-terminated string
    - n_values: uint32
    - n_calibration_tokens: uint32
    - values: float32[n_values]  -- one value per column (input channel)
```

The values are the raw sum of squared input activations per channel (our `sum_sq`). llama.cpp does not store per-block scores, combined metrics, outlier flags, or rich metadata. The MXQ format is a strict superset of what llama.cpp stores, providing much richer information for the bit allocation algorithm.

---

## 6. From Importance Scores to Bit Allocation

### The Bit Allocation Problem

Given:
- An importance matrix with a score for every weight block
- A target average bit width (e.g., 2.5 bits)
- A set of available bit widths per block (2, 3, 4, 5, 6, 8)
- Layer-type priors (minimum bits for specific layer types)

Find:
- The bit width assignment for every block that minimizes total quantization error, subject to the average bit width constraint.

This is a constrained optimization problem. The formal statement:

```
Minimize:    sum_k  error(block_k, bits_k)
Subject to:  (1/K) * sum_k bits_k = target_bits
             bits_k in {2, 3, 4, 5, 6, 8} for all k
             bits_k >= min_bits(layer_type(block_k)) for all k

where K is the total number of blocks
```

### Estimating Quantization Error from Importance Scores

The error function `error(block_k, bits_k)` estimates how much error quantizing block `k` to `bits_k` bits will introduce. We do not need to actually quantize each block to estimate this; we can derive an analytical approximation.

For a block of weights with importance score `s_k`, the quantization error at `b` bits is approximately:

```
error(block_k, b) = s_k * variance(block_k) / (2^(2b) - 1)
```

The intuition: quantization with `b` bits introduces rounding error proportional to the step size, which is inversely proportional to the number of quantization levels (`2^b`). The squared step size (which relates to mean squared error) is inversely proportional to `(2^b)^2 = 2^(2b)`. The importance score `s_k` scales the error by how much this block's error matters for the model's output. The variance of the block's weights determines the magnitude of the rounding error (higher variance means larger dynamic range, meaning larger step sizes at the same bit width).

This formulation allows us to solve the allocation problem analytically, without trial-and-error quantization.

### Algorithm 1: Greedy Bit Allocation

The simplest effective algorithm. It produces near-optimal results and is easy to implement and debug.

```
Algorithm: Greedy Bit Allocation

Input:
  importance[k] for k = 1..K  (importance score per block)
  target_bits                   (target average, e.g., 2.5)
  min_bits[k] for k = 1..K    (per-block minimum from layer-type priors)
  available_bits = {2, 3, 4, 5, 6, 8}

Output:
  bits[k] for k = 1..K  (bit width assignment per block)

Procedure:
  1. Initialize: bits[k] = min_bits[k] for all k
  2. Compute current_avg = mean(bits[k])
  3. While current_avg < target_bits:
     a. For each block k where bits[k] < max(available_bits):
        Compute upgrade_priority[k] = importance[k] / cost_of_upgrade[k]
        where cost_of_upgrade[k] = next_bits(bits[k]) - bits[k]
        and next_bits(b) = smallest element of available_bits greater than b
     b. Select k* = argmax(upgrade_priority)
     c. bits[k*] = next_bits(bits[k*])
     d. Update current_avg = mean(bits[k])
  4. Return bits[k]
```

The priority metric `importance[k] / cost_of_upgrade[k]` is a "bang for the buck" ratio: it prefers upgrading blocks that are both important and cheap to upgrade (e.g., upgrading from 2 to 3 bits costs 1 bit, while upgrading from 4 to 5 also costs 1 bit, but the importance-per-bit-spent is what matters).

```python
def greedy_bit_allocation(importance_scores, target_bits,
                           min_bits_per_block, block_sizes,
                           available_bits=(2, 3, 4, 5, 6, 8)):
    """
    Greedy bit allocation algorithm.

    Args:
        importance_scores: flat array of per-block importance, shape (K,)
        target_bits: target average bit width (e.g., 2.5)
        min_bits_per_block: flat array of minimum bits per block, shape (K,)
        block_sizes: flat array of actual block size per block, shape (K,)
                     (last block in a row may be smaller)
        available_bits: tuple of allowed bit widths

    Returns:
        bits: array of bit width assignments, shape (K,)
    """
    K = len(importance_scores)
    available = sorted(available_bits)

    # Initialize at minimum bits
    bits = min_bits_per_block.copy()

    # Total bits budget
    total_weights = sum(block_sizes)
    target_total_bits = target_bits * total_weights
    current_total_bits = sum(
        bits[k] * block_sizes[k] for k in range(K)
    )

    while current_total_bits < target_total_bits:
        best_k = -1
        best_priority = -1.0

        for k in range(K):
            current_b = bits[k]
            if current_b >= available[-1]:
                continue  # Already at max bits

            # Find next available bit width
            next_b = None
            for ab in available:
                if ab > current_b:
                    next_b = ab
                    break
            if next_b is None:
                continue

            cost = (next_b - current_b) * block_sizes[k]
            priority = importance_scores[k] / cost

            if priority > best_priority:
                best_priority = priority
                best_k = k

        if best_k == -1:
            break  # All blocks at max bits

        # Upgrade best block
        current_b = bits[best_k]
        for ab in available:
            if ab > current_b:
                next_b = ab
                break

        old_bits = bits[best_k] * block_sizes[best_k]
        bits[best_k] = next_b
        new_bits = bits[best_k] * block_sizes[best_k]
        current_total_bits += (new_bits - old_bits)

    return bits
```

**Complexity:** `O(K^2)` in the worst case (each iteration scans all K blocks, and there can be up to K iterations). For a 72B model with K = 2.2 million blocks, this is about 5 * 10^12 operations, which is too slow.

**Optimized implementation:** Use a max-heap (priority queue) to avoid scanning all blocks each iteration:

```python
import heapq

def greedy_bit_allocation_fast(importance_scores, target_bits,
                                min_bits_per_block, block_sizes,
                                available_bits=(2, 3, 4, 5, 6, 8)):
    """
    Optimized greedy bit allocation using a priority queue.
    Complexity: O(K log K) instead of O(K^2).
    """
    K = len(importance_scores)
    available = sorted(available_bits)
    bits = min_bits_per_block.copy()

    total_weights = sum(block_sizes)
    target_total_bits = target_bits * total_weights
    current_total_bits = sum(
        bits[k] * block_sizes[k] for k in range(K)
    )

    def get_next_bits(current_b):
        for ab in available:
            if ab > current_b:
                return ab
        return None

    # Build max-heap (negate priority for min-heap behavior)
    heap = []
    for k in range(K):
        next_b = get_next_bits(bits[k])
        if next_b is not None:
            cost = (next_b - bits[k]) * block_sizes[k]
            priority = importance_scores[k] / cost
            heapq.heappush(heap, (-priority, k))

    while current_total_bits < target_total_bits and heap:
        neg_priority, k = heapq.heappop(heap)

        # Verify the priority is still current (bits[k] may have
        # changed since this entry was pushed)
        next_b = get_next_bits(bits[k])
        if next_b is None:
            continue

        expected_cost = (next_b - bits[k]) * block_sizes[k]
        expected_priority = importance_scores[k] / expected_cost
        if abs(-neg_priority - expected_priority) > 1e-10:
            # Stale entry, re-push with correct priority
            heapq.heappush(heap, (-expected_priority, k))
            continue

        # Upgrade this block
        old_bits_total = bits[k] * block_sizes[k]
        bits[k] = next_b
        new_bits_total = bits[k] * block_sizes[k]
        current_total_bits += (new_bits_total - old_bits_total)

        # Push next upgrade for this block
        next_next_b = get_next_bits(bits[k])
        if next_next_b is not None:
            cost = (next_next_b - bits[k]) * block_sizes[k]
            priority = importance_scores[k] / cost
            heapq.heappush(heap, (-priority, k))

    return bits
```

**Complexity:** `O(K log K)` -- each block is pushed to the heap at most once per bit level, and there are at most `len(available_bits)` bit levels. Total heap operations: `O(K * len(available_bits) * log K)`.

### Algorithm 2: Dynamic Programming (Knapsack) Allocation

The greedy algorithm is fast but not guaranteed to be optimal. The bit allocation problem can be formulated as a variant of the bounded knapsack problem and solved exactly with dynamic programming.

**Formulation:**

```
Items:     K blocks, each with weight = block_size_k
Capacity:  target_total_bits = target_bits * total_weights
Choices:   For each block, choose from len(available_bits) options
Value:     error_reduction(block_k, bits)
           = error(block_k, min_bits_k) - error(block_k, bits)
Cost:      additional_bits(block_k, bits)
           = (bits - min_bits_k) * block_size_k

Maximize:  sum_k error_reduction(block_k, bits_k)
Subject to: sum_k additional_bits(block_k, bits_k) <= budget
            budget = target_total_bits - sum_k(min_bits_k * block_size_k)
```

**DP solution:** Standard 0-1 knapsack DP, except each item has multiple choices (one per available bit width).

```
dp[k][b] = maximum total error reduction using blocks 0..k-1
            with total additional bits <= b

For each block k, for each total bits b:
  dp[k+1][b] = max over all bit choices c in available_bits
                where c >= min_bits[k]:
    dp[k][b - additional_cost(k, c)] + error_reduction(k, c)
```

**Complexity:** `O(K * B * len(available_bits))` where `B` is the total budget in bits. For a 72B model, `K = 2.2M` and `B` could be in the billions, making exact DP infeasible.

**Practical DP approach:** Group blocks by layer type and solve the DP at the layer level rather than block level. This reduces `K` from millions to hundreds.

```
Layer-level DP:
  Items: L_total layers (e.g., 80 * 7 = 560 for a 72B model)
  For each layer, precompute the error at each possible average bit width
  Choices per layer: average bits in
    {2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0}
  Use DP to find the layer-level bit allocation that minimizes
  total error

Then within each layer, use the greedy algorithm to distribute the
layer's bit budget across its individual blocks.
```

This two-level approach (DP across layers, greedy within layers) gives near-optimal results at practical computational cost.

### Layer-Type Priors

Certain layer types are systematically more sensitive to quantization than others. These priors are applied as minimum bit constraints before the allocation algorithm runs.

```
Layer-type minimum bits:

  embed_tokens (embedding layer): min 4-bit
    Reason: the embedding table directly maps token IDs to vectors.
    Low-bit quantization here corrupts the basic token representations
    that all subsequent computation depends on. Errors here cannot
    be corrected by later layers.

  lm_head (output projection): min 6-bit
    Reason: the lm_head projects hidden states to vocabulary logits.
    At inference time, the top-1 (or top-k) token is selected based
    on these logits. Even small errors can change which token is
    selected, directly affecting generation quality. The lm_head is
    often tied to embed_tokens (shared weights); if so, both get
    the higher minimum.

  self_attn.q_proj (query projection): min 3-bit
    Reason: queries are multiplied with keys to compute attention
    scores. Quantization errors in Q compound with errors in K,
    producing O(error^2) distortion in attention patterns.

  self_attn.k_proj (key projection): min 3-bit
    Reason: same as q_proj -- attention score = Q * K^T means
    errors in K are as harmful as errors in Q.

  self_attn.v_proj (value projection): min 3-bit
    Reason: values are weighted by attention scores and summed.
    Errors in V directly corrupt the output of each attention head.
    Slightly less sensitive than Q/K because the attention weights
    provide some averaging.

  self_attn.o_proj (output projection): min 3-bit
    Reason: projects concatenated attention heads back to the
    residual dimension. Less sensitive than Q/K but still important.

  mlp.gate_proj, mlp.up_proj: min 2-bit
    Reason: MLP layers are the most compressible. The SwiGLU
    activation (gate * up) provides error correction: if gate
    is slightly wrong, the output is still approximately correct
    because the multiplication with up_proj provides a "soft" gate.

  mlp.down_proj: min 2-bit
    Reason: projects MLP hidden states back to residual dimension.
    Similar compressibility to gate_proj and up_proj.

  First 2 transformer layers (layers 0, 1): +1 bit bonus
    Reason: errors in early layers propagate through all subsequent
    layers. The model has no opportunity to correct errors that
    occur at the input. Empirically, quantizing early layers too
    aggressively causes disproportionate quality loss.

  Last 2 transformer layers (layers N-2, N-1): +1 bit bonus
    Reason: the last layers are closest to the output and have
    the most direct influence on the generated tokens. There are
    no subsequent layers to correct errors introduced here.
```

**Applying priors to the allocation algorithm:**

```python
def compute_min_bits(model_config, n_layers,
                     available_bits=(2, 3, 4, 5, 6, 8)):
    """
    Compute per-block minimum bits based on layer-type priors.

    Returns:
        dict mapping layer_name -> minimum bit width
    """
    min_bits = {}

    for layer_idx in range(n_layers):
        # Base minimums by layer type
        prefix = f"model.layers.{layer_idx}"
        min_bits[f"{prefix}.self_attn.q_proj"] = 3
        min_bits[f"{prefix}.self_attn.k_proj"] = 3
        min_bits[f"{prefix}.self_attn.v_proj"] = 3
        min_bits[f"{prefix}.self_attn.o_proj"] = 3
        min_bits[f"{prefix}.mlp.gate_proj"] = 2
        min_bits[f"{prefix}.mlp.up_proj"] = 2
        min_bits[f"{prefix}.mlp.down_proj"] = 2

        # First/last layer bonus
        if layer_idx < 2 or layer_idx >= n_layers - 2:
            for key in list(min_bits.keys()):
                if f"layers.{layer_idx}." in key:
                    min_bits[key] = min(
                        min_bits[key] + 1, max(available_bits)
                    )

    # Embedding and lm_head
    min_bits["model.embed_tokens"] = 4
    min_bits["lm_head"] = 6

    return min_bits
```

### Bit Budget Arithmetic

For a target average of 2.5 bits on a 72B model:

```
Total parameters: ~72 billion
Total bits at bf16: 72B * 16 = 1,152 billion bits = 144 GB
Total bits at 2.5-bit average: 72B * 2.5 = 180 billion bits = 22.5 GB

Compression ratio: 144 / 22.5 = 6.4x

Bit budget after applying layer-type priors:
  embed_tokens: 150M params * 4 bits = 600M bits (0.33%)
  lm_head: 150M params * 6 bits = 900M bits (0.50%)
  Attention (Q,K,V,O): ~28B params * 3 bits min = 84B bits (46.7%)
  MLP (gate,up,down): ~43B params * 2 bits min = 86B bits (47.8%)
  First/last layer bonus: ~3.6B params * 1 extra bit = 3.6B bits (2.0%)

Total at minimum priors: ~175.1B bits
Target total: 180B bits
Remaining budget for upgrades: 4.9B bits

This means we can upgrade approximately:
  4.9B / 1 bit = 4.9B weights from their minimum to +1 bit
  or equivalently, ~6.8% of all weights get one extra bit above minimum
```

This shows that at 2.5-bit average, the bit budget is very tight. The allocation algorithm must be precise about which blocks deserve the extra bits. This is exactly why high-quality importance scoring matters so much at low bit widths.

---

## 7. Validation and Quality Measurement

### Perplexity Assessment

Perplexity is the standard metric for evaluating language model quality. It measures how well the model predicts the next token in a sequence.

**Definition:**

```
PPL = exp(-(1/N) * sum_{i=1}^{N} log P(token_i | token_1, ..., token_{i-1}))

where N is the total number of tokens in the evaluation dataset
```

Lower perplexity is better. A perplexity of 1.0 would mean the model perfectly predicts every token. A perplexity of V (vocabulary size, e.g., 152,000) would mean the model is no better than random guessing.

**Interpretation for quantization:** The key metric is the relative perplexity increase:

```
PPL_ratio = PPL(quantized) / PPL(full_precision)

PPL_ratio = 1.00: quantization introduces no error (impossible in practice)
PPL_ratio = 1.01-1.03: excellent (typical for MXQ-3 and above)
PPL_ratio = 1.03-1.05: good (target for MXQ-2.5)
PPL_ratio = 1.05-1.10: acceptable (aggressive compression)
PPL_ratio > 1.10: concerning (quality degradation is noticeable)
PPL_ratio > 1.50: poor (model outputs are noticeably degraded)
PPL_ratio > 2.00: unusable (model produces incoherent text)
```

**Standard evaluation datasets:**

- **Wikitext-2:** ~245K tokens of Wikipedia articles. The most widely used benchmark for perplexity. Small enough to run quickly. Cons: narrow domain (encyclopedic text), does not test code/math/chat.

- **C4 validation set:** ~365K tokens of cleaned web text. More diverse than Wikitext-2. Better for evaluating general-purpose models.

- **LAMBADA:** 5,153 passages requiring last-word prediction. Tests whether the model maintains long-range coherence. Useful for detecting quantization damage to long-range attention patterns.

**Perplexity measurement implementation:**

```python
def measure_perplexity(model, dataset, seq_length=2048, stride=512):
    """
    Measure model perplexity on a dataset.

    Args:
        model: the model to assess (quantized or full-precision)
        dataset: tokenized text as a flat array of token IDs
        seq_length: context window for the run
        stride: sliding window stride (overlap = seq_length - stride)

    Returns:
        perplexity: float
    """
    total_log_likelihood = 0.0
    total_tokens = 0

    n_tokens = len(dataset)

    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + seq_length, n_tokens)
        input_ids = mx.array(dataset[begin:end]).reshape(1, -1)

        logits = model(input_ids)  # (1, seq_len, vocab_size)

        # Only count tokens in the non-overlapping region
        # (except for the first window, which counts all tokens)
        if begin == 0:
            target_start = 0
        else:
            target_start = seq_length - stride

        # Shift for next-token prediction
        shift_logits = logits[0, target_start:-1, :]  # (n, vocab)
        shift_targets = input_ids[0, target_start + 1:]  # (n,)

        # Cross-entropy loss
        log_probs = mx.log_softmax(shift_logits, axis=-1)
        token_log_probs = mx.take_along_axis(
            log_probs,
            shift_targets[:, None],
            axis=-1
        ).squeeze(-1)

        total_log_likelihood += mx.sum(token_log_probs).item()
        total_tokens += len(shift_targets)

        if end >= n_tokens:
            break

    avg_neg_log_likelihood = -total_log_likelihood / total_tokens
    perplexity = float(np.exp(avg_neg_log_likelihood))

    return perplexity
```

### KL Divergence

KL divergence measures how much the quantized model's output distribution differs from the full-precision model's output distribution. Unlike perplexity (which requires ground truth labels), KL divergence directly compares the two models.

```
KL(P_full || P_quant) = sum_v P_full(v) * log(P_full(v) / P_quant(v))

where the sum is over all vocabulary tokens v
```

**Per-token KL divergence:** Computed at each token position, then averaged:

```
mean_KL = (1/N) * sum_{i=1}^{N} KL(P_full(.|context_i) || P_quant(.|context_i))
```

**Interpretation:**
```
mean_KL < 0.001: negligible distribution shift
mean_KL 0.001-0.01: minimal shift (typical for MXQ-3+)
mean_KL 0.01-0.1: noticeable shift (typical for MXQ-2.5)
mean_KL 0.1-1.0: significant shift (aggressive compression)
mean_KL > 1.0: severe shift (model behavior substantially altered)
```

### Per-Layer Error Analysis

Beyond aggregate metrics, MXQ tracks the quantization error at each layer individually. This helps diagnose problems: if one layer has disproportionate error, the importance scores or bit allocation for that layer may need adjustment.

**Relative Frobenius norm error:**

```
layer_error = ||W_original - W_quantized||_F / ||W_original||_F

where ||.||_F is the Frobenius norm (sqrt of sum of squared elements)
```

This measures what fraction of the weight's total magnitude is lost to quantization error. A relative error of 0.01 (1%) is excellent; 0.10 (10%) is typical for 3-bit quantization; 0.30 (30%) or higher indicates problematic quantization.

**Per-layer error budget:** The importance-based allocation should produce a roughly uniform per-layer KL divergence contribution. If one layer's contribution to total KL divergence is 10x higher than another, the allocation is suboptimal: the high-error layer needs more bits, and the low-error layer can spare bits.

```python
def per_layer_error_analysis(original_model, quantized_model):
    """
    Compute per-layer quantization error metrics.

    Returns:
        dict mapping layer_name -> {
            "frobenius_relative": float,
            "max_abs_error": float,
            "mean_abs_error": float,
        }
    """
    errors = {}

    for (name_o, param_o), (name_q, param_q) in zip(
        original_model.named_parameters(),
        quantized_model.named_parameters()
    ):
        assert name_o == name_q

        # Dequantize if needed
        w_orig = param_o.astype(mx.float32)
        w_quant = param_q.astype(mx.float32)

        diff = w_orig - w_quant
        frobenius_orig = mx.sqrt(mx.sum(w_orig ** 2)).item()
        frobenius_diff = mx.sqrt(mx.sum(diff ** 2)).item()

        errors[name_o] = {
            "frobenius_relative": (
                frobenius_diff / (frobenius_orig + 1e-10)
            ),
            "max_abs_error": mx.max(mx.abs(diff)).item(),
            "mean_abs_error": mx.mean(mx.abs(diff)).item(),
        }

    return errors
```

### Task-Specific Benchmarks

Perplexity and KL divergence are aggregate measures. They tell us the overall quality but not how specific capabilities are affected. MXQ validates quantized models on task-specific benchmarks:

**MMLU (Massive Multitask Language Understanding).** 57 subjects, 14,042 multiple-choice questions. Tests factual knowledge and reasoning. Quantization typically affects MMLU by 1-3% at 3-bit average, 3-7% at 2.5-bit average.

**HumanEval (code generation).** 164 Python programming problems. Tests code generation quality. Code is disproportionately affected by quantization because:
- Syntax requires exact token prediction (a single wrong token breaks the program)
- Variable names and function signatures must be precisely recalled
- Logical operators (==, !=, >=) differ by a single token

MXQ addresses this by including code in the calibration dataset, ensuring code-relevant weights receive sufficient bits.

**GSM8K (math reasoning).** 1,319 grade-school math word problems. Tests arithmetic and step-by-step reasoning. Math is sensitive to quantization because:
- Numerical computations require precise intermediate values
- Chain-of-thought reasoning is sequential (errors compound)

**ARC (AI2 Reasoning Challenge).** 7,787 science questions requiring reasoning. Tests common-sense and scientific reasoning. Less sensitive to quantization than code or math.

### Calibration Validation

The importance matrix itself must be validated for stability and correctness.

**Stability test:** Run calibration twice with different random subsets of the calibration dataset. Compare the resulting importance scores.

```
stability_metric = correlation(scores_run1, scores_run2)

For Pearson correlation:
  > 0.99: excellent stability (use 256+ samples)
  0.95-0.99: good stability (use 128-256 samples)
  0.90-0.95: marginal stability (use more samples)
  < 0.90: unstable (need significantly more samples or better dataset)
```

If the importance scores are unstable (correlation < 0.95), the calibration dataset is too small or not representative enough. Adding more samples will improve stability up to a point; beyond that, the dataset composition needs improvement.

**Cross-domain test:** Calibrate on one dataset (e.g., code-only), assess on another (e.g., prose). If the code-calibrated model performs much worse on prose than a prose-calibrated model, the calibration dataset is not diverse enough.

**Rank correlation test:** Compare the block ranking (sorted by importance score) between different calibration runs. Use Spearman rank correlation:

```
spearman_rho = correlation(rank(scores_run1), rank(scores_run2))

Since bit allocation depends on ranking (not absolute scores), rank
correlation is more relevant than Pearson correlation.

spearman_rho > 0.98: allocation will be nearly identical between runs
spearman_rho > 0.95: allocation will differ by at most a few blocks
spearman_rho < 0.90: allocation is unstable, need more calibration data
```

---

## 8. Practical Tips for High-Quality Calibration

### Sample Count

**Minimum:** 128 samples. Below this, importance scores are too noisy for reliable bit allocation at 2-3 bit widths. Random fluctuations in which tokens appear dominate the statistics.

**Recommended:** 256-512 samples. This range provides good stability (Spearman rank correlation > 0.98 between runs) at reasonable computational cost.

**Diminishing returns:** Beyond 1024 samples, the marginal improvement in score stability is negligible. The scores converge well before this point for most models.

**MXQ default:** 512 samples. This balances quality and compute cost. For a 72B model at 2048 tokens per sample, this processes approximately 1 million calibration tokens.

### Sequence Length Matching

Use calibration sequences that match the model's typical inference use. If the model will primarily be used for chat (short exchanges of 100-500 tokens), calibrate with short sequences. If the model will be used for document analysis (8K-32K tokens), include long sequences.

The risk of mismatched sequence length is that position-dependent activation patterns are not captured. Modern LLMs use RoPE (Rotary Position Embedding), which causes activation patterns to vary with position. Weights that are important at position 0 may not be important at position 8000, and vice versa.

### Adversarial Samples

Include calibration samples that stress rare but important model capabilities:

- **Edge-case tokens:** Rare Unicode characters, emoji, mathematical symbols, code with unusual syntax
- **Long-range dependencies:** Texts where a detail mentioned early is crucial thousands of tokens later
- **Mixed modality:** Code interspersed with natural language, tables with text, structured JSON/XML
- **Adversarial prompts:** Inputs designed to trigger unusual activation patterns (e.g., repeated tokens, very long words, base64-encoded text)

These samples ensure that weights responsible for handling edge cases receive appropriate importance scores. Without adversarial samples, these weights may be scored as unimportant (because they are rarely activated in typical text) and quantized too aggressively.

### Verification Protocol

Before using an importance matrix for production quantization, run this verification:

```
1. Quantize a SMALL model (1B-7B params) using the importance matrix
2. Compare against uniform quantization at the same average bit width
3. The importance-aware quantization MUST be better:
   - PPL(importance-aware) < PPL(uniform) at same average bits
   - If not, the importance matrix is broken

4. Run the stability test:
   - Calibrate twice with different random seeds
   - Spearman rank correlation > 0.95
   - If not, increase calibration samples

5. Run the cross-domain test:
   - Calibrate on general text, assess on code
   - Calibrate on code, assess on general text
   - If the difference is > 10% perplexity, the dataset needs
     more diversity in the underperforming domain
```

### Caching and Reproducibility

Importance matrices are expensive to compute. For a 72B model with 512 calibration samples, calibration takes 4-8 hours on a high-end Mac Studio (192GB unified memory). Cache the results.

**Reproducibility requirements:**
- Store the calibration dataset hash (SHA-256 of the exact data used)
- Store the random seed if any randomness was involved (sample ordering, etc.)
- Store the model identifier (exact HuggingFace model ID or weight hash)
- Store the MXQ version used for calibration
- Store all hyperparameters (block_size, scoring_method, scoring_weights)

Given the same inputs and hyperparameters, the importance matrix should be bit-for-bit identical. If it is not (due to floating-point non-determinism on GPU), it should be stable enough that the resulting bit allocation is identical.

**Cache naming convention:**

```
imatrix_<model_hash>_<dataset_hash>_<block_size>_<scoring_method>.safetensors

Example:
imatrix_qwen35_72b_bf16_a1b2c3_mxqcalib_v1_d4e5f6_bs64_awq_hessian.safetensors
```

### Computational Cost Summary

For a 72B model on a Mac Studio (192GB, M4 Ultra):

```
Calibration (512 samples, 2048 seq_len):
  Forward passes: 512
  Time per forward pass: ~30 seconds
  Total forward pass time: ~4.3 hours
  Statistics computation: negligible (online, during forward passes)
  Importance scoring: ~5 minutes (element-wise operations)
  Total calibration time: ~4.5 hours

Bit allocation (greedy with priority queue):
  Total blocks: ~2.2 million
  Allocation time: ~30 seconds (fast with heap)

For comparison, Fisher information scoring:
  Forward + backward passes: 512 * 3 = 1,536 effective forward passes
  Total time: ~13 hours
  Memory: ~2x peak (need to store gradients)
  Quality improvement: ~5-10% error reduction

For comparison, full sensitivity analysis:
  Forward passes: millions (impractical for 72B)
  Only feasible for validation on small models (1B-7B)
```

### End-to-End Calibration Pipeline

The complete MXQ calibration pipeline, from raw model to importance matrix:

```
Step 1: Load full-precision model
  mxq calibrate --model Qwen/Qwen3.5-72B --dtype bfloat16

Step 2: Load calibration dataset
  --dataset mxq-calib-v1 (built-in)
  or --dataset-path /path/to/custom.jsonl

Step 3: Tokenize calibration samples
  - Use model's tokenizer
  - Truncate/pad to target sequence length
  - Shuffle samples (for stochastic stability)

Step 4: Register activation hooks on all Linear layers

Step 5: Run forward passes (the expensive step)
  - Process samples one at a time (batch_size=1 for memory)
  - Collect activation statistics online (Welford's algorithm)
  - Periodically call mx.eval() to free computation graph
  - Log progress every 10 samples

Step 6: Compute importance scores
  - AWQ scores: channel_l1_norm * |weight|, averaged per block
  - Hessian scores: sum_sq * weight^2, averaged per block
  - Combined: 0.5 * normalized_awq + 0.5 * normalized_hessian
  - Apply outlier channel protection

Step 7: Save importance matrix
  - Safetensors format with full metadata
  - Include calibration dataset hash for reproducibility
  - Include all channel statistics (for debugging and re-scoring)

Step 8: Validate (optional but recommended)
  - Stability test: re-run with different seed, check rank correlation
  - Sanity check: verify that lm_head and embedding have highest scores
  - Distribution check: verify scores are not degenerate (all same value)

Output: imatrix.safetensors (~1-2 GB for a 72B model)
```

This importance matrix is then consumed by Phase 2 (bit allocation and quantization) to produce the final MXQ model. A single importance matrix supports all bit width targets (MXQ-2, MXQ-2.5, MXQ-3, MXQ-4, etc.) -- only the bit allocation changes, not the importance scores.

---

## Summary of Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary scoring method | AWQ + Hessian combined | 90-95% quality of Fisher/sensitivity at 0% extra cost |
| Scoring weights | alpha=0.5, beta=0.5 | Equal weighting; robust default |
| Block size | 64 | Matches MLX quantization granularity |
| Calibration samples | 512 | Good stability, reasonable compute |
| Sequence length | 2048 primary, some 4096-8192 | Covers typical + long-context use |
| Statistics format | Safetensors | Memory-mapped, typed, metadata, ecosystem |
| Outlier detection | 6-sigma threshold | Catches true outliers without false positives |
| Outlier protection | Guaranteed top-10% importance | Ensures minimum 4-bit for outlier channels |
| Bit allocation | Greedy with priority queue | O(K log K), near-optimal, simple to debug |
| Layer-type priors | Embed 4-bit, lm_head 6-bit, attn 3-bit, MLP 2-bit | Empirically validated sensitivity hierarchy |
| First/last layer bonus | +1 bit | Error propagation / output proximity |
| Calibration dataset | Custom curated mix | Covers all model capabilities |
| Validation metric | Perplexity ratio PPL_quant/PPL_full | Standard, interpretable, fast to compute |

---

## References

- **AWQ: Activation-aware Weight Quantization.** Lin et al., 2023. The foundation for activation-magnitude scoring.
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** Frantar et al., 2023. Hessian-based quantization with optimal brain surgeon updates.
- **SqueezeLLM: Dense-and-Sparse Quantization.** Kim et al., 2023. Sensitivity-based importance scoring with outlier protection.
- **EXL2 / ExLlamaV2.** Turboderp, 2023. Mixed-precision per-block quantization with calibrated bit allocation (the closest prior art to MXQ's approach, but for CUDA).
- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** Dettmers et al., 2022. First systematic study of outlier channels in LLMs.
- **Optimal Brain Damage.** LeCun, Denker, Solla, 1989. Hessian diagonal for weight pruning (the theoretical foundation for Hessian-based importance scoring).
- **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.** Tseng et al., 2024. State-of-the-art low-bit quantization with incoherence processing.
- **AQLM: Extreme Compression of Large Language Models via Additive Quantization.** Egiazarian et al., 2024. Additive quantization codes for extreme compression.
- **K-Quants in llama.cpp.** ggerganov, 2023. Mixed-precision quantization for CPU inference with imatrix support.
