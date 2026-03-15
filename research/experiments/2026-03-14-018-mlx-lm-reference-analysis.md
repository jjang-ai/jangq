# MLX-LM Reference Implementation Analysis for Qwen2

**Date:** 2026-03-14
**Purpose:** Understand exactly how MLX-LM implements the Qwen2 forward pass, find differences with MXQ runtime that cause wrong output.

**Source:** `/opt/homebrew/lib/python3.14/site-packages/mlx_lm/` (installed mlx-lm package)

---

## 1. Qwen2 Model Class -- Forward Pass Structure

**File:** `models/qwen2.py`

### Top-level Model

```
Model.__call__(inputs, cache):
  out = Qwen2Model(inputs, cache)     # run transformer
  out = embed_tokens.as_linear(out)    # lm_head (tied embeddings) or lm_head(out)
  return out                           # (B, L, vocab_size)
```

### Qwen2Model (the transformer)

```
Qwen2Model.__call__(inputs, cache):
  h = embed_tokens(inputs)              # (B, L) -> (B, L, hidden_size)
  mask = create_attention_mask(h, cache[0])  # "causal" string or array
  for layer, c in zip(layers, cache):
    h = layer(h, mask, c)               # TransformerBlock
  return norm(h)                        # final RMSNorm
```

### TransformerBlock (per-layer)

The exact order of operations per layer:

```
TransformerBlock.__call__(x, mask, cache):
  r = self_attn(input_layernorm(x), mask, cache)   # step 1-2: norm then attention
  h = x + r                                         # step 3: residual add
  r = mlp(post_attention_layernorm(h))              # step 4-5: norm then MLP
  out = h + r                                        # step 6: residual add
  return out
```

**Critical detail:** The residual connection uses the ORIGINAL input `x`, not the normalized version. The norm output feeds into attention/MLP, but the residual bypasses the norm. This matches MXQ's implementation.

### MLP (SwiGLU)

```
MLP.__call__(x):
  return down_proj(swiglu(gate_proj(x), up_proj(x)))
```

Where `swiglu(gate, x) = silu(gate) * x` -- compiled with `mx.compile(shapeless=True)`.

---

## 2. Attention Implementation -- GQA

**File:** `models/qwen2.py`, lines 32-84

### Projection and Reshape

```python
B, L, D = x.shape

# Linear projections
queries = q_proj(x)   # (B, L, n_heads * head_dim)
keys    = k_proj(x)   # (B, L, n_kv_heads * head_dim)
values  = v_proj(x)   # (B, L, n_kv_heads * head_dim)

# Reshape: (B, L, n*d) -> (B, L, n, d) -> transpose -> (B, n, L, d)
queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
keys    = keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
values  = values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
```

### RoPE Application

```python
queries = rope(queries, offset=cache.offset)   # applied AFTER reshape, in (B,H,L,D) layout
keys    = rope(keys, offset=cache.offset)
```

RoPE is applied to Q and K separately, both in the `(B, heads, seq, head_dim)` layout. The offset comes from the KV cache (number of tokens already processed).

### KV Cache Update

```python
keys, values = cache.update_and_fetch(keys, values)
# After this: keys shape = (B, n_kv_heads, total_seq_so_far, head_dim)
#             values shape = same
```

### Attention Computation

```python
output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=scale, mask=mask)
# Uses mx.fast.scaled_dot_product_attention(Q, K, V, scale, mask)
# which handles GQA natively -- no manual K/V tiling needed
```

**GQA handling:** `mx.fast.scaled_dot_product_attention` natively supports grouped query attention. When `n_q_heads != n_kv_heads`, the function internally maps query heads to KV heads. No explicit `repeat_kv` or tiling is needed.

### Output Reshape

```python
output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)  # (B, n_heads, L, d) -> (B, L, hidden)
return o_proj(output)  # (B, L, hidden_size)
```

### Qwen2-specific: Q/K/V have bias, O projection has no bias

```python
q_proj = nn.Linear(dim, n_heads * head_dim, bias=True)
k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=True)
o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
```

### Attention scale

```python
scale = head_dim ** -0.5   # = 1 / sqrt(head_dim)
```

---

## 3. RoPE Implementation

**Files:** `models/rope_utils.py`, MLX framework `nn.RoPE`

### Qwen2 Configuration

```python
rope_theta: float = 1000000       # base frequency
rope_traditional: bool = False     # DEFAULT IS FALSE = non-traditional
rope_scaling: Optional[Dict] = None  # no scaling by default
```

For Qwen2 with no `rope_scaling`, `initialize_rope()` returns:

```python
nn.RoPE(dims=head_dim, traditional=False, base=1000000, scale=1.0)
```

### nn.RoPE.__call__

```python
def __call__(self, x, offset=0):
    return mx.fast.rope(x, self.dims, traditional=False, base=1000000, scale=1.0, offset=offset)
```

### mx.fast.rope -- Non-Traditional Mode (traditional=False)

**This is the DEFAULT for Qwen2 and most modern models.**

Input `x` has shape `(..., head_dim)`. The rotation operates on the last dimension.

Frequency computation:
```
freqs[i] = 1.0 / base^(2*i / dims)    for i = 0, 1, ..., dims/2 - 1
angle[i] = (position + offset) * freqs[i]
```

Non-traditional rotation (split halves):
```
first_half  = x[..., :dims/2]           # positions 0..63 (for head_dim=128)
second_half = x[..., dims/2:]           # positions 64..127

out[..., :dims/2]  = first_half * cos(angle) - second_half * sin(angle)
out[..., dims/2:]  = second_half * cos(angle) + first_half * sin(angle)
```

Dimension pairing: position `i` is paired with position `i + dims/2`.

### mx.fast.rope -- Traditional Mode (traditional=True)

Traditional rotation (consecutive pairs):
```
For pair i (positions 2i and 2i+1):
  out[..., 2*i]   = x[..., 2*i] * cos(angle[i]) - x[..., 2*i+1] * sin(angle[i])
  out[..., 2*i+1] = x[..., 2*i] * sin(angle[i]) + x[..., 2*i+1] * cos(angle[i])
```

Dimension pairing: position `2i` is paired with position `2i+1`.

### CRITICAL: The frequency formulas are identical between traditional and non-traditional

Both modes compute:
```
freq[i] = 1.0 / base^(2*i / dims)
```

The only difference is which dimensions get paired:
- **Non-traditional (Qwen2 default):** pairs `(i, i + dims/2)` -- i.e., `(0,64), (1,65), ..., (63,127)`
- **Traditional:** pairs `(2i, 2i+1)` -- i.e., `(0,1), (2,3), ..., (126,127)`

Verified experimentally: with `head_dim=128, base=1e6, position=5`, the two modes produce completely different outputs because they rotate different dimension pairs despite using identical frequencies.

---

## 4. RMSNorm

**File:** MLX framework `nn.RMSNorm`

```python
class RMSNorm:
    def __init__(self, dims, eps=1e-5):
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)
```

Formula: `y = x / sqrt(mean(x^2) + eps) * weight`

- Accumulation for mean is done in float32 precision (per docstring).
- `eps` default is `1e-5`, but Qwen2 config specifies `rms_norm_eps` (typically `1e-6`).
- In `qwen2.py`: `nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)` -- uses model config value.
- MXQ reads `rms_norm_eps` from `config.json` and passes it as `config.normEps`.

**MXQ's RMSNorm kernel** (`mxq_rms_norm`): Computes `sum_sq` in float32, then `rms = rsqrt(sum_sq/hidden + eps)`, then `output = x * rms * gamma`. This matches the MLX formula.

---

## 5. Generate Loop -- Prefill vs Decode

**File:** `generate.py`, `generate_step()` function (line 303)

### Prefill Phase

Processes prompt tokens in chunks of `prefill_step_size` (default: 2048):

```python
while total_prompt_tokens - prompt_processed_tokens > 1:
    n_to_process = min(prefill_step_size, remaining - 1)
    model(prompt[:n_to_process][None], cache=prompt_cache)     # process chunk
    mx.eval_state([c.state for c in prompt_cache])              # force computation
    prompt = prompt[n_to_process:]                              # advance
```

All prompt tokens except the last one are processed in the prefill loop. The last token goes through `_step()` which also samples the first output token.

Key points:
- Prompt tokens are processed ALL AT ONCE (up to `prefill_step_size`), not one at a time.
- The model sees the full chunk as a batch with shape `(1, chunk_len)`.
- `create_attention_mask` returns `"causal"` (a string) which tells `mx.fast.scaled_dot_product_attention` to use efficient causal masking.
- After each chunk, computation is forced and `mx.clear_cache()` releases intermediate buffers.

### Decode Phase

Single-token autoregressive:

```python
y, logprobs = _step(last_prompt_token)     # first output token
while n < max_tokens:
    next_y, next_logprobs = _step(y)       # each subsequent token
    mx.async_eval(next_y, next_logprobs)   # async for pipeline overlap
    yield y.item(), logprobs
    y, logprobs = next_y, next_logprobs
```

During decode, the model is called with a single token `(1, 1)`. The mask is `None` (returned by `create_attention_mask` when `N == 1` -- no causal mask needed for a single query).

### Double-buffering / Pipelining

MLX uses `mx.async_eval()` and a dedicated `generation_stream` to pipeline computation. While the current token is being yielded, the next token's computation is already running asynchronously.

---

## 6. KV Cache Implementation

**File:** `models/cache.py`, `KVCache` class (line 308)

### Pre-allocation Strategy

```python
class KVCache:
    step = 256  # grow in chunks of 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            # Grow the buffer by 'step' entries
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            new_k = mx.zeros((B, n_kv_heads, n_steps * self.step, head_dim))
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
            else:
                self.keys = new_k

        # Write new keys/values at current offset
        self.offset += keys.shape[2]
        self.keys[..., prev:self.offset, :] = keys
        self.values[..., prev:self.offset, :] = values

        # Return only the valid portion
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]
```

Shape: `(B, n_kv_heads, seq_len, head_dim)` -- matches the `(B, H, T, D)` layout used by attention.

Grows in steps of 256 to avoid frequent reallocations. Uses slice assignment for efficient in-place updates.

---

## 7. Buffer Management -- MLX vs MXQ

### MLX Approach (Functional / Lazy)

- MLX uses **lazy computation** -- operations build a computation graph, only run when forced.
- Every operation creates a **new tensor**. No in-place mutation except KV cache slice assignment.
- Memory is managed by MLX's allocator with reference counting. `mx.clear_cache()` releases unused buffers.
- The computation graph enables **fusion** -- MLX can fuse multiple operations into a single GPU kernel.

### MXQ Approach (Imperative / Buffer Reuse)

- MXQ **pre-allocates** fixed buffers (`hiddenBuffer`, `normBuffer`, `qBuffer`, etc.) at init time.
- Each kernel dispatches reads from input buffers and writes to output buffers within a single `MTLCommandBuffer`.
- All kernels for one forward pass are encoded into one command buffer, submitted once.
- Buffers are reused across layers (e.g., `normBuffer` is used for both input_layernorm output and as scratch for o_proj output).

### Potential Issues with MXQ Buffer Reuse

The MXQ code reuses `normBuffer` as temporary storage for the O projection output (line 188-189 of MXQInference.swift):
```swift
// 6. O projection
dispatchDequantGEMV(input: attnOutBuffer, weight: layer.oProj,
                    output: normBuffer,  // reuse normBuffer as temp
                    K: hiddenSize, N: hiddenSize)

// 7. Residual add: hidden = residual + attn_output
dispatchAdd(a: residualBuffer, b: normBuffer, output: hiddenBuffer, count: hiddenSize)
```

This should be safe since `normBuffer` is not read after the O projection writes to it, until it is re-populated by the post-attention norm.

---

## 8. CRITICAL FINDING: RoPE Mode Mismatch

### The Bug

**MXQ implements TRADITIONAL RoPE (consecutive pairs), but Qwen2 uses NON-TRADITIONAL RoPE (split halves).**

MXQ's `mxq_rope` Metal kernel (`MXQCompute.metal`, line 61-94):
```metal
uint idx0 = base_idx + pair * 2;       // consecutive: 0,2,4,...
uint idx1 = base_idx + pair * 2 + 1;   // consecutive: 1,3,5,...

qk[idx0] = half(v0 * cos_val - v1 * sin_val);
qk[idx1] = half(v0 * sin_val + v1 * cos_val);
```

This rotates consecutive pairs `(0,1), (2,3), (4,5), ...` -- this is **traditional** mode.

MLX's Qwen2 uses `rope_traditional=False`, which pairs positions `(i, i+half_dim)`:
- `(0,64), (1,65), (2,66), ...` for head_dim=128.

### Impact

With `head_dim=128` and `base=1e6`, at position 5, the outputs are completely different between the two modes. Every Q and K vector in MXQ will have wrong values, which means:
- Every attention score will be wrong
- KV cache stores incorrectly rotated K vectors
- Errors compound across layers

This is **the primary bug** explaining why MXQ produces wrong output despite correct individual kernels.

### Fix

Change the `mxq_rope` kernel to use split-half indexing:

```metal
// Non-traditional RoPE (split halves) -- correct for Qwen2
uint half_dim = head_dim / 2;
uint idx_first  = base_idx + pair;              // first half: 0,1,2,...,63
uint idx_second = base_idx + pair + half_dim;   // second half: 64,65,...,127

float v_first  = float(qk[idx_first]);
float v_second = float(qk[idx_second]);

qk[idx_first]  = half(v_first * cos_val - v_second * sin_val);
qk[idx_second] = half(v_second * cos_val + v_first * sin_val);
```

The frequency computation is already correct:
```metal
float freq = 1.0f / pow(theta_base, float(pair) / float(half_dim));
```
This equals `1/base^(pair/half_dim) = 1/base^(2*pair/head_dim)`, matching MLX.

---

## 9. Secondary Differences (May or May Not Matter)

### 9a. RMSNorm Epsilon Source

MXQ reads `rms_norm_eps` from `config.json` and uses it. For Qwen2 models this is typically `1e-6`. MLX does the same. **No issue here** as long as the config is parsed correctly.

### 9b. Attention Scale

Both use `1/sqrt(head_dim)`. MLXQ: `1.0 / sqrt(Float(headDim))`. MLX: `head_dim ** -0.5`. **Equivalent.**

### 9c. SwiGLU Activation

MLX: `silu(gate) * up` (compiled with `mx.compile`).
MLXQ: `silu_g = g / (1 + exp(-g)); output = silu_g * u` in Metal. **Equivalent formula.**

### 9d. Tied Embeddings for LM Head

MLX: `embed_tokens.as_linear(out)` -- uses the embedding weight matrix transposed as a linear layer.
MLXQ: `lmHeadWeight = model.lmHead ?? model.embedTokens` -- uses the raw embedding table with `dispatchDequantGEMV`.

Both should compute `hidden @ embed_weight.T`. Need to verify MXQ's GEMV kernel transposes correctly for tied embeddings. The embedding table has shape `(vocab_size, hidden_size)`, and we need `output[v] = sum_d(hidden[d] * embed[v][d])`. MXQ's GEMV computes `output[n] = sum_k(input[k] * weight[n][k])` which is correct if `weight` has shape `(N, K) = (vocab_size, hidden_size)`.

### 9e. Prefill: MXQ processes tokens one at a time

MLX processes the entire prompt at once (up to 2048 tokens per chunk) with causal masking. MXQ's `forward(tokenId:)` processes one token at a time. This is functionally equivalent for autoregressive generation but much slower for prefill. However, for correctness, processing one token at a time with a growing KV cache should produce the same result as batch prefill with causal masking -- **as long as RoPE offsets are correct**.

### 9f. KV Cache Layout

MLX: `(B, n_kv_heads, seq, head_dim)` -- heads before sequence.
MLXQ: `(seq, n_kv_heads, head_dim)` -- sequence before heads (no batch dim).

Both layouts are valid; the attention kernel just needs to index correctly. MXQ's attention kernel indexes as:
```metal
K_cache + pos * n_kv_heads * head_dim + kv_head * head_dim
```
This correctly accesses `K_cache[pos][kv_head][:]`, matching the `(seq, kv_heads, head_dim)` layout.

### 9g. Precision

MLX operates in float16 for weights, float32 for accumulation in SDPA softmax (`precise=True` in `base.py` line 97).
MXQ's attention kernel uses `float` (32-bit) for score computation, softmax, and weighted sum, then casts back to `half` for output. **Equivalent.**

### 9h. Mask Handling for Single Token

During decode (single token), MLX returns `mask=None` from `create_attention_mask` (line 51 of base.py: `if N == 1: return None`). `mx.fast.scaled_dot_product_attention` with `mask=None` means no masking -- the single query can attend to all keys. MXQ's attention kernel has no masking for decode (all `seq_len` positions are attended to). **Equivalent.**

---

## 10. Summary of Findings

| Aspect | MLX-LM (Reference) | MXQ (Our Runtime) | Match? |
|--------|--------------------|--------------------|--------|
| Layer order | norm -> attn -> residual -> norm -> MLP -> residual | Same | YES |
| RMSNorm formula | `x / sqrt(mean(x^2) + eps) * weight` | Same | YES |
| RMSNorm eps | From config (`rms_norm_eps`) | From config | YES |
| Q/K/V bias | Q,K,V have bias; O has no bias | Q,K,V bias applied; O no bias | YES |
| **RoPE mode** | **Non-traditional (split halves)** | **Traditional (consecutive pairs)** | **NO -- BUG** |
| RoPE frequency | `1/base^(2i/dims)` | `1/base^(pair/half_dim)` (same) | YES |
| RoPE base | From config (`rope_theta`, default 1e6) | From config | YES |
| Attention scale | `head_dim^(-0.5)` | `1/sqrt(head_dim)` | YES |
| GQA mapping | Native in `mx.fast.sdpa` | `h * n_kv / n_heads` | YES |
| KV cache layout | `(B, heads, seq, dim)` | `(seq, heads, dim)` | OK |
| SwiGLU | `silu(gate) * up` | Same | YES |
| Tied embeddings | `embed.as_linear(x)` | GEMV with embed table | Verify |
| Prefill | Batch (up to 2048 tokens) | Token-by-token | Slower but correct |
| Precision | fp16 weights, fp32 accum | fp16 weights, fp32 accum | YES |

### Root Cause of Wrong Output

**The RoPE dimension pairing is wrong.** MXQ uses traditional/consecutive-pair rotation `(0,1), (2,3), ...` but Qwen2 requires non-traditional/split-half rotation `(0,64), (1,65), ...`. This produces completely different Q and K vectors at every position, corrupting all attention scores and propagating errors through every layer.

### Recommended Fix

1. Change `mxq_rope` kernel to use split-half indexing (see Section 8).
2. Consider adding a `traditional` flag to the kernel for future model support.
3. After fixing RoPE, if output is still wrong, verify the tied-embedding GEMV produces correct logits.
