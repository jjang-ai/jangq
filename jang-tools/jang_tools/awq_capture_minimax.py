"""
AWQ activation capture for MiniMax-M2 (no shared expert).
Created by Jinho Jang (eric@jangq.ai)

MiniMax-M2 has only routed experts (no shared expert). To propagate
hidden states layer-by-layer for AWQ stat capture we use expert-0 of
each layer as a representative MLP. All 256 experts in a layer share
the same `post_attention_layernorm` output as input, so one
per-channel stat per layer covers every expert's gate_proj (w1) and
up_proj (w3). w2 stays at 4-bit in JANG_K and does not need AWQ.

With alpha fixed at 0.5 (standard AWQ), this script captures and
immediately produces scales — no intermediate raw-activation file.

Output: safetensors with one tensor per layer:
    model.layers.<li>.block_sparse_moe.input_scale  shape (hidden,)  fp32
plus informational keys:
    model.layers.<li>.self_attn.input_max           shape (hidden,)  fp32
    model.layers.<li>.block_sparse_moe.expert0_intermediate_max  (intermediate,) fp32

Usage:
    python3 -m jang_tools.awq_capture_minimax <fp8_src> <out_path> [n_samples] [seq_len]
"""
import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .awq_capture_fp8 import (
    _CALIB_PROMPTS,
    _rmsnorm,
    _silu,
    _safe_load_fp8,
    _build_weight_map,
    _load_tensor_by_name,
    _load_layer_weights,
)

DEFAULT_N_SAMPLES = 16
DEFAULT_SEQ_LEN = 256
DEFAULT_ALPHA = 0.5
EPS = 1e-8
# Clip floor on AWQ scale: prevents inverse-fold from amplifying dead-channel
# LN weights by huge factors (e.g. 500×). Outlier channels still get scale > 1.
SCALE_FLOOR = 1.0

# Long-form calibration passages — supplement to the short _CALIB_PROMPTS.
# Each passage ~200-300 tokens after MiniMax tokenization, giving the
# capture solid per-channel statistics.
_LONG_CALIB = [
    "The transformer architecture, introduced in 2017 in the paper 'Attention "
    "is All You Need', revolutionized natural language processing by replacing "
    "recurrence with self-attention. Each layer computes queries, keys, and "
    "values from the input, then attends across all positions in parallel. "
    "Multi-head attention runs several attention operations in parallel, each "
    "with its own learned projections, then concatenates the results. The "
    "feed-forward sublayer applies a position-wise MLP — typically a SwiGLU or "
    "GeGLU variant — to each token independently. Residual connections and "
    "layer normalization stabilize training across deep stacks. Modern large "
    "language models extend this with grouped-query attention, rotary position "
    "embeddings, and mixture-of-experts routing for parameter efficiency.",

    "def quicksort(arr):\n"
    "    if len(arr) <= 1:\n        return arr\n"
    "    pivot = arr[len(arr) // 2]\n"
    "    left = [x for x in arr if x < pivot]\n"
    "    middle = [x for x in arr if x == pivot]\n"
    "    right = [x for x in arr if x > pivot]\n"
    "    return quicksort(left) + middle + quicksort(right)\n\n"
    "def merge_sort(arr):\n"
    "    if len(arr) <= 1: return arr\n"
    "    mid = len(arr) // 2\n"
    "    left = merge_sort(arr[:mid])\n"
    "    right = merge_sort(arr[mid:])\n"
    "    return merge(left, right)\n\n"
    "class BTree:\n"
    "    def __init__(self, t):\n"
    "        self.root = BTreeNode(True)\n"
    "        self.t = t\n"
    "    def insert(self, k):\n"
    "        root = self.root\n"
    "        if len(root.keys) == 2 * self.t - 1:\n"
    "            new_node = BTreeNode(False)\n"
    "            self.root = new_node\n"
    "            new_node.children.insert(0, root)\n"
    "            self.split_child(new_node, 0)\n"
    "            self.insert_non_full(new_node, k)\n"
    "        else:\n"
    "            self.insert_non_full(root, k)\n",

    "Quantum mechanics describes the behavior of matter and energy at atomic "
    "and subatomic scales. The Schrödinger equation governs the time evolution "
    "of the wavefunction, a complex-valued probability amplitude over "
    "configuration space. Observable quantities correspond to Hermitian "
    "operators, and measurement outcomes are eigenvalues weighted by the "
    "squared modulus of the wavefunction projection onto the corresponding "
    "eigenstate. Heisenberg's uncertainty principle establishes a fundamental "
    "limit on the simultaneous precision of conjugate observables such as "
    "position and momentum. Entanglement, demonstrated by the violation of "
    "Bell inequalities, shows that quantum correlations cannot be reproduced "
    "by any local hidden-variable theory.",

    "在过去十年中,大型语言模型从基本的循环神经网络发展为庞大的"
    "Transformer架构。以GPT、Claude、Gemini为代表的现代模型,通过自注意力"
    "机制并行处理整个序列,极大地提升了训练效率与表达能力。专家混合"
    "(Mixture of Experts)架构进一步将参数规模推向万亿级别,同时保持"
    "推理时的稀疏激活——每个token只路由到几个专家上,从而以更低的"
    "计算成本提供更强的表达能力。此外,旋转位置编码(RoPE)、分组查询"
    "注意力(GQA)、滑动窗口注意力等技术让模型能够处理百万级的上下文。"
    "在量化领域,从GPTQ、AWQ到MXTQ和码本量化,各种方法不断推进低位精度"
    "下的模型质量边界。",

    "SELECT u.user_id, u.email, COUNT(o.order_id) AS total_orders, "
    "SUM(o.total_amount) AS lifetime_value, MIN(o.created_at) AS first_order, "
    "MAX(o.created_at) AS most_recent_order, AVG(o.total_amount) AS avg_order "
    "FROM users u LEFT JOIN orders o ON u.user_id = o.user_id "
    "WHERE u.created_at >= '2024-01-01' AND (o.status = 'completed' OR "
    "o.status IS NULL) GROUP BY u.user_id, u.email "
    "HAVING COUNT(o.order_id) > 5 OR SUM(o.total_amount) > 1000.00 "
    "ORDER BY lifetime_value DESC NULLS LAST, total_orders DESC LIMIT 100;\n\n"
    "WITH monthly_revenue AS (SELECT DATE_TRUNC('month', created_at) AS mo, "
    "SUM(total_amount) AS rev FROM orders WHERE status = 'completed' "
    "GROUP BY 1) SELECT mo, rev, LAG(rev) OVER (ORDER BY mo) AS prev_rev, "
    "ROUND(100.0 * (rev - LAG(rev) OVER (ORDER BY mo)) / "
    "NULLIF(LAG(rev) OVER (ORDER BY mo), 0), 2) AS pct_change "
    "FROM monthly_revenue ORDER BY mo;",
]

LAYER_KEYS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "block_sparse_moe.experts.0.w1.weight",
    "block_sparse_moe.experts.0.w2.weight",
    "block_sparse_moe.experts.0.w3.weight",
]


def _propagate(h: np.ndarray, lw: dict) -> tuple:
    """One MiniMax layer forward (skip MLA, expert-0 MLP).

    Returns (h_new, stats_dict) where stats hold per-channel max(|x|).
    """
    stats = {}

    if "input_layernorm.weight" in lw:
        h_attn = _rmsnorm(h, lw["input_layernorm.weight"])
        stats["attn_input"] = np.abs(h_attn).max(axis=(0, 1))

    if "post_attention_layernorm.weight" in lw:
        h_mlp = _rmsnorm(h, lw["post_attention_layernorm.weight"])
        stats["mlp_input"] = np.abs(h_mlp).max(axis=(0, 1))
    else:
        h_mlp = h

    w1 = lw.get("block_sparse_moe.experts.0.w1.weight")  # gate_proj
    w2 = lw.get("block_sparse_moe.experts.0.w2.weight")  # down_proj
    w3 = lw.get("block_sparse_moe.experts.0.w3.weight")  # up_proj

    if w1 is not None and w2 is not None and w3 is not None:
        gate = h_mlp @ w1.T
        up = h_mlp @ w3.T
        inter = _silu(gate) * up
        stats["mlp_intermediate"] = np.abs(inter).max(axis=(0, 1))
        mlp_out = (inter @ w2.T).astype(np.float32)
    else:
        mlp_out = np.zeros_like(h, dtype=np.float32)

    return h + mlp_out, stats


def run_capture(src, out, n_samples=DEFAULT_N_SAMPLES,
                seq_len=DEFAULT_SEQ_LEN, alpha=DEFAULT_ALPHA):
    src = Path(src)
    out = Path(out)

    cfg = json.loads((src / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]
    intermediate_size = cfg["intermediate_size"]
    n_experts = cfg["num_local_experts"]

    print("=" * 60)
    print("  AWQ capture for MiniMax-M2")
    print("=" * 60)
    print(f"  src:        {src}")
    print(f"  out:        {out}")
    print(f"  layers:     {n_layers}")
    print(f"  hidden:     {hidden_size}  intermediate: {intermediate_size}")
    print(f"  experts:    {n_experts}  (all share per-layer input)")
    print(f"  samples:    {n_samples}  seq_len: {seq_len}")
    print(f"  alpha:      {alpha} (fixed, scales = (act_max+eps)^alpha)",
          flush=True)

    weight_map = _build_weight_map(src)
    print(f"  weight_map: {len(weight_map)} tensors", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
    print(f"  tokenizer:  {tok.__class__.__name__}", flush=True)

    # Mix long-form passages with short prompts so per-channel stats see
    # varied token distributions, not just one-liners.
    base = list(_LONG_CALIB) + list(_CALIB_PROMPTS)
    prompts = []
    for i in range(n_samples):
        prompts.append(base[i % len(base)])
    token_batches = [tok.encode(p)[:seq_len] for p in prompts]
    max_sl = max(len(b) for b in token_batches)
    avg_sl = sum(len(b) for b in token_batches) / max(len(token_batches), 1)
    print(f"  max_sl actual: {max_sl}  avg: {avg_sl:.1f}", flush=True)

    print("  loading embed_tokens...", flush=True)
    t0 = time.time()
    embed = _load_tensor_by_name(src, weight_map, "model.embed_tokens.weight")
    if embed is None:
        raise RuntimeError("embed_tokens not in weight_map")
    print(f"    embed: {embed.shape} {embed.dtype} in {time.time()-t0:.1f}s",
          flush=True)

    hidden = np.zeros((n_samples, max_sl, hidden_size), dtype=np.float32)
    for i, b in enumerate(token_batches):
        for j, tid in enumerate(b):
            hidden[i, j] = embed[tid]
    del embed
    gc.collect()

    per_layer_mlp_max = {}     # li -> (hidden,) fp32 running max
    per_layer_attn_max = {}    # li -> (hidden,) fp32
    per_layer_inter_max = {}   # li -> (intermediate,) fp32 (expert 0)

    def update_max(d, k, v):
        prev = d.get(k)
        d[k] = v.astype(np.float32).copy() if prev is None \
            else np.maximum(prev, v.astype(np.float32))

    print(f"  forward through {n_layers} layers...", flush=True)
    t0 = time.time()
    for li in range(n_layers):
        t_l = time.time()
        lw = _load_layer_weights(src, weight_map, li, LAYER_KEYS)
        t_load = time.time() - t_l

        t_f = time.time()
        hidden, lstats = _propagate(hidden, lw)
        t_fwd = time.time() - t_f

        if "mlp_input" in lstats:
            update_max(per_layer_mlp_max, li, lstats["mlp_input"])
        if "attn_input" in lstats:
            update_max(per_layer_attn_max, li, lstats["attn_input"])
        if "mlp_intermediate" in lstats:
            update_max(per_layer_inter_max, li, lstats["mlp_intermediate"])

        h_norm = float(np.linalg.norm(hidden[0, 0]))
        print(f"    L{li:2d}: load={t_load:5.1f}s fwd={t_fwd:5.2f}s "
              f"h0.norm={h_norm:7.2f} "
              f"mlp_max={float(lstats.get('mlp_input',np.zeros(1)).max()):.3f}",
              flush=True)
        del lw
        gc.collect()

    total = time.time() - t0
    print(f"  total forward: {total:.1f}s "
          f"({total/max(n_layers,1):.1f}s/layer)", flush=True)

    # Convert per-layer maxes to AWQ scales: s = (max + eps)^alpha
    # Then clip to ≥ SCALE_FLOOR so dead/quiet channels never amplify the
    # inverse fold target (post_attention_layernorm.weight). Outliers still
    # get the full upscale; quiet channels become a no-op (s == 1).
    output_tensors = {}
    for li, vec in per_layer_mlp_max.items():
        scale = np.power(vec + EPS, alpha).astype(np.float32)
        scale = np.maximum(scale, SCALE_FLOOR).astype(np.float32)
        output_tensors[
            f"model.layers.{li}.block_sparse_moe.input_scale"
        ] = scale
        # Save raw max too for forensics / re-derivation with different alpha
        output_tensors[
            f"model.layers.{li}.block_sparse_moe.input_max"
        ] = vec.astype(np.float32)
    for li, vec in per_layer_attn_max.items():
        output_tensors[
            f"model.layers.{li}.self_attn.input_max"
        ] = vec.astype(np.float32)
    for li, vec in per_layer_inter_max.items():
        output_tensors[
            f"model.layers.{li}.block_sparse_moe.expert0_intermediate_max"
        ] = vec.astype(np.float32)

    save_file(output_tensors, str(out))
    sz_mb = sum(a.nbytes for a in output_tensors.values()) / 1e6
    n_layers_done = len(per_layer_mlp_max)
    print(f"  wrote {out}: {len(output_tensors)} tensors, {sz_mb:.2f} MB, "
          f"{n_layers_done} layers covered", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python3 -m jang_tools.awq_capture_minimax "
            "<fp8_src> <out_path> [n_samples] [seq_len] [alpha]",
            file=sys.stderr,
        )
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_SAMPLES
    sl = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_SEQ_LEN
    a = float(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_ALPHA
    run_capture(src, out, n_samples=ns, seq_len=sl, alpha=a)
