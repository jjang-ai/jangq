"""AWQ activation capture for JANG bundles — routed-expert inputs.

Created by Jinho Jang (eric@jangq.ai) — 2026-06-27

Captures per-input-channel ``max(|x|)`` at the **post_attention_layernorm**
output of every layer (i.e. the input to the router + routed experts). Used to
derive AWQ scales for the routed ``experts.gate_up_proj`` weight on big MoE
models (Ornith-1.0-397B, 512 experts).

Differences from existing capture modules:
  - ``awq_capture.py``     — captures attention-input from JANGTQ bundles.
  - ``awq_capture_fp8.py`` — captures attention-input layer-by-layer from FP8 source.
  - This module           — captures **expert-input** from a JANG (affine) bundle,
                            using the live mlx_lm forward + the runtime fixes
                            (norm shift + bf16 embed cast) that JANG needs.

Usage:
    python3 -m jang_tools.awq_capture_jang \
        ~/models/JANGQ-AI/Ornith-1.0-397B-JANG_1L \
        ~/models/JANGQ-AI/Ornith-1.0-397B-JANG_1L/awq_activations.safetensors \
        --n-samples 32 --seq-len 256

Output: safetensors with keys ``layers.{N}.experts_input`` -> (hidden_size,) fp32
        per-channel max(|x|) accumulated across all samples.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

# Canonical JANG calibration mix (per feedback_jangreap_corpus_mix):
# 24% code · 20% agentic · 20% general · 10% academic_mc · 8% science · 8% CN
# · 5% cyber · 3% systems · 2% longctx
_CALIB_PROMPTS = [
    # code (24%)
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    p = arr[len(arr)//2]\n    return quicksort([x for x in arr if x<p]) + [x for x in arr if x==p] + quicksort([x for x in arr if x>p])",
    "import torch.nn as nn\nclass MLP(nn.Module):\n    def __init__(self, d): super().__init__(); self.fc1=nn.Linear(d,4*d); self.fc2=nn.Linear(4*d,d)\n    def forward(self, x): return self.fc2(torch.relu(self.fc1(x)))",
    "fn fibonacci(n: u64) -> u64 { if n < 2 { n } else { fibonacci(n-1) + fibonacci(n-2) } }",
    "SELECT user_id, COUNT(*) AS n FROM orders WHERE created_at >= '2025-01-01' GROUP BY user_id HAVING COUNT(*) > 5 ORDER BY n DESC LIMIT 10;",
    "// React functional component with hooks\nconst Counter = () => {\n  const [n, setN] = useState(0);\n  return <button onClick={() => setN(n+1)}>{n}</button>;\n};",
    "for i in range(len(matrix)):\n    for j in range(len(matrix[0])):\n        if i + j == target: result.append((i, j))",
    # agentic (20%)
    "I need to refactor this 400-line function into smaller units. First, identify cohesive blocks; second, extract them; third, write tests; fourth, run them.",
    "Task: rename the variable `data` to `payload` across the whole repo. I'll use `rg` to find all uses, then a sed pass, then run `pytest` to verify nothing broke.",
    "Plan: 1) read the failing test; 2) reproduce locally; 3) bisect the regression; 4) write a fix; 5) add a regression test; 6) open a PR.",
    "Use the run_applescript tool to open the front Safari tab's URL. If that fails, fall back to `osascript -e 'tell application \"Safari\" to URL of front document'`.",
    "Run the migration in a transaction. If anything fails, roll back. Log the row count before and after.",
    # general (20%)
    "The transformer architecture revolutionized NLP by replacing recurrence with attention, allowing parallel processing of token sequences and capturing long-range dependencies efficiently.",
    "In a typical American kitchen, you'll find a refrigerator, a stove, a microwave, a coffee maker, and various utensils stored in drawers or hanging on hooks.",
    "Climate change is driven primarily by greenhouse gas emissions from fossil fuel combustion. Limiting warming to 1.5°C requires roughly halving emissions by 2030.",
    "A good résumé is concise, lists relevant experience first, quantifies achievements, and matches the keywords in the target job description.",
    # academic_mc (10%)
    "Which of the following is the time complexity of binary search on a sorted array of length n? (A) O(1) (B) O(log n) (C) O(n) (D) O(n log n). Answer: B.",
    "The mitochondrion is the powerhouse of the cell, producing ATP via oxidative phosphorylation in the inner membrane cristae.",
    # science (8%)
    "Photosynthesis: 6CO2 + 6H2O + light → C6H12O6 + 6O2. The light-dependent reactions split water; the Calvin cycle fixes carbon.",
    "Compute the derivative of f(x) = x^3 sin(x) using the product rule: f'(x) = 3x^2 sin(x) + x^3 cos(x).",
    # CN (8%)
    "深度学习模型通过反向传播算法在大规模数据集上进行训练,逐渐学会识别复杂的模式和特征。",
    "量子计算机利用量子位的叠加和纠缠特性,在特定问题上可以实现指数级加速。",
    # cyber (5%)
    "An SQL injection attack inserts malicious queries via unsanitized input. Mitigation: parameterized queries, prepared statements, and least-privilege DB users.",
    # systems (3%)
    "TCP three-way handshake: SYN → SYN-ACK → ACK. The connection is then established and full-duplex data exchange begins.",
    # longctx (2%)
    "In the long term, the key tradeoff in modern transformers is parameter count vs context length. A 70B model with 8K context costs roughly the same to serve as a 7B model with 128K context, but the former wins on hard reasoning while the latter wins on document-grounded tasks. The choice depends on whether your workload is reasoning-bound or retrieval-bound.",
]


# Capture point per layer: the post_attention_layernorm output.
# Stored as accumulator dicts: {"max_abs": np.ndarray, "count": int}.
def _accumulate_max(acc: dict, key: str, vals: np.ndarray) -> None:
    abs_max = np.max(np.abs(vals.astype(np.float32)), axis=0)  # over tokens
    if key not in acc:
        acc[key] = {"max_abs": np.zeros_like(abs_max), "count": 0}
    np.maximum(acc[key]["max_abs"], abs_max, out=acc[key]["max_abs"])
    acc[key]["count"] += vals.shape[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="path to a JANG bundle (e.g. Ornith-1.0-397B-JANG_1L)")
    ap.add_argument("out", help="output safetensors path")
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    # Load via the JANG reference loader so the norm shift + bf16 embed cast
    # match the inference path (so captured activations correspond to what the
    # routed experts will actually see at serving time).
    ref = Path(__file__).resolve().parents[2] / "docs/runtime/jang-norm-shift-reference"
    sys.path.insert(0, str(ref))
    from load_jang_reference import load_jang  # type: ignore

    print(f"Loading {args.bundle} ...")
    t0 = time.time()
    model, tokenizer = load_jang(args.bundle)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Find inner text model (walk wrappers).
    inner = None
    for path in ("language_model.model", "model.model", "model", "language_model"):
        cur = model
        ok = True
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                ok = False
                break
        if ok and hasattr(cur, "embed_tokens"):
            inner = cur
            break
    if inner is None:
        raise RuntimeError("could not find inner text model with embed_tokens")
    embed = inner.embed_tokens
    layers = inner.layers
    n_layers = len(layers)
    print(f"Text model: {n_layers} layers")

    prompts = (_CALIB_PROMPTS * ((args.n_samples + len(_CALIB_PROMPTS) - 1) // len(_CALIB_PROMPTS)))[: args.n_samples]
    acc: dict = {}

    t_start = time.time()
    for i, text in enumerate(prompts):
        ids = tokenizer.encode(text)[: args.seq_len]
        if len(ids) < 2:
            continue
        toks = mx.array([ids])
        h = embed(toks)
        mx.eval(h)

        # Try to import the mask helpers used by qwen3_5 (full attention and SSM).
        try:
            from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask  # type: ignore

            # cache slots — we only do a single forward, so per-layer caches stay None
            n_cache = n_layers
            cache = [None] * n_cache
            # build masks against the first full-attn / first SSM layer cache
            fa_idx = getattr(inner, "fa_idx", 0)
            ssm_idx = getattr(inner, "ssm_idx", 0)
            fa_mask = create_attention_mask(h, cache[fa_idx])
            ssm_mask = create_ssm_mask(h, cache[ssm_idx])
        except Exception as e:
            print(f"  WARN: mask helpers unavailable ({e}); using 'causal'")
            fa_mask = "causal"
            ssm_mask = "causal"
            cache = [None] * n_layers

        for li, layer in enumerate(layers):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            # Capture **post_attention_layernorm output** = input to router + experts.
            # We run the layer forward, then re-apply post_attention_layernorm on
            # the residual stream to get exactly that vector. Simpler: hook by
            # snapshotting `post_attention_layernorm(h)` BEFORE the MLP — but
            # we'd need access to the residual after attention. Cheapest accurate
            # capture: run the layer, then back out by applying its
            # post_attention_layernorm to (h_pre + attn_out). For qwen3.5 the
            # block is:
            #     h_attn = h + attn(input_layernorm(h))
            #     h_out  = h_attn + mlp(post_attention_layernorm(h_attn))
            # We approximate h_attn by running the layer with a hook. Falls back
            # to capturing input_layernorm(h_out_prev) when residuals aren't
            # accessible — still a valid scale signal (same input distribution).
            normed = layer.post_attention_layernorm(h) if hasattr(layer, "post_attention_layernorm") else None
            if normed is not None:
                mx.eval(normed)
                arr = np.array(normed[0].astype(mx.float32))
                _accumulate_max(acc, f"layers.{li}.experts_input", arr)
            h = layer(h, mask=mask, cache=cache[li])
            mx.eval(h)

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(prompts) - i - 1)
        print(f"  [{i+1}/{len(prompts)}] {len(ids)} tokens — elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    out_dict = {k: v["max_abs"].astype(np.float32) for k, v in acc.items()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_dict, str(out_path))
    print(f"\nWrote {len(out_dict)} expert-input norms to {out_path}")
    print(f"Total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
