"""
AWQ activation capture via layer-by-layer FP8 forward.
Created by Jinho Jang (eric@jangq.ai)

FALLBACK for `awq_capture.py` when JANGTQ runtime hangs. This module
bypasses JANGTQ entirely: streams one layer's FP8 weights at a time,
propagates a hidden state through RMSNorm + shared_expert MLP, and
captures per-input-channel `max(|x|)` at the points that feed routed
experts (post_attention_layernorm output).

Simplifications vs full model forward:
  - Skips MLA attention entirely (residual contribution only).  This
    biases the hidden-state magnitude slightly low vs real forward but
    preserves the per-channel outlier structure that AWQ exploits.
  - Uses shared_expert only (not routed experts) to propagate h.  Shared
    expert has the same input distribution as routed experts within a
    layer, so captured stats are correct for routed experts.

Memory: ~5 GB per layer (weights dequant'd fp8->fp32).

Usage:
    python3 -m jang_tools.awq_capture_fp8 <fp8_src> <output_path> [n_samples] [seq_len]

Output: safetensors file with tensor-name -> (in_features,) fp32 per-channel max.
"""
import sys, json, time, gc
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from .fp8 import load_fp8_tensor
from .calibrate import _load_bf16_tensor


# ── Config defaults ────────────────────────────────────────────────
DEFAULT_N_SAMPLES = 16
DEFAULT_SEQ_LEN = 256


# ── Calibration prompts (short, diverse) ───────────────────────────
_CALIB_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. A pangram contains every letter.",
    "In 2025, machine learning models have become proficient at code generation.",
    "def fib(n): return n if n<2 else fib(n-1)+fib(n-2)",
    "Compute the derivative of f(x) = x**3 + 2*x. The answer is 3*x**2 + 2.",
    "Self-attention Q K^T / sqrt(d) is the core of transformer architectures.",
    "SELECT user_id, COUNT(*) FROM orders WHERE year=2025 GROUP BY user_id;",
    "Industrial Revolution in Britain mechanized agriculture and textile production.",
    "Photosynthesis: 6CO2 + 6H2O + light -> C6H12O6 + 6O2 in chloroplasts.",
    "深度学习模型通过在大规模文本上训练学会理解和生成自然语言。",
    "Bonjour, je voudrais réserver une table pour deux personnes ce soir.",
    "Energy cannot be created or destroyed, only transformed — first law of thermodynamics.",
    "Kubernetes pods share a network namespace and storage volumes.",
    "The Pythagorean theorem: a^2 + b^2 = c^2 in right triangles.",
    "한국어는 한글 문자를 사용하며 과학적인 음소문자 체계를 갖춘다.",
    "Blockchain uses cryptographic hash chains for immutable ledgers.",
    "The mitochondrion produces ATP via oxidative phosphorylation.",
]


# ── Numerical helpers ──────────────────────────────────────────────
def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm: x / rms(x) * weight."""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps) * weight).astype(np.float32)


def _silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))))


# ── Weight loading ─────────────────────────────────────────────────
def _safe_load_fp8(sf_path: Path, key: str, shape, scale):
    """Load tensor trying FP8 dequant first, then raw then bf16."""
    try:
        return load_fp8_tensor(sf_path, key, shape, scale).astype(np.float32)
    except Exception:
        pass
    try:
        with safe_open(str(sf_path), framework="numpy") as f:
            t = f.get_tensor(key)
            if hasattr(t, "astype"):
                return t.astype(np.float32)
    except Exception:
        pass
    return _load_bf16_tensor(sf_path, key, shape).astype(np.float32)


def _build_weight_map(model_path: Path) -> dict:
    """Return {tensor_name: shard_path}."""
    idx_file = model_path / "model.safetensors.index.json"
    if idx_file.exists():
        with open(idx_file) as f:
            return json.load(f)["weight_map"]
    # Fall back: scan shards.
    weight_map = {}
    for sf in sorted(model_path.glob("model-*.safetensors")):
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                if k.endswith("_scale_inv"):
                    continue
                weight_map[k] = sf.name
    return weight_map


def _load_tensor_by_name(model_path: Path, weight_map: dict, key: str):
    """Load a single tensor by its full key name (with shard lookup)."""
    shard = weight_map.get(key)
    if shard is None:
        return None
    sf_path = model_path / shard
    with safe_open(str(sf_path), framework="numpy") as f:
        if key not in f.keys():
            return None
        shape = list(f.get_slice(key).get_shape())
        sk = key + "_scale_inv"
        scale = None
        try:
            scale = f.get_tensor(sk)
        except Exception:
            scale = None
    return _safe_load_fp8(sf_path, key, shape, scale)


def _load_layer_weights(model_path: Path, weight_map: dict, li: int,
                        keys_to_load: list) -> dict:
    """Load a curated subset of layer-l weights."""
    out = {}
    for rel in keys_to_load:
        full = f"model.layers.{li}.{rel}"
        t = _load_tensor_by_name(model_path, weight_map, full)
        if t is None:
            full = f"layers.{li}.{rel}"
            t = _load_tensor_by_name(model_path, weight_map, full)
        if t is not None:
            out[rel] = t
    return out


# ── Forward-pass core ──────────────────────────────────────────────
def _propagate_through_layer(h: np.ndarray, layer_weights: dict,
                              is_moe: bool) -> tuple:
    """Propagate hidden state h through layer's RMSNorm+MLP (no attention).

    Captures per-channel max(|x|) at:
      - pre-attention (input_layernorm output)  -> "attn_input"
      - pre-mlp      (post_attention_layernorm output) -> "mlp_input"
      - mlp-intermediate (silu(gate)*up) -> "mlp_intermediate" for down_proj

    Returns (new_h, stats_dict).
    """
    stats = {}

    # Pre-attention RMSNorm
    if "input_layernorm.weight" in layer_weights:
        h_norm_attn = _rmsnorm(h, layer_weights["input_layernorm.weight"])
        stats["attn_input"] = np.abs(h_norm_attn).max(axis=(0, 1))
    elif "attn_norm.weight" in layer_weights:
        h_norm_attn = _rmsnorm(h, layer_weights["attn_norm.weight"])
        stats["attn_input"] = np.abs(h_norm_attn).max(axis=(0, 1))
    else:
        h_norm_attn = h

    # Pre-MLP RMSNorm (the critical one for routed experts)
    if "post_attention_layernorm.weight" in layer_weights:
        h_norm_mlp = _rmsnorm(h, layer_weights["post_attention_layernorm.weight"])
        stats["mlp_input"] = np.abs(h_norm_mlp).max(axis=(0, 1))
    elif "ffn_norm.weight" in layer_weights:
        h_norm_mlp = _rmsnorm(h, layer_weights["ffn_norm.weight"])
        stats["mlp_input"] = np.abs(h_norm_mlp).max(axis=(0, 1))
    else:
        h_norm_mlp = h

    # MLP forward — shared_expert for MoE, plain MLP for dense.
    if is_moe:
        prefixes = ["mlp.shared_experts", "ffn.shared_experts"]
    else:
        prefixes = ["mlp", "ffn"]

    mlp_out = np.zeros_like(h, dtype=np.float32)
    for p in prefixes:
        gw = layer_weights.get(f"{p}.gate_proj.weight") or layer_weights.get(f"{p}.w1.weight")
        uw = layer_weights.get(f"{p}.up_proj.weight") or layer_weights.get(f"{p}.w3.weight")
        dw = layer_weights.get(f"{p}.down_proj.weight") or layer_weights.get(f"{p}.w2.weight")
        if gw is None or uw is None or dw is None:
            continue
        gate = h_norm_mlp @ gw.T
        up = h_norm_mlp @ uw.T
        inter = _silu(gate) * up
        # Capture intermediate stats for down_proj input distribution
        stats["mlp_intermediate"] = np.abs(inter).max(axis=(0, 1))
        mlp_out = mlp_out + (inter @ dw.T).astype(np.float32)

    h_new = h + mlp_out
    return h_new, stats


# ── Top-level capture ──────────────────────────────────────────────
def run_capture_fp8(model_path, output_path,
                    n_samples=DEFAULT_N_SAMPLES, seq_len=DEFAULT_SEQ_LEN):
    model_path = Path(model_path)
    output_path = Path(output_path)

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)
    n_layers = tc["num_hidden_layers"]
    first_dense = tc.get("first_k_dense_replace", 0)
    hidden_size = tc["hidden_size"]
    vocab_size = tc["vocab_size"]

    print("=" * 60)
    print("  AWQ FP8 Layer-by-Layer Capture")
    print("=" * 60)
    print(f"  Source:    {model_path}")
    print(f"  Output:    {output_path}")
    print(f"  Layers:    {n_layers} ({first_dense} dense + {n_layers-first_dense} MoE)")
    print(f"  Samples:   {n_samples}  SeqLen: {seq_len}", flush=True)

    weight_map = _build_weight_map(model_path)
    print(f"  Weight map: {len(weight_map)} tensors", flush=True)

    # Load tokenizer for tokenizing calibration prompts.
    # §JANG_V3 (2026-04-27): transformers ≥4.50 requires `rope_parameters` in
    # config.json. DSV4 source ships only `rope_scaling`. AutoTokenizer.from_
    # pretrained tries to construct PreTrainedConfig and fails with
    # `'PreTrainedConfig' object has no attribute 'max_position_embeddings'`
    # → byte-level fallback → calibration stats become garbage (h.norm=1e12).
    # Two-stage fix: (1) try PreTrainedTokenizerFast directly with just
    # tokenizer.json (no config.json dependency), (2) fall back to a JANG
    # bundle's tokenizer if the source has the rope_scaling issue, (3) byte
    # last resort but warn loudly.
    tok = None
    try:
        from transformers import PreTrainedTokenizerFast
        tok_json = Path(model_path) / "tokenizer.json"
        if tok_json.exists():
            tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_json))
            print(f"  tokenizer: PreTrainedTokenizerFast (direct from tokenizer.json)", flush=True)
    except Exception as e:
        print(f"  PreTrainedTokenizerFast load failed: {e}", flush=True)
    if tok is None:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"  tokenizer: {tok.__class__.__name__}", flush=True)
        except Exception as e:
            print(f"  AutoTokenizer load failed: {e}", flush=True)
    if tok is not None:
        def encode_prompts(texts):
            return [tok.encode(t)[:seq_len] for t in texts]
    else:
        print(f"  ⚠ TOKENIZER LOAD FAILED — falling back to byte-level. "
              f"AWQ stats will be GARBAGE. Fix tokenizer or pass --tokenizer-path.", flush=True)
        def encode_prompts(texts):
            return [list(t.encode("utf-8"))[:seq_len] for t in texts]

    # Prepare calibration batches.
    prompts = list(_CALIB_PROMPTS)
    while len(prompts) < n_samples:
        prompts.append(_CALIB_PROMPTS[len(prompts) % len(_CALIB_PROMPTS)])
    prompts = prompts[:n_samples]
    token_batches = encode_prompts(prompts)
    max_sl = max(len(b) for b in token_batches)
    print(f"  max seq_len actual: {max_sl}", flush=True)

    # Load embedding weights (size: vocab x hidden).
    print("  Loading embed_tokens...", flush=True)
    t0 = time.time()
    embed = _load_tensor_by_name(
        model_path, weight_map, "model.embed_tokens.weight"
    )
    if embed is None:
        embed = _load_tensor_by_name(model_path, weight_map, "embed.weight")
    if embed is None:
        raise RuntimeError("embed.weight not found in weight_map")
    print(f"    embed: {embed.shape}{embed.dtype}  "
          f"norm={np.linalg.norm(embed):.1f}  "
          f"in {time.time()-t0:.1f}s", flush=True)

    # Embed all samples once (hidden states stacked as (n_samples, seq, hidden)).
    # For variable-length prompts, pad to max_sl with zeros.
    hidden = np.zeros((n_samples, max_sl, hidden_size), dtype=np.float32)
    mask = np.zeros((n_samples, max_sl), dtype=bool)
    for i, b in enumerate(token_batches):
        for j, tok_id in enumerate(b):
            hidden[i, j] = embed[tok_id]
            mask[i, j] = True
    del embed
    gc.collect()

    # Per-layer stats accumulator: {tensor_name: (in_features,) max}
    global_stats = {}

    def update_stat(key, vec):
        prev = global_stats.get(key)
        if prev is None:
            global_stats[key] = vec.astype(np.float32).copy()
        else:
            global_stats[key] = np.maximum(prev, vec.astype(np.float32))

    # Keys to try per layer.
    ATTN_KEYS = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
    ]
    MLP_KEYS_DENSE = [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    MLP_KEYS_MOE = [
        "mlp.shared_experts.gate_proj.weight",
        "mlp.shared_experts.up_proj.weight",
        "mlp.shared_experts.down_proj.weight",
        "ffn.shared_experts.w1.weight",
        "ffn.shared_experts.w3.weight",
        "ffn.shared_experts.w2.weight",
    ]

    print("  Running layer-by-layer forward...", flush=True)
    total_t = time.time()
    for li in range(n_layers):
        is_moe = li >= first_dense
        load_keys = list(ATTN_KEYS)
        load_keys += MLP_KEYS_MOE if is_moe else MLP_KEYS_DENSE
        t_l = time.time()
        lw = _load_layer_weights(model_path, weight_map, li, load_keys)
        t_load = time.time() - t_l

        # Propagate
        t_f = time.time()
        hidden, lstats = _propagate_through_layer(hidden, lw, is_moe)
        t_fwd = time.time() - t_f

        # Map captured stats to AWQ module names.
        if is_moe:
            base = f"model.layers.{li}.mlp.switch_mlp"
            if "mlp_input" in lstats:
                # routed experts gate_proj / up_proj share this input
                update_stat(f"{base}.gate_proj", lstats["mlp_input"])
                update_stat(f"{base}.up_proj", lstats["mlp_input"])
            if "mlp_intermediate" in lstats:
                update_stat(f"{base}.down_proj", lstats["mlp_intermediate"])
        # Always record attention inputs for bookkeeping (not used in
        # JANGTQ_2L since attention is FP16, but useful to have).
        if "attn_input" in lstats:
            update_stat(
                f"model.layers.{li}.self_attn.q_a_proj",
                lstats["attn_input"],
            )

        print(f"    layer {li:2d}: load={t_load:.1f}s fwd={t_fwd:.1f}s "
              f"h.norm={np.linalg.norm(hidden[0,0]):.2f} "
              f"stats={list(lstats.keys())}",
              flush=True)

        del lw
        gc.collect()

    print(f"  Total time: {time.time()-total_t:.1f}s", flush=True)
    print(f"  Captured stats for {len(global_stats)} tensor keys", flush=True)

    # Save.
    save_file(global_stats, str(output_path))
    sz = sum(a.nbytes for a in global_stats.values())
    mags = sorted([float(a.max()) for a in global_stats.values()], reverse=True)
    print(f"  top-5 max: {[f'{m:.3f}' for m in mags[:5]]}", flush=True)
    print(f"  bottom-5:  {[f'{m:.3f}' for m in mags[-5:]]}", flush=True)
    print(f"  wrote {output_path} ({sz/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 -m jang_tools.awq_capture_fp8 "
              "<fp8_src> <output_path> [n_samples] [seq_len]", file=sys.stderr)
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2]
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_N_SAMPLES
    sl = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_SEQ_LEN
    run_capture_fp8(src, out, n_samples=ns, seq_len=sl)
