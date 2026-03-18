#!/usr/bin/env python3
"""
CRACK Surgery for JANG Models — Mixed-precision aware abliteration.

Applies CRACK refusal vector surgery to JANG quantized models, respecting
per-tensor bit widths. Dequantizes target tensors at their actual bit width
(not the config.json default), applies surgery, requantizes at the same width.

Created by Jinho Jang (eric@jangq.ai)
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file


def infer_bits(weight_shape, scales_shape, group_size):
    """Infer quantization bit width from tensor shapes.

    MLX packs N-bit values into uint32. The relationship:
      weight_cols = (in_features * bits) / 32  (packed)
      scales_cols = in_features / group_size
    So: bits = (weight_cols * 32) / (scales_cols * group_size)
    """
    w_cols = weight_shape[-1]
    s_cols = scales_shape[-1]
    bits = (w_cols * 32) / (s_cols * group_size)
    return int(round(bits))


def dequantize_tensor(weight, scales, biases, bits, group_size):
    """Dequantize a quantized tensor to float16."""
    w_mx = mx.array(weight)
    s_mx = mx.array(scales)
    b_mx = mx.array(biases)

    if w_mx.ndim == 3:
        n_exp = w_mx.shape[0]
        results = []
        for e in range(n_exp):
            dq = mx.dequantize(w_mx[e], s_mx[e], b_mx[e],
                               group_size=group_size, bits=bits)
            results.append(dq)
        return mx.stack(results)

    return mx.dequantize(w_mx, s_mx, b_mx, group_size=group_size, bits=bits)


def quantize_tensor(weight_fp, bits, group_size):
    """Quantize a float tensor back to MLX native format."""
    if weight_fp.ndim == 3:
        n_exp = weight_fp.shape[0]
        all_w, all_s, all_b = [], [], []
        for e in range(n_exp):
            qw, qs, qb = mx.quantize(weight_fp[e], group_size=group_size, bits=bits)
            all_w.append(qw)
            all_s.append(qs)
            all_b.append(qb)
        return mx.stack(all_w), mx.stack(all_s), mx.stack(all_b)

    return mx.quantize(weight_fp, group_size=group_size, bits=bits)


def apply_surgery(weight_fp, refusal_vec, strength):
    """Apply CRACK ablation: W' = W - s * v_hat @ (v_hat^T @ W)

    Standard ablation (not MPOA) — more aggressive, proven for MoE.
    """
    v = mx.array(refusal_vec).astype(mx.float32)
    v = v / mx.sqrt(mx.sum(v * v))  # ensure unit norm

    W = weight_fp.astype(mx.float32)

    if W.ndim == 3:
        # Batched experts: v in residual-stream space (out_dim)
        # W shape: (n_experts, out_dim, in_dim)
        vTW = mx.einsum("o,eoi->ei", v, W)
        proj = mx.einsum("o,ei->eoi", v, vTW)
        W_new = W - strength * proj
    else:
        # 2D: W shape (out_dim, in_dim), v shape (out_dim,)
        vTW = v @ W
        proj = mx.outer(v, vTW)
        W_new = W - strength * proj

    return W_new.astype(mx.float16)


def run_surgery(
    model_path: str,
    vector_path: str,
    output_path: str,
    direction_layer: int = 43,
    target_layers: list[int] = None,
    strength: float = 8.60,
    group_size: int = 128,
):
    model_path = Path(model_path)
    output_path = Path(output_path)
    vector_path = Path(vector_path)

    if target_layers is None:
        target_layers = [23, 27, 31, 35, 39, 43]

    print(f"\n{'='*60}")
    print(f"  CRACK Surgery for JANG Model")
    print(f"{'='*60}")
    print(f"  Model:     {model_path}")
    print(f"  Vector:    {vector_path}")
    print(f"  Output:    {output_path}")
    print(f"  Direction: L{direction_layer}")
    print(f"  Targets:   {target_layers}")
    print(f"  Strength:  {strength}")
    print(f"  Group sz:  {group_size}")
    print(f"{'='*60}\n")

    # Load refusal vector (may be bf16 which numpy can't handle)
    try:
        with safe_open(str(vector_path), framework="numpy") as f:
            v = f.get_tensor(str(direction_layer)).astype(np.float32)
    except TypeError:
        # bf16 fallback: load via MLX which handles bf16 natively
        vecs = mx.load(str(vector_path))
        v_key = str(direction_layer)
        v = np.array(vecs[v_key].astype(mx.float32))
    print(f"  Loaded direction vector L{direction_layer}: shape={v.shape}, "
          f"max|v|={np.max(np.abs(v)):.4f}")

    # Build target tensor patterns
    target_patterns = []
    for layer_idx in target_layers:
        target_patterns.append(
            f"model.layers.{layer_idx}.self_attn.o_proj")
        # VL/JANG key format (language_model prefix, no extra model.)
        target_patterns.append(
            f"model.language_model.layers.{layer_idx}.self_attn.o_proj")

    # Read config
    config = json.loads((model_path / "config.json").read_text())
    cfg_group_size = config.get("quantization", {}).get("group_size", group_size)
    if cfg_group_size != group_size:
        print(f"  Note: config group_size={cfg_group_size}, using {group_size}")

    # Copy model to output (if different path)
    if output_path != model_path:
        print(f"\n  Copying model to output directory...")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(model_path, output_path)
        print(f"  Copied {model_path.name} -> {output_path.name}")
    else:
        print(f"\n  In-place surgery on {model_path.name}")

    # Load shard index
    index_path = output_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
    else:
        weight_map = {}
        for sf in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    weight_map[k] = sf.name

    # Find target tensors and their shards
    targets_by_shard = {}
    for pattern in target_patterns:
        weight_key = f"{pattern}.weight"
        if weight_key in weight_map:
            shard = weight_map[weight_key]
            layer_idx = int(pattern.split("layers.")[1].split(".")[0])
            if shard not in targets_by_shard:
                targets_by_shard[shard] = []
            targets_by_shard[shard].append((pattern, layer_idx))

    if not targets_by_shard:
        print("  ERROR: No target tensors found in model!")
        sample = list(weight_map.keys())[:5]
        print(f"  Sample keys: {sample}")
        sys.exit(1)

    total_targets = sum(len(v) for v in targets_by_shard.values())
    print(f"\n  Found {total_targets} target tensors across "
          f"{len(targets_by_shard)} shards")

    # Process each shard
    modified = 0
    for shard_name, targets in targets_by_shard.items():
        shard_path = output_path / shard_name
        print(f"\n  Processing {shard_name}...")

        # Load ALL tensors from this shard
        all_tensors = {}
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

        for base_name, layer_idx in targets:
            w_key = f"{base_name}.weight"
            s_key = f"{base_name}.scales"
            b_key = f"{base_name}.biases"

            if w_key not in all_tensors or s_key not in all_tensors:
                print(f"    SKIP {base_name} (missing weight/scales)")
                continue

            weight = all_tensors[w_key]
            scales = all_tensors[s_key]
            biases = all_tensors.get(b_key)

            # Infer actual bit width from tensor shapes
            bits = infer_bits(weight.shape, scales.shape, group_size)
            print(f"    L{layer_idx} o_proj: {bits}-bit, "
                  f"shape={weight.shape}", end="")

            # Handle missing biases (mxfp4)
            if biases is None:
                print(" [no biases]", end="")
                biases = np.zeros_like(scales)

            # Dequantize at actual bit width
            W_fp = dequantize_tensor(weight, scales, biases, bits, group_size)

            # Apply CRACK surgery
            W_new = apply_surgery(W_fp, v, strength)
            mx.synchronize()

            # Requantize at same bit width
            qw, qs, qb = quantize_tensor(W_new, bits, group_size)
            mx.synchronize()

            # Compute change stats
            old_dq = dequantize_tensor(weight, scales, biases, bits, group_size)
            delta = mx.abs(W_new - old_dq)
            change_pct = (mx.mean(delta) /
                          (mx.mean(mx.abs(old_dq)) + 1e-8)).item() * 100
            print(f" -> surgery applied, delta={change_pct:.2f}%")

            # Replace in tensor dict
            all_tensors[w_key] = np.array(qw)
            all_tensors[s_key] = np.array(qs).astype(np.float16)
            all_tensors[b_key] = np.array(qb).astype(np.float16)

            modified += 1
            mx.clear_cache()

        # Write modified shard
        print(f"    Writing {shard_name}...")
        clean = {}
        for k, arr in all_tensors.items():
            if isinstance(arr, mx.array):
                arr = np.array(arr)
            clean[k] = arr
        np_save_file(clean, str(shard_path))
        print(f"    Done ({shard_name})")

    # Update jang_config with CRACK info
    jang_cfg_path = output_path / "jang_config.json"
    if jang_cfg_path.exists():
        jcfg = json.loads(jang_cfg_path.read_text())
        jcfg["crack_surgery"] = {
            "vector": str(vector_path.name),
            "direction_layer": direction_layer,
            "target_layers": target_layers,
            "strength": strength,
            "modified_tensors": modified,
            "mode": "standard",
        }
        jang_cfg_path.write_text(json.dumps(jcfg, indent=2))

    print(f"\n{'='*60}")
    print(f"  CRACK Surgery Complete")
    print(f"  Modified: {modified} tensors")
    print(f"  Output:   {output_path}")
    print(f"{'='*60}\n")

    return modified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRACK Surgery for JANG Models")
    parser.add_argument("model",
                        help="Path to JANG model directory")
    parser.add_argument("-v", "--vector", required=True,
                        help="Path to refusal vector .safetensors")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("-d", "--direction-layer", type=int, default=43,
                        help="Direction layer (default: 43)")
    parser.add_argument("-l", "--layers", type=str,
                        default="23,27,31,35,39,43",
                        help="Target layers (comma-separated)")
    parser.add_argument("-s", "--strength", type=float, default=8.60,
                        help="Surgery strength (default: 8.60)")
    parser.add_argument("-g", "--group-size", type=int, default=128,
                        help="Quantization group size (default: 128)")

    args = parser.parse_args()
    target_layers = [int(x) for x in args.layers.split(",")]

    run_surgery(
        model_path=args.model,
        vector_path=args.vector,
        output_path=args.output,
        direction_layer=args.direction_layer,
        target_layers=target_layers,
        strength=args.strength,
        group_size=args.group_size,
    )
