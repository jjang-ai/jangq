#!/usr/bin/env python3
"""
CRACK Surgery for JANG Models — Multi-Projection Edition.

Supports surgery on ALL attention projections (q/k/v/o_proj),
with correct axis handling per projection type:
- o_proj: v in OUTPUT space → W' = W - s * v @ (vT @ W)
- q/k/v_proj: v in INPUT space → W' = W - s * (W @ v) @ vT

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
    w_cols = weight_shape[-1]
    s_cols = scales_shape[-1]
    return int(round((w_cols * 32) / (s_cols * group_size)))


def apply_surgery(weight_fp, refusal_vec, strength, proj_type):
    """Apply CRACK ablation with axis-aware formula.

    o_proj: v in output space (axis 0) → W' = W - s * v @ (vT @ W)
    q/k/v_proj: v in input space (axis 1) → W' = W - s * (W @ v) @ vT
    """
    v = mx.array(refusal_vec).astype(mx.float32)
    v = v / mx.sqrt(mx.sum(v * v))
    W = weight_fp.astype(mx.float32)

    if proj_type == "o_proj":
        # v is in output dimension (W rows)
        if W.ndim == 3:
            vTW = mx.einsum("o,eoi->ei", v, W)
            proj = mx.einsum("o,ei->eoi", v, vTW)
        else:
            vTW = v @ W
            proj = mx.outer(v, vTW)
        W_new = W - strength * proj
    else:
        # q/k/v_proj: v is in input dimension (W columns)
        if W.ndim == 3:
            Wv = mx.einsum("eoi,i->eo", W, v)
            proj = mx.einsum("eo,i->eoi", Wv, v)
        else:
            Wv = W @ v
            proj = mx.outer(Wv, v)
        W_new = W - strength * proj

    return W_new.astype(mx.float16)


def run_surgery(
    model_path, vector_path, output_path,
    direction_layer=61, target_layers=None,
    strength=2.0, group_size=128,
    target_projs=None,
):
    model_path = Path(model_path)
    output_path = Path(output_path)
    vector_path = Path(vector_path)

    if target_layers is None:
        target_layers = list(range(50, 62))
    if target_projs is None:
        target_projs = ["q_proj", "k_proj", "v_proj", "o_proj"]

    print(f"\n{'='*60}")
    print(f"  CRACK Multi-Projection Surgery for JANG Model")
    print(f"{'='*60}")
    print(f"  Model:     {model_path}")
    print(f"  Vector:    {vector_path}")
    print(f"  Output:    {output_path}")
    print(f"  Direction: L{direction_layer}")
    print(f"  Targets:   {target_layers}")
    print(f"  Projs:     {target_projs}")
    print(f"  Strength:  {strength}")
    print(f"  Group sz:  {group_size}")
    print(f"{'='*60}\n")

    # Load refusal vector (handle bf16)
    try:
        with safe_open(str(vector_path), framework="numpy") as f:
            v = f.get_tensor(str(direction_layer)).astype(np.float32)
    except TypeError:
        vecs = mx.load(str(vector_path))
        v = np.array(vecs[str(direction_layer)].astype(mx.float32))
    print(f"  Vector L{direction_layer}: shape={v.shape}, "
          f"max|v|={np.max(np.abs(v)):.4f}")

    # Build target patterns
    target_patterns = []
    for layer_idx in target_layers:
        for proj in target_projs:
            target_patterns.append(
                (f"model.layers.{layer_idx}.self_attn.{proj}", layer_idx, proj))
            target_patterns.append(
                (f"model.language_model.layers.{layer_idx}.self_attn.{proj}",
                 layer_idx, proj))

    # Read config
    config = json.loads((model_path / "config.json").read_text())
    cfg_gs = config.get("quantization", {}).get("group_size", group_size)

    # Copy model
    if output_path != model_path:
        print(f"\n  Copying model...")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(model_path, output_path)
        print(f"  Copied -> {output_path.name}")

    # Load shard index
    index_path = output_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text()).get("weight_map", {})
    else:
        weight_map = {}
        for sf in sorted(output_path.glob("*.safetensors")):
            with safe_open(str(sf), framework="numpy") as f:
                for k in f.keys():
                    weight_map[k] = sf.name

    # Find targets by shard
    targets_by_shard = {}
    for base_name, layer_idx, proj_type in target_patterns:
        w_key = f"{base_name}.weight"
        if w_key in weight_map:
            shard = weight_map[w_key]
            if shard not in targets_by_shard:
                targets_by_shard[shard] = []
            targets_by_shard[shard].append((base_name, layer_idx, proj_type))

    total = sum(len(v) for v in targets_by_shard.values())
    if not targets_by_shard:
        print("  ERROR: No targets found!")
        sample = list(weight_map.keys())[:5]
        print(f"  Sample keys: {sample}")
        sys.exit(1)
    print(f"\n  Found {total} targets across {len(targets_by_shard)} shards")

    # Process
    modified = 0
    for shard_name, targets in targets_by_shard.items():
        shard_path = output_path / shard_name
        print(f"\n  Processing {shard_name} ({len(targets)} targets)...")

        all_tensors = {}
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

        for base_name, layer_idx, proj_type in targets:
            w_key = f"{base_name}.weight"
            s_key = f"{base_name}.scales"
            b_key = f"{base_name}.biases"

            if w_key not in all_tensors or s_key not in all_tensors:
                continue

            weight = all_tensors[w_key]
            scales = all_tensors[s_key]
            biases = all_tensors.get(b_key)
            if biases is None:
                biases = np.zeros_like(scales)

            bits = infer_bits(weight.shape, scales.shape, group_size)

            # Dequantize
            W_fp = mx.dequantize(
                mx.array(weight), mx.array(scales), mx.array(biases),
                group_size=group_size, bits=bits)

            # Surgery with correct axis
            W_new = apply_surgery(W_fp, v, strength, proj_type)
            mx.synchronize()

            # Requantize
            qw, qs, qb = mx.quantize(W_new, group_size=group_size, bits=bits)
            mx.synchronize()

            print(f"    L{layer_idx} {proj_type}: {bits}-bit, "
                  f"shape={weight.shape}", flush=True)

            all_tensors[w_key] = np.array(qw)
            all_tensors[s_key] = np.array(qs).astype(np.float16)
            all_tensors[b_key] = np.array(qb).astype(np.float16)
            modified += 1
            mx.clear_cache()

        # Write
        print(f"    Writing {shard_name}...")
        clean = {k: (np.array(v) if isinstance(v, mx.array) else v)
                 for k, v in all_tensors.items()}
        np_save_file(clean, str(shard_path), metadata={"format": "mlx"})

    # Update jang_config
    jcfg_path = output_path / "jang_config.json"
    if jcfg_path.exists():
        jcfg = json.loads(jcfg_path.read_text())
        jcfg["crack_surgery"] = {
            "vector": str(vector_path.name),
            "direction_layer": direction_layer,
            "target_layers": target_layers,
            "target_projs": target_projs,
            "strength": strength,
            "modified_tensors": modified,
            "mode": "standard_multi_proj",
        }
        jcfg_path.write_text(json.dumps(jcfg, indent=2))

    print(f"\n{'='*60}")
    print(f"  CRACK Multi-Proj Surgery Complete")
    print(f"  Modified: {modified} tensors")
    print(f"  Output:   {output_path}")
    print(f"{'='*60}\n")
    return modified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CRACK Multi-Projection Surgery for JANG")
    parser.add_argument("model")
    parser.add_argument("-v", "--vector", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-d", "--direction-layer", type=int, default=61)
    parser.add_argument("-l", "--layers", type=str, default="50,51,52,53,54,55,56,57,58,59,60,61")
    parser.add_argument("-p", "--projs", type=str, default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("-s", "--strength", type=float, default=2.0)
    parser.add_argument("-g", "--group-size", type=int, default=128)
    args = parser.parse_args()

    run_surgery(
        model_path=args.model,
        vector_path=args.vector,
        output_path=args.output,
        direction_layer=args.direction_layer,
        target_layers=[int(x) for x in args.layers.split(",")],
        strength=args.strength,
        group_size=args.group_size,
        target_projs=args.projs.split(","),
    )
