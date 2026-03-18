#!/usr/bin/env python3
"""
Optimized MiniMax CRACK surgery — signal-aware, minimal damage.

Builds multiple configs for A/B testing via vMLX API.
Each config targets different layer ranges and projection sets.

Usage:
  python3 crack_minimax_optimized.py --config A  # Top 11 layers, 4 projs
  python3 crack_minimax_optimized.py --config B  # Top 20 layers, 4 projs
  python3 crack_minimax_optimized.py --config C  # Top 11, o+q only
  python3 crack_minimax_optimized.py --config E  # Signal-weighted all 62

Created by Jinho Jang (eric@jangq.ai)
"""

import sys, json, shutil, argparse
sys.path.insert(0, '/Users/eric/jang/jang-tools')
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file
from pathlib import Path

MODEL = "/Users/eric/.mlxstudio/models/MiniMax-M2.5-JANG_2L"
VECTOR = "/Users/eric/CRACK_abliteration/vectors/minimax_m25_jang_projected_vectors.safetensors"
GROUP_SIZE = 128

# Load signal scores for weighting
SCORES = json.load(open(VECTOR.replace('.safetensors', '_scores.json')))['scores']

CONFIGS = {
    "A": {
        "name": "Top 11 layers (L51-61) × 4 projs",
        "layers": list(range(51, 62)),
        "projs": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "strength": 5.0,
        "weighted": False,
    },
    "B": {
        "name": "Top 20 layers (L42-61) × 4 projs",
        "layers": list(range(42, 62)),
        "projs": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "strength": 3.5,
        "weighted": False,
    },
    "C": {
        "name": "Top 11 (L51-61) × o+q only",
        "layers": list(range(51, 62)),
        "projs": ["q_proj", "o_proj"],
        "strength": 6.0,
        "weighted": False,
    },
    "E": {
        "name": "Signal-weighted ALL 62 × 4 projs",
        "layers": list(range(62)),
        "projs": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "strength": 5.0,
        "weighted": True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=CONFIGS.keys())
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    output = f"/Users/eric/.mlxstudio/models/MiniMax-M2.5-JANG_2L-CRACK"

    print(f"\nConfig {args.config}: {cfg['name']}", flush=True)
    print(f"Layers: {len(cfg['layers'])}, Projs: {len(cfg['projs'])}, "
          f"Base strength: {cfg['strength']}, Weighted: {cfg['weighted']}", flush=True)

    # Copy base model
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(MODEL, output)
    print("Copied base model", flush=True)

    # Load vectors
    all_vecs = mx.load(VECTOR)
    per_layer_v = {}
    max_signal = max(float(SCORES[str(li)]) for li in cfg['layers'])
    for li in cfg['layers']:
        v = np.array(all_vecs[str(li)].astype(mx.float32))
        v_mx = mx.array(v).astype(mx.float32)
        n = mx.sqrt(mx.sum(v_mx * v_mx))
        per_layer_v[li] = v_mx / mx.maximum(n, mx.array(1e-8))

    # Build targets
    idx = json.load(open(f"{output}/model.safetensors.index.json"))
    wm = idx.get("weight_map", {})

    targets_by_shard = {}
    for li in cfg['layers']:
        for proj in cfg['projs']:
            wk = f"model.layers.{li}.self_attn.{proj}.weight"
            if wk in wm:
                shard = wm[wk]
                if shard not in targets_by_shard:
                    targets_by_shard[shard] = []
                targets_by_shard[shard].append(
                    (f"model.layers.{li}.self_attn.{proj}", li, proj))

    total = sum(len(v) for v in targets_by_shard.values())
    print(f"Targets: {total} tensors in {len(targets_by_shard)} shards", flush=True)

    modified = 0
    for shard_name, tgts in sorted(targets_by_shard.items()):
        shard_path = f"{output}/{shard_name}"
        print(f"Processing {shard_name} ({len(tgts)} targets)...", flush=True)
        all_tensors = {}
        with safe_open(shard_path, framework="numpy") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

        for base, li, proj in tgts:
            wk, sk, bk = f"{base}.weight", f"{base}.scales", f"{base}.biases"
            if wk not in all_tensors or sk not in all_tensors:
                continue
            weight, scales = all_tensors[wk], all_tensors[sk]
            biases = all_tensors.get(bk, np.zeros_like(scales))
            bits = int(round((weight.shape[-1] * 32) / (scales.shape[-1] * GROUP_SIZE)))

            v_mx = per_layer_v[li]

            # Compute effective strength
            s = cfg['strength']
            if cfg['weighted']:
                signal = float(SCORES[str(li)])
                s = cfg['strength'] * (signal / max_signal)

            W = mx.dequantize(mx.array(weight), mx.array(scales),
                              mx.array(biases), group_size=GROUP_SIZE, bits=bits
                              ).astype(mx.float32)
            if proj == "o_proj":
                W_new = (W - s * mx.outer(v_mx, v_mx @ W)).astype(mx.float16)
            else:
                W_new = (W - s * mx.outer(W @ v_mx, v_mx)).astype(mx.float16)
            mx.synchronize()
            qw, qs, qb = mx.quantize(W_new, group_size=GROUP_SIZE, bits=bits)
            mx.synchronize()

            all_tensors[wk] = np.array(qw)
            all_tensors[sk] = np.array(qs).astype(np.float16)
            all_tensors[bk] = np.array(qb).astype(np.float16)
            modified += 1
            mx.clear_cache()

        clean = {k: (np.array(v) if isinstance(v, mx.array) else v)
                 for k, v in all_tensors.items()}
        np_save_file(clean, shard_path, metadata={"format": "mlx"})
        print(f"  Written ({modified} total)", flush=True)

    # Update config
    jcfg = json.load(open(f"{output}/jang_config.json"))
    jcfg["crack_surgery"] = {
        "config": args.config,
        "description": cfg['name'],
        "layers": cfg['layers'],
        "projs": cfg['projs'],
        "base_strength": cfg['strength'],
        "weighted": cfg['weighted'],
        "modified_tensors": modified,
        "method": "projected_per_layer",
    }
    json.dump(jcfg, open(f"{output}/jang_config.json", "w"), indent=2)

    # Copy branding
    import os
    for f in ["dealign_mascot.png", "dealign_logo.png"]:
        src = f"/Users/eric/.mlxstudio/models/dealignai/GPT-OSS-120B-MLX-CRACK/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output}/{f}")

    print(f"\nDONE: Config {args.config}, {modified} tensors → {output}", flush=True)


if __name__ == "__main__":
    main()
