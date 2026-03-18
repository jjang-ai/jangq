#!/usr/bin/env python3
"""
Probe a JANG model with PROJECTED method for cleaner refusal vectors.

Projected method: removes harmless component from refusal direction,
producing vectors that isolate refusal WITHOUT entangling knowledge.

v_diff = mean(harmful) - mean(harmless)
v_harmless_norm = mean(harmless) / ||mean(harmless)||
v_projected = v_diff - (v_diff . v_harmless_norm) * v_harmless_norm

Created by Jinho Jang (eric@jangq.ai)
"""

import sys, time, argparse, json
sys.path.insert(0, '/Users/eric/jang/jang-tools')
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from jang_tools.loader import load_jang_model
from safetensors.numpy import save_file


def create_causal_mask(seq_len):
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    return mask.astype(mx.float16)


def manual_forward(model, input_ids):
    m = model.model
    h = m.embed_tokens(input_ids)
    seq_len = input_ids.shape[1]
    mask = create_causal_mask(seq_len)
    states = []
    for layer in m.layers:
        h = layer(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        last_h = h[:, -1, :].astype(mx.float32)
        mx.synchronize()
        states.append(np.array(last_h).flatten())
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--n-prompts", type=int, default=200)
    args = parser.parse_args()

    print(f"Loading: {args.model}", flush=True)
    model, tokenizer = load_jang_model(args.model)
    print(f"GPU: {mx.get_active_memory()/1024**3:.1f} GB", flush=True)

    cfg = json.load(open(f"{args.model}/config.json"))
    tc = cfg.get("text_config", cfg)
    hidden_size = tc.get("hidden_size", 3072)
    n_layers = tc.get("num_hidden_layers", 62)
    print(f"Hidden: {hidden_size}, Layers: {n_layers}", flush=True)

    # Load prompts — support both .txt (one per line) and .jsonl (prompt field)
    def load_prompts(path, n):
        prompts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{'):
                    try:
                        d = json.loads(line)
                        prompts.append(d.get('prompt', d.get('text', line)))
                    except json.JSONDecodeError:
                        prompts.append(line)
                else:
                    prompts.append(line)
                if len(prompts) >= n:
                    break
        return prompts

    harmful = load_prompts(args.harmful, args.n_prompts)
    harmless = load_prompts(args.harmless, args.n_prompts)
    print(f"Prompts: {len(harmful)} harmful, {len(harmless)} harmless\n", flush=True)

    def run_prompts(prompts, label):
        means = [np.zeros(hidden_size, dtype=np.float64) for _ in range(n_layers)]
        count = 0
        for pi, prompt in enumerate(prompts):
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    enable_thinking=False, tokenize=False)
            except Exception:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)

            tokens = mx.array([tokenizer.encode(text)])
            try:
                states = manual_forward(model, tokens)
                for li in range(min(len(states), n_layers)):
                    if len(states[li]) == hidden_size:
                        means[li] += states[li].astype(np.float64)
                count += 1
            except Exception as e:
                if pi < 3:
                    print(f"  Error [{pi}]: {e}", flush=True)

            if (pi + 1) % 20 == 0:
                print(f"  {label}: {pi+1}/{len(prompts)}", flush=True)
            mx.clear_cache()

        for li in range(n_layers):
            means[li] /= max(count, 1)
        return means, count

    print("Probing harmful...", flush=True)
    h_means, h_count = run_prompts(harmful, "harmful")
    print(f"\nProbing harmless...", flush=True)
    hl_means, hl_count = run_prompts(harmless, "harmless")

    # Compute PROJECTED vectors (the key improvement)
    print(f"\nComputing PROJECTED vectors ({h_count} harmful, {hl_count} harmless):",
          flush=True)
    vectors = {}
    scores = {}

    for li in range(n_layers):
        diff = h_means[li] - hl_means[li]

        # PROJECTED method: remove harmless component
        harmless_norm = np.linalg.norm(hl_means[li])
        if harmless_norm > 1e-9:
            harmless_unit = hl_means[li] / harmless_norm
            projection_scalar = np.dot(diff, harmless_unit)
            projected = diff - projection_scalar * harmless_unit
        else:
            projected = diff

        proj_norm = np.linalg.norm(projected)
        scores[li] = float(np.max(np.abs(projected)))

        if proj_norm > 1e-8:
            vectors[str(li)] = (projected / proj_norm).astype(np.float32)
        else:
            vectors[str(li)] = projected.astype(np.float32)

        # Also compute difference method score for comparison
        diff_score = float(np.max(np.abs(diff)))

        print(f"  L{li}: proj={scores[li]:.4f} (diff={diff_score:.4f}, "
              f"removed={abs(projection_scalar) if harmless_norm > 1e-9 else 0:.4f})",
              flush=True)

    save_file(vectors, args.output)
    best = max(scores, key=scores.get)
    scores_path = args.output.replace('.safetensors', '_scores.json')
    json.dump({
        "n_harmful": h_count, "n_harmless": hl_count,
        "n_layers": n_layers, "hidden_size": hidden_size,
        "method": "projected",
        "scores": {str(k): v for k, v in scores.items()},
        "top10": sorted(scores.items(), key=lambda x: -x[1])[:10],
        "best_layer": best, "best_score": scores[best],
    }, open(scores_path, 'w'), indent=2)
    print(f"\nSaved to {args.output}", flush=True)
    print(f"Best: L{best} (max|v|={scores[best]:.4f})", flush=True)


if __name__ == "__main__":
    main()
