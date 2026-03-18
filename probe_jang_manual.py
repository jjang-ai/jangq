#!/usr/bin/env python3
"""
Probe a JANG model via manual forward pass.

Manually runs embeddings → each layer → captures hidden states.
This bypasses MLX's compiled dispatch which ignores __call__ hooks.

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
    """Create a causal attention mask (float16 for MiniMax compatibility)."""
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    return mask.astype(mx.float16)


def manual_forward(model, input_ids):
    """Run manual forward pass, return hidden state after each layer."""
    # Get model internals
    m = model.model  # MiniMaxModel / Qwen3_5MoeModel etc.

    # Embed
    h = m.embed_tokens(input_ids)

    # Create causal mask
    seq_len = input_ids.shape[1]
    mask = create_causal_mask(seq_len)

    states = []
    for i, layer in enumerate(m.layers):
        h = layer(h, mask=mask, cache=None)
        if isinstance(h, tuple):
            h = h[0]
        # Capture last-token hidden state
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
    parser.add_argument("-n", "--n-prompts", type=int, default=64)
    args = parser.parse_args()

    print(f"Loading: {args.model}", flush=True)
    model, tokenizer = load_jang_model(args.model)
    print(f"GPU: {mx.get_active_memory()/1024**3:.1f} GB", flush=True)

    cfg = json.load(open(f"{args.model}/config.json"))
    tc = cfg.get("text_config", cfg)
    hidden_size = tc.get("hidden_size", 3072)
    n_layers = tc.get("num_hidden_layers", 62)
    print(f"Hidden: {hidden_size}, Layers: {n_layers}", flush=True)

    with open(args.harmful) as f:
        harmful = [l.strip() for l in f if l.strip()][:args.n_prompts]
    with open(args.harmless) as f:
        harmless = [l.strip() for l in f if l.strip()][:args.n_prompts]
    print(f"Prompts: {len(harmful)} harmful, {len(harmless)} harmless\n", flush=True)

    def run_prompts(prompts, label):
        means = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
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
                        means[li] += states[li]
                count += 1
            except Exception as e:
                if pi < 3:
                    print(f"  Error [{pi}]: {e}", flush=True)

            if (pi + 1) % 8 == 0:
                print(f"  {label}: {pi+1}/{len(prompts)}", flush=True)
            mx.clear_cache()

        for li in range(n_layers):
            means[li] /= max(count, 1)
        return means, count

    print("Probing harmful...", flush=True)
    h_means, h_count = run_prompts(harmful, "harmful")
    print(f"\nProbing harmless...", flush=True)
    hl_means, hl_count = run_prompts(harmless, "harmless")

    print(f"\nVectors ({h_count} harmful, {hl_count} harmless):", flush=True)
    vectors = {}
    scores = {}
    for li in range(n_layers):
        diff = h_means[li] - hl_means[li]
        norm = np.linalg.norm(diff)
        scores[li] = float(np.max(np.abs(diff)))
        vectors[str(li)] = (diff / max(norm, 1e-8)).astype(np.float32)
        print(f"  L{li}: max|v|={scores[li]:.4f}, norm={norm:.4f}", flush=True)

    save_file(vectors, args.output)
    best = max(scores, key=scores.get)
    scores_path = args.output.replace('.safetensors', '_scores.json')
    json.dump({
        "n_harmful": h_count, "n_harmless": hl_count,
        "n_layers": n_layers, "hidden_size": hidden_size,
        "scores": {str(k): v for k, v in scores.items()},
        "top10": sorted(scores.items(), key=lambda x: -x[1])[:10],
        "best_layer": best, "best_score": scores[best],
    }, open(scores_path, 'w'), indent=2)
    print(f"\nSaved to {args.output}", flush=True)
    print(f"Best: L{best} (max|v|={scores[best]:.4f})", flush=True)


if __name__ == "__main__":
    main()
