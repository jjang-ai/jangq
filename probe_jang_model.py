#!/usr/bin/env python3
"""
Probe a JANG model for refusal vectors.

Uses the JANG loader + monkey-patched layer forwarding to capture
residual stream activations at each layer boundary.

Created by Jinho Jang (eric@jangq.ai)
"""

import sys, time
sys.path.insert(0, '/Users/eric/jang/jang-tools')
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import numpy as np
from jang_tools.loader import load_jang_model
from mlx_lm import generate
from safetensors.numpy import save_file
import json, argparse


def probe_model(model_path, harmful_path, harmless_path, output_path, n_prompts=64):
    print(f"Loading model: {model_path}", flush=True)
    model, tokenizer = load_jang_model(model_path)
    print(f"Loaded! GPU: {mx.get_active_memory()/1024**3:.1f} GB", flush=True)

    # Find the layers list
    layers = None
    for attr_path in ['model.layers', 'model.model.layers',
                      'model.language_model.layers',
                      'model.model.language_model.layers']:
        obj = model
        try:
            for part in attr_path.split('.'):
                obj = getattr(obj, part)
            layers = obj
            print(f"Found layers at {attr_path}: {len(layers)}", flush=True)
            break
        except AttributeError:
            continue

    if layers is None:
        print("ERROR: Cannot find layers!", flush=True)
        sys.exit(1)

    n_layers = len(layers)

    # Get hidden_size from config
    cfg = json.load(open(f"{model_path}/config.json"))
    tc = cfg.get("text_config", cfg)
    hidden_size = tc.get("hidden_size", 3072)
    print(f"Hidden size: {hidden_size}, Layers: {n_layers}", flush=True)

    # Load prompts
    with open(harmful_path) as f:
        harmful = [l.strip() for l in f if l.strip()][:n_prompts]
    with open(harmless_path) as f:
        harmless = [l.strip() for l in f if l.strip()][:n_prompts]
    print(f"Prompts: {len(harmful)} harmful, {len(harmless)} harmless", flush=True)

    # Monkey-patch layers to capture hidden states
    captured_states = {}

    def make_capture_wrapper(layer_idx, original_call):
        def wrapper(*args, **kwargs):
            result = original_call(*args, **kwargs)
            if isinstance(result, tuple):
                h = result[0]
            else:
                h = result
            if h is not None and h.ndim >= 2:
                last_h = h[:, -1, :].astype(mx.float32)
                mx.synchronize()
                captured_states[layer_idx] = np.array(last_h).flatten()
            return result
        return wrapper

    # Install hooks
    original_calls = {}
    for i, layer in enumerate(layers):
        original_calls[i] = layer.__call__
        layer.__call__ = make_capture_wrapper(i, layer.__call__)

    def run_prompts(prompts, label):
        means = [np.zeros(hidden_size, dtype=np.float32) for _ in range(n_layers)]
        count = 0
        for pi, prompt in enumerate(prompts):
            captured_states.clear()
            msgs = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True,
                    enable_thinking=False, tokenize=False)
            except Exception:
                text = tokenizer.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)

            try:
                generate(model, tokenizer, prompt=text, max_tokens=1)
            except Exception as e:
                if pi == 0:
                    print(f"  Generate error: {e}", flush=True)

            if captured_states:
                for li in range(n_layers):
                    if li in captured_states and len(captured_states[li]) == hidden_size:
                        means[li] += captured_states[li]
                count += 1

            if (pi + 1) % 16 == 0:
                print(f"  {label}: {pi+1}/{len(prompts)} "
                      f"(captured {len(captured_states)} layers)", flush=True)
            mx.clear_cache()

        for li in range(n_layers):
            means[li] /= max(count, 1)
        return means, count

    print(f"\nProbing harmful prompts...", flush=True)
    harmful_means, h_count = run_prompts(harmful, "harmful")

    print(f"\nProbing harmless prompts...", flush=True)
    harmless_means, hl_count = run_prompts(harmless, "harmless")

    # Restore original calls
    for i, layer in enumerate(layers):
        layer.__call__ = original_calls[i]

    # Compute difference vectors
    print(f"\nComputing vectors ({h_count} harmful, {hl_count} harmless)...",
          flush=True)
    vectors = {}
    scores = {}
    for li in range(n_layers):
        diff = harmful_means[li] - harmless_means[li]
        norm = np.linalg.norm(diff)
        scores[li] = float(np.max(np.abs(diff)))
        if norm > 1e-8:
            vectors[str(li)] = (diff / norm).astype(np.float32)
        else:
            vectors[str(li)] = diff.astype(np.float32)
        print(f"  L{li}: max|v|={scores[li]:.4f}, norm={norm:.4f}", flush=True)

    # Save
    save_file(vectors, output_path)
    print(f"\nSaved {len(vectors)} vectors to {output_path}", flush=True)

    # Save scores
    scores_path = output_path.replace('.safetensors', '_scores.json')
    best = max(scores, key=scores.get)
    json.dump({
        "n_harmful": h_count,
        "n_harmless": hl_count,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "model": model_path,
        "scores": {str(k): v for k, v in scores.items()},
        "top10": sorted(scores.items(), key=lambda x: -x[1])[:10],
        "best_layer": best,
        "best_score": scores[best],
    }, open(scores_path, 'w'), indent=2)
    print(f"Saved scores to {scores_path}", flush=True)
    print(f"Best direction: L{best} (max|v|={scores[best]:.4f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe JANG model for refusal vectors")
    parser.add_argument("model", help="Path to JANG model")
    parser.add_argument("--harmful", required=True)
    parser.add_argument("--harmless", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-n", "--n-prompts", type=int, default=64)
    args = parser.parse_args()

    probe_model(args.model, args.harmful, args.harmless, args.output, args.n_prompts)
