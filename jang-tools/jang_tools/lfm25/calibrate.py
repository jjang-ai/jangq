"""LFM2.5 calibration capture — AWQ activation stats + QAT input samples.

Created by Jinho Jang (eric@jangq.ai) — 2026-08-04.

Runs the bf16 source through the live mlx_lm lfm2 forward (the exact serving
math) over the canonical JANG calibration mix, and records per layer:

  layers.{N}.ffn_input_max         (hidden,)  f32   max|x| at ffn_norm output
                                                    (input to w1/w3)
  layers.{N}.ffn_intermediate_max  (inter,)   f32   max|x| at silu(w1x)*w3x
                                                    (input to w2)
  layers.{N}.x1                    (K, hidden) f16  token samples of the
                                                    w1/w3 input, for GPTQ

Prompts are rendered through the bundle chat template with the generation
prompt appended (so the pre-opened ``<think>`` token is in-distribution), and
each sequence is extended with a short greedy continuation so captured
activations cover the model's own thinking-mode output, not just prompt text.

The source config.json lacks ``block_ff_dim`` (mlx_lm ModelArgs requires it);
it is patched in memory the same way the vendor's own MLX export stamps it
(block_ff_dim = intermediate_size).

Usage:
    python -m jang_tools.lfm25.calibrate \
        --src ~/.mlxstudio/models/LiquidAI/LFM2.5-2.6B \
        --out /path/to/lfm25_calib.safetensors \
        --gen-tokens 128 --seq-len 512 --max-x1-rows 8192
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.numpy import save_file

# Canonical JANG calibration mix (per feedback_jangreap_corpus_mix).
from jang_tools.awq_capture_jang import _CALIB_PROMPTS


def load_source_model(src: Path):
    from mlx_lm.models import lfm2
    from mlx_lm.utils import load_tokenizer

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    config.setdefault("block_ff_dim", config["intermediate_size"])
    args = lfm2.ModelArgs.from_dict(config)
    model = lfm2.Model(args)
    weights: dict[str, mx.array] = {}
    for f in sorted(src.glob("model-*.safetensors")):
        weights.update(mx.load(str(f)))
    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(src)
    return model, tokenizer, config


class _Recorder:
    """Accumulates AWQ max-abs stats and x1 token samples per layer."""

    def __init__(self, n_layers: int, hidden: int, inter: int):
        self.enabled = False
        self.ffn_input_max = [np.zeros(hidden, dtype=np.float32) for _ in range(n_layers)]
        self.inter_max = [np.zeros(inter, dtype=np.float32) for _ in range(n_layers)]
        self.x1_rows: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.token_count = 0

    def record(self, idx: int, x: mx.array, inter: mx.array) -> None:
        if not self.enabled:
            return
        x2d = np.asarray(x.reshape(-1, x.shape[-1]).astype(mx.float32))
        i2d = np.asarray(inter.reshape(-1, inter.shape[-1]).astype(mx.float32))
        np.maximum(self.ffn_input_max[idx], np.abs(x2d).max(axis=0),
                   out=self.ffn_input_max[idx])
        np.maximum(self.inter_max[idx], np.abs(i2d).max(axis=0),
                   out=self.inter_max[idx])
        self.x1_rows[idx].append(x2d.astype(np.float16))
        if idx == 0:
            self.token_count += x2d.shape[0]


def patch_mlp(recorder: _Recorder, mlp_to_idx: dict[int, int]):
    """Patch lfm2.MLP.__call__ to record w1/w3 input + swiglu intermediate."""
    from mlx_lm.models import lfm2

    orig = lfm2.MLP.__call__

    def patched(self, x):
        idx = mlp_to_idx.get(id(self))
        if idx is None or not recorder.enabled:
            return orig(self, x)
        g = self.w1(x)
        u = self.w3(x)
        inter = nn.silu(g) * u
        mx.eval(inter)
        recorder.record(idx, x, inter)
        return self.w2(inter)

    lfm2.MLP.__call__ = patched
    return orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="greedy continuation length per prompt (thinking-mode coverage)")
    ap.add_argument("--seq-len", type=int, default=512,
                    help="max forward length per captured sequence")
    ap.add_argument("--max-x1-rows", type=int, default=8192,
                    help="token samples kept per layer for GPTQ")
    args = ap.parse_args()

    src = args.src.expanduser()
    model, tokenizer, config = load_source_model(src)
    n_layers = config["num_hidden_layers"]
    hidden = config["hidden_size"]
    inter = config["intermediate_size"]
    eos_ids = config["eos_token_id"]
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    mlp_to_idx = {id(l.feed_forward): i for i, l in enumerate(model.layers)}
    recorder = _Recorder(n_layers, hidden, inter)
    patch_mlp(recorder, mlp_to_idx)

    from mlx_lm.models.cache import make_prompt_cache

    print(f"Capture: {len(_CALIB_PROMPTS)} canonical-mix prompts, "
          f"+{args.gen_tokens} greedy tokens each, seq_len<={args.seq_len}")
    t0 = time.time()
    sequences: list[list[int]] = []
    for p_i, prompt_text in enumerate(_CALIB_PROMPTS):
        messages = [{"role": "user", "content": prompt_text}]
        ids = list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        # greedy continuation (capture disabled) so thinking-mode activations
        # appear in the capture pass below
        recorder.enabled = False
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]))
        gen = []
        for _ in range(args.gen_tokens):
            gen.append(tok)
            if tok in eos_ids:
                break
            logits = model(mx.array([[tok]]), cache=cache)
            tok = int(mx.argmax(logits[0, -1]))
        seq = (ids + gen)[: args.seq_len]
        sequences.append(seq)

    # capture pass: single forward per sequence, recorder on
    for s_i, seq in enumerate(sequences):
        recorder.enabled = True
        model(mx.array([seq]))
        recorder.enabled = False
        mx.clear_cache()
    print(f"  captured {recorder.token_count} tokens/layer "
          f"in {time.time() - t0:.1f}s")

    rng = np.random.default_rng(0)
    tensors: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        x1 = np.concatenate(recorder.x1_rows[i], axis=0)
        if x1.shape[0] > args.max_x1_rows:
            keep = rng.choice(x1.shape[0], size=args.max_x1_rows, replace=False)
            keep.sort()
            x1 = x1[keep]
        tensors[f"layers.{i}.ffn_input_max"] = recorder.ffn_input_max[i]
        tensors[f"layers.{i}.ffn_intermediate_max"] = recorder.inter_max[i]
        tensors[f"layers.{i}.x1"] = x1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata={
        "source": src.name,
        "tokens_per_layer": str(recorder.token_count),
        "x1_rows": str(min(recorder.token_count, args.max_x1_rows)),
        "corpus": "canonical-mix-v2026-04-23 + greedy thinking continuations",
    })
    size_mb = args.out.stat().st_size / 1e6
    print(f"  wrote {args.out} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
