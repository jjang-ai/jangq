"""Calibration capture for Nemotron-3-Nano-Omni-30B-A3B (nemotron_h backbone).

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-22.

One forward pass feeds all three consumers of calibration data:

  * **diagonal** per-channel second moment ``E[x^2]`` for every quantized
    Linear -> imatrix fit, AWQ scales, and Hessian-trace bit allocation;
  * **full ``H = X^T X``** (float64) for the routed-expert projections only
    -> GPTQ's error-compensated rounding, which needs the whole Hessian and
    not just its diagonal.

Why this module exists rather than a flag on an existing one
-----------------------------------------------------------
`awq_capture` hardcodes ``embed_tokens`` / ``input_layernorm`` and
`ornith_moe_hessians` patches ``SwitchGLU`` at ``layer.mlp.switch_mlp``.
Nemotron-H matches neither: the MoE block is ``layer.mixer.switch_mlp`` and it
is a **`SwitchMLP`** (``fc1`` -> ``activation`` -> ``fc2``, no gate), because the
expert MLP is gate-less ``relu2``.

Two Hessians per MoE layer, because two different activations feed the two
expert projections:

    backbone.layers.N.norm(h)  -> fc1   (hidden 2688)
    relu2(fc1(x))              -> fc2   (moe_intermediate 1856)

The ``fc2`` input is taken by swapping ``switch_mlp.activation`` for a tap
object: it sits exactly where ``nn.ReLU2`` does, so it observes the real input
without re-deriving ``SwitchMLP``'s internals (``expand_dims``, ``_gather_sort``
and the sorted-index plumbing are all easy to get wrong).

All experts in a layer share one H, matching `gptq_mlx`. Because the tap is on
the *block* input rather than per-expert, that shared H sees **every** token, so
its conditioning is ``tokens / d_in`` (not ``tokens*k/E / d_in``) -- at 200 k
tokens that is 74x over-determined on the 2688-wide ``fc1``.

Accumulation is float64 throughout: the DSV4 lesson is that f32 inversion of a
rank-deficient H silently falls back to RTN, i.e. you get no GPTQ at all and no
error saying so.

Memory: 2688^2 f64 = 57.8 MiB per MoE layer for ``in`` plus 1856^2 = 27.6 MiB
for ``mid`` -> ~2.0 GiB across 23 MoE layers, on top of the ~59 GiB bf16 LLM.

    PYTHONPATH=~/jang/jang-tools \
    python -m jang_tools.nemotron_omni_calibrate \
        --src <bf16_omni_dir> --out <dir> \
        [--corpus kimi_v3_calib/corpus_v3.jsonl] [--tokens 300000]
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Routed-expert source key: language_model.backbone.layers.N.mixer.experts.E.{up,down}_proj.weight
EXPERT_KEY_RE = re.compile(
    r"^(?:language_model\.)?backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\."
    r"(up_proj|down_proj)\.weight$"
)


class _Diag:
    """Running per-channel second moment E[x^2], float64."""

    __slots__ = ("s", "n")

    def __init__(self) -> None:
        self.s: np.ndarray | None = None
        self.n = 0

    def add(self, x: mx.array) -> None:
        a = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1])
        if a.shape[0] == 0:
            return
        acc = np.einsum("ij,ij->j", a, a, dtype=np.float64)
        if self.s is None:
            self.s = acc
        else:
            self.s += acc
        self.n += a.shape[0]


class _Full:
    """Running H = X^T X, float64, plus a token counter for conditioning."""

    __slots__ = ("H", "n")

    def __init__(self) -> None:
        self.H: np.ndarray | None = None
        self.n = 0

    def add(self, x: mx.array) -> None:
        a = np.asarray(x.astype(mx.float32)).reshape(-1, x.shape[-1]).astype(np.float64)
        if a.shape[0] == 0:
            return
        if self.H is None:
            self.H = np.zeros((a.shape[1], a.shape[1]), dtype=np.float64)
        self.H += a.T @ a
        self.n += a.shape[0]


def load_llm(src: Path):
    """Build the nemotron_h LLM from the omni checkpoint, weights only.

    Reads only ``language_model.*`` (the towers are never quantized, so they are
    not calibrated), strips the prefix, and runs the module's own ``sanitize``
    so the per-expert tensors stack into ``switch_mlp.{fc1,fc2}``.
    """
    from mlx_lm.models.nemotron_h import Model, ModelArgs

    full = json.loads((src / "config.json").read_text())
    llm_cfg = full.get("llm_config", full)
    args = ModelArgs.from_dict(llm_cfg)
    model = Model(args)

    # mx.load mmaps the shard: entries stay lazy until touched, so the 59 GiB of
    # LLM weights are never simultaneously resident as both a source dict and a
    # stacked copy. Eager numpy reads here peaked near 88 GiB (source + the
    # expert stacks sanitize() builds) on a 128 GiB box.
    weights: dict[str, mx.array] = {}
    shards = sorted(src.glob("model-*.safetensors")) or sorted(src.glob("*.safetensors"))
    for sf in shards:
        shard = mx.load(str(sf))
        for k, v in shard.items():
            if k.startswith("language_model."):
                weights[k[len("language_model."):]] = v
        del shard
    if not weights:
        raise RuntimeError(
            f"no language_model.* tensors in {src} — wrong source layout, "
            "refusing rather than calibrating an empty model")

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())
    del weights
    gc.collect()
    return model, args, full


def _blocks(model):
    inner = getattr(model, "backbone", None) or getattr(model, "model", model)
    return getattr(inner, "layers", [])


def patch(model) -> tuple[dict, dict]:
    """Tap every calibration point. Returns (full_hessians, diagonals).

    Raises rather than returning empty accumulators — an empty capture that
    reports success is the failure mode this whole module exists to avoid.
    """
    from mlx_lm.models.switch_layers import SwitchMLP, SwitchLinear

    full: dict[tuple[int, str], _Full] = {}
    diag: dict[str, _Diag] = {}
    registry: dict[int, _Full] = {}

    n_moe = 0
    for idx, layer in enumerate(_blocks(model)):
        mixer = getattr(layer, "mixer", None)
        smlp = getattr(mixer, "switch_mlp", None) if mixer is not None else None
        if smlp is None or not isinstance(smlp, SwitchMLP):
            continue
        f_in, f_mid = _Full(), _Full()
        full[(idx, "in")] = f_in
        full[(idx, "mid")] = f_mid
        registry[id(smlp)] = f_in

        class _TapAct:
            """Sits exactly where nn.ReLU2 does, so it sees fc2's input."""

            def __init__(self, inner, sink):
                self._inner = inner
                self._sink = sink

            def __call__(self, x):
                out = self._inner(x)
                self._sink.add(out)
                return out

        smlp.activation = _TapAct(smlp.activation, f_mid)
        n_moe += 1

    if n_moe == 0:
        raise RuntimeError(
            "no MoE layers patched — layer.mixer.switch_mlp not found. Refusing "
            "rather than writing empty Hessians.")

    # SwitchMLP.__call__ tap -> fc1 input (shared H for all experts in the layer)
    if not getattr(SwitchMLP, "_jang_tapped", False):
        _orig_smlp = SwitchMLP.__call__

        def _tapped_smlp(self, x, indices, *a, **k):
            sink = registry.get(id(self))
            if sink is not None:
                sink.add(x)
            return _orig_smlp(self, x, indices, *a, **k)

        SwitchMLP.__call__ = _tapped_smlp
        SwitchMLP._jang_tapped = True

    # Diagonal for every dense Linear (attention q/k/v/o, mamba in/out_proj,
    # shared expert up/down, router gate, lm_head). Keyed by module identity and
    # resolved to tensor names afterwards.
    dense_reg: dict[int, _Diag] = {}
    named = {}
    def _walk(mod, prefix=""):
        for name, child in getattr(mod, "children", lambda: {})().items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (list, tuple)):
                for i, c in enumerate(child):
                    _walk(c, f"{path}.{i}")
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, SwitchLinear):
                d = _Diag()
                diag[path] = d
                dense_reg[id(child)] = d
                named[path] = child
            _walk(child, path)
    _walk(model)

    if not getattr(nn.Linear, "_jang_tapped", False):
        _orig_lin = nn.Linear.__call__

        def _tapped_lin(self, x, *a, **k):
            sink = dense_reg.get(id(self))
            if sink is not None:
                sink.add(x)
            return _orig_lin(self, x, *a, **k)

        nn.Linear.__call__ = _tapped_lin
        nn.Linear._jang_tapped = True

    print(f"  patched {n_moe} MoE layers (full H) + {len(diag)} dense Linears (diag)",
          flush=True)
    return full, diag


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path.home() / ".cache" / "jang" / "corpus_v3.jsonl",
    )
    p.add_argument("--tokens", type=int, default=300_000,
                   help="target calibration tokens (shared-H rank = tokens/d_in)")
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(a.src), trust_remote_code=True)

    from jang_tools.qwen36_calibrate import build_text_corpus
    corpus, total_tok, spent = build_text_corpus(
        tok, a.corpus, a.tokens, max_prompt_tokens=a.max_prompt_tokens)
    print(f"  corpus: {len(corpus)} prompts, {total_tok:,} tokens", flush=True)
    print(f"  mix: {spent}", flush=True)

    print("  loading BF16 LLM (~59 GiB)...", flush=True)
    t0 = time.time()
    model, args, full_cfg = load_llm(a.src)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    full, diag = patch(model)

    t0 = time.time()
    for i, text in enumerate(corpus, 1):
        ids = mx.array([tok.encode(text)])
        model(ids)          # prefill only — activations are what we need
        mx.eval(mx.zeros(1))
        del ids
        if i % 25 == 0 or i == len(corpus):
            print(f"    {i}/{len(corpus)} ({time.time()-t0:.0f}s)", flush=True)
            gc.collect()

    # ---- persist -------------------------------------------------------
    d_in = args.hidden_size
    payload = {}
    ranks = {}
    for (idx, kind), acc in full.items():
        if acc.H is None:
            raise RuntimeError(f"empty Hessian for layer {idx} {kind} — refusing to write")
        payload[f"H.{idx}.{kind}"] = acc.H.astype(np.float64)
        ranks[f"{idx}.{kind}"] = round(acc.n / acc.H.shape[0], 2)
    np.savez(a.out / "expert_hessians.npz", **payload)

    dpayload = {}
    for name, acc in diag.items():
        if acc.s is None:
            continue        # module never executed (e.g. unused head)
        dpayload[name] = acc.s.astype(np.float64)
    np.savez(a.out / "diag_second_moment.npz", **dpayload)

    meta = {
        "source": str(a.src),
        "corpus": str(a.corpus),
        "prompts": len(corpus),
        "tokens": total_tok,
        "mix": spent,
        "moe_layers": len([k for k in full if k[1] == "in"]),
        "dense_modules": len(dpayload),
        "hidden_size": d_in,
        "moe_intermediate_size": args.moe_intermediate_size,
        "hessian_rank_ratio": ranks,
        "min_rank_ratio": min(ranks.values()) if ranks else None,
        "norm_convention": "plain_rmsnorm",   # weight * x, no +1 offset
    }
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  wrote {a.out}/expert_hessians.npz "
          f"({len(payload)} Hessians), diag_second_moment.npz "
          f"({len(dpayload)} modules)")
    print(f"  min Hessian rank ratio: {meta['min_rank_ratio']}x "
          f"(>=1.0 required for GPTQ; <1 means singular -> silent RTN)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
