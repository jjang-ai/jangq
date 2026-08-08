"""Load an Inkling JANG bundle into the MLX runtime.

Dedicated loader rather than the generic `jang_tools.loader`, because the bundle
deliberately keeps the checkpoint's own tensor names (`wq_du`, `unembed`, fused
`w13_weight`) while the model tree uses readable ones. That means the
quantization map's keys must go through the SAME rename as the weights, or
`nn.quantize`'s class predicate silently matches nothing and every module is
built dense while the file holds packed uint32 — a load that "succeeds" and
produces garbage. `Model.map_key` is shared by both paths so they cannot drift.

Usage:
    from jang_tools.inkling.load import load_inkling
    model, tok = load_inkling("~/models/JANGQ-AI/Inkling-Small-JANG_1L")
"""

from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from .model import Model, ModelArgs


def _load_tq_spec(bundle: Path) -> dict | None:
    """Routed-expert TQ widths, or None for a plain affine bundle."""
    cfg = json.loads((bundle / "config.json").read_text())
    if cfg.get("weight_format") != "mxtq":
        return None
    jc_path = bundle / "jang_config.json"
    jc = json.loads(jc_path.read_text()) if jc_path.is_file() else {}
    spec = dict(jc.get("routed_expert_bits") or {})
    spec.setdefault("gate_proj", jc.get("mxtq_bits", 2))
    spec.setdefault("up_proj", jc.get("mxtq_bits", 2))
    spec.setdefault("down_proj", jc.get("mxtq_bits", 2))
    spec["seed"] = jc.get("seed", 42)
    return spec


def _load_quant_map(bundle: Path) -> dict[str, dict]:
    """Per-module {bits, group_size} for AFFINE modules only, keyed by
    MODEL-tree module path. JANGTQ modules are excluded: they are not
    nn.quantize-able, they arrive already packed as `packed`/`norms`."""
    cfg = json.loads((bundle / "config.json").read_text())
    q = cfg.get("quantization", {}) or {}
    out: dict[str, dict] = {}
    inter = None
    tc = cfg.get("text_config", cfg)
    inter = tc.get("intermediate_size", tc.get("moe_intermediate_size"))
    for base, v in q.items():
        # The current text-only reference model intentionally has no MTP
        # modules. Keep their quantization metadata in the bundle for the
        # official engines without reporting 8x20 false "unmatched" warnings.
        if base.startswith("model.mtp") or ".mtp." in base:
            continue
        if not isinstance(v, dict) or "bits" not in v:
            continue
        if v.get("mode") == "mxtq":
            continue
        spec = {"bits": int(v["bits"]), "group_size": int(v.get("group_size", 64))}
        mapped = Model.map_key(base)
        # Fused tensors are split by sanitize(); both halves inherit the spec.
        if base.endswith("mlp.experts.w13_weight"):
            b = Model.map_key(base[: -len("experts.w13_weight")])
            out[b + "switch_mlp.gate_proj"] = spec
            out[b + "switch_mlp.up_proj"] = spec
        elif base.endswith("mlp.experts.w2_weight"):
            b = Model.map_key(base[: -len("experts.w2_weight")])
            out[b + "switch_mlp.down_proj"] = spec
        elif base.endswith("shared_experts.shared_w13_weight"):
            b = Model.map_key(base[: -len("shared_w13_weight")])
            out[b + "gate_proj"] = spec
            out[b + "up_proj"] = spec
        elif base.endswith("shared_experts.shared_w2_weight"):
            b = Model.map_key(base[: -len("shared_w2_weight")])
            out[b + "down_proj"] = spec
        elif base.endswith("mlp.w13_dn"):
            b = Model.map_key(base[: -len("w13_dn")])
            out[b + "gate_proj"] = spec
            out[b + "up_proj"] = spec
        elif base.endswith("mlp.w2_md"):
            b = Model.map_key(base[: -len("w2_md")])
            out[b + "down_proj"] = spec
        else:
            out[mapped] = spec
    return out


def load_inkling(bundle: str | Path, verbose: bool = True,
                 dtype: mx.Dtype = mx.bfloat16):
    """Build the model, apply per-module quantization, load weights.

    `dtype` casts the floating parameters (scales/biases/norms) and therefore the
    activations. It defaults to **bfloat16, and fp16 genuinely does not work**:
    the bundle ships fp16, and measured on this model the residual stream grows
    monotonically with depth — absmax 121 at L2, 831 at L7, 2.9e4 at L20 — and
    blows past fp16's 65504 ceiling at layer 24, turning the whole hidden state
    NaN. Every prompt then decodes to token 0 at a perfectly healthy 37 tok/s.
    bfloat16 has fp32's exponent range, so the same magnitudes are unremarkable.
    Matches `project_bfloat16_fix` (many-expert MoE needs bf16 activations) and
    the reference, which keeps sconv and the shared-expert sum in fp32.
    """
    bundle = Path(bundle).expanduser()
    t0 = time.time()
    cfg = json.loads((bundle / "config.json").read_text())
    args = ModelArgs.from_dict(cfg)
    tq = _load_tq_spec(bundle)
    model = Model(args, tq=tq)
    if verbose and tq:
        print(f"  JANGTQ routed experts: gate{tq['gate_proj']} up{tq['up_proj']} "
              f"down{tq['down_proj']} (seed {tq['seed']})")

    qmap = _load_quant_map(bundle)
    if verbose:
        print(f"  quantization map: {len(qmap)} modules")

    matched: list[str] = []

    def class_predicate(path: str, module) -> Any:
        spec = qmap.get(path)
        if spec is None:
            return False
        if not hasattr(module, "to_quantized"):
            return False
        matched.append(path)
        return spec

    nn.quantize(model, group_size=64, bits=8, class_predicate=class_predicate)
    if verbose:
        print(f"  nn.quantize matched {len(matched)}/{len(qmap)} modules")
    if not matched:
        raise RuntimeError(
            "quantization predicate matched NOTHING — the quant map keys do not "
            "line up with the module tree. Loading would silently build dense "
            "modules over packed uint32 weights and emit garbage."
        )
    missed = sorted(set(qmap) - set(matched))
    if missed and verbose:
        print(f"  WARNING: {len(missed)} quant-map modules unmatched, e.g. {missed[:3]}")

    # Sanitize PER SHARD and evaluate as we go. Loading all 84 shards into one
    # dict first would hold the full 89.7 GB, and splitting the fused `w13`
    # tensors (48.3 GB of it) then adds two halves on top before the parents can
    # be released — enough to exceed a 137 GB box. Per-shard keeps the peak near
    # one shard plus the running model.
    weights: dict[str, mx.array] = {}
    shards = sorted(glob.glob(str(bundle / "*.safetensors")))
    n_raw = 0
    for i, shard in enumerate(shards):
        raw = mx.load(shard)
        n_raw += len(raw)
        part = model.sanitize(raw)
        mx.eval(list(part.values()))     # materialize slices, release the parents
        weights.update(part)
        del raw, part
        if verbose and (i + 1) % 20 == 0:
            print(f"    shard {i+1}/{len(shards)}")
    if verbose:
        print(f"  read {n_raw} tensors from {len(shards)} shards")
    expected = set(dict(tree_flatten(model.parameters())))
    got = set(weights)
    missing, extra = sorted(expected - got), sorted(got - expected)
    if verbose:
        print(f"  after sanitize: {len(got)} tensors | missing {len(missing)} | extra {len(extra)}")
    if missing:
        raise RuntimeError(f"missing {len(missing)} parameters, e.g. {missing[:5]}")
    if extra:
        # Unbound keys mean a rename gap: they would be silently dropped.
        raise RuntimeError(f"{len(extra)} unbound tensors, e.g. {extra[:5]}")

    model.update(tree_unflatten(list(weights.items())))
    del weights
    if dtype is not None:
        # Quantizer metadata is part of the encoded weight, not an activation
        # parameter. The converter emits affine scales/biases and TQ norms as
        # fp16; rounding them again to bf16 changes every reconstructed matrix
        # and invalidates converter-side quality measurements.
        quant_storage = [
            (name, value)
            for name, value in tree_flatten(model.parameters())
            if name.endswith((".scales", ".biases", ".norms"))
            and value.dtype == mx.float16
        ]
        # The correction bias is a selection-only FP32 checkpoint parameter.
        # Preserve it so tiny top-k margins do not flip experts across backends.
        # Do NOT preserve gate.global_scale: it participates in the BF16 expert
        # arithmetic and keeping it FP32 promotes shared-expert activations.
        fp32_router_biases = []
        for layer in model.layers:
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is not None:
                fp32_router_biases.append((gate, gate.bias.astype(mx.float32)))
        # set_dtype's default predicate casts floating arrays only, leaving
        # uint32 packed quant weights untouched.
        model.set_dtype(dtype)
        if quant_storage:
            model.update(tree_unflatten(quant_storage))
        for gate, bias in fp32_router_biases:
            gate.bias = bias
        if verbose:
            print(f"  preserved {len(fp32_router_biases)} FP32 router biases")
            print(f"  preserved {len(quant_storage)} FP16 quantizer tensors")
    mx.eval(model.parameters())
    model.eval()
    if verbose:
        seen_dt = {str(v.dtype) for _, v in tree_flatten(model.parameters())}
        print(f"  parameter dtypes: {sorted(seen_dt)}")
    if verbose:
        print(f"  loaded in {time.time() - t0:.1f}s")

    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    except Exception as exc:  # tokenizer is optional for a raw logits probe
        if verbose:
            print(f"  tokenizer unavailable ({type(exc).__name__}) — id-level probes only")
    return model, tok


def greedy(model: Model, prompt_ids: list[int], n: int = 32,
           cache: Optional[Any] = None) -> list[int]:
    """Greedy decode. Returns generated ids."""
    cache = cache if cache is not None else model.make_cache()
    logits = model(mx.array([prompt_ids]), cache)
    mx.eval(logits)
    out = [int(mx.argmax(logits[0, -1]).item())]
    for _ in range(n - 1):
        logits = model(mx.array([[out[-1]]]), cache)
        mx.eval(logits)
        out.append(int(mx.argmax(logits[0, -1]).item()))
    return out
