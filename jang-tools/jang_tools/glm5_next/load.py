"""Load / sanitize glm5_next checkpoints into the MLX runtime.

Canonical (sanitized) naming = HF names with these transforms:
  model.language_model.X            -> model.X
  layers.N.hc_{attn,ffn}_{base,fn,scale} -> layers.N.{attn,ffn}_hc.hc_{...}
  layers.N.mlp.experts.E.{p}_proj   -> layers.N.mlp.switch_mlp.{p}_proj  [stacked E-first]
  layers.N.mlp.gate.e_score_correction_bias -> layers.N.mlp.e_score_correction_bias
  layers.N.self_attn.{q,k,v}_conv1d [C,1,W] -> [C,W]
  layers.N.self_attn.o_norm.weight  -> layers.N.self_attn.o_norm
  layers.45.* / model.visual.*      -> DROPPED for text eval (converter keeps)

fp32 keeps at load: A_log, dt_bias, e_score_correction_bias, hc_base, hc_scale.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import mlx.core as mx

from .modeling import Glm5Args, Glm5NextForCausalLM

FP32_SUFFIXES = ("A_log", "dt_bias", "e_score_correction_bias", "hc_base", "hc_scale")


def sanitize(weights: dict, num_layers: int = 45) -> dict:
    out = {}
    experts: dict = {}
    for k, v in weights.items():
        if k.startswith("model.visual."):
            continue
        k = k.replace("model.language_model.", "model.")
        m = re.match(r"model\.layers\.(\d+)\.", k)
        if m and int(m.group(1)) >= num_layers:
            continue  # MTP block — eval runtime drops it (HF does the same)
        if ".self_attn.indexer." in k:
            continue  # dense DSA bypass — indexer unused for T <= index_topk
        m = re.match(r"(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|up|down)_proj\.weight", k)
        if m:
            experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = v
            continue
        k = re.sub(r"\.hc_(attn|ffn)_(base|fn|scale)$", r".\1_hc.hc_\2", k)
        k = k.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
        if k.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            k = k[: -len(".weight")]
            v = v.reshape(v.shape[0], v.shape[-1])
        if k.endswith("self_attn.o_norm.weight"):
            k = k[: -len(".weight")]
        if any(k.endswith(s) for s in FP32_SUFFIXES):
            v = v.astype(mx.float32)
        out[k] = v
    for (base, proj), parts in experts.items():
        E = max(parts) + 1
        stacked = mx.stack([parts[i] for i in range(E)], axis=0)
        out[f"{base}.switch_mlp.{proj}_proj.weight"] = stacked
    return out


def load_bundle(bundle_dir: str) -> Glm5NextForCausalLM:
    """Load a quantized JANG glm5_next bundle (runtime naming, per-module
    quantization recorded in config['quantization'] at build time)."""
    import mlx.nn as nn

    d = Path(bundle_dir).expanduser()
    cfg = json.loads((d / "config.json").read_text())
    args = Glm5Args.from_config(cfg)
    qmap = {k: v for k, v in (cfg.get("quantization") or {}).items()
            if isinstance(v, dict)}
    assert qmap, "bundle config has no per-module quantization block"

    weights = {}
    for f in sorted(glob.glob(str(d / "model-*.safetensors"))):
        weights.update(mx.load(f))
    weights = {k: v for k, v in weights.items()
               if not k.startswith("visual.")
               and not re.match(rf"model\.layers\.{args.num_hidden_layers}\.", k)
               and ".self_attn.indexer." not in k}

    model = Glm5NextForCausalLM(args)

    def class_predicate(path, module):
        spec = qmap.get(path)
        if spec is None:
            return False
        return {"group_size": spec["group_size"], "bits": spec["bits"]}

    nn.quantize(model, class_predicate=class_predicate)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


def load_model(model_dir: str, dtype=mx.bfloat16) -> Glm5NextForCausalLM:
    d = Path(model_dir).expanduser()
    cfg = json.loads((d / "config.json").read_text())
    args = Glm5Args.from_config(cfg)
    weights = {}
    for f in sorted(glob.glob(str(d / "model*.safetensors"))):
        weights.update(mx.load(f))
    weights = sanitize(weights, args.num_hidden_layers)
    cast = {k: (v if any(k.endswith(s) for s in FP32_SUFFIXES) or v.dtype == mx.uint32
                else v.astype(dtype)) for k, v in weights.items()}
    model = Glm5NextForCausalLM(args)
    model.load_weights(list(cast.items()), strict=True)
    mx.eval(model.parameters())
    return model
