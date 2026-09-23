"""Load third-party mlx-quantized glm5_next bundles (HF naming, per-expert
quant triples) into OUR runtime for apples-to-apples KL evaluation.

Their layout (e.g. orcarouter/GLM-5.3-Flash-MLX):
  model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.{weight,scales,biases}
  + config['quantization'] = {group_size, bits, <module>: {...} | False, ...}

We stack the QUANTIZED triples across experts (no dequantization — their
fidelity is measured exactly), translate names through our sanitize rules,
and build the model with nn.quantize driven by their per-module config.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import mlx.core as mx

from .modeling import Glm5Args, Glm5NextForCausalLM


def _their_spec(qcfg: dict, hf_module: str):
    """Resolve (bits, gs) for an HF module name from their config block."""
    e = qcfg.get(hf_module)
    if e is False:
        return None
    if isinstance(e, dict):
        return {"group_size": e["group_size"], "bits": e["bits"]}
    return {"group_size": qcfg.get("group_size", 64), "bits": qcfg.get("bits", 4)}


def load_foreign(bundle_dir: str) -> Glm5NextForCausalLM:
    d = Path(bundle_dir).expanduser()
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        cfg_path = d.parent / "config.json"
    cfg = json.loads(cfg_path.read_text())
    args = Glm5Args.from_config(cfg)
    qcfg = cfg.get("quantization") or cfg.get("quantization_config") or {}
    assert qcfg, "foreign bundle has no quantization config"

    raw = {}
    for f in sorted(glob.glob(str(d / "model*.safetensors"))):
        raw.update(mx.load(f))

    out = {}
    experts: dict = {}
    spec_by_module: dict = {}
    for k, v in raw.items():
        if k.startswith("model.visual."):
            continue
        rk = k.replace("model.language_model.", "model.")
        m = re.match(r"model\.layers\.(\d+)\.", rk)
        if m and int(m.group(1)) >= args.num_hidden_layers:
            continue
        if ".self_attn.indexer." in rk:
            continue
        me = re.match(r"(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|scales|biases)", rk)
        if me:
            experts.setdefault((me.group(1), me.group(3), me.group(4)), {})[int(me.group(2))] = v
            if me.group(4) == "weight":
                hf_mod = k[: -len(".weight")]
                spec_by_module.setdefault(
                    f"{me.group(1)}.switch_mlp.{me.group(3)}_proj",
                    _their_spec(qcfg, hf_mod))
            continue
        rk = re.sub(r"\.hc_(attn|ffn)_(base|fn|scale)$", r".\1_hc.hc_\2", rk)
        rk = rk.replace(".mlp.gate.e_score_correction_bias", ".mlp.e_score_correction_bias")
        if rk.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")):
            rk = rk[: -len(".weight")]
            v = v.reshape(v.shape[0], v.shape[-1])
        if rk.endswith("self_attn.o_norm.weight"):
            rk = rk[: -len(".weight")]
        out[rk] = v
        if rk.endswith((".weight",)) and (rk[: -len(".weight")] + ".scales") in (
                k2.replace("model.language_model.", "model.") for k2 in raw):
            pass
        if k.endswith(".weight") and k[: -len(".weight")] + ".scales" in raw:
            spec_by_module[rk[: -len(".weight")]] = _their_spec(qcfg, k[: -len(".weight")])
    for (base, proj, part), parts in experts.items():
        E = max(parts) + 1
        stacked = mx.stack([parts[i] for i in range(E)], axis=0)
        mx.eval(stacked)  # bounded graphs — a deferred all-at-once eval of
        mx.clear_cache()  # 378 stacks trips the Metal watchdog
        out[f"{base}.switch_mlp.{proj}_proj.{part}"] = stacked

    model = Glm5NextForCausalLM(args)
    import mlx.nn as nn

    # Derive (bits, gs) per module from THEIR shapes vs the model's true
    # in-dims (config-key conventions vary; shapes don't lie):
    #   bits = 32 * packed_cols / in_dim ; gs = in_dim / scales_cols
    quant_bases = {k[: -len(".scales")] for k in out if k.endswith(".scales")}
    from mlx.utils import tree_flatten
    fp_shapes = {k: v.shape for k, v in tree_flatten(model.parameters())}
    derived = {}
    for base in quant_bases:
        wk = base + ".weight" if (base + ".weight") in fp_shapes else base
        if wk not in fp_shapes:
            continue
        in_dim = fp_shapes[wk][-1]
        packed = out[base + ".weight" if (base + ".weight") in out else base].shape[-1]
        groups = out[base + ".scales"].shape[-1]
        bits = 32 * packed // in_dim
        gs = in_dim // groups
        assert bits in (2, 3, 4, 5, 6, 8) and gs in (32, 64, 128), \
            f"{base}: derived bits={bits} gs={gs} (in={in_dim})"
        derived[base if not base.endswith(".weight") else base[:-7]] = \
            {"group_size": gs, "bits": bits}

    def class_predicate(path, module):
        spec = derived.get(path)
        return spec if spec else False

    nn.quantize(model, class_predicate=class_predicate)
    model.load_weights(list(out.items()), strict=True)
    flat = tree_flatten(model.parameters())
    for i in range(0, len(flat), 200):
        mx.eval([v for _, v in flat[i:i + 200]])
    return model
