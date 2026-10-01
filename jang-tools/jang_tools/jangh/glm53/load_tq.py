"""Load a GLM-5.3-Flash JANGTQ v2 bundle into the parity-proven jang_tools glm5_next reference model.
Text-only eval path (vision + DSA indexer + MTP not constructed, same as jang_tools.glm5_next.load.load_bundle).
Routed experts -> TQSwitchGLU (fused kernels); other modules quantized per the bundle's per-module config
(affine 8-bit g64 / mxfp8 g32). Fails closed on any missing or unexpected tensor (strict load)."""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


from jang_tools.glm5_next.modeling import Glm5Args, Glm5NextForCausalLM  # noqa: E402
from jang_tools.jangh.switch import TQSwitchGLU  # noqa: E402

FP32_SUFFIXES = ("A_log", "dt_bias", "e_score_correction_bias", "hc_base", "hc_scale")


def _tq_moe_call(self, x):
    """MoEBlock.__call__ using the fused weighted routed path (identical routing math to the reference)."""
    logits = x.astype(mx.float32) @ self.gate.weight.astype(mx.float32).T
    scores = mx.sigmoid(logits)
    choice = scores + self.e_score_correction_bias.astype(mx.float32)
    idx = mx.argpartition(-choice, kth=self.k - 1, axis=-1)[..., : self.k]
    w = mx.take_along_axis(scores, idx, axis=-1)
    if self.norm_topk:
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
    w = w * self.scaling
    routed = self.switch_mlp.routed(x, idx, w)
    return routed.astype(x.dtype) + self.shared_experts(x)


def load_tq_bundle(bundle_dir: str, verbose: bool = True):
    t0 = time.time()
    d = Path(bundle_dir).expanduser()
    cfg = json.loads((d / "config.json").read_text())
    assert cfg.get("jangtq", {}).get("version") == 2, "not a JANGTQ v2 bundle"
    qcfg = cfg["quantization"]
    args = Glm5Args.from_config(cfg)
    model = Glm5NextForCausalLM(args)
    moe_cls = None
    for i, layer in enumerate(model.model.layers):
        name = f"model.layers.{i}.mlp.switch_mlp"
        if hasattr(layer.mlp, "switch_mlp"):
            gu, dn = qcfg[f"{name}.gate_proj"], qcfg[f"{name}.down_proj"]
            assert gu["mode"] == dn["mode"] == "jangtq2" and qcfg[f"{name}.up_proj"]["bits"] == gu["bits"]
            layer.mlp.switch_mlp = TQSwitchGLU(args.hidden_size, args.moe_intermediate_size, args.n_routed_experts,
                                               gu["bits"], dn["bits"], args.swiglu_limit,
                                               rotation_gate_up=gu.get("rotation", "none"),
                                               rotation_down=dn.get("rotation", "none"))
            moe_cls = type(layer.mlp)
    moe_cls.__call__ = _tq_moe_call

    def pred(path, module):
        spec = qcfg.get(path)
        if not isinstance(spec, dict) or spec.get("mode") == "jangtq2":
            return False
        return {"group_size": spec["group_size"], "bits": spec["bits"], "mode": spec.get("mode", "affine")}
    nn.quantize(model, class_predicate=pred)
    weights = {}
    for f in sorted(glob.glob(str(d / "model-*.safetensors"))):
        weights.update(mx.load(f))
    L = args.num_hidden_layers
    weights = {k: (v.astype(mx.float32) if any(k.endswith(s) for s in FP32_SUFFIXES) else v) for k, v in weights.items()
               if not k.startswith("visual.") and ".self_attn.indexer." not in k and not re.match(rf"model\.layers\.{L}\.", k)}
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    if verbose:
        print(f"loaded {d.name}: {len(weights)} tensors in {time.time()-t0:.1f}s", flush=True)
    return model
