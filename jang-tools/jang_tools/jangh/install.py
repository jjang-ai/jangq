"""vMLX runtime hook: install JANGTQ v2 routed experts into a constructed model BEFORE nn.quantize.

Contract (fail closed):
  * config["jangtq"]["version"] == 2 and per-module entries {"mode": "jangtq2", "bits": b} for every
    <layer>.mlp.switch_mlp.{gate,up,down}_proj of every routed-MoE layer.
  * gate_proj and up_proj may use different bits (mixed-bit fused kernels, 2026-09-27); their rotation must match.
  * Every SwitchGLU that has a jangtq2 entry is replaced; any jangtq2 entry that matches no module is an error,
    and any MoE SwitchGLU left without an entry in a jangtq2 bundle is an error (no silent mixed stacks).
"""
from __future__ import annotations

import logging

import mlx.nn as nn

from .switch import TQSwitchGLU

logger = logging.getLogger(__name__)


def _cfg_key(module_path: str) -> str:
    """Runtime module path -> on-disk config key (the bundle uses model.layers.N... naming)."""
    for pre in ("language_model.model.", "model.language_model.", "language_model."):
        if module_path.startswith(pre):
            return "model." + module_path[len(pre):]
    return module_path


def is_jangtq2(config: dict) -> bool:
    return isinstance(config.get("jangtq"), dict) and int(config["jangtq"].get("version", 0)) == 2


def install_jangtq2(model: nn.Module, config: dict) -> int:
    from mlx_lm.models.switch_layers import SwitchGLU
    q = config.get("quantization") or {}
    want = {k[: -len(".gate_proj")] for k, v in q.items()
            if isinstance(v, dict) and v.get("mode") == "jangtq2" and k.endswith(".switch_mlp.gate_proj")}
    tcfg = config.get("text_config", config)
    limit = float(tcfg.get("swiglu_limit", 0.0) or 0.0)
    replaced, seen = 0, set()
    for path, mod in list(model.named_modules()):
        sw = getattr(mod, "switch_mlp", None)
        if not isinstance(sw, SwitchGLU):
            continue
        key = _cfg_key(f"{path}.switch_mlp")
        if key not in want:
            continue
        gu, up, dn = q[f"{key}.gate_proj"], q[f"{key}.up_proj"], q[f"{key}.down_proj"]
        if not (gu.get("mode") == up.get("mode") == dn.get("mode") == "jangtq2"):
            raise ValueError(f"jangtq2: inconsistent entries for {key}: {gu} {up} {dn}")
        E, I, D = sw.gate_proj.weight.shape[0], sw.gate_proj.weight.shape[1], sw.down_proj.weight.shape[1]
        if gu.get("rotation", "none") != up.get("rotation", "none"):
            raise ValueError(f"jangtq2: gate/up rotation differs for {key}")
        mod.switch_mlp = TQSwitchGLU(D, I, E, int(gu["bits"]), int(dn["bits"]), limit,
                                     rotation_gate_up=str(gu.get("rotation", "none")),
                                     rotation_down=str(dn.get("rotation", "none")), bits_up=int(up["bits"]))
        mod.switch_mlp.is_jangtq2 = True
        seen.add(key); replaced += 1
    missing = want - seen
    if missing:
        raise ValueError(f"jangtq2: {len(missing)} config entries matched no SwitchGLU module, e.g. {sorted(missing)[:3]}")
    # a jangtq2 bundle must not leave any routed SwitchGLU on the stock path
    left = [p for p, m in model.named_modules() if isinstance(getattr(m, "switch_mlp", None), SwitchGLU)
            and not p.endswith(".mtp") and "mtp" not in p]
    if left:
        raise ValueError(f"jangtq2: {len(left)} routed SwitchGLU modules have no jangtq2 entry, e.g. {left[:3]}")
    logger.info("JANGTQ v2: installed %d TQSwitchGLU modules (codebook %s)", replaced, config["jangtq"].get("codebook_family"))
    return replaced


def alias_runtime_quant_keys(quant: dict, prefix: str = "language_model.") -> int:
    """mlx_vlm's quantize predicate matches per-module entries by RUNTIME path ('language_model.model.layers...'),
    while JANG bundles key them by checkpoint path ('model.layers...', 'lm_head'). Without aliases every module falls
    back to the global default (affine 8/g64 here), which breaks the mxfp8 modules. Adds aliases IN PLACE (the dict
    object is the one mlx_vlm reads). Returns the number of aliases added."""
    n = 0
    for k in list(quant.keys()):
        v = quant[k]
        if not isinstance(v, dict) or v.get("mode") == "jangtq2":
            continue
        if k.startswith("model.") or k == "lm_head":
            alias = prefix + k
            if alias not in quant:
                quant[alias] = v
                n += 1
    return n
