"""Native MLX image encoder for Nemotron-3.5-Bolt-74B-A9B (C-RADIO v4-H tower + relu2 projector).

The tower body is the C-RADIO ViT-H/16 of ``jang_tools.nemotron_omni.radio`` (CPE patch generator, 10 cls/register
tokens, no final norm); on Bolt's own weights it matches NVIDIA's torch ``modeling_radio.RadioModel`` on a 576x800 image
(all 1,810 tokens, cls/register cos 1.0, patch cos >= 0.99995, rel-L2 0.35 % in bf16). Bolt-specific parts:

  * projector ``mlp1`` = RMSNorm(eps 1e-5, fp32 math) -> linear1 (5120 -> 20480) -> relu2 -> linear2 (-> 4096), no bias,
    after pixel shuffle (ps_version v2, ratio 0.5) of the patch features (the 10 leading tokens dropped);
  * images are prepared by the bundle's own processor (dynamic resolution, 1024..13312 patches, normalization done
    there, so the tower's input conditioner is not applied); ``encode`` returns the expanded token ids and the LLM-space
    rows for every ``<image>`` (img_context_token_id) position;
  * linear weights may be stored MLX-affine-quantized (``.scales`` / ``.biases`` present): those modules are quantized
    with the bundle's per-module bits before loading;
  * Nemotron-3.5-Super-VL (same tower family): when the LLM has MTP (``llm_config.num_nextn_predict_layers > 0``) the
    RADIO patch features get a final LayerNorm (``vision_projector.vision_final_layernorm.*``, eps = vision_config
    layer_norm_eps) before the pixel shuffle — Megatron's shared-config final LN, modeling_nemotron_h_omni.py
    ``_project``. There is NO config key for it: it is derived from the MTP setting, exactly like the HF code.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from jang_tools.nemotron_omni.radio import RADIOVisionModel, map_radio_weights


class _RMSNorm(nn.Module):
    def __init__(self, n: int, eps: float):
        super().__init__()
        self.weight = mx.ones((n,)); self.eps = eps

    def __call__(self, x):
        dt = x.dtype; x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (self.weight.astype(mx.float32) * x).astype(dt)


class BoltProjector(nn.Module):
    def __init__(self, vit_hidden: int, proj_hidden: int, llm_hidden: int, ratio: float = 0.5):
        super().__init__()
        self.ratio = ratio
        self.norm = _RMSNorm(vit_hidden * int(1 / ratio) ** 2, 1e-5)
        self.linear1 = nn.Linear(vit_hidden * int(1 / ratio) ** 2, proj_hidden, bias=False)
        self.linear2 = nn.Linear(proj_hidden, llm_hidden, bias=False)

    @staticmethod
    def pixel_shuffle(x, s):
        n, w, h, c = x.shape
        x = x.reshape(n, w, int(h * s), int(c / s)).transpose(0, 2, 1, 3)
        x = x.reshape(n, int(h * s), int(w * s), int(c / (s * s)))
        return x.transpose(0, 2, 1, 3)                                    # ps_version v2: swap back

    def __call__(self, feats, gh: int, gw: int):
        x = feats.reshape(feats.shape[0], gh, gw, -1)
        x = self.pixel_shuffle(x, self.ratio)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return self.linear2(mx.square(nn.relu(self.linear1(self.norm(x)))))


class BoltVision:
    def __init__(self, bundle: str | Path, dtype=mx.bfloat16, processor=None):
        self.bundle = Path(bundle)
        omni = json.loads((self.bundle / ("config_omni.json" if (self.bundle / "config_omni.json").exists() else "config.json")).read_text())
        vc = omni["vision_config"]
        self.has_final_ln = (omni.get("llm_config", {}).get("num_nextn_predict_layers") or 0) > 0
        assert omni.get("ps_version", "v2") == "v2"
        self.img_token = int(omni["img_context_token_id"])
        self.patch = int(omni["patch_size"])
        self.n_skip = int(vc["num_cls_tokens"]) + int(vc["num_registers"])
        self.tower = RADIOVisionModel(embed_dim=vc["hidden_size"], num_blocks=vc["num_hidden_layers"], num_heads=vc["num_attention_heads"],
                                      mlp_ratio=vc["mlp_ratio"], patch_size=self.patch, num_cls_tokens=self.n_skip,
                                      max_grid=vc["args"]["cpe_max_size"] // self.patch)
        llm_h = json.loads((self.bundle / "config.json").read_text()).get("hidden_size") or omni["llm_config"]["hidden_size"]
        self.proj = BoltProjector(omni["vit_hidden_size"], omni["projector_hidden_size"], llm_h, omni["downsample_ratio"])
        idx = json.loads((self.bundle / "model.safetensors.index.json").read_text())["weight_map"]
        W = {}
        pref = ("vision_model.", "mlp1.", "vision_projector.")
        for f in sorted({v for k, v in idx.items() if k.startswith(pref)}):
            W.update({k: v for k, v in mx.load(str(self.bundle / f)).items() if k.startswith(pref)})
        lnw = {k.rsplit(".", 1)[1]: W.pop(k) for k in [k for k in W if k.startswith("vision_projector.vision_final_layernorm.")]}
        unknown = sorted(k for k in W if k.startswith("vision_projector."))
        assert not unknown, f"vision projector tensors this port does not know: {unknown}"
        self.final_ln = None
        if self.has_final_ln:
            assert set(lnw) == {"weight", "bias"}, f"MTP model without vision_projector.vision_final_layernorm.{{weight,bias}} ({sorted(lnw)})"
            self.final_ln = nn.LayerNorm(omni["vit_hidden_size"], eps=float(vc["layer_norm_eps"]))
            self.final_ln.load_weights([("weight", lnw["weight"]), ("bias", lnw["bias"])], strict=True)
        else:
            assert not lnw, "vision final LayerNorm weights present but the model has no MTP (unexpected layout)"
        quant = json.loads((self.bundle / "config.json").read_text()).get("quantization", {})
        tw = map_radio_weights(W)                    # prefix rules: packed .weight / .scales / .biases map alike
        pw = {}
        for k, v in W.items():
            if k.startswith("mlp1."):
                n = k.replace("mlp1.0.", "norm.").replace("mlp1.1.", "linear1.").replace("mlp1.3.", "linear2.")
                pw[n] = v
        self._quantize_like(self.tower, tw, quant, "vision_model.radio_model.model.")
        self._quantize_like(self.proj, pw, quant, None)
        self.tower.load_weights(list(tw.items()), strict=False); self.proj.load_weights(list(pw.items()), strict=True)
        from mlx.utils import tree_flatten
        missing = set(k for k, _ in tree_flatten(self.tower.parameters())) - set(tw)
        assert not missing, f"Bolt vision: tower parameters not in the bundle: {sorted(missing)[:6]}"
        self.tower.set_dtype(dtype); self.proj.set_dtype(dtype)
        if self.final_ln is not None:
            self.final_ln.set_dtype(dtype); mx.eval(self.final_ln.parameters())
        mx.eval(self.tower.parameters(), self.proj.parameters())
        self.dtype = dtype
        self.processor = processor

    @staticmethod
    def _quantize_like(module, weights, quant, prefix):
        """Quantize exactly the Linear modules whose packed weights the bundle carries (per-module bits)."""
        targets = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}
        if not targets:
            return

        def pred(path, m):
            if path not in targets or not isinstance(m, nn.Linear):
                return False
            key = (prefix + path) if prefix else ("mlp1." + {"norm": "0", "linear1": "1", "linear2": "3"}.get(path, path))
            q = quant.get(key, {})
            return {"group_size": q.get("group_size", 64), "bits": q.get("bits", 8)}
        nn.quantize(module, class_predicate=pred)

    def load_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(str(self.bundle), trust_remote_code=True)
        return self.processor

    def embed_pixels(self, pixel_values: np.ndarray) -> mx.array:
        """(1, 3, H, W) normalized pixels -> (H/32 * W/32, hidden) LLM-space rows."""
        pv = mx.array(pixel_values).astype(self.dtype)
        _, _, H, Wd = pv.shape
        feats = self.tower(pv)[:, self.n_skip:]
        if self.final_ln is not None:                                  # per token: same as before cls stripping
            feats = self.final_ln(feats)
        out = self.proj(feats, H // self.patch, Wd // self.patch)
        return out.reshape(-1, out.shape[-1])

    def encode(self, text: str, images: list):
        """Rendered chat text with '<image>' markers + PIL images -> (input ids, image positions, (n, H) embeddings)."""
        proc = self.load_processor()
        enc = proc(text=[text], images=images, return_tensors="np")
        ids = [int(t) for t in np.asarray(enc["input_ids"])[0]]
        pvs = enc["pixel_values"]
        pvs = list(pvs) if isinstance(pvs, (list, tuple)) else [pvs]
        rows = [self.embed_pixels(np.asarray(p if np.asarray(p).ndim == 4 else np.asarray(p)[None]).astype(np.float32)) for p in pvs]
        emb = mx.concatenate(rows, axis=0) if len(rows) > 1 else rows[0]
        pos = [i for i, t in enumerate(ids) if t == self.img_token]
        assert len(pos) == emb.shape[0], f"{len(pos)} <image> rows vs {emb.shape[0]} projector rows"
        return ids, pos, emb
