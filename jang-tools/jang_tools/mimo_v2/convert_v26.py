"""MiMo-V2.6 (Flash-RL) -> JANG mixed-precision bundle.

Plan-driven: every routed-expert UNIT (one layer's gate/up/down projection
across all experts, i.e. one stacked ``switch_mlp.<proj>`` module) gets its own
spec, because a stacked module has exactly one quantization mode:

    {"mode": "affine", "bits": 2, "group_size": 128}
    {"mode": "mxfp4"}          # native: the source MXFP4 bytes, copied bit-exact

Everything else is fixed:
    text/MTP linears (qkv, o_proj, dense MLP, eh_proj), embed, lm_head
        -> affine 8-bit, group 64, bf16 scales/biases (imatrix fit when available)
    router weight + e_score_correction_bias -> fp32
    norms, attention sink bias             -> bf16
    visual.*, audio_encoder.*, speech_embeddings.* -> bf16 verbatim
    dflash/ (DFlash drafter), audio_tokenizer/, assets/ -> copied verbatim as subfolders

Source decoding (all measured 2026-09-22, see weight_loader / mxfp4_codec):
    fused qkv = 4 TP-rank blocks, FP8-quantized per rank -> rank-blocked dequant
    + de-interleave to [q|k|v]; experts = MXFP4, low nibble first, e8m0 bias 127.

Bundle layout rules (AGENTS.md): root holds ONLY model-*.safetensors + ONE
model.safetensors.index.json (+ json/tokenizer files); every shard is rewritten
aligned and verified (0 unaligned tensors or the build fails).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from ..format.aligned_safetensors import rewrite_aligned_safetensors, verify_safetensors_alignment
from .mxfp4_codec import mxfp4_raw_to_mlx
from .v26_quant import quantize_affine
from .v26_gptq_provenance import describe_run
from .weight_loader import MiMoShardIndex

PLAN_SCHEMA = "mimo-v26-jang-plan-v1"
PROJS = ("gate_proj", "up_proj", "down_proj")
NONEXPERT = {"bits": 8, "group_size": 64}
EXPERT_CHUNK = 32
_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def _to_mx(t: torch.Tensor, dtype) -> mx.array:
    if t.dtype == torch.bfloat16:
        return mx.array(t.view(torch.int16).numpy()).view(mx.bfloat16).astype(dtype)
    return mx.array(t.float().numpy()).astype(dtype)


# ----------------------------------------------------------------------------- plan
def load_plan(path: Path, n_layers: int, moe_layers: list[int]) -> dict:
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema {plan.get('schema')!r} != {PLAN_SCHEMA}")
    default = plan["expert_default"]
    units = {}
    for L in moe_layers:
        over = plan.get("experts", {}).get(str(L), {})
        for p in PROJS:
            spec = dict(over.get(p, default))
            if spec["mode"] == "affine":
                if spec["bits"] not in (2, 3, 4, 5, 6, 8) or spec["group_size"] not in (32, 64, 128):
                    raise ValueError(f"bad affine spec L{L} {p}: {spec}")
            elif spec["mode"] != "mxfp4":
                raise ValueError(f"bad mode L{L} {p}: {spec}")
            units[(L, p)] = spec
    extra = {int(k) for k in plan.get("experts", {})} - set(moe_layers)
    if extra:
        raise ValueError(f"plan names non-MoE layers {sorted(extra)}")
    plan["_units"] = units
    return plan


# ------------------------------------------------------------------------ calibration
class Calib:
    """Calibration inputs, shared with v26_sweep so the build equals what was measured.

    plan["stats"]      : imatrix.safetensors from v26_calibrate
    plan["awq_alpha"]  : {"L": alpha} per MoE layer (0 / absent = no AWQ fold)
    """

    def __init__(self, plan: dict):
        from .v26_sweep import awq_scale, expert_importance
        self._awq_scale, self._expert_imp = awq_scale, expert_importance
        self.stats = mx.load(plan["stats"]) if plan.get("stats") else None
        self.alpha = {int(k): float(v) for k, v in (plan.get("awq_alpha") or {}).items()}

    def awq_for(self, L: int):
        if self.stats is None or not self.alpha.get(L):
            return None
        return self._awq_scale(self.stats, L, self.alpha[L])

    def expert_importance(self, L: int, proj: str, s):
        if self.stats is None:
            return None
        return self._expert_imp(self.stats, L, proj, s)

    def imatrix(self, weight_name: str):
        """Map a non-expert weight to its captured input statistic (None = RTN)."""
        if self.stats is None:
            return None
        m = re.match(r"^model\.layers\.(\d+)\.(self_attn\.(qkv|o)_proj|mlp\.(gate|up|down)_proj)\.weight$", weight_name)
        if m:
            L = m.group(1)
            if m.group(3) == "qkv":
                key = f"{L}.attn_in"
            elif m.group(3) == "o":
                key = f"{L}.o_in"
            elif m.group(4) in ("gate", "up"):
                key = "0.dense_in"
            else:
                key = "0.dense_down_in"
            return self.stats.get(key)
        if weight_name == "lm_head.weight":
            return self.stats.get("final_norm")
        return None  # embed_tokens (lookup), MTP (never run in calibration)


# ---------------------------------------------------------------------------- writer
class ShardWriter:
    def __init__(self, dst: Path, max_bytes: int):
        self.dst, self.max_bytes = dst, max_bytes
        self.buf, self.nbytes, self.idx = {}, 0, 0
        self.tensor_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.files: list[str] = []

    def add(self, name: str, arr: mx.array):
        if name in self.weight_map or name in self.buf:
            raise ValueError(f"duplicate tensor {name}")
        mx.eval(arr)
        self.buf[name] = arr
        self.nbytes += arr.nbytes
        self.tensor_bytes += arr.nbytes
        if self.nbytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.idx += 1
        fname = f"model-{self.idx:05d}-of-XXXXX.safetensors"
        path = self.dst / fname
        mx.save_safetensors(str(path), self.buf, metadata={"format": "mlx"})
        rewrite_aligned_safetensors(path)
        n, bad = verify_safetensors_alignment(path)
        if bad:
            raise RuntimeError(f"{fname}: {bad}/{n} tensors unaligned after rewrite")
        for k in self.buf:
            self.weight_map[k] = fname
        self.files.append(fname)
        print(f"  shard {self.idx}: {len(self.buf)} tensors {self.nbytes/2**30:.2f} GiB (aligned {n}/{n})", flush=True)
        self.buf, self.nbytes = {}, 0
        mx.clear_cache()

    def finalize(self) -> int:
        self.flush()
        total = len(self.files)
        final_map = {}
        size = self.tensor_bytes
        for i, fname in enumerate(self.files, 1):
            new = fname.replace("XXXXX", f"{total:05d}")
            (self.dst / fname).rename(self.dst / new)
            for k, v in self.weight_map.items():
                if v == fname:
                    final_map[k] = new
        (self.dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size}, "weight_map": dict(sorted(final_map.items()))}, indent=2))
        return size


# ---------------------------------------------------------------------------- units
def quantize_linear(w: mx.array, imp: mx.array | None, spec=NONEXPERT):
    q, s, b = quantize_affine(w, bits=spec["bits"], group_size=spec["group_size"], importance=imp)
    return q, s, b


def expert_unit(idx: MiMoShardIndex, L: int, proj: str, spec: dict, n_exp: int,
                awq: mx.array | None, imp: mx.array | None):
    """Return tensors for model.layers.L.mlp.switch_mlp.<proj> (stacked over experts)."""
    names = [f"model.layers.{L}.mlp.experts.{e}.{proj}.weight" for e in range(n_exp)]
    if spec["mode"] == "mxfp4":
        if awq is not None and proj != "down_proj":
            raise ValueError(f"L{L} {proj}: native MXFP4 cannot carry an AWQ fold; plan must disable AWQ for this layer")
        ws, ss = zip(*(mxfp4_raw_to_mlx(*idx.read_mxfp4_raw(n)) for n in names))
        return {"weight": mx.array(np.stack(ws)), "scales": mx.array(np.stack(ss))}
    outs = {"weight": [], "scales": [], "biases": []}
    for c0 in range(0, n_exp, EXPERT_CHUNK):
        chunk = names[c0:c0 + EXPERT_CHUNK]
        raw = [mxfp4_raw_to_mlx(*idx.read_mxfp4_raw(n)) for n in chunk]
        w = mx.dequantize(mx.array(np.stack([r[0] for r in raw])), mx.array(np.stack([r[1] for r in raw])),
                          group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
        if awq is not None and proj != "down_proj":
            w = w * awq.astype(mx.float32)
        cimp = None if imp is None else imp[c0:c0 + EXPERT_CHUNK]
        q, s, b = quantize_affine(w, bits=spec["bits"], group_size=spec["group_size"], importance=cimp)
        mx.eval(q, s, b)
        outs["weight"].append(q); outs["scales"].append(s); outs["biases"].append(b)
        del w
    return {k: mx.concatenate(v, axis=0) for k, v in outs.items()}


# ---------------------------------------------------------------------------- metadata
def write_metadata(src: Path, dst: Path, plan: dict, overrides: dict, size: int, counts: dict, src_sha: str):
    cfg = json.loads((src / "config.json").read_text())
    gen = json.loads((src / "generation_config.json").read_text())
    eos = list(gen["eos_token_id"])
    cfg.pop("quantization_config", None)
    cfg["eos_token_id"] = eos
    quant = {"group_size": NONEXPERT["group_size"], "bits": NONEXPERT["bits"], "mode": "affine"}
    quant.update(overrides)
    cfg["quantization"] = quant
    cfg["jang_profile"] = plan["profile"]
    media = bool(plan.get("media_runtime"))  # flip only after live media proof
    mods = ["text", "vision", "video", "audio"] if media else ["text"]
    cfg["capabilities"] = {
        "family": "mimo_v2",
        "modalities": mods,
        "preserved_modalities": ["vision", "video", "audio"],
        "unwired_modalities": [] if media else ["vision", "video", "audio"],
        "multimodal_status": "mimo_v2_multimodal_runtime" if media else "weights_preserved_text_runtime",
        "reasoning": {"supported": True, "default": True, "parser": "think_xml"},
        "tools": {"supported": True, "parser": "xml_function"},
        "cache_type": "kv",
    }
    cfg["runtime"] = {
        "multimodal_mode": "mimo_v2_multimodal_runtime" if media else "weights_preserved_text_runtime",
        "mtp_mode": "preserved_disabled", "bundle_has_mtp": True,
        "dflash": "preserved_disabled (dflash/ sidecar)",
        "quantization_profile": plan["profile"],
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    # generation_config: vendor recommends T=1.0/top_p=0.95 (README) but ships do_sample=false
    # (greedy) and max_new_tokens=2048 (truncates reasoning). Stamp the card's sampling.
    g2 = {"bos_token_id": gen.get("bos_token_id"), "eos_token_id": eos, "do_sample": True,
          "temperature": 1.0, "top_p": 0.95}
    (dst / "generation_config.json").write_text(json.dumps(g2, indent=2))

    units = plan["_units"]
    unit_modes = {}
    for (L, p), s in units.items():
        key = "mxfp4" if s["mode"] == "mxfp4" else f"affine{s['bits']}/g{s['group_size']}"
        unit_modes[key] = unit_modes.get(key, 0) + 1
    jc = {
        "version": 2,
        "weight_format": "mixed_affine_mxfp4",
        "profile": plan["profile"],
        "source_model": {"name": "MiMo-V2.6-Flash-RL", "architecture": "mimo_v2",
                         "repo": "XiaomiMiMo/MiMo-V2.6-Flash-RL", "revision": src_sha},
        "has_vision": media, "has_audio": media, "has_video": media,
        "preserved_modalities": ["vision", "video", "audio"],
        "tool_calling": {
            "dialect": "xml_function",
            "tool_call_start": "<tool_call>", "tool_call_end": "</tool_call>",
            "format": "<tool_call><function=NAME><parameter=ARG>VALUE</parameter></function></tool_call>",
            "tools_block": "first system turn, before any user system prompt (tool-set change = full re-prefill)",
        },
        "chat": {
            "sampling_defaults": {"temperature": 1.0, "top_p": 0.95, "do_sample": True,
                                  "source": "XiaomiMiMo/MiMo-V2.6-Flash-RL README 'Recommended sampling'"},
            "eos_token_ids": eos,
            "thinking": {
                "supported": True, "template_flag": "enable_thinking", "default": True,
                "on": "prompt ends '<|im_start|>assistant\\n'; model emits <think>...</think>",
                "off": "template appends '<think></think>'",
                "history": "every past assistant turn is rendered '<think>'+reasoning_content+'</think>'+content; "
                           "runtimes must round-trip reasoning_content or the prompt diverges from what was generated",
            },
        },
        "quantization": {
            "method": "mixed", "modes": sorted({"affine"} | ({"mxfp4"} if "mxfp4" in unit_modes else set())),
            "scale_dtype": "bfloat16", "norm_convention": "rmsnorm_no_plus_one",
            "routed_expert_units": unit_modes,
            "native_mxfp4_units": sorted(f"L{L}.{p}" for (L, p), s in units.items() if s["mode"] == "mxfp4"),
            "nonexpert": dict(NONEXPERT, note="qkv/o_proj/dense/MTP/eh_proj/embed/lm_head"),
            "router": "fp32", "vision_audio": "bf16 verbatim",
            "awq": {"applied": any(v for v in (plan.get("awq_alpha") or {}).values()),
                    "statistic": "max|x| of post_attention_layernorm output, ^alpha, geomean-normalized, clip [0.5, 2]",
                    "fold_sites": ["post_attention_layernorm -> router (fp32) + expert gate/up"],
                    "alpha_per_layer": plan.get("awq_alpha") or {}},
            "gptq": describe_run(plan.get("gptq_dir")),
            "imatrix": {"applied": bool(plan.get("stats")),
                        "statistic": "E[x_c^2] per input channel; per expert for routed units (router-selected tokens)",
                        "fit": "alternating least squares affine grid, bf16 scale/bias storage"},
            "source_decode": {"qkv": "4 TP-rank blocks, per-rank FP8 block scales, de-interleaved to [q|k|v]",
                              "experts": "OCP MXFP4, low nibble first, e8m0 bias 127"},
            **counts,
        },
        "runtime": {"total_weight_bytes": size, "total_weight_gib": round(size / 2**30, 3),
                    "mtp": "preserved_disabled (3 layers, in model shards)",
                    "dflash": "preserved_disabled (bf16 sidecar in dflash/)"},
        "capabilities": {
            "family": "mimo_v2", "reasoning_parser": "think_xml", "tool_parser": "xml_function",
            "think_in_template": False, "supports_tools": True, "supports_thinking": True,
            "modality": "omni" if media else "text",
            "modalities": {"text": True, "vision": media, "audio": media, "video": media},
            "multimodal_status": "mimo_v2_multimodal_runtime" if media else "weights_preserved_text_runtime",
            "cache_type": "kv", "attention": "hybrid full + SWA(128) with sink bias on SWA",
        },
    }
    (dst / "jang_config.json").write_text(json.dumps(jc, indent=2))
    if plan.get("gptq_dir") and (Path(plan["gptq_dir"]) / "sequential_run.json").exists():
        provenance = dst / "quantization"
        provenance.mkdir(exist_ok=True)
        for name in ("gptq_run.json", "sequential_run.json", "gptq_report.json", "hessian_capture_report.json"):
            shutil.copyfile(Path(plan["gptq_dir"]) / name, provenance / name)


def copy_aux(src: Path, dst: Path):
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json", "configuration_mimo_v2.py", "modeling_mimo_v2.py", "LICENSE", "LICENSE.txt"):
        if (src / fn).exists():
            shutil.copy2(src / fn, dst / fn)
    # Use one authoritative native template in both loader conventions.
    # The source copies differ by a blank line; do not propagate divergence.
    tc = dst / "tokenizer_config.json"
    jt = dst / "chat_template.jinja"
    if tc.exists() and jt.exists():
        config = json.loads(tc.read_text())
        config["chat_template"] = jt.read_text()
        tc.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    for sub in ("dflash", "audio_tokenizer", "assets"):
        if (src / sub).is_dir():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)


# ---------------------------------------------------------------------------- main
def convert(src: Path, dst: Path, plan_path: Path, max_shard_gib: float = 4.0):
    t0 = time.time()
    idx = MiMoShardIndex(src)
    cfg = idx.config
    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg["n_routed_experts"]
    moe_layers = [L for L in range(n_layers) if cfg["moe_layer_freq"][L]]
    plan = load_plan(plan_path, n_layers, moe_layers)
    if plan.get("gptq_dir"):
        from .v26_gptq_provenance import validate_conversion
        validate_conversion(plan["gptq_dir"], plan, src)
    calib = Calib(plan)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} is not empty; never overwrite a bundle in place")
    dst.mkdir(parents=True, exist_ok=True)
    wr = ShardWriter(dst, int(max_shard_gib * 2**30))
    overrides, counts = {}, {"quantized_linear": 0, "expert_units": 0, "bf16": 0, "fp32": 0}

    def lin(name: str, w: mx.array, imp_key: str | None = None):
        base = name[: -len(".weight")]
        imp = calib.imatrix(imp_key or name)
        q, s, b = quantize_linear(w.astype(mx.float32), imp)
        wr.add(f"{base}.weight", q); wr.add(f"{base}.scales", s); wr.add(f"{base}.biases", b)
        counts["quantized_linear"] += 1

    expert_seen = set()
    for i, name in enumerate(idx.weight_keys):
        m = _EXPERT_RE.match(name)
        if m:
            expert_seen.add(name)
            continue  # written per unit below
        if name.startswith("model.layers.") and ".mlp.gate." in name:
            L = int(name.split(".")[2])
            t = _to_mx(idx.read_passthrough(name, out_dtype=torch.float32), mx.float32)
            a = calib.awq_for(L)
            if a is not None and name.endswith(".mlp.gate.weight"):
                t = t * a.astype(mx.float32)  # router consumes the same normed input as gate/up
            wr.add(name, t); counts["fp32"] += 1
            continue
        if name.startswith(("visual.", "audio_encoder.", "speech_embeddings.")) or \
           name.endswith(("norm.weight", "attention_sink_bias")) or name.endswith(".bias"):
            t = _to_mx(idx.read_tensor(name, out_dtype=torch.float32), mx.float32)
            if name.endswith("post_attention_layernorm.weight") and name.startswith("model.layers."):
                a = calib.awq_for(int(name.split(".")[2]))
                if a is not None:
                    t = t / a.astype(mx.float32)
            wr.add(name, t.astype(mx.bfloat16)); counts["bf16"] += 1
            continue
        if name.endswith(".weight") and (name.startswith(("model.layers.", "model.mtp.")) or
                                          name in ("model.embed_tokens.weight", "lm_head.weight")):
            t = _to_mx(idx.read_tensor(name, out_dtype=torch.float32), mx.float32)
            if t.ndim != 2:
                raise ValueError(f"unexpected non-2D text weight {name} {t.shape}")
            lin(name, t)
            continue
        raise ValueError(f"unclassified source tensor {name}")

    if len(expert_seen) != len(moe_layers) * n_exp * 3:
        raise RuntimeError(f"expert tensors {len(expert_seen)} != {len(moe_layers)}*{n_exp}*3")
    for L in moe_layers:
        a = calib.awq_for(L)
        natives = [p for p in ("gate_proj", "up_proj") if plan["_units"][(L, p)]["mode"] == "mxfp4"]
        if a is not None and natives:
            raise ValueError(f"L{L}: AWQ alpha set but {natives} native MXFP4 (cannot absorb the fold)")
        for p in PROJS:
            spec = plan["_units"][(L, p)]
            gdir = plan.get("gptq_dir")
            gfile = Path(gdir) / f"L{L}.{p}.safetensors" if gdir else None
            if spec["mode"] == "affine" and gfile is not None:
                if not gfile.exists():
                    raise FileNotFoundError(f"plan uses GPTQ codes but {gfile} is missing")
                t, meta = mx.load(str(gfile), return_metadata=True)
                want = (str(spec["bits"]), str(spec["group_size"]), str(a is not None))
                got = (meta.get("bits"), meta.get("group_size"), meta.get("awq"))
                if want != got:
                    raise ValueError(f"GPTQ codes {gfile} were solved for {got}, plan wants {want}")
                counts["gptq_units"] = counts.get("gptq_units", 0) + 1
            else:
                imp = calib.expert_importance(L, p, a) if spec["mode"] == "affine" else None
                t = expert_unit(idx, L, p, spec, n_exp, a, imp)
            base = f"model.layers.{L}.mlp.switch_mlp.{p}"
            for k, v in t.items():
                wr.add(f"{base}.{k}", v)
            if spec["mode"] == "mxfp4":
                overrides[base] = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
            else:
                overrides[base] = {"group_size": spec["group_size"], "bits": spec["bits"], "mode": "affine"}
            counts["expert_units"] += 1
        print(f"[L{L}] units done ({time.time()-t0:.0f}s)", flush=True)
    size = wr.finalize()
    src_sha = plan.get("source_revision", "unknown")
    write_metadata(src, dst, plan, overrides, size, counts, src_sha)
    copy_aux(src, dst)
    print(f"[convert] DONE {size/2**30:.3f} GiB in {len(wr.files)} shards, {time.time()-t0:.0f}s; {counts}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--max-shard-gib", type=float, default=4.0)
    a = ap.parse_args(argv)
    convert(a.src.expanduser(), a.dst.expanduser(), a.plan.expanduser(), a.max_shard_gib)


if __name__ == "__main__":
    main()
