"""dots3-note fp8 source -> JANG v2 (MLX-native affine/mxfp4) bundle converter.

Streaming, plan-driven, bounded-memory. Adapts the DSV4 0731 converter
contract to dots3 tensor names:

- routed experts stacked per (layer, proj) into
  `model.layers.{i}.mlp.switch_mlp.{proj}.weight` (+ .scales / .biases),
  one quantized unit per (layer, proj) — the gather_qmm ABI unit;
- every other 2-D Linear quantized per the plan (mode affine or mxfp4);
- norms / routers / e_score biases / convs / LayerNorm biases pass through
  in source dtype;
- MTP (layer 46 + model.mtp.embed_tokens) preserved into dedicated shards;
  mtp embed is DEDUPED against model.embed_tokens when byte-identical;
- per-module quantization overrides written into config.json/quantization
  (MLX convention) so stock mlx loaders can map every tensor;
- lazy-mmap self-clobber safe: fresh output dir, mx.eval before save,
  temp file + atomic rename per shard.

    python -m jang_tools.dots3.convert_dots3_jang <src> <dst> --plan plan.json
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import Dots3Config
from .folds import Folds
from .fp8 import ShardIndex

SHARD_BYTES = 4 * 1024**3
PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- classify
def classify(name: str) -> str:
    """Return the plan class for a source tensor name."""
    if name.endswith("_scale_inv"):
        return "skip"
    if name in ("model.embed_tokens.weight", "lm_head.weight",
                "model.mtp.embed_tokens.weight"):
        return "bookend"
    if ".mlp.experts." in name and name.startswith("model."):
        return "routed"
    if ".mlp.shared_experts." in name:
        return "shared_expert"
    if name.startswith("model.layers.46."):
        # MTP block: eh_proj + attention + dense mlp all live here
        if any(k in name for k in ("layernorm", "norm")):
            return "passthrough"
        if name.endswith(".weight") and ".mlp.gate." not in name:
            return "mtp_linear"
        return "passthrough"
    if name.startswith("model.layers.") and ".self_attn.indexer." in name:
        if ".k_norm." in name:
            return "passthrough"
        return "attention"
    if name.startswith("model.layers.") and ".self_attn." in name:
        if "layernorm" in name:
            return "passthrough"
        return "attention"
    if ".mlp.gate.weight" in name or "e_score_correction_bias" in name:
        return "passthrough"
    if name.startswith("model.layers.") and ".mlp." in name and name.endswith(".weight"):
        return "dense_mlp"            # layer 0 dense FFN
    if name.startswith(("model.norm", "model.layers.")):
        return "passthrough"
    if name.startswith("vision_encoder."):
        if ".mlp.experts." in name:
            return "vision_expert"
        if any(k in name for k in ("norm", "gate_weight", "router_bias",
                                   "patch_embed.proj", "ln_q")):
            return "passthrough"
        if name.endswith((".weight",)) and ".bias" not in name:
            return "vision_linear"
        return "passthrough"
    if name.startswith("audio_encoder."):
        if any(k in name for k in ("conv2d", "layer_norm", "norm.")):
            return "passthrough"
        if name.endswith(".weight") and "proj.0." not in name:
            return "audio_linear"
        return "passthrough"
    return "passthrough"


def routed_key(name: str) -> tuple[int, str] | None:
    # model.layers.{i}.mlp.experts.{e}.{proj}.weight — BACKBONE only
    # (vision_encoder.blocks.*.mlp.experts.* must NOT match).
    if not name.startswith("model.layers."):
        return None
    parts = name.split(".")
    if len(parts) >= 7 and parts[3] == "mlp" and parts[4] == "experts":
        return int(parts[2]), parts[6]
    return None


# ---------------------------------------------------------------- plan
class Plan:
    def __init__(self, path: Path):
        p = json.loads(path.read_text())
        self.defaults: dict = p["defaults"]
        self.routed_overrides: dict = p.get("routed_overrides", {})
        self.raw = p

    def for_class(self, cls: str) -> dict | None:
        spec = self.defaults.get(cls)
        return dict(spec) if spec else None

    def for_routed(self, layer: int, proj: str) -> dict:
        spec = dict(self.defaults["routed"])
        ov = self.routed_overrides.get(f"{layer}:{proj}")
        if ov:
            spec.update(ov)
        return spec


def compatible_group(in_dim: int, gs: int) -> int | None:
    for g in (gs, 64, 32):
        if g <= gs and in_dim % g == 0:
            return g
    return None


# ---------------------------------------------------------------- writer
class ShardWriter:
    def __init__(self, dst: Path, tag: str = "model"):
        self.dst = dst
        self.tag = tag
        self.buf: dict[str, mx.array] = {}
        self.bytes = 0
        self.n = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, name: str, arr: mx.array):
        mx.eval(arr)
        self.buf[name] = arr
        nbytes = arr.size * arr.dtype.size
        self.bytes += nbytes
        self.total_bytes += nbytes
        if self.bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.n += 1
        fname = f"{self.tag}-{self.n:05d}.safetensors"
        # NB: mx.save_safetensors APPENDS .safetensors when missing — the
        # temp name must already end with it.
        tmp = self.dst / f"{self.tag}-{self.n:05d}.tmp.safetensors"
        mx.save_safetensors(str(tmp), self.buf)
        tmp.rename(self.dst / fname)
        for k in self.buf:
            self.weight_map[k] = fname
        print(f"    wrote {fname} ({self.bytes/1e9:.2f} GB, {len(self.buf)} tensors)",
              flush=True)
        self.buf.clear()
        self.bytes = 0
        gc.collect()
        mx.clear_cache()

    def finalize(self, extra_maps: dict[str, str] | None = None):
        self.flush()
        wm = dict(self.weight_map)
        if extra_maps:
            wm.update(extra_maps)
        index = {"metadata": {"total_size": self.total_bytes},
                 "weight_map": wm}
        (self.dst / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=1, sort_keys=True))


# ---------------------------------------------------------------- quantize
def quantize_tensor(w: np.ndarray, spec: dict, name: str, audit: dict,
                    force_dtype=None) -> dict[str, mx.array]:
    """Quantize a (out, in) matrix per spec -> {suffix: array} pieces."""
    bits = spec["bits"]
    mode = spec.get("mode", "affine")
    gs = compatible_group(w.shape[-1], spec["group_size"])
    if gs is None or bits >= 16:
        audit[name] = {"passthrough": True, "reason": "no compatible group"}
        return {"": mx.array(w).astype(force_dtype or mx.float16)}
    if mode == "mxfp4":
        gs = 32
        if w.shape[-1] % 32:
            audit[name] = {"passthrough": True, "reason": "mxfp4 needs /32"}
            return {"": mx.array(w).astype(mx.float16)}
    wm = mx.array(w)
    out = mx.quantize(wm, group_size=gs, bits=bits, mode=mode)
    audit[name] = {"bits": bits, "group_size": gs, "mode": mode}
    if mode == "affine":
        wq, scales, biases = out
        # f16 sidecars: storage convention (DSV4 F16-corrected precedent) —
        # halves sidecar bytes and defines the grid GPTQ codes live on.
        return {"": wq, ".scales__": scales.astype(mx.float16),
                ".biases__": biases.astype(mx.float16)}
    wq, scales = out
    return {"": wq, ".scales__": scales}


def emit_quantized(writer: ShardWriter, base: str, pieces: dict, audit: dict,
                   overrides: dict, name_for_audit: str):
    """base like 'model.layers.3.mlp.switch_mlp.gate_proj.weight'."""
    a = audit[name_for_audit]
    if a.get("passthrough"):
        writer.add(base, pieces[""])
        return
    stem = base[: -len(".weight")]
    writer.add(base, pieces[""])
    writer.add(stem + ".scales", pieces[".scales__"])
    if ".biases__" in pieces:
        writer.add(stem + ".biases", pieces[".biases__"])
    overrides[stem] = {"group_size": a["group_size"], "bits": a["bits"],
                       **({"mode": a["mode"]} if a["mode"] != "affine" else {})}


# ---------------------------------------------------------------- main
def convert(src: Path, dst: Path, plan: Plan, drop_mtp: bool = False,
            limit_layers: int | None = None,
            folds: Folds | None = None) -> None:
    cfg = Dots3Config.load(src)
    idx = ShardIndex(src)
    folds = folds or Folds.none()
    dst.mkdir(parents=True, exist_ok=False)

    audit: dict = {}
    overrides: dict = {}
    writer = ShardWriter(dst)
    t0 = time.time()

    # -- backbone layers (0..45) + optional MTP layer 46 ----------------------
    n_layers = cfg.num_hidden_layers if limit_layers is None else limit_layers
    layer_ids = list(range(n_layers)) + ([] if drop_mtp else [46])
    for i in layer_ids:
        p = f"model.layers.{i}."
        is_mtp = i >= cfg.num_hidden_layers
        print(f"[layer {i}]", flush=True)
        names = sorted(n for n in idx.names()
                       if n.startswith(p) and not n.endswith("_scale_inv"))
        done_routed: set[tuple[int, str]] = set()
        for name in names:
            rk = routed_key(name)
            if rk is not None:
                if rk in done_routed:
                    continue
                done_routed.add(rk)
                layer_i, proj = rk
                spec = plan.for_routed(layer_i, proj)
                shape = idx.info(p + f"mlp.experts.0.{proj}.weight")[2]
                stack = np.empty((cfg.n_routed_experts, *shape), dtype=np.float32)
                for e in range(cfg.n_routed_experts):
                    en = p + f"mlp.experts.{e}.{proj}.weight"
                    stack[e] = folds.apply(en, idx.read_dequant(en))
                unit_name = p + f"mlp.switch_mlp.{proj}.weight"
                pieces = quantize_tensor(stack, spec, unit_name, audit)
                emit_quantized(writer, unit_name, pieces, audit, overrides,
                               unit_name)
                del stack, pieces
                gc.collect()
                mx.clear_cache()
                continue
            cls = classify(name)
            if cls == "skip":
                continue
            if cls == "passthrough":
                arr = idx.read(name)
                _, dtype, _ = idx.info(name)
                if name.endswith(("post_attention_layernorm.weight",)) or \
                        ".mlp.gate.weight" in name:
                    arr = folds.apply(name, arr.astype(np.float32))
                tgt = mx.bfloat16 if dtype == "BF16" else (
                    mx.float32 if dtype == "F32" else mx.float16)
                writer.add(name, mx.array(arr).astype(tgt))
                continue
            spec = plan.for_class(
                "mtp_linear" if is_mtp and cls in ("attention", "dense_mlp",
                                                   "mtp_linear") else cls)
            if spec is None:
                raise KeyError(f"no plan class for {name} ({cls})")
            w = folds.apply(name, idx.read_dequant(name))
            pieces = quantize_tensor(w, spec, name, audit)
            emit_quantized(writer, name, pieces, audit, overrides, name)
            del w, pieces
        writer.flush()

    # -- bookends -------------------------------------------------------------
    print("[bookends]", flush=True)
    writer.add("model.norm.weight",
               mx.array(idx.read("model.norm.weight")).astype(mx.bfloat16))
    emb = idx.read_dequant("model.embed_tokens.weight")
    spec = plan.for_class("bookend")
    pieces = quantize_tensor(emb, spec, "model.embed_tokens.weight", audit)
    emit_quantized(writer, "model.embed_tokens.weight", pieces, audit,
                   overrides, "model.embed_tokens.weight")
    head = idx.read_dequant("lm_head.weight")
    pieces = quantize_tensor(head, spec, "lm_head.weight", audit)
    emit_quantized(writer, "lm_head.weight", pieces, audit, overrides,
                   "lm_head.weight")
    mtp_embed_shared = False
    if not drop_mtp and idx.has("model.mtp.embed_tokens.weight"):
        raw_main, _, _ = idx.read_raw("model.embed_tokens.weight")
        raw_mtp, _, _ = idx.read_raw("model.mtp.embed_tokens.weight")
        if raw_main.shape == raw_mtp.shape and (raw_main == raw_mtp).all():
            mtp_embed_shared = True
            audit["model.mtp.embed_tokens.weight"] = {"deduped": True}
            print("  mtp.embed_tokens BYTE-IDENTICAL to embed_tokens — deduped")
        else:
            me = idx.read_dequant("model.mtp.embed_tokens.weight")
            pieces = quantize_tensor(me, spec, "model.mtp.embed_tokens.weight",
                                     audit)
            emit_quantized(writer, "model.mtp.embed_tokens.weight", pieces,
                           audit, overrides, "model.mtp.embed_tokens.weight")
            del me
        del raw_main, raw_mtp
    del emb, head
    gc.collect(); mx.clear_cache()

    # -- towers ----------------------------------------------------------------
    for tower, cls_map in (("vision_encoder.", ("vision_expert", "vision_linear")),
                           ("audio_encoder.", ("audio_linear",))):
        print(f"[{tower[:-1]}]", flush=True)
        for name in sorted(n for n in idx.names() if n.startswith(tower)):
            cls = classify(name)
            if cls == "skip":
                continue
            if cls == "passthrough":
                arr = idx.read(name)
                _, dtype, _ = idx.info(name)
                tgt = mx.bfloat16 if dtype == "BF16" else (
                    mx.float32 if dtype == "F32" else mx.float16)
                writer.add(name, mx.array(arr).astype(tgt))
                continue
            spec = plan.for_class(cls)
            w = idx.read_dequant(name)
            pieces = quantize_tensor(w, spec, name, audit)
            emit_quantized(writer, name, pieces, audit, overrides, name)
            del w
        writer.flush()

    writer.finalize()

    # -- configs ----------------------------------------------------------------
    cfg_json = json.loads((src / "config.json").read_text())
    cfg_json.pop("quantization_config", None)
    dflt = plan.defaults["routed"]
    cfg_json["quantization"] = {
        "group_size": dflt["group_size"], "bits": dflt["bits"],
        "mode": "affine", **{k: v for k, v in overrides.items()}}
    cfg_json["quantization_config"] = cfg_json["quantization"]
    (dst / "config.json").write_text(json.dumps(cfg_json, indent=1))

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json", "merges.txt", "vocab.json",
              "added_tokens.json", "preprocessor_config.json",
              "chat_template.jinja", "README.md", "LICENSE"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)

    jang = {
        "format": "jang-v2",
        "model": "dots3-note-prev",
        "source": str(src),
        "plan": plan.raw,
        "mtp_embed_shared": mtp_embed_shared,
        "converted_bytes": writer.total_bytes,
    }
    (dst / "jang_config.json").write_text(json.dumps(jang, indent=1))
    (dst / "conversion_audit.json").write_text(json.dumps(audit, indent=1))
    print(f"DONE {writer.total_bytes/1e9:.2f} GB "
          f"({writer.total_bytes/2**30:.2f} GiB) in {(time.time()-t0)/60:.1f} min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--folds", type=Path, default=None,
                    help="folds.npz (AWQ + diag-imatrix scale vectors)")
    ap.add_argument("--drop-mtp", action="store_true")
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="debug: convert only the first N backbone layers")
    a = ap.parse_args()
    mx.set_memory_limit(int(46 * 1024**3))
    folds = Folds.load(a.folds) if a.folds else Folds.none()
    convert(a.src.expanduser(), a.dst.expanduser(), Plan(a.plan),
            drop_mtp=a.drop_mtp, limit_layers=a.limit_layers, folds=folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
