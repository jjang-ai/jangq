"""DeepSeek-V4.1-Flash(-1017) -> JANG mixed affine bundle (MLX mx.quantize, group 64).

Default plan (~211 GB resident, fits a 256 GB Mac with KV/runtime headroom):

  routed experts (543.6B params)   gate/up q3 on layers 0-7, q2 on 8-39; down q3 everywhere
  attention / shared expert /
  indexer wq_b / engram wkv /
  embed / head                     8-bit affine
  router, mHC, sinks, norms,
  compressor, indexer wk/weights   source precision (F32 / BF16) passthrough
  engram n-gram tables (~203 GB)   NOT copied: stay FP8 on disk, read by row at runtime
  DSpark MTP (mtp.*)               dropped (autoregressive runtime only)

Routed experts are written pre-stacked as
  layers.N.ffn.experts.{gate,up,down}_proj.{weight,scales,biases}   [384, out, in*bits/32]
Everything else keeps its source name.

Output layout: one shard per routed layer plus shards for the non-routed tensors, so a
killed run resumes by skipping shards whose names are already recorded in
`.convert_progress.json`.

Usage:
  python -m jang_tools.dsv41.convert_dsv41_jang \
      --src ~/models/DeepSeek-V4.1-Flash-1017 \
      --dst ~/models/DeepSeek-V4.1-Flash-1017-JANG
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.dsv41.st_io import Shards, dequant_fp4

GROUP_SIZE = 64
PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
EXPERT_CHUNK = 64  # experts dequantized + quantized at once (~2.4 GB bf16 per chunk)


def default_plan(n_layers: int) -> dict[str, dict[int, int]]:
    hi_layers = set(range(8))
    return {
        "gate_proj": {l: 3 if l in hi_layers else 2 for l in range(n_layers)},
        "up_proj": {l: 3 if l in hi_layers else 2 for l in range(n_layers)},
        "down_proj": {l: 3 for l in range(n_layers)},
    }


def load_plan(path: str | None, n_layers: int) -> dict[str, dict[int, int]]:
    plan = default_plan(n_layers)
    if path:
        user = json.loads(Path(path).read_text())
        for proj, layers in user.items():
            for l, b in layers.items():
                plan[proj][int(l)] = int(b)
    return plan


# --------------------------------------------------------------------------- classify

PASSTHROUGH_RE = re.compile(
    r"(norm|attn_sink|hc_attn|hc_ffn|\.gate\.(weight|bias|bias_vl)$|compressor\.(wkv|wgate)"
    r"|indexer\.(wk|weights_proj)\.weight$|engram\.(q_weight|k_weight)$"
    r"|^image_(start|end|newline)$|^vision\.|^aligner\.)"
)


def classify(name: str, shards: Shards) -> str | None:
    """'skip' | 'routed' | 'q8' | 'pass' | None (scale sibling, consumed with its weight)."""
    if name.startswith("mtp.") or ".engram.embed." in name:
        return "skip"
    if name.endswith(".scale"):
        return None
    if re.search(r"\.ffn\.experts\.\d+\.w[123]\.weight$", name):
        return "routed"
    if PASSTHROUGH_RE.search(name):
        return "pass"
    if name.endswith(".weight") and len(shards.meta[name].shape) == 2:
        return "q8"
    return "pass"


# --------------------------------------------------------------------------- writer

class BundleWriter:
    def __init__(self, dst: Path):
        self.dst = dst
        self.progress_path = dst / ".convert_progress.json"
        self.progress = json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {"shards": {}}

    def done(self, key: str) -> bool:
        info = self.progress["shards"].get(key)
        return info is not None and (self.dst / info["file"]).exists()

    def write(self, key: str, tensors: dict[str, mx.array]):
        fname = f"part-{key}.safetensors"
        tmp = self.dst / (fname + ".tmp.safetensors")
        mx.save_safetensors(str(tmp), tensors, metadata={"format": "mlx"})
        tmp.rename(self.dst / fname)
        self.progress["shards"][key] = {
            "file": fname,
            "tensors": sorted(tensors),
            "bytes": (self.dst / fname).stat().st_size,
        }
        self.progress_path.write_text(json.dumps(self.progress, indent=1))

    def finalize(self) -> dict[str, str]:
        """Rename parts to model-XXXXX-of-NNNNN in key order and write the index."""
        keys = sorted(self.progress["shards"], key=_shard_sort_key)
        n = len(keys)
        weight_map, total = {}, 0
        for i, key in enumerate(keys, 1):
            info = self.progress["shards"][key]
            final = f"model-{i:05d}-of-{n:05d}.safetensors"
            src = self.dst / info["file"]
            if src.exists() and info["file"] != final:
                src.rename(self.dst / final)
            info["file"] = final
            total += info["bytes"]
            for t in info["tensors"]:
                weight_map[t] = final
        self.progress_path.write_text(json.dumps(self.progress, indent=1))
        (self.dst / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2)
        )
        return weight_map


def _shard_sort_key(key: str):
    kind, _, num = key.partition("-")
    return (0 if kind == "base" else 1, int(num) if num.isdigit() else 0)


# --------------------------------------------------------------------------- conversion

def quantize(w: mx.array, bits: int) -> dict[str, mx.array]:
    qw, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
    return {"weight": qw, "scales": s, "biases": b}


def convert_routed_layer(shards: Shards, layer: int, plan, pool: ThreadPoolExecutor, n_experts: int):
    out: dict[str, mx.array] = {}
    stats = {}
    for src_proj, proj in PROJ.items():
        bits = plan[proj][layer]
        names = [f"layers.{layer}.ffn.experts.{e}.{src_proj}.weight" for e in range(n_experts)]
        raw_w = list(pool.map(shards.raw, names))
        raw_s = list(pool.map(shards.raw, [n[: -len(".weight")] + ".scale" for n in names]))
        parts = {"weight": [], "scales": [], "biases": []}
        err_num = err_den = 0.0
        for c in range(0, n_experts, EXPERT_CHUNK):
            w = dequant_fp4(mx.array(np.stack(raw_w[c : c + EXPERT_CHUNK])), mx.array(np.stack(raw_s[c : c + EXPERT_CHUNK])))
            q = quantize(w, bits)
            if c == 0:  # cheap fidelity probe on the first chunk
                deq = mx.dequantize(q["weight"], q["scales"], q["biases"], group_size=GROUP_SIZE, bits=bits)
                d = (deq.astype(mx.float32) - w.astype(mx.float32))
                err_num = float(mx.sqrt(mx.sum(d * d)))
                err_den = float(mx.sqrt(mx.sum(w.astype(mx.float32) ** 2)))
            mx.eval(q)
            for k in parts:
                parts[k].append(q[k])
            del w, q
        base = f"layers.{layer}.ffn.experts.{proj}"
        for k, v in parts.items():
            out[f"{base}.{k}"] = mx.concatenate(v, axis=0)
        mx.eval(out)
        stats[proj] = {"bits": bits, "rel_l2_first_chunk": err_num / max(err_den, 1e-30)}
        del raw_w, raw_s, parts
    return out, stats


def convert_base(shards: Shards, names: list[str]):
    out: dict[str, mx.array] = {}
    kinds = {"q8": 0, "pass": 0}
    for name in names:
        kind = classify(name, shards)
        if kind == "q8":
            w = shards.dequant(name, mx.bfloat16)
            base = name[: -len(".weight")]
            for k, v in quantize(w, 8).items():
                out[f"{base}.{k}"] = v
        else:
            m = shards.meta[name]
            if m.dtype in ("F8_E4M3", "I8"):
                raise ValueError(f"refusing to pass through quantized source tensor {name}")
            out[name] = shards.to_mx(name)
        kinds[kind] += 1
        mx.eval(list(out.values())[-3:])
    return out, kinds


def write_metadata(src: Path, dst: Path, plan, n_layers: int, stats: dict, excluded: dict):
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    text = cfg["text_config"]
    quant = {"group_size": GROUP_SIZE, "bits": 8, "mode": "affine"}
    for proj, layers in plan.items():
        for l in range(n_layers):
            quant[f"layers.{l}.ffn.experts.{proj}"] = {"group_size": GROUP_SIZE, "bits": layers[l], "mode": "affine"}
    cfg["quantization"] = quant
    text["num_nextn_predict_layers"] = 0
    cfg["runtime"] = {
        "bundle_has_mtp": False,
        "mtp_mode": "dropped",
        "engram_tables": "disk",
        "engram_dir": "engram",
    }
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    routed_bits = sum(
        plan[p][l] for p in plan for l in range(n_layers)
    ) / (3 * n_layers)
    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "affine",
        "profile": "JANG_DSV41_Q2Q3_D3_L0-7G3U3_A8",
        "source_model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "source_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "has_vision": True,
        "critical_f32_preserved": True,
        "affine_group_size": GROUP_SIZE,
        "affine_bits": {
            "routed_expert_mean": round(routed_bits, 4),
            "attention": 8,
            "shared_expert": 8,
            "indexer_wq_b": 8,
            "engram_wkv": 8,
            "embed_tokens": 8,
            "lm_head": 8,
            "norms_router_hc_compressor": "source",
        },
        "routed_expert_bit_plan": {p: {str(l): b for l, b in v.items()} for p, v in plan.items()},
        "routed_layout": "prestacked layers.N.ffn.experts.{gate,up,down}_proj",
        "engram": {
            "storage": "fp8_e4m3 rows + e8m0 scales per 32 cols, unmodified source shards",
            "location": "engram/ (see engram/README.md)",
            "resident": False,
        },
        "excluded": excluded,
        "quant_error_probe": stats,
    }, indent=2))

    for fn in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (src / fn).exists():
            shutil.copy2(src / fn, dst / fn)
    if (src / "encoding").exists():
        shutil.copytree(src / "encoding", dst / "encoding", dirs_exist_ok=True)
    (dst / "engram").mkdir(exist_ok=True)
    readme = dst / "engram" / "README.md"
    if not readme.exists():
        readme.write_text(
            "Engram n-gram tables (FP8, ~203 GB) are read from disk row by row, never loaded into RAM.\n\n"
            "Place the two unmodified source shards that hold `layers.{1,14}.engram.embed.{weight,scale}`\n"
            "here (model-00047-of-00048.safetensors, model-00048-of-00048.safetensors), or pass\n"
            "`--engram-dir <source dir>` to the runtime. They can be moved, not copied, from the source download.\n"
        )


def convert(src: Path, dst: Path, plan_path: str | None, only_layers: list[int] | None = None):
    dst.mkdir(parents=True, exist_ok=True)
    shards = Shards(src, allow_missing=True)
    if shards.missing:
        print(f"[convert] shards not present yet: {shards.missing} (routed layers proceed; rerun to finish)")
    text = json.loads((src / "config.json").read_text())["text_config"]
    n_layers, n_experts = text["num_hidden_layers"], text["n_routed_experts"]
    plan = load_plan(plan_path, n_layers)
    writer = BundleWriter(dst)
    stats = writer.progress.setdefault("stats", {})
    t0 = time.time()

    with ThreadPoolExecutor(16) as pool:
        for layer in only_layers if only_layers is not None else range(n_layers):
            key = f"routed-{layer}"
            if writer.done(key):
                continue
            out, st = convert_routed_layer(shards, layer, plan, pool, n_experts)
            writer.progress.setdefault("stats", {})[str(layer)] = st
            writer.write(key, out)
            gb = writer.progress["shards"][key]["bytes"] / 1e9
            print(f"[{time.time() - t0:7.0f}s] layer {layer:2d}: {gb:.2f} GB {st}", flush=True)
            del out
            mx.clear_cache()

    base_names = sorted(n for n in shards.keys() if classify(n, shards) in ("q8", "pass"))
    excluded = {
        "mtp": sum(n.startswith("mtp.") for n in shards.keys()),
        "engram_embed": sum(".engram.embed." in n for n in shards.keys()),
    }
    # non-routed tensors, split in a few deterministic groups so each shard stays a few GB
    groups: dict[str, list[str]] = {}
    for n in base_names:
        m = re.match(r"layers\.(\d+)\.", n)
        key = f"base-{int(m.group(1)) // 10 + 1}" if m else "base-0"
        groups.setdefault(key, []).append(n)
    for key, names in sorted(groups.items()):
        if writer.done(key) or only_layers is not None:
            continue
        if any(shards.meta.get(n) is None for n in names) or shards.missing:
            print(f"[convert] {key}: waiting for missing source shards")
            continue
        out, kinds = convert_base(shards, names)
        writer.write(key, out)
        print(f"[{time.time() - t0:7.0f}s] {key}: {len(names)} tensors {kinds}", flush=True)
        del out
        mx.clear_cache()

    if only_layers is not None or shards.missing:
        return
    writer.finalize()
    write_metadata(src, dst, plan, n_layers, writer.progress.get("stats", {}), excluded)
    total = sum(v["bytes"] for v in writer.progress["shards"].values())
    print(f"done in {time.time() - t0:.0f}s, {total / 1e9:.2f} GB ({total / 2**30:.2f} GiB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--plan", help="JSON {proj: {layer: bits}} overriding the default plan")
    ap.add_argument("--layers", help="comma list: convert only these routed layers (no finalize)")
    a = ap.parse_args()
    only = [int(x) for x in a.layers.split(",")] if a.layers else None
    convert(Path(a.src).expanduser(), Path(a.dst).expanduser(), a.plan, only)


if __name__ == "__main__":
    main()
