"""GLM-5.3-Flash (glm5_next) -> JANG v2 converter.

Ported from the proven qwen4_exp converter. Key differences:
  * Bundle keeps EVERYTHING: vision tower, MTP layer 45, DSA indexer — all in
    the runtime (sanitized) naming, so load_bundle/vmlx consume directly.
  * AWQ fold (mlp side only): x-side scales fold into post_attention_layernorm
    (a plain RMSNorm — divide weight by s) with EXACT compensation of every
    consumer of that activation: routed gate/up (stacked, last axis), shared
    gate/up, dense gate/up, AND the router gate.weight (fp32 — exact).
  * All fold/sanitize outputs cast back to the source dtype INLINE
    (2026-08-26 incident invariant #1). No post-hoc dtype passes, ever.
  * Per-module `quantization` block emitted into config.json AT BUILD TIME
    (invariant #3) — recorded from what is actually written.

HARD KEEPS (never quantized): conv1d taps, A_log, dt_bias, f_a/f_b, g_a/g_b,
b_proj (KDA gate producers — vendor keeps them BF16 even in the FP8 release),
o_norm + all norms, hc_* (mHC), router gate + e_score bias (fp32), kpool ape.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import time
from pathlib import Path

import mlx.core as mx

SHARD_BYTES = 5 * 2**30

KEEP_SUFFIXES = (
    "q_conv1d", "k_conv1d", "v_conv1d", "A_log", "dt_bias", "o_norm",
    "e_score_correction_bias", "hc_fn", "hc_base", "hc_scale",
    "index_kpool_compress_ape",
)
KEEP_CONTAINS = (
    ".f_a_proj.", ".f_b_proj.", ".g_a_proj.", ".g_b_proj.", ".b_proj.",
    "layernorm", ".norm.", "norm.weight", "norm.bias", ".mlp.gate.weight",
    ".enorm.", ".hnorm.", ".shared_head.",
)


def match_spec(path: str, bit_map: dict):
    best, best_len = None, -1
    for prefix, spec in bit_map.items():
        if path.startswith(prefix) and len(prefix) > best_len:
            best, best_len = spec, len(prefix)
    return best


def quantizable(path: str, w) -> bool:
    if w.ndim < 2 or w.dtype == mx.uint32:
        return False
    if any(path.endswith(s) or (s + ".") in path for s in KEEP_SUFFIXES):
        return False
    if any(c in path for c in KEEP_CONTAINS):
        return False
    if not path.endswith(".weight"):
        return False
    return w.shape[-1] % 64 == 0


def sanitize_bundle(weights: dict, no_mtp: bool = False) -> dict:
    """Full-bundle sanitize: keep vision + indexer (+ MTP unless no_mtp),
    runtime naming."""
    out, experts = {}, {}
    for k, v in weights.items():
        k = k.replace("model.language_model.", "model.")
        k = k.replace("model.visual.", "visual.")
        if no_mtp and re.match(r"model\.layers\.45\.", k):
            continue
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
        out[k] = v
    for (base, proj), parts in experts.items():
        E = max(parts) + 1
        out[f"{base}.switch_mlp.{proj}_proj.weight"] = mx.stack(
            [parts[i] for i in range(E)], axis=0)
    return out


def apply_awq(weights: dict, scales: dict):
    """Fold mlp-input AWQ scales. scales: {'model.layers.N.mlp': s[4096]}.
    post_attention_layernorm.weight /= s ; every consumer's input columns *= s.
    All outputs cast back to the source dtype INLINE."""
    def scale_cols(key, s):
        w = weights[key]
        weights[key] = (w * s.astype(mx.float32).reshape(
            *([1] * (w.ndim - 1)), -1).astype(w.dtype)).astype(w.dtype)

    n = 0
    for site, s in scales.items():
        m = re.match(r"model\.layers\.(\d+)\.mlp$", site)
        if not m:
            continue
        base = site[: -len(".mlp")]
        s32 = s.astype(mx.float32)
        nk = f"{base}.post_attention_layernorm.weight"
        nw = weights[nk]
        weights[nk] = (nw.astype(mx.float32) / s32).astype(nw.dtype)
        for c in (f"{site}.switch_mlp.gate_proj.weight",
                  f"{site}.switch_mlp.up_proj.weight",
                  f"{site}.shared_experts.gate_proj.weight",
                  f"{site}.shared_experts.up_proj.weight",
                  f"{site}.gate_proj.weight",          # dense layers
                  f"{site}.up_proj.weight",
                  f"{site}.gate.weight"):              # router (fp32 keep)
            if c in weights:
                scale_cols(c, s32)
        n += 1
    return n


def refit_quantize(w: mx.array, imatrix, group_size: int, bits: int):
    if imatrix is None:
        return mx.quantize(w, group_size=group_size, bits=bits)
    from jang_tools.qwen4_exp.affine_mx import refit_quantize_mx

    # GLM's stacked expert tensors are 2.4B params; one ALS graph at that size
    # saturates Metal long enough to starve watchdogd (kernel panic 01:39
    # 2026-08-29, "no checkins from watchdogd in 90 seconds"). Chunk the
    # expert axis so each graph stays small and the system breathes.
    if w.ndim == 3 and w.shape[0] > 32:
        outs = []
        im3 = imatrix if getattr(imatrix, "ndim", 1) == 2 else None
        for i in range(0, w.shape[0], 32):
            im_c = im3[i:i + 32] if im3 is not None else imatrix
            part = refit_quantize_mx(w[i:i + 32], im_c, group_size, bits)
            mx.eval(part)
            outs.append(part)
            mx.clear_cache()
        return tuple(mx.concatenate([o[j] for o in outs], axis=0) for j in range(3))
    return refit_quantize_mx(w, imatrix, group_size, bits)


class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.buf: dict = {}
        self.buf_bytes = 0
        self.idx = 0
        self.weight_map: dict = {}
        self.total = 0

    def add(self, name: str, arr: mx.array):
        mx.eval(arr)
        self.buf[name] = arr
        self.buf_bytes += arr.nbytes
        self.total += arr.nbytes
        if self.buf_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.idx += 1
        fname = f"model-{self.idx:05d}.safetensors"
        mx.save_safetensors(str(self.out_dir / fname), self.buf)
        for k in self.buf:
            self.weight_map[k] = fname
        self.buf = {}
        self.buf_bytes = 0
        mx.clear_cache()

    def finish(self):
        self.flush()
        files = sorted(self.out_dir.glob("model-*.safetensors"))
        n = len(files)
        renames = {}
        for i, f in enumerate(files, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            renames[f.name] = new
            f.rename(self.out_dir / new)
        wm = {k: renames[v] for k, v in self.weight_map.items()}
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": wm}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit-map", required=True)
    ap.add_argument("--awq-scales", default=None)
    ap.add_argument("--imatrix", default=None,
                    help="augmented diag (translated names + down_expert_diag)")
    ap.add_argument("--tensor-start", type=int, default=0)
    ap.add_argument("--tensor-end", type=int, default=0)
    ap.add_argument("--segment", default=None)
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()

    imatrix = mx.load(args.imatrix) if args.imatrix else {}

    def resolve_imatrix(name: str):
        mod = name[: -len(".weight")] if name.endswith(".weight") else name
        if mod + ".diag" in imatrix:
            return imatrix[mod + ".diag"]              # [in]
        if mod + ".expert_diag" in imatrix:
            return imatrix[mod + ".expert_diag"]       # [E, in]
        return None

    model_dir, out_dir = Path(args.model), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bit_map = json.loads(Path(args.bit_map).read_text())

    weights = {}
    for f in sorted(glob.glob(str(model_dir / "model-*.safetensors"))):
        weights.update(mx.load(f))
    weights = sanitize_bundle(weights, no_mtp=args.no_mtp)

    awq_applied = False
    if args.awq_scales:
        n = apply_awq(weights, mx.load(args.awq_scales))
        print(f"AWQ folded at {n} mlp sites")
        awq_applied = True

    seg_dir = out_dir / args.segment if args.segment else out_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(seg_dir)
    report = {"quantized": 0, "kept": 0, "bytes_by_spec": {}}
    quant_entries: dict = {}
    t0 = time.time()
    items = sorted(weights.items())
    if args.tensor_end:
        items = items[args.tensor_start: args.tensor_end]
    elif args.tensor_start:
        items = items[args.tensor_start:]
    for i, (name, w) in enumerate(items):
        prev = writer.total
        spec = match_spec(name, bit_map)
        if spec in ("keep", "fp16", None) or not quantizable(name, w):
            report["kept"] += 1
            writer.add(name, w)
            key = "keep"
        else:
            bits, gs = spec["bits"], spec["group_size"]
            report["quantized"] += 1
            im = resolve_imatrix(name)
            wq, sc, bi = refit_quantize(w.astype(mx.float32), im, gs, bits)
            base = name[: -len(".weight")]
            writer.add(base + ".weight", wq)
            writer.add(base + ".scales", sc.astype(mx.float16))
            writer.add(base + ".biases", bi.astype(mx.float16))
            quant_entries[base] = {"group_size": gs, "bits": bits}
            key = f"{bits}b_gs{gs}"
        report["bytes_by_spec"][key] = report["bytes_by_spec"].get(key, 0) + writer.total - prev
        if i % 10 == 0:
            mx.clear_cache()
        if i % 50 == 0:
            print(f"{i}/{len(items)} tensors, {writer.total/2**30:.1f} GiB out, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    if args.segment:
        writer.flush()
        (seg_dir / "map.json").write_text(json.dumps(
            {"weight_map": writer.weight_map, "total": writer.total,
             "quant_entries": quant_entries}))
        print(f"segment {args.segment}: {writer.total/2**30:.2f} GiB, "
              f"{len(writer.weight_map)} entries")
        print(json.dumps(report["bytes_by_spec"], indent=1))
        return

    writer.finish()
    _finalize_configs(model_dir, out_dir, bit_map, quant_entries, args, awq_applied)
    print(json.dumps(report, indent=1))
    print(f"total out: {writer.total/2**30:.2f} GiB in {(time.time()-t0)/60:.1f} min")


def _finalize_configs(model_dir: Path, out_dir: Path, bit_map: dict,
                      quant_entries: dict, args, awq_applied: bool):
    for f in ("config.json", "generation_config.json", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja",
              "processor_config.json", "preprocessor_config.json",
              "video_preprocessor_config.json", "merges.txt", "vocab.json"):
        src = model_dir / f
        if src.exists():
            shutil.copy(src, out_dir / f)
    cfg = json.loads((out_dir / "config.json").read_text())
    from collections import Counter
    if quant_entries:
        mg = Counter(v["group_size"] for v in quant_entries.values()).most_common(1)[0][0]
        mb = Counter(v["bits"] for v in quant_entries.values()).most_common(1)[0][0]
        q = {"group_size": mg, "bits": mb}
        q.update(dict(sorted(quant_entries.items())))
        cfg["quantization"] = q
    cfg["jang_config"] = {
        "format": "jang_v2",
        "family": "glm5_next",
        "norm_convention": "standard_rmsnorm",
        "bit_map": bit_map,
        "created": time.strftime("%Y-%m-%d"),
        "quantization": {
            "calibrated": bool(args.imatrix),
            "imatrix": Path(args.imatrix).name if args.imatrix else None,
            "imatrix_refit": bool(args.imatrix),
            "awq_folded": awq_applied,
            "awq_scales": Path(args.awq_scales).name if args.awq_scales else None,
            "hessian_allocation": Path(args.bit_map).name,
            "gptq": False,
        },
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
