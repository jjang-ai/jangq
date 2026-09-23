"""JANG v2 bundle converter for Qwen3.8-Flash-Next (qwen4_exp).

Streams the 360 GB checkpoint lazily: sanitize → (optional) AWQ fold →
per-tensor quantize → incremental shard writes. Nothing bigger than one
tensor is ever materialized.

Bit map JSON format (module-path → spec), resolved longest-prefix-first:
  {
    "default":                       {"bits": 4, "group_size": 64},
    "language_model.layers.*.mlp.switch_mlp": {"bits": 4, "group_size": 64},
    "...ple.ngram_embedding":        {"bits": 4, "group_size": 32},
    "lm_head":                       {"bits": 6, "group_size": 64},
    "...linear_attn.in_proj_a":      "fp16",
    ...
  }
"*" matches any single path segment. Values: {"bits","group_size"} | "fp16" | "keep".

Norm convention: bundle stores RUNTIME (+1-applied) norm weights — recorded
in jang_config as norm_convention=runtime_plus1_applied. The Python runtime
loads bundles without any shift.

Usage:
  python -m jang_tools.qwen4_exp.convert --model ~/models/Qwen3.8-Flash-Next \
      --out ~/.mlxstudio/models/JANGQ-AI/Qwen3.8-Flash-Next-JANG_4M \
      --bit-map maps/jang_4m.json [--awq-scales scales.safetensors] [--mtp]
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .load import PLUS_ONE_SUFFIXES, _SHARD_RE

SHARD_BYTES = 5 * 1024**3

# never quantize regardless of map (tiny / structurally critical / non-2D)
FORCE_KEEP_SUFFIXES = (
    "A_log", "dt_bias", "conv1d.weight", "conv1d_weight",
    ".hc_norm.weight", ".norm.weight", ".norm_key.weight", ".norm_query.weight",
    ".norm_conv.weight", ".q_norm.weight", ".k_norm.weight",
    ".q_layernorm.weight", ".k_layernorm.weight",
    "layer_multipliers", "ngram_heads_offsets", "ngram_heads_vocab_sizes",
    ".block_inject_weight.weight", ".shared_expert_gate.weight",
    ".input_mix_weight_down.weight", ".input_mix_weight_up.weight",
    ".pos_embed.weight", ".patch_embed.proj.bias",
)


def match_spec(path: str, bit_map: dict):
    best, best_len = None, -1
    for pat, spec in bit_map.items():
        if pat == "default":
            continue
        if fnmatch.fnmatch(path, pat) or path.startswith(pat) or fnmatch.fnmatch(path, pat + "*"):
            if len(pat) > best_len:
                best, best_len = spec, len(pat)
    return best if best is not None else bit_map.get("default", {"bits": 4, "group_size": 64})


def quantizable(path: str, w) -> bool:
    if any(path.endswith(s) for s in FORCE_KEEP_SUFFIXES):
        return False
    if w.ndim < 2 or w.dtype in (mx.int64, mx.int32, mx.uint32):
        return False
    if w.shape[-1] % 32 != 0:
        return False
    return True


def sanitize_stream(weights: dict, keep_mtp: bool):
    """Name transforms mirroring load.sanitize, but LAZY (no eval)."""
    out = {}
    for k, v in weights.items():
        if k.startswith("mtp."):
            if keep_mtp:
                out[k] = v
            continue
        if k.startswith("model.visual."):
            out[k.replace("model.visual.", "visual.")] = v
            continue
        if k.startswith("model.language_model."):
            k = k.replace("model.language_model.", "language_model.")
        m = _SHARD_RE.match(k)
        if m:
            out[f"{m.group(1)}.ngram_embedding.shards.{int(m.group(2))}.weight"] = v
            continue
        if k.endswith("ple.ple_embedding.layer_multipliers") or k.endswith(
            "ple.ple_embedding.ngram_heads_offsets"
        ) or k.endswith("ple.ple_embedding.ngram_heads_vocab_sizes"):
            out[k.replace(".ple_embedding.", ".")] = v
            continue
        if k.endswith("ple.conv1d.weight"):
            out[k.replace("ple.conv1d.weight", "ple.conv1d_weight")] = v.squeeze(1)
            continue
        if k.endswith("linear_attn.conv1d.weight"):
            out[k] = v.moveaxis(2, 1)
            continue
        if k.endswith("mlp.experts.gate_up_proj"):
            h = v.shape[1] // 2
            base = k.replace("mlp.experts.gate_up_proj", "mlp.switch_mlp")
            out[base + ".gate_proj.weight"] = v[:, :h, :]
            out[base + ".up_proj.weight"] = v[:, h:, :]
            continue
        if k.endswith("mlp.experts.down_proj"):
            out[k.replace("mlp.experts.down_proj", "mlp.switch_mlp.down_proj.weight")] = v
            continue
        out[k] = v
    for k in list(out):
        if any(k.endswith(s) for s in PLUS_ONE_SUFFIXES):
            v = out[k]
            out[k] = (v + 1.0).astype(v.dtype)
    return out


def apply_awq(weights: dict, scales: dict, hc_count: int = 4):
    """scales: site-path → s [hidden] (mx arrays). Site paths:
    '<layer>.attn', '<layer>.mlp', 'final'. Mirrors awq_fold consumer sets,
    operating directly on the weight dict (lazy)."""

    def scale_in(key, s):
        if key in weights:
            w = weights[key]
            weights[key] = (w * s).astype(w.dtype)

    def tile(s):
        return mx.tile(s, (hc_count,))

    for site, s in scales.items():
        s = s.astype(mx.float32)
        if site == "final":
            base = "language_model.hyper_connection_mixer"
            consumers = ["lm_head.weight"]
        elif site.endswith(".attn"):
            lp = site[: -len(".attn")]
            base = f"{lp}.attn_hyper_connection"
            if f"{lp}.linear_attn.in_proj_qkv.weight" in weights:
                consumers = [f"{lp}.linear_attn.in_proj_{x}.weight" for x in ("qkv", "z", "a", "b")]
            else:
                consumers = [f"{lp}.self_attn.{x}.weight" for x in
                             ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")]
        elif site.endswith(".mlp"):
            lp = site[: -len(".mlp")]
            base = f"{lp}.mlp_hyper_connection"
            consumers = [f"{lp}.mlp.{x}.weight" for x in
                         ("gate", "switch_mlp.gate_proj", "switch_mlp.up_proj",
                          "shared_expert.gate_proj", "shared_expert.up_proj",
                          "shared_expert_gate")]
        else:
            raise ValueError(f"unknown AWQ site {site}")
        st = tile(s)
        hn = weights[f"{base}.hc_norm.weight"]
        weights[f"{base}.hc_norm.weight"] = (hn / st).astype(hn.dtype)
        scale_in(f"{base}.input_mix_weight_down.weight", st)
        if f"{base}.block_inject_weight.weight" in weights:
            scale_in(f"{base}.block_inject_weight.weight", st)
        for c in consumers:
            scale_in(c, s)


def _refit_slice(args):
    """Process-pool worker: one expert slice through the weighted affine fit."""
    w2, imp, bits, gs = args
    from jang_tools.affine import quantize_imatrix_affine_numpy

    packed, scales, biases, _ = quantize_imatrix_affine_numpy(
        w2, imp, bits=bits, group_size=gs)
    return packed, scales, biases


def refit_quantize(w: mx.array, imatrix: mx.array | None, group_size: int, bits: int):
    """Affine quantize; with an imatrix, use the PROVEN weighted alternating
    least-squares fit from jang_tools.affine (native MLX storage ABI for all
    of 2/3/4/5/6/8-bit — err 0.44→0.24 on Qwen3.6-27B). Runs on the
    ALREADY-FOLDED weight, so AWQ can never be silently reverted by refit.
    imatrix: per-input-channel E[x²] proportional weights [in], or None.
    """
    if imatrix is None:
        return mx.quantize(w, group_size=group_size, bits=bits)

    # GPU-batched weighted ALS (verified identical to the numpy reference)
    from jang_tools.qwen4_exp.affine_mx import refit_quantize_mx

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
        nb = arr.nbytes
        self.buf[name] = arr
        self.buf_bytes += nb
        self.total += nb
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
        # renumber to -of- format
        files = sorted(self.out_dir.glob("model-*.safetensors"))
        n = len(files)
        renames = {}
        for i, f in enumerate(files, 1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            renames[f.name] = new
            f.rename(self.out_dir / new)
        wm = {k: renames[v] for k, v in self.weight_map.items()}
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total}, "weight_map": wm}, indent=1)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bit-map", required=True)
    ap.add_argument("--awq-scales", default=None)
    ap.add_argument("--imatrix", default=None, help="capture diag.safetensors for weighted refit")
    ap.add_argument("--mtp", action="store_true", help="include mtp.* (same default spec)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tensor-start", type=int, default=0)
    ap.add_argument("--tensor-end", type=int, default=0, help="0 = all")
    ap.add_argument("--segment", default=None,
                    help="write into <out>/<segment>/ with a local map.json "
                         "(merged later by merge_segments)")
    args = ap.parse_args()

    imatrix = mx.load(args.imatrix) if args.imatrix else {}

    def resolve_imatrix(name: str):
        """Per-input-channel importance for a weight tensor, or None."""
        mod = name[: -len(".weight")] if name.endswith(".weight") else name
        if mod + ".diag" in imatrix:
            return imatrix[mod + ".diag"]
        if mod + ".expert_diag" in imatrix:
            ed = imatrix[mod + ".expert_diag"]        # [E, d_in]
            rows = imatrix.get(mod + ".expert_rows")
            if mod.endswith("down_proj"):
                return ed                              # per-expert (input differs)
            w = rows.astype(mx.float32) if rows is not None else mx.ones(ed.shape[0])
            return (ed * w[:, None]).sum(0) / mx.maximum(w.sum(), 1)  # shared input
        # gate/up importance also serves its sibling
        for sib in ("gate_proj", "up_proj"):
            alt = mod.rsplit(".", 1)[0] + f".{sib}.expert_diag"
            if alt in imatrix:
                ed = imatrix[alt]
                rows = imatrix.get(alt.replace("expert_diag", "expert_rows"))
                w = rows.astype(mx.float32) if rows is not None else mx.ones(ed.shape[0])
                return (ed * w[:, None]).sum(0) / mx.maximum(w.sum(), 1)
        return None

    model_dir, out_dir = Path(args.model), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bit_map = json.loads(Path(args.bit_map).read_text())

    weights = {}
    for f in sorted(glob.glob(str(model_dir / "model-*.safetensors"))):
        weights.update(mx.load(f))
    weights = sanitize_stream(weights, keep_mtp=args.mtp)

    awq_applied = False
    if args.awq_scales:
        raw = mx.load(args.awq_scales)
        apply_awq(weights, raw)
        awq_applied = True

    seg_dir = out_dir / args.segment if args.segment else out_dir
    seg_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(seg_dir)
    report = {"quantized": 0, "kept": 0, "bytes_by_spec": {}}
    t0 = time.time()
    items = sorted(weights.items())
    if args.tensor_end:
        items = items[args.tensor_start: args.tensor_end]
    elif args.tensor_start:
        items = items[args.tensor_start:]
    for i, (name, w) in enumerate(items):
        prev_bytes = writer.total
        spec = match_spec(name, bit_map)
        if spec in ("keep", "fp16") or not quantizable(name, w):
            report["kept"] += 1
            if not args.dry_run:
                writer.add(name, w)  # source dtype passthrough
            key = "keep"
        else:
            bits, gs = spec["bits"], spec["group_size"]
            report["quantized"] += 1
            if not args.dry_run:
                im = resolve_imatrix(name)
                wf = w.astype(mx.float32)
                # GPU ALS handles both shared [in] and per-expert [E, in]
                # importance in one batched program
                wq, sc, bi = refit_quantize(wf, im, gs, bits)
                base = name[: -len(".weight")] if name.endswith(".weight") else name
                writer.add(base + ".weight", wq)
                writer.add(base + ".scales", sc.astype(mx.float16))
                writer.add(base + ".biases", bi.astype(mx.float16))
            key = f"{spec['bits']}b_gs{spec['group_size']}"
        report["bytes_by_spec"][key] = (
            report["bytes_by_spec"].get(key, 0) + writer.total - prev_bytes
        )
        if i % 25 == 0:
            # long single-process Metal sessions accumulate command-buffer
            # state and eventually SIGABRT (draft build died at ~tensor 520
            # twice; the same range replays clean in a fresh process)
            mx.clear_cache()
        if i % 100 == 0:
            print(f"{i}/{len(items)} tensors, {writer.total/2**30:.1f} GiB out, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

    if args.segment:
        writer.flush()
        (seg_dir / "map.json").write_text(json.dumps(
            {"weight_map": writer.weight_map, "total": writer.total}))
        print(f"segment {args.segment}: {writer.total/2**30:.2f} GiB, "
              f"{len(writer.weight_map)} entries")
        print(json.dumps(report["bytes_by_spec"], indent=1))
        return

    if not args.dry_run:
        writer.finish()
        for f in ("config.json", "generation_config.json", "tokenizer.json",
                  "tokenizer_config.json", "vocab.json", "merges.txt",
                  "chat_template.jinja", "preprocessor_config.json",
                  "video_preprocessor_config.json"):
            src = model_dir / f
            if src.exists():
                shutil.copy(src, out_dir / f)
        cfg = json.loads((out_dir / "config.json").read_text())
        cfg["jang_config"] = {
            "format": "jang_v2",
            "family": "qwen4_exp",
            "norm_convention": "runtime_plus1_applied",
            "bit_map": bit_map,
            "created": time.strftime("%Y-%m-%d"),
            # calibration attestation — bundles must record their own method
            # (Ornith-1.5 shipped jang_config.quantization=null and the whole
            # verification had to be redone against raw weights)
            "quantization": {
                "calibrated": bool(args.imatrix),
                "imatrix": Path(args.imatrix).name if args.imatrix else None,
                "imatrix_refit": bool(args.imatrix),
                "awq_folded": awq_applied,
                "awq_scales": Path(args.awq_scales).name if args.awq_scales else None,
                "hessian_allocation": Path(args.bit_map).name,
                "gptq": False,  # set true by the GPTQ code-rewrite pass
            },
        }
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(report, indent=1))
    print(f"total out: {writer.total/2**30:.2f} GiB in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
