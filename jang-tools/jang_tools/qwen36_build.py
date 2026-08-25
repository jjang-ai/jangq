"""Build a Qwen3.6-27B JANG bundle from a MEASURED bit map.

Consumes the allocation produced by `qwen36_allocate.py` (Hessian-trace ranked)
and applies it per-module via `nn.quantize(class_predicate=...)`, which mlx_vlm's
loader mirrors exactly — it reads the same per-module dict back out of
`config.json["quantization"]`.

Why affine and not mxfp4 for the 4-bit tier: measured on real weights, mxfp4@gs32
and affine4@gs128 are both 4.25 bpw but mxfp4's relative error is ~15 % higher,
with no decode speed advantage. See docs/internal/qwen36-27b-prep/
02-CALIBRATION-RESULTS.md. `class_predicate` *can* return a per-module `mode`,
so switching later is a one-line change if a future kernel makes MX win.

    python -m jang_tools.qwen36_build <src> <bitmap.json> <out_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Files that must ride along for vision + video + tokenisation to work.
SIDECARS = [
    "chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "generation_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "configuration.json", "LICENSE",
]


def main(argv) -> int:
    if len(argv) < 4:
        print(__doc__)
        return 1
    src, bitmap_p, out = Path(argv[1]), Path(argv[2]), Path(argv[3])
    mode = "affine"
    awq_calib, awq_alpha = None, 0.25
    for i, a in enumerate(argv):
        if a == "--mode":
            mode = argv[i + 1]
        # AWQ is opt-in and needs the capture; without --awq-calib nothing
        # about the existing behaviour changes.
        if a == "--awq-calib":
            awq_calib = argv[i + 1]
        if a == "--awq-alpha":
            awq_alpha = float(argv[i + 1])
    plan = json.loads(bitmap_p.read_text())
    gs = plan["group_size"]
    bits_by_path = {m["path"]: m["bits"] for m in plan["modules"]}

    from mlx_vlm import load
    from mlx_vlm.utils import save_config, save_weights

    print(f"  loading {src.name} ...", flush=True)
    t0 = time.time()
    model, proc = load(str(src))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # ── AWQ (opt-in): salient-channel scaling, absorbed into the producing
    # norm. MUST run while the model is still sanitized — the in-memory norm
    # is (stored + 1), and the un-sanitize step below writes back
    # (stored + 1)/s - 1, which is the verified absorption formula.
    awq_info = None
    if awq_calib:
        from .ornith_awq_fold import apply_awq
        # Scales are written NEXT TO THE CALIB (not into the bundle): a stray
        # .safetensors inside a bundle is picked up by anything that globs the
        # directory. qwen36_imatrix_refit MUST be given this path or it will
        # revert W*s to W from source while the norms stay folded.
        out.mkdir(parents=True, exist_ok=True)
        scales_p = str(Path(awq_calib).with_suffix("")) + f"-awq-{out.name}.safetensors"
        awq_info = apply_awq(model, awq_calib, alpha=awq_alpha,
                             scales_out=scales_p)

    applied: dict[str, dict] = {}
    skipped: list[str] = []

    def predicate(path: str, module):
        b = bits_by_path.get(path)
        if b is None:
            # Embedding isn't in the capture (not an nn.Linear). Give the
            # token embedding the same floor as the untied lm_head rather than
            # leaving it at source precision or guessing low.
            if isinstance(module, nn.Embedding):
                # mxfp8 only supports bits=8; affine keeps the 4-bit floor
                eb = 8 if mode == "mxfp8" else 4
                spec = {"group_size": gs, "bits": eb, "mode": mode}
                applied[path] = spec
                return spec
            skipped.append(path)
            return False
        if b >= 16:
            skipped.append(path)      # fp16 passthrough (e.g. in_features 4304)
            return False
        if getattr(module, "weight", None) is not None:
            in_f = module.weight.shape[-1]
            if in_f % gs != 0:
                skipped.append(path)
                return False
        spec = {"group_size": gs, "bits": int(b), "mode": mode}
        applied[path] = spec
        return spec

    print("  quantizing with measured bit map ...", flush=True)
    t0 = time.time()
    nn.quantize(model, group_size=gs, bits=plan["base_bits"],
                mode=mode, class_predicate=predicate)
    mx.eval(model.parameters())
    print(f"  quantized {len(applied)} modules, {len(skipped)} left at source "
          f"precision ({time.time()-t0:.1f}s)", flush=True)

    import collections
    dist = collections.Counter(v["bits"] for v in applied.values())
    print(f"  bit distribution: {dict(sorted(dist.items()))}")

    out.mkdir(parents=True, exist_ok=True)

    # config with per-module overrides, in the exact shape mlx_vlm reads back
    cfg = json.loads((src / "config.json").read_text())
    q = {"group_size": gs, "bits": plan["base_bits"], "mode": mode}
    for path, spec in applied.items():
        q[path] = dict(spec)
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    if awq_info:
        # Provenance: AWQ is folded into the weights + norms, so it leaves no
        # sidecar. Record it or the bundle is indistinguishable from a non-AWQ
        # build.
        cfg["awq"] = dict(awq_info, folded_into="producing RMSNorm",
                          formula="(stored+1)/s - 1")

    # ── UNDO sanitize() before saving ────────────────────────────────────
    # mlx_vlm's qwen3_5 sanitize() is NOT idempotent: it adds +1.0 to every
    # 1-D norm weight and moveaxis(2,1) on conv1d. load() already applied both,
    # so saving the in-memory params would bake them in and the NEXT load would
    # apply them a SECOND time — norms end up +2, conv1d transposed back.
    # Measured: bundle norm mean was exactly source + 1.0000, and conv1d shape
    # flipped (10240,1,4) -> (10240,4,1). Output was garbage while every Linear
    # weight verified correct, which is what makes this bug so easy to miss.
    # Bundles must store SOURCE convention; the runtime owns the shift.
    NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                     "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
    from mlx.utils import tree_flatten, tree_unflatten
    flat = dict(tree_flatten(model.parameters()))
    n_norm = n_conv = 0
    for k, v in list(flat.items()):
        if v is None or not hasattr(v, "ndim"):
            continue
        if any(k.endswith(sfx) for sfx in NORM_SUFFIXES) and v.ndim == 1:
            flat[k] = v - 1.0
            n_norm += 1
        elif "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] == 1:
            flat[k] = v.moveaxis(1, 2)
            n_conv += 1
    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    print(f"  un-sanitized before save: {n_norm} norms (-1.0), "
          f"{n_conv} conv1d (moveaxis back)", flush=True)

    print("  saving weights ...", flush=True)
    save_weights(out, model, donate_weights=True)

    save_config(cfg, out / "config.json")

    for f in SIDECARS:
        s = src / f
        if s.exists():
            shutil.copy(s, out / f)

    total = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"\n  DONE  {out}")
    print(f"  weight bytes: {total/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
