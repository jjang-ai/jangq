"""Inkling-Small -> JANGTQ bundle (routed experts in Lloyd-Max codebook TQ).

Why this exists: MLX affine cannot hold this model at ~90 GB. Measured on real
activations, expert-output cosine vs fp32 at layer 2/26:

    affine 2/3 gs128  (2.58 bits)  0.777 / 0.793   <- 89.7 GB bundle = word salad
    affine 3/3 gs128  (3.25 bits)  0.928 / 0.930   <- 111 GB
    JANGTQ tq2/tq2    (2.00 bits)  0.799 / 0.811
    JANGTQ tq2/tq4    (2.67 bits)  0.851 / 0.860   <- this build, ~92.5 GB
    JANGTQ tq4/tq4    (4.00 bits)  0.982           <- ~135 GB

AWQ was evaluated on top and moves this by +0.001..+0.006 — Inkling's MoE-input
activations only show a 3.5x max/median channel spread, so there is little
saliency for AWQ to exploit. Not applied.

Layout (matches what `load_jangtq.py` already expects):
    routed experts -> `...mlp.switch_mlp.{gate_proj,up_proj,down_proj}.tq_packed`
                      + `.tq_norms` + `.tq_bits`
    everything else -> affine 8-bit gs64, or fp16 passthrough

`w13_weight` is the fused [E, 2*inter, hidden] gate+up with native rows ordered
`g0,u0,g1,u1,...`; it is de-interleaved BEFORE quantization so gate and up can
carry different widths later if needed.

Usage:
    python -m jang_tools.convert_inkling_jangtq <src> <out> [--w13-bits 2] [--w2-bits 4]
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open

from .convert_inkling_jang_affine import repair_inkling_bundle_metadata
from .turboquant.linear import tq_quantize_weight

SEED = 42
SHARD_BYTES = 4_000_000_000

# fp16 passthrough: 1-D things, the router, the short convs, the rel-bias bank.
PASSTHROUGH_SUBSTR = ("_norm.weight", "norm.weight", "_sconv.weight",
                      "rel_logits_proj.proj", "global_scale", ".gate.bias",
                      ".gate.weight")
TOWER_PREFIX = ("model.visual", "model.audio")


def _is_passthrough(name: str, shape) -> bool:
    if len(shape) == 1:
        return True
    if any(s in name for s in PASSTHROUGH_SUBSTR):
        return True
    if name.startswith(TOWER_PREFIX):
        return True          # vision/audio towers stay fp16 (69M params, 139 MB)
    return False


def _load(f, name, shape):
    t = f.get_tensor(name)
    return np.asarray(t.to(__import__("torch").float32).numpy()
                      if hasattr(t, "to") else t, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--w13-bits", type=int, default=2, choices=[2, 4])
    ap.add_argument("--w2-bits", type=int, default=4, choices=[2, 4])
    ap.add_argument("--attn-bits", type=int, default=8)
    a = ap.parse_args()
    SRC, OUT = Path(a.src), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((SRC / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    inter = tc.get("intermediate_size", 2048)
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]

    print(f"  src {SRC}\n  out {OUT}")
    print(f"  routed experts: gate/up tq{a.w13_bits}, down tq{a.w2_bits}; "
          f"everything else affine {a.attn_bits}-bit gs64")

    shard, shard_bytes, shard_idx = {}, 0, 0
    weight_map, quant_meta = {}, {}
    counts = {"tq": 0, "affine": 0, "fp16": 0, "fp32": 0, "skipped": 0}

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        fn = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        mx.save_safetensors(str(OUT / fn), {k: mx.array(v) for k, v in shard.items()})
        for k in shard:
            weight_map[k] = fn
        print(f"    shard {shard_idx}: {len(shard)} tensors, {shard_bytes/1e9:.2f} GB", flush=True)
        shard, shard_bytes = {}, 0

    def add(name, arr):
        nonlocal shard_bytes
        shard[name] = arr
        shard_bytes += arr.nbytes
        if shard_bytes >= SHARD_BYTES:
            flush()

    def tq_stack(w3d, bits, out_base):
        """TQ-quantize a stacked [E, out, in] expert tensor, expert by expert."""
        packed, norms = [], []
        for e in range(w3d.shape[0]):
            r = tq_quantize_weight(np.ascontiguousarray(w3d[e]), bits=bits, seed=SEED)
            packed.append(r["packed"]); norms.append(r["norms"])
        add(f"{out_base}.tq_packed", np.stack(packed))
        add(f"{out_base}.tq_norms", np.stack(norms))
        add(f"{out_base}.tq_bits", np.array([bits], dtype=np.uint8))
        quant_meta[out_base] = {"mode": "mxtq", "bits": bits, "seed": SEED}
        counts["tq"] += 1

    t0 = time.time()
    names = [n for n in idx if not (n.startswith("model.mtp") or ".mtp." in n)]
    # experts last: they dominate runtime, so the cheap tensors land early
    names.sort(key=lambda n: ("experts" in n, n))
    files = {}
    for i, name in enumerate(names):
        fn = idx[name]
        if fn not in files:
            files[fn] = safe_open(str(SRC / fn), framework="pt")
        f = files[fn]
        t = f.get_tensor(name)
        shape = tuple(t.shape)

        if _is_passthrough(name, shape):
            torch = __import__("torch")
            # The routed correction bias and learned global scale are FP32 in
            # the source. The bias controls top-k selection and must not be
            # rounded to fp16/bf16.
            passthrough_dtype = torch.float32 if t.dtype == torch.float32 else torch.float16
            arr = t.to(passthrough_dtype).numpy()
            if name.endswith("_sconv.weight") and arr.ndim == 3 and arr.shape[-1] != 1:
                arr = np.ascontiguousarray(arr.transpose(0, 2, 1))   # (C,1,K)->(C,K,1)
            add(name, arr)
            counts["fp32" if passthrough_dtype == torch.float32 else "fp16"] += 1
        elif name.endswith("mlp.experts.w13_weight"):
            w = t.to(__import__("torch").float32).numpy()
            base = name[: -len("experts.w13_weight")]
            paired = w.reshape(w.shape[0], inter, 2, w.shape[-1])
            tq_stack(np.ascontiguousarray(paired[:, :, 0, :]), a.w13_bits,
                     base + "switch_mlp.gate_proj")
            tq_stack(np.ascontiguousarray(paired[:, :, 1, :]), a.w13_bits,
                     base + "switch_mlp.up_proj")
            del w
        elif name.endswith("mlp.experts.w2_weight"):
            w = t.to(__import__("torch").float32).numpy()
            base = name[: -len("experts.w2_weight")]
            tq_stack(w, a.w2_bits, base + "switch_mlp.down_proj")
            del w
        else:
            w = mx.array(t.to(__import__("torch").float32).numpy())
            qw, qs, qb = mx.quantize(w, group_size=64, bits=a.attn_bits)
            base = name[:-len(".weight")] if name.endswith(".weight") else name
            add(f"{base}.weight", np.array(qw))
            add(f"{base}.scales", np.array(qs).astype(np.float16))
            add(f"{base}.biases", np.array(qb).astype(np.float16))
            quant_meta[base] = {"mode": "affine", "bits": a.attn_bits, "group_size": 64}
            counts["affine"] += 1
            del w, qw, qs, qb

        del t
        if i % 100 == 0:
            gc.collect()
            print(f"  [{i}/{len(names)}] {counts}  {time.time()-t0:.0f}s", flush=True)

    flush()
    for i in range(1, shard_idx + 1):
        (OUT / f"model-{i:05d}-of-XXXXX.safetensors").rename(
            OUT / f"model-{i:05d}-of-{shard_idx:05d}.safetensors")
    weight_map = {k: v.replace("XXXXX", f"{shard_idx:05d}") for k, v in weight_map.items()}

    total = sum((OUT / f).stat().st_size for f in set(weight_map.values()))
    (OUT / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))

    out_cfg = dict(cfg)
    out_cfg["quantization"] = quant_meta
    out_cfg["weight_format"] = "mxtq"
    (OUT / "config.json").write_text(json.dumps(out_cfg, indent=1))
    (OUT / "jang_config.json").write_text(json.dumps({
        "format": "jangtq", "format_version": 2, "weight_format": "mxtq",
        "source_model": str(SRC), "seed": SEED,
        "preserve_source_fp32_controls": True,
        "routed_expert_bits": {"gate_proj": a.w13_bits, "up_proj": a.w13_bits,
                               "down_proj": a.w2_bits},
        "mxtq_bits": a.w13_bits,
        "bundle_has_mtp": False, "mtp_layers": [],
    }, indent=1))
    for extra in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "processor_config.json",
                  "generation_config.json"):
        if (SRC / extra).is_file():
            shutil.copy2(SRC / extra, OUT / extra)
    if (SRC / "tiktoken").is_dir():
        shutil.copytree(SRC / "tiktoken", OUT / "tiktoken", dirs_exist_ok=True)

    metadata_result = repair_inkling_bundle_metadata(OUT)
    print(
        "  metadata: "
        f"eos={metadata_result['eos_token_id']} "
        f"reasoning={metadata_result['reasoning_parser']} "
        f"tools={metadata_result['tool_call_parser']} "
        f"default_effort={metadata_result['default_reasoning_effort']}",
        flush=True,
    )

    print(f"\n  done in {(time.time()-t0)/60:.1f} min — {counts}")
    print(f"  bundle: {total/1e9:.2f} GB / {total/2**30:.1f} GiB in {shard_idx} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
