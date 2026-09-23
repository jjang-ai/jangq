"""NemotronLabs VoiceChat 11B -> MXFP8, straight from safetensors.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-20.

Why this exists separately: every other converter in this tree loads the model
through `mlx_vlm.load()` to run it. VoiceChat has **no Swift/our-stack runtime
yet**, and `nemotron_voicechat` only exists in upstream mlx-vlm >= 0.6.15, which
is not our deliverable. MXFP8 is uniform 8-bit and needs NO calibration, so it
can be produced by pure tensor ops with no model class at all — which makes it
the one quant we can ship before the runtime lands, and gives the Swift work a
real quantized artifact to load.

JANG 2/4/6 deliberately are NOT here: they require Hessian + AWQ + imatrix
(`feedback_calibrated_trio_mandatory`), which require forward passes, which
require the runtime.

🚨 TWO TENSORS MUST NEVER BE QUANTIZED — VoiceChat-specific:

  * `tts_model.rvq_embs` [31, 1024, 512] — the residual-VQ **codebook**. These
    are lookup entries, not a projection: quantizing them moves every codebook
    centroid and corrupts decoded audio globally. Only 0.016 B, so protecting
    them is free.
  * `tts_model.audio_prompt_latents.*` [1, 37, 1152] — the **speaker latent**,
    i.e. the entire identity of a voice, ~170 KB. Also the thing custom voices
    are made of, so it must survive byte-exact.

Same class of trap as Ornith's vision `linear_fc2` at in=4304: small tensors
whose corruption is invisible to a structural check and fatal to output.

    python -m jang_tools.convert_voicechat_mxfp8 <src_bf16> <out> [--group-size 32]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx

# Substrings that force fp passthrough regardless of shape.
PROTECTED = (
    "rvq_embs",              # RVQ codebook — lookup table, not a projection
    "audio_prompt_latents",  # speaker identity (and what custom voices ARE)
    "_control_codes",
    "codec_silence_tokens",
    # 🚨 READ RAW, bypassing the module forward. tts.py:136 does
    #     self.proj_mus.weight.reshape(num_predictions, low_rank, -1)[component]
    # so a quantized (packed uint32) weight yields a 4x-too-small last dim:
    # "[matmul] (1,64,288) must match (1,1152,1)". 65536x1152 = 75 M params, the
    # single largest thing we deliberately leave in fp.
    "mog_head.proj_mus",
    # 🚨 tts.py:342 allocates its output buffer with
    #     dtype=self.embed_tokens.weight.dtype
    # On a quantized embedding that is uint32, so float values would be written
    # into an integer buffer and silently truncated. Tiny table (257x1152).
    "embed_subword.embed_tokens",
)

# Lookup tables with very few rows: quantization saves nothing (group stats over
# a handful of rows) and risks everything. e.g. subword_flag_emb.cont_emb is
# [2, 1152] and bos_eos_emb.special_emb is [3, 1152].
MIN_ROWS_TO_QUANTIZE = 64

SHARD_BYTES = 4 * 1024**3


def _is_protected(key: str) -> bool:
    return any(p in key for p in PROTECTED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--bits", type=int, default=8)
    a = ap.parse_args()

    src_files = sorted(a.src.glob("*.safetensors"))
    if not src_files:
        print(f"  no safetensors in {a.src}")
        return 1
    a.out.mkdir(parents=True, exist_ok=True)

    quantized: dict[str, dict] = {}
    n_q = n_pass = n_protected = n_undiv = n_tiny = 0
    out_tensors: dict[str, mx.array] = {}
    t0 = time.time()

    for f in src_files:
        w = mx.load(str(f))
        for k, v in w.items():
            base = k[: -len(".weight")] if k.endswith(".weight") else k

            if _is_protected(k):
                out_tensors[k] = v
                n_protected += 1
                continue
            # 🚨 Only quantize actual Linear WEIGHTS.
            #
            # `ndim >= 2` is NOT sufficient: the Conformer carries bare 2-D
            # parameters such as `self_attn.pos_bias_u` / `pos_bias_v`
            # (n_heads, head_dim). Quantizing those replaces an array with a
            # quantized-module dict and attention fails with
            # "Cannot perform addition on an mlx.core.array and dict".
            # Requiring a `.weight` suffix excludes bare parameters; requiring
            # ndim == 2 excludes 3-D conv kernels, which MLX quantizes
            # differently.
            if not k.endswith(".weight") or v.ndim != 2:
                out_tensors[k] = v
                n_pass += 1
                continue
            if v.shape[-1] % a.group_size != 0:
                # Cannot form whole groups along the input axis.
                out_tensors[k] = v
                n_undiv += 1
                continue
            if v.shape[0] < MIN_ROWS_TO_QUANTIZE:
                out_tensors[k] = v
                n_tiny += 1
                continue

            # NOTE: mxfp8 returns (codes, scales) — TWO values. Its e8m0 group
            # scale needs no bias, unlike affine which returns three. Unpacking
            # three here is a ValueError, which is at least loud; the dangerous
            # version of this mistake is writing an empty `.biases` that a
            # loader then reads as zeros.
            q, s = mx.quantize(v, group_size=a.group_size, bits=a.bits,
                               mode="mxfp8")
            mx.eval(q, s)
            out_tensors[f"{base}.weight"] = q
            out_tensors[f"{base}.scales"] = s
            quantized[base] = {"group_size": a.group_size, "bits": a.bits,
                               "mode": "mxfp8"}
            n_q += 1
        del w
        print(f"  {f.name}: running total {n_q} quantized", flush=True)

    print(f"\n  quantized {n_q} | fp passthrough: {n_pass} non-Linear, "
          f"{n_protected} PROTECTED, {n_undiv} not group-divisible, "
          f"{n_tiny} tiny tables ({time.time()-t0:.0f}s)")

    # ── shard + index ─────────────────────────────────────────────────────
    keys = sorted(out_tensors)
    shards: list[list[str]] = [[]]
    cur = 0
    for k in keys:
        nbytes = out_tensors[k].nbytes
        if cur + nbytes > SHARD_BYTES and shards[-1]:
            shards.append([])
            cur = 0
        shards[-1].append(k)
        cur += nbytes

    weight_map: dict[str, str] = {}
    total = 0
    for i, group in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        payload = {k: out_tensors[k] for k in group}
        mx.save_safetensors(str(a.out / name), payload, metadata={"format": "pt"})
        for k in group:
            weight_map[k] = name
            total += out_tensors[k].nbytes
        print(f"  wrote {name} ({len(group)} tensors)", flush=True)

    (a.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))

    # ── config + sidecars ─────────────────────────────────────────────────
    cfg = json.loads((a.src / "config.json").read_text())
    q = {"group_size": a.group_size, "bits": a.bits, "mode": "mxfp8"}
    for base, spec in quantized.items():
        q[base] = dict(spec)
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    cfg["jang_protected_fp_tensors"] = list(PROTECTED)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for extra in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "README.md"):
        p = a.src / extra
        if p.exists():
            shutil.copy2(p, a.out / extra)
    rt = a.src / "rnnt_tokenizer"
    if rt.is_dir():
        shutil.copytree(rt, a.out / "rnnt_tokenizer", dirs_exist_ok=True)

    print(f"\n  DONE  {a.out}")
    print(f"  weight bytes: {total / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
