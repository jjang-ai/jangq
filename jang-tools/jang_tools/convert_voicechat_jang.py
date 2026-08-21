"""VoiceChat 11B -> JANG (Hessian-allocated + AWQ + imatrix), from safetensors.

Created by Jinho Jang (eric@osaurus.ai) — 2026-08-20.

All three mandated methods, driven by the ONE capture in
`examples/voicechat/calibrate.py` (562 modules, 3.38 M row-samples over real
conversational audio, both directions exercised):

  * Hessian   tr(H) = sum_c E[x_c^2]; score = tr(H) * ||W||_F^2 -> bit map
  * AWQ       salient-channel scaling, absorbed into the producing RMSNorm
  * imatrix   activation-weighted affine fit instead of RTN codes

MEASURED sensitivity (see docs 04), which drives the floors below:

    TTS subword enc   mean 169,885  max 1,281,301   <- MOST sensitive in model
    lm_head            mean  22,157
    function_head      mean  22,157                 <- IDENTICAL to lm_head
    LLM backbone       mean  18,397  max   539,633  (late layers dominate)
    TTS backbone       mean   2,380
    perception         mean     934  max   140,123  (pre_encode.out outlier)
    MoG head           mean     411
    RNN-T              mean      61

🚨 PROTECTED tensors (never quantized) are inherited from
`convert_voicechat_mxfp8.PROTECTED` — RVQ codebook, speaker latents,
`mog_head.proj_mus` (read raw), `embed_subword.embed_tokens` (dtype read).
See docs 04 §2 for why each one breaks.

    python -m jang_tools.convert_voicechat_jang <src_bf16> <out> <calib.json> \
        <calib.safetensors> --base-bits 4 [--group-size 32] [--awq-alpha 0.25]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from .affine import quantize_imatrix_affine_numpy
from .convert_voicechat_mxfp8 import PROTECTED, MIN_ROWS_TO_QUANTIZE, _is_protected

SHARD_BYTES = 4 * 1024**3

# Bit floors by component. Derived from the MEASURED traces above, not names.
FLOORS = {
    "embed_subword": 8,          # most sensitive component; only 8 modules
    # The backbone's OWN token embedding, which the duplex loop feeds back into
    # itself every frame: fused = embed(prev_text) + audio + w*embed(prev_func).
    # At 2 bits the model cannot tell "I just emitted PAD" from "I emitted <s>",
    # so it never transitions into speech — JANG_2 transcribed perfectly and
    # chose <PAD> on 61/62 frames at p=1.000. lm_head was already floored; this
    # is its input-side twin and was missed.
    "stt_model.embed_tokens": 8,
    "pre_encode.out": 8,         # perception bottleneck, 140k trace outlier
    "lm_head": 6,
    "function_head": 6,          # SAME as lm_head — identical measured trace
    "rnnt": 6,                   # tiny; no reason to squeeze
}
# Late LLM layers carry the largest backbone traces.
LATE_LLM_FROM = 46
LATE_LLM_BONUS = 1


def _is_offset_norm(norm_key: str) -> bool:
    """True when this norm applies (1.0 + weight) rather than weight.

    TTS uses OffsetRMSNorm (Gemma-style); the nemotron_h backbone uses plain
    nn.RMSNorm. Getting this backwards silently kills text generation.
    """
    return norm_key.startswith("tts_model.")


def _norm_for(path: str) -> str | None:
    """The RMSNorm whose output feeds this Linear, for AWQ absorption."""
    if ".llm.layers." in path:
        pre = path.split(".mixer.")[0]
        # every nemotron_h block: layers.N.norm -> layers.N.mixer.*
        if ".mixer." in path and any(
            path.endswith(s) for s in
            ("in_proj.weight", "q_proj.weight", "k_proj.weight",
             "v_proj.weight", "up_proj.weight")
        ):
            return f"{pre}.norm"
        return None
    if "tts_model.backbone.layers." in path:
        pre = path.split(".self_attn.")[0].split(".mlp.")[0]
        if ".self_attn." in path and path.endswith(
            ("q_proj.weight", "k_proj.weight", "v_proj.weight")
        ):
            return f"{pre}.input_layernorm"
        if ".mlp." in path and path.endswith(("gate_proj.weight", "up_proj.weight")):
            return f"{pre}.pre_feedforward_layernorm"
        return None
    return None


def _floor_for(path: str, base: int, llm_floor: int = 0) -> int:
    b = base
    # The turn-taking decision (<s> vs <PAD>) is a narrow margin in the
    # backbone, and at 2 bits it collapses permanently to <PAD>: the model
    # transcribes fine and never speaks. Measured on JANG_2, <PAD> won 61/62
    # frames at p=1.000 with perfectly healthy logits, while the 4-bit build
    # broke out at frame 30 on <s> 0.643 vs <PAD> 0.357. `--llm-floor` raises
    # the backbone so that margin survives.
    if llm_floor and ".llm.layers." in path:
        b = max(b, llm_floor)
    for key, f in FLOORS.items():
        if key in path:
            b = max(b, f)
    if ".llm.layers." in path:
        try:
            li = int(path.split(".layers.")[1].split(".")[0])
            if li >= LATE_LLM_FROM:
                b = max(b, min(8, base + LATE_LLM_BONUS))
        except (IndexError, ValueError):
            pass
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("calib_json", type=Path)
    ap.add_argument("calib_st", type=Path)
    ap.add_argument("--base-bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--awq-alpha", type=float, default=0.25)
    ap.add_argument("--llm-floor", type=int, default=0,
                    help="minimum bits for .llm.layers.* (0 = leave as base)")
    ap.add_argument("--floor", action="append", default=[], metavar="SUBSTR=BITS",
                    help="raise the floor for any module path containing SUBSTR")
    a = ap.parse_args()

    stats = json.loads(a.calib_json.read_text())["stats"]
    second = {k[: -len(".second_moment")]: v
              for k, v in load_file(str(a.calib_st)).items()
              if k.endswith(".second_moment")}
    print(f"  calib: {len(stats)} modules")

    # ── load all weights ──────────────────────────────────────────────────
    W: dict[str, mx.array] = {}
    for f in sorted(a.src.glob("*.safetensors")):
        W.update(mx.load(str(f)))
    print(f"  weights: {len(W)} tensors")

    # ── which tensors are quantizable at all ──────────────────────────────
    def quantizable(k, v):
        if _is_protected(k):
            return False
        if not k.endswith(".weight") or v.ndim != 2:
            return False
        if v.shape[-1] % a.group_size != 0:
            return False
        if v.shape[0] < MIN_ROWS_TO_QUANTIZE:
            return False
        return True

    cands = [k for k, v in W.items() if quantizable(k, v)]
    print(f"  quantizable: {len(cands)}")

    # ── AWQ: one scale per producing norm, shared by all its consumers ─────
    groups: dict[str, list[str]] = {}
    for k in cands:
        nrm = _norm_for(k)
        if nrm and f"{nrm}.weight" in W:
            groups.setdefault(nrm, []).append(k)

    awq_scales: dict[str, np.ndarray] = {}
    n_folded = 0
    for nrm, consumers in groups.items():
        mods = [k[: -len(".weight")] for k in consumers]
        sm = [second[m] for m in mods if m in second]
        if not sm:
            continue
        mean_sq = np.mean(np.stack(sm, 0), 0)
        nk = f"{nrm}.weight"
        if mean_sq.shape[0] != W[nk].shape[0]:
            continue
        mag = np.sqrt(np.maximum(mean_sq, 0.0)) + 1e-8
        s = np.power(mag, a.awq_alpha)
        s = (s / np.exp(np.mean(np.log(s)))).clip(1e-2, 1e2).astype(np.float32)
        s_mx = mx.array(s)
        for k in consumers:
            W[k] = (W[k].astype(mx.float32) * s_mx).astype(W[k].dtype)
            awq_scales[k] = s
            n_folded += 1
        # 🚨 VoiceChat uses TWO DIFFERENT RMSNorm CONVENTIONS in one model:
        #
        #   LLM backbone (stt_model.llm.*) -> mlx nn.RMSNorm
        #       applies  weight            -> fold as   w / s
        #   TTS side     (tts_model.*)     -> OffsetRMSNorm (Gemma-style,
        #       tts.py:26-37) applies 1.0 + weight
        #                                  -> fold as  (w + 1)/s - 1
        #
        # Applying the offset formula to the plain norms corrupts every LLM
        # layer. MEASURED: doing so produced `agent text: ''` (empty) with
        # rms 0.0007 vs 0.0112 — while ASR still worked, rel-err read a healthy
        # 0.0488, and NO tensor contained NaN or Inf. Nothing structural
        # flagged it; only running the model did.
        wn = W[nk].astype(mx.float32)
        if _is_offset_norm(nk):
            W[nk] = (((wn + 1.0) / s_mx) - 1.0).astype(W[nk].dtype)
        else:
            W[nk] = (wn / s_mx).astype(W[nk].dtype)
    print(f"  AWQ alpha={a.awq_alpha}: folded {len(groups)} norm groups / "
          f"{n_folded} linears "
          f"| s min={min((v.min() for v in awq_scales.values()), default=1):.3f} "
          f"max={max((v.max() for v in awq_scales.values()), default=1):.3f}")

    # ── Hessian-scored bit allocation ─────────────────────────────────────
    bits_of: dict[str, int] = {}
    for k in cands:
        m = k[: -len(".weight")]
        st = stats.get(m)
        b = _floor_for(k, a.base_bits, a.llm_floor)
        for spec in a.floor:
            sub, _, bits = spec.partition("=")
            if sub in k:
                b = max(b, int(bits))
        if st is not None:
            w = np.array(W[k].astype(mx.float32))
            fro = float((w * w).sum())
            score = st["trace"] * fro
            # top-decile modules by score get +1 bit (capped at 8)
            bits_of[k] = (b, score)
        else:
            bits_of[k] = (b, 0.0)
    scored = sorted((v[1] for v in bits_of.values()), reverse=True)
    cutoff = scored[max(len(scored) // 10 - 1, 0)] if scored else 0.0
    # 🚨 The imatrix fit supports {2,3,4,5,6,8} only — NOT 7. Landing on 7
    # raises mid-build after AWQ has already been folded into the weights,
    # which would otherwise waste the whole pass. Round 7 UP to 8: these are
    # top-decile-sensitivity modules, so rounding down would be the wrong way.
    SUPPORTED = (2, 3, 4, 5, 6, 8)

    def _snap(b: int) -> int:
        return b if b in SUPPORTED else min(
            (s for s in SUPPORTED if s >= b), default=8)

    final_bits = {}
    for k, (b, score) in bits_of.items():
        raw = min(8, b + 1) if (score >= cutoff and score > 0) else b
        final_bits[k] = _snap(raw)

    import collections
    print(f"  bit distribution: {dict(sorted(collections.Counter(final_bits.values()).items()))}")

    # ── quantize with imatrix fit ─────────────────────────────────────────
    out_t: dict[str, mx.array] = {}
    qmap: dict[str, dict] = {}
    errs = []
    t0 = time.time()
    done = 0
    for k, v in W.items():
        if k not in final_bits:
            out_t[k] = v
            continue
        base = k[: -len(".weight")]
        bits = final_bits[k]
        w = np.array(v.astype(mx.float32))
        imp = second.get(base)
        if imp is not None and imp.shape[0] == w.shape[-1]:
            packed, sc, bi, werr = quantize_imatrix_affine_numpy(
                w, imp.astype(np.float32), bits=bits, group_size=a.group_size)
            errs.append(werr)
            out_t[f"{base}.weight"] = mx.array(packed)
            out_t[f"{base}.scales"] = mx.array(sc).astype(mx.bfloat16)
            out_t[f"{base}.biases"] = mx.array(bi).astype(mx.bfloat16)
        else:
            q, sc, bi = mx.quantize(mx.array(w), group_size=a.group_size,
                                    bits=bits, mode="affine")
            mx.eval(q, sc, bi)
            out_t[f"{base}.weight"] = q
            out_t[f"{base}.scales"] = sc.astype(mx.bfloat16)
            out_t[f"{base}.biases"] = bi.astype(mx.bfloat16)
        qmap[base] = {"group_size": a.group_size, "bits": bits, "mode": "affine"}
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{len(final_bits)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  quantized {done} modules in {time.time()-t0:.0f}s")
    if errs:
        print(f"  mean weighted rel-err after imatrix fit: {float(np.mean(errs)):.4f}")

    # ── shard + write ─────────────────────────────────────────────────────
    a.out.mkdir(parents=True, exist_ok=True)
    keys = sorted(out_t)
    shards: list[list[str]] = [[]]
    cur = 0
    for k in keys:
        nb = out_t[k].nbytes
        if cur + nb > SHARD_BYTES and shards[-1]:
            shards.append([]); cur = 0
        shards[-1].append(k); cur += nb
    wmap, total = {}, 0
    for i, grp in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(a.out / name), {k: out_t[k] for k in grp},
                            metadata={"format": "pt"})
        for k in grp:
            wmap[k] = name; total += out_t[k].nbytes
        print(f"  wrote {name} ({len(grp)} tensors)")
    (a.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": wmap}, indent=2))

    cfg = json.loads((a.src / "config.json").read_text())
    q = {"group_size": a.group_size, "bits": a.base_bits, "mode": "affine"}
    q.update({k: dict(v) for k, v in qmap.items()})
    cfg["quantization"] = q
    cfg["quantization_config"] = q
    cfg["jang_protected_fp_tensors"] = list(PROTECTED)
    cfg["jang_calibration"] = {
        "methods": ["hessian_trace_allocation", "awq_norm_fold", "imatrix_refit"],
        "awq_alpha": a.awq_alpha, "awq_groups": len(groups),
        "awq_linears": n_folded, "base_bits": a.base_bits,
    }
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for extra in ("tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "README.md"):
        p = a.src / extra
        if p.exists():
            shutil.copy2(p, a.out / extra)
    rt = a.src / "rnnt_tokenizer"
    if rt.is_dir():
        shutil.copytree(rt, a.out / "rnnt_tokenizer", dirs_exist_ok=True)

    print(f"\n  DONE  {a.out}\n  weight bytes: {total/2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
