"""Activation-weighted (imatrix) requantization of DSV4.1-1017 routed experts.

For each routed expert projection, the per-input-channel importance imp_c = mean
x_c^2 (from calib_stats) weights an AWQ-style per-group clip search: each group of
64 input channels is clipped to [center +/- alpha*halfrange] for the alpha that
minimises the importance-weighted dequant error, then the clipped weights are
affine-quantized with MLX's own packer (so the bundle format is unchanged). This
is the sound, hardware-fitting subset of "awq/gptq/imatrix" for this model:

  - imatrix clip search is applied to gate/up/down of every routed expert;
  - plain AWQ scale-folding does NOT apply cleanly here (gate/up share ffn_norm
    with the router, so a folded per-channel scale would change routing; down's
    input passes the SwiGLU +/-10 clamp, which breaks the fold) -- so we clip
    rather than fold;
  - GPTQ error-feedback needs per-expert input covariance (XX^T, ~40 GB/layer
    for all experts at once) and a separate capture; left as a follow-up.

Experts with too few calibration tokens fall back to plain affine (uniform imp).
Non-routed tensors, Engram and metadata are copied from the RTN bundle unchanged.

  python -m jang_tools.dsv41.calib_requant --src <source> --rtn <rtn-bundle> \
      --stats calib/dsv41-1017/stats.npz --dst <out-bundle>
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from jang_tools.dsv41.convert_dsv41_jang import GROUP_SIZE, PROJ, default_plan
from jang_tools.dsv41.st_io import Shards, dequant_fp4

ALPHAS = mx.array(np.linspace(1.0, 0.5, 11, dtype=np.float32))  # 1.0 = no clip
PLAIN_ALPHAS = mx.array([1.0], dtype=np.float32)  # --plain: alpha=1 only == plain affine (RTN)


def imatrix_affine(w: mx.array, imp: mx.array, bits: int, alphas: mx.array = ALPHAS):
    """Clip each group to the importance-optimal range, then MLX-affine quantize.

    w [out, in]; imp [in] (>=0, per input channel). For every clip strength alpha the
    TRUE packed+dequantized error (so the search sees MLX's rounding and the bf16 cast
    exactly) is scored per group with the importance weights; the best alpha per group
    is kept. alpha=1.0 is plain affine, so the result never loses to it. Returns
    (packed, scales, biases) in MLX's affine layout plus the chosen clipped weights."""
    out, din = w.shape
    ng = din // GROUP_SIZE
    wf = w.astype(mx.float32)
    wg = wf.reshape(out, ng, GROUP_SIZE)
    impg = imp.reshape(1, ng, GROUP_SIZE) + 1e-12
    gmin = mx.min(wg, axis=0, keepdims=True)  # [1,ng,1->G broadcast]
    gmax = mx.max(wg, axis=0, keepdims=True)
    center = (gmin + gmax) / 2
    half = (gmax - gmin) / 2
    best_err = best_pk = best_sc = best_bi = None
    for ai in range(ALPHAS.shape[0]):
        a = ALPHAS[ai]
        lo = center - a * half
        hi = center + a * half
        wc = mx.clip(wg, lo, hi).reshape(out, din).astype(mx.bfloat16)
        pk, sc, bi = mx.quantize(wc, group_size=GROUP_SIZE, bits=bits)
        deq = mx.dequantize(pk, sc, bi, group_size=GROUP_SIZE, bits=bits).astype(mx.float32).reshape(out, ng, GROUP_SIZE)
        err = mx.sum(impg * (wg - deq) ** 2, axis=(0, 2), keepdims=True)  # [1,ng,1] per group
        if best_err is None:
            best_err, best_pk, best_sc, best_bi = err, pk, sc, bi
        else:
            take = err < best_err  # [1,ng,1]
            best_err = mx.where(take, err, best_err)
            tk = take[0, :, 0]  # [ng]; pk is [out, ng*wpg], sc/bi are [out, ng]
            wpg = pk.shape[1] // ng
            best_pk = mx.where(tk[None, :, None], pk.reshape(out, ng, wpg), best_pk.reshape(out, ng, wpg)).reshape(pk.shape)
            best_sc = mx.where(tk[None, :], sc, best_sc)
            best_bi = mx.where(tk[None, :], bi, best_bi)
    return best_pk, best_sc.astype(mx.bfloat16), best_bi.astype(mx.bfloat16), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="original FP4/FP8 source")
    ap.add_argument("--rtn", required=True, help="RTN JANG bundle to copy non-routed tensors from")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--plan", help="JSON {proj:{layer:bits}}; default = RTN plan")
    ap.add_argument("--min-tokens", type=int, default=64, help="below this, an expert uses uniform importance")
    ap.add_argument("--plain", action="store_true",
                    help="plain affine (RTN): restrict the clip search to alpha=1 (a no-op clip), i.e. no imatrix. "
                         "Bit-identical to convert_dsv41_jang's routed quantization; stats are ignored.")
    ap.add_argument("--supersede", action="store_true",
                    help="disk-safe in-place: hardlink non-routed shards from --rtn, and delete each RTN routed "
                         "shard right after its imatrix replacement is written (consumes the RTN bundle; eval it first)")
    a = ap.parse_args()

    src = Path(a.src).expanduser()
    rtn = Path(a.rtn).expanduser()
    dst = Path(a.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    st = np.load(a.stats, allow_pickle=True)
    meta = json.loads(str(st["meta"]))
    # Engram shards (47/48) were moved into the RTN bundle at RTN-build time, so they are
    # absent from the source tree; the requant only reads routed experts (main FP4 shards),
    # so allow_missing skips the Engram shards instead of failing on them.
    shards = Shards(src, allow_missing=True)
    text = json.loads((src / "config.json").read_text())["text_config"]
    n_layers, n_experts = text["num_hidden_layers"], text["n_routed_experts"]
    from jang_tools.dsv41.convert_dsv41_jang import load_plan

    plan = load_plan(a.plan, n_layers)
    # importance per (layer, expert, channel): mean x^2, with per-channel floor
    gu_imp = st["gu_sq"] / np.maximum(st["gu_cnt"][..., None], 1.0)  # [L,E,D]
    dn_imp = st["dn_sq"] / np.maximum(st["gu_cnt"][..., None], 1.0)  # [L,E,F] (down shares the routed count)
    cnt = st["gu_cnt"]

    index = json.loads((rtn / "model.safetensors.index.json").read_text())["weight_map"]
    # copy every non-routed shard file from the RTN bundle verbatim (hardlink under --supersede: 0 extra bytes)
    routed_prefixes = tuple(f"layers.{l}.ffn.experts." for l in range(n_layers))
    keep_files = sorted({f for t, f in index.items() if not t.startswith(routed_prefixes)})
    # layer -> the RTN routed shard file holding that layer's experts (for --supersede deletion)
    rtn_routed_file = {}
    for t, f in index.items():
        if t.startswith(routed_prefixes):
            rtn_routed_file[int(t.split(".")[1])] = f
    dst_is_separate = dst.resolve() != rtn.resolve()
    new_map = {}
    for f in keep_files:
        if not (dst / f).exists():
            if a.supersede:
                _link_or_copy(str(rtn / f), str(dst / f))
            else:
                shutil.copy2(rtn / f, dst / f)
        for t, ff in index.items():
            if ff == f:
                new_map[t] = f
    for fn in ("config.json", "jang_config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (rtn / fn).exists():
            shutil.copy2(rtn / fn, dst / fn)
    for sub in ("encoding", "engram"):
        if (rtn / sub).exists() and not (dst / sub).exists():
            if (rtn / sub).is_symlink():
                (dst / sub).symlink_to((rtn / sub).resolve())
            else:
                shutil.copytree(rtn / sub, dst / sub, copy_function=_link_or_copy)

    t0 = time.time()
    shard_i = max([int(f.split("-")[1]) for f in keep_files], default=0)
    report = {}
    for layer in range(n_layers):
        out = {}
        for sp, proj in PROJ.items():
            bits = plan[proj][layer]
            imp_all = (dn_imp if proj == "down_proj" else gu_imp)[layer]  # [E, d]
            names = [f"layers.{layer}.ffn.experts.{e}.{sp}.weight" for e in range(n_experts)]
            pks, scs, bis = [], [], []
            err_num = err_den = 0.0
            for e in range(n_experts):
                w = shards.dequant(names[e], mx.bfloat16)  # [out,in]
                if cnt[layer, e] >= a.min_tokens:
                    imp = mx.array(imp_all[e])
                else:
                    imp = mx.ones((w.shape[1],), mx.float32)
                pk, sc, bi, wc = imatrix_affine(w, imp, bits, PLAIN_ALPHAS if a.plain else ALPHAS)
                pks.append(pk); scs.append(sc); bis.append(bi)
                if e < 8:  # cheap fidelity probe
                    deq = mx.dequantize(pk, sc, bi, group_size=GROUP_SIZE, bits=bits)
                    err_num += float(mx.sum((deq.astype(mx.float32) - w.astype(mx.float32)) ** 2))
                    err_den += float(mx.sum(w.astype(mx.float32) ** 2))
                mx.eval(pk, sc, bi)
            base = f"layers.{layer}.ffn.experts.{proj}"
            out[f"{base}.weight"] = mx.stack(pks)
            out[f"{base}.scales"] = mx.stack(scs)
            out[f"{base}.biases"] = mx.stack(bis)
            report[f"{layer}.{proj}"] = {"bits": bits, "rel_l2_first8": (err_num / max(err_den, 1e-30)) ** 0.5}
        shard_i += 1
        fn = f"model-routed-{layer:05d}.safetensors"
        mx.save_safetensors(str(dst / fn), out, metadata={"format": "mlx"})
        for k in out:
            new_map[k] = fn
        freed = ""
        if a.supersede and dst_is_separate:
            # the FP4 source (shards) is the requant input, so dropping the RTN copy of this
            # layer's experts is safe and keeps peak disk at ~one extra routed shard.
            old = rtn / rtn_routed_file[layer]
            if old.exists():
                gb = old.stat().st_size / 1e9
                old.unlink()
                freed = f"  (freed RTN {old.name}, {gb:.1f} GB)"
        print(f"[{time.time()-t0:6.0f}s] layer {layer:2d}: {report[f'{layer}.down_proj']}{freed}", flush=True)
        del out
        mx.clear_cache()

    total = sum((dst / f).stat().st_size for f in set(new_map.values()))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))}, indent=2))
    jc = json.loads((dst / "jang_config.json").read_text())
    if a.plain:
        jc.pop("imatrix_rel_l2_first8", None)
        jc["calibration"] = {"method": "plain affine (RTN); imatrix evaluated and not adopted (no held-out gain)"}
    else:
        jc["profile"] = jc.get("profile", "") + "_IMATRIX"
        jc["calibration"] = {"method": "imatrix clip-search (activation x^2 weighted)", "alphas": [1.0, 0.5, 11],
                             "stats_tokens": meta["tokens"], "stats_rows": meta["rows"],
                             "min_tokens_for_imatrix": a.min_tokens}
        jc["imatrix_rel_l2_first8"] = report
    (dst / "jang_config.json").write_text(json.dumps(jc, indent=2))
    print(f"done in {time.time()-t0:.0f}s -> {dst} ({total/1e9:.1f} GB routed+copied)")


def _link_or_copy(s, d):
    try:
        import os

        os.link(s, d)
    except OSError:
        shutil.copy2(s, d)


if __name__ == "__main__":
    main()
