"""RTN MLX-affine CONTROL bundle of Naive-N0.5-Flash at the same size budget — the speed and quality
comparator ("RTN mx affine"): stock mlx_lm SwitchGLU, experts affine 2-bit g128 / 3-bit g64 / 4-bit g64 (bf16 scales),
bits per (layer, projection) from a knapsack on the affine RTN held-out curves (curves.json affine_* columns);
non-experts exactly as in the JANGH build. No calibration of any kind. MLX_ENABLE_TF32=0 not required (no matmuls)."""
from __future__ import annotations

import argparse
import heapq
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import mlx.core as mx



from jang_tools.jangh.n05.plan import classify  # noqa: E402

E, D, I = 256, 4096, 2048
GS = {2: 128, 3: 64, 4: 64}


def unit_bytes(p, bits):
    n, k = (I, D) if p in ("gate", "up") else (D, I)
    return E * n * (k * bits // 8 + (k // GS[bits]) * 4)      # packed + bf16 scale and bias per group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--headers", required=True); ap.add_argument("--curves", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--total-gib", type=float, default=96.0)
    a = ap.parse_args()
    from jang_tools.jangh.glm53.convert_tq import ShardWriter
    src, out = Path(a.model), Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    H = json.loads(Path(a.headers).read_text()); curves = json.loads(Path(a.curves).read_text())
    fixed = 0
    for k, (f, dt, shp) in H.items():
        c = classify(k, dt, shp); n = 1
        for s in shp:
            n *= s
        fixed += n + n / 64 * 4 if c == "q8" else (n * (4 if dt == "F32" else 2) if c == "keep" else 0)
    budget = a.total_gib * 2**30 - fixed
    # affine RTN curves exist for gate_up (2, 3) and down (2, 3): tie gate/up, bits in {2, 3}
    state, spent, heap = {}, 0, []
    for L in range(1, 48):
        r = curves[str(L)]
        for unit, err in (("gate_up", {2: r["affine_gate_up2"], 3: r["affine_gate_up3"]}), ("down", {2: r["affine_down2"], 3: r["affine_down3"]})):
            b2 = sum(unit_bytes(p, 2) for p in (("gate", "up") if unit == "gate_up" else ("down",)))
            b3 = sum(unit_bytes(p, 3) for p in (("gate", "up") if unit == "gate_up" else ("down",)))
            state[(L, unit)] = 2; spent += b2
            heapq.heappush(heap, (-(err[2] - err[3]) / (b3 - b2), L, unit, b3 - b2))
    assert spent <= budget, f"all-2-bit affine {spent/2**30:.2f} GiB exceeds the expert budget {budget/2**30:.2f} GiB"
    while heap:
        _, L, unit, delta = heapq.heappop(heap)
        if spent + delta <= budget:
            spent += delta; state[(L, unit)] = 3
    bits = {}
    for (L, unit), b in state.items():
        for p in (("gate", "up") if unit == "gate_up" else ("down",)):
            bits[f"{L}:{p}"] = b
    print(f"affine control: fixed {fixed/2**30:.3f} GiB + experts {spent/2**30:.3f} GiB = {(fixed+spent)/2**30:.3f} GiB; "
          f"3-bit units: gate_up {sum(1 for (L,u),b in state.items() if u=='gate_up' and b==3)} down {sum(1 for (L,u),b in state.items() if u=='down' and b==3)}", flush=True)
    cache = {}

    def get(name):
        f = H[name][0]
        if f not in cache:
            cache.clear(); mx.clear_cache(); cache[f] = mx.load(str(src / f))
        return cache[f][name]

    W = ShardWriter(out); qcfg = {"group_size": 64, "bits": 8, "mode": "affine"}; t0 = time.time()
    for k in sorted(k for k in H if ".mlp.experts." not in k):
        c = classify(k, H[k][1], H[k][2]); w = get(k)
        if c == "keep":
            W.add(k, w)
        else:
            mod = k[: -len(".weight")]
            wq, sc, bi = mx.quantize(w.astype(mx.bfloat16), group_size=64, bits=8)
            W.add(mod + ".weight", wq); W.add(mod + ".scales", sc); W.add(mod + ".biases", bi)
            qcfg[mod] = {"group_size": 64, "bits": 8, "mode": "affine"}
    for L in range(1, 48):
        for p in ("gate", "up", "down"):
            b = bits[f"{L}:{p}"]; qs, ss, bs = [], [], []
            for e0 in range(0, E, 32):
                Wc = mx.stack([get(f"model.layers.{L}.mlp.experts.{e}.{p}_proj.weight") for e in range(e0, e0 + 32)]).astype(mx.bfloat16)
                q, s, bi = mx.quantize(Wc, group_size=GS[b], bits=b); mx.eval(q, s, bi); qs.append(q); ss.append(s); bs.append(bi)
            m = f"model.layers.{L}.mlp.switch_mlp.{p}_proj"
            W.add(m + ".weight", mx.concatenate(qs)); W.add(m + ".scales", mx.concatenate(ss)); W.add(m + ".biases", mx.concatenate(bs))
            qcfg[m] = {"group_size": GS[b], "bits": b, "mode": "affine"}
        print(f"L{L:2d} gate/up {bits[f'{L}:gate']} down {bits[f'{L}:down']}  total {(time.time()-t0)/60:.1f} min", flush=True)
    n = W.finish()
    idx = json.loads((out / "model.safetensors.index.json").read_text()); idx["metadata"]["format"] = "mlx-affine"
    (out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=1))
    cfg = json.loads((src / "config.json").read_text()); cfg.pop("auto_map", None)
    cfg["quantization"] = qcfg; cfg["quantization_config"] = qcfg
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    (out / "control_build.json").write_text(json.dumps({"kind": "RTN MLX affine control (no calibration)", "expert_bits": bits,
                                                         "total_gib": (fixed + spent) / 2**30}, indent=1))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    print(f"DONE {n} shards {W.total/2**30:.3f} GiB {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
