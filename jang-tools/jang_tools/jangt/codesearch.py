"""JANGT code search: score code families + constants by trellis MSE on Gaussian data vs scalar Lloyd-Max and the
rate-distortion bound. Usage: python -m jang_tools.jangt.codesearch OUT.json [--trials 64] [--K 2,3] [--tiles 16]"""
from __future__ import annotations
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .codes import FAMILIES
from .trellis import gaussian_mse, LLOYD_MAX, RD_BOUND


def _odd(r):
    return int(r.integers(1 << 20, 1 << 32)) | 1


def candidates(trials: int, seed: int = 7, families=None):
    r = np.random.default_rng(seed)
    out = [("random_gaussian", {"seed": 0})]
    if families == "w16":
        for _ in range(trials):
            out.append(("pop16", {"mult": int(r.integers(1, 1 << 16)) | 1, "add": int(r.integers(0, 1 << 16))}))
            out.append(("pop16x2", {"m1": int(r.integers(1, 1 << 16)) | 1, "m2": int(r.integers(1, 1 << 16)) | 1}))
        return out
    if families == "cheap":
        for _ in range(trials):
            out.append(("pop1", {"mult": _odd(r), "add": int(r.integers(0, 1 << 32))}))
            out.append(("pop_wsum", {"mult": _odd(r), "m": int(r.integers(0, 1 << 32)), "w": float(r.choice([0.25, 0.5]))}))
            out.append(("byte2", {"mult": _odd(r), "add": int(r.integers(0, 1 << 32))}))
        return out
    for _ in range(trials):
        m1 = int(r.integers(0, 1 << 32)); out.append(("popmix", {"mult": _odd(r), "m1": m1, "m2": (~m1) & 0xFFFFFFFF, "w2": float(r.choice([1.5, 2.0, 3.0]))}))
        out.append(("nibble_sum", {"mult": _odd(r), "add": int(r.integers(0, 1 << 32))}))
        out.append(("byte_pair", {"mult": _odd(r), "add": int(r.integers(0, 1 << 32))}))
    return out


def score(item, Ks, tiles):
    fam, params = item
    code, ops = FAMILIES[fam](**params)
    res = {"family": fam, "params": params, "ops": ops, "levels": int(len(np.unique(np.round(code, 6))))}
    from .trellis import LLOYD_MAX as _LM
    for K in Ks:
        mse, sc = gaussian_mse(code, K, tiles=tiles)
        res[f"mse_K{K}"] = mse; res[f"scale_K{K}"] = sc; res[f"vs_lloyd_K{K}"] = mse / LLOYD_MAX[K]
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--K", default="2,3"); ap.add_argument("--tiles", type=int, default=16); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--families", default=None, help="'cheap' = <=5-op families only")
    a = ap.parse_args(); Ks = [int(k) for k in a.K.split(",")]
    cands = candidates(a.trials, families=a.families); t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(score, cands, [Ks] * len(cands), [a.tiles] * len(cands)))
    res.sort(key=lambda r: r[f"mse_K{Ks[0]}"])
    summary = {"K": Ks, "lloyd_max": {k: LLOYD_MAX[k] for k in Ks}, "rd_bound": {k: RD_BOUND[k] for k in Ks},
               "seconds": round(time.time() - t0), "top": res[:15],
               "best_by_family": {f: next((r for r in res if r["family"] == f), None) for f in FAMILIES}}
    json.dump(summary, open(a.out, "w"), indent=1)
    for r in res[:10]:
        print(r["family"], r["ops"], r["levels"], {k: round(v, 4) for k, v in r.items() if k.startswith("mse_") or k.startswith("vs_")})


if __name__ == "__main__":
    main()
