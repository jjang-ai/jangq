"""Vector-trellis (V=2) code search: quality of one-state-two-weights codes vs V=1 pop1 and scalar.
Usage: python -m jang_tools.jangt.codesearch_v2 OUT.json [--trials 48] [--K 2,3]"""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .codes import v2_halfbits, v2_pop, v2_random
from .trellis import gaussian_mse_v, LLOYD_MAX


def score(item, Ks):
    fam, p = item
    fn = {"v2_halfbits": v2_halfbits, "v2_pop": v2_pop, "v2_random": v2_random}[fam]
    code, ops = fn(**p)
    if not np.isfinite(code).all():
        return None
    r = {"family": fam, "params": p, "ops_per_weight": ops}
    for K in Ks:
        mse, sc = gaussian_mse_v(code, K, 2, tiles=12); r[f"mse_K{K}"] = mse; r[f"scale_K{K}"] = sc
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--trials", type=int, default=48); ap.add_argument("--K", default="2,3")
    a = ap.parse_args(); Ks = [int(k) for k in a.K.split(",")]; r = np.random.default_rng(11)
    cands = [("v2_random", {"seed": 0})]
    for _ in range(a.trials):
        mult = int(r.integers(1 << 20, 1 << 32)) | 1
        # mantissa random (10 bits), sign random (bit 15), exponent: some bits fixed via orbits, some random via mask
        ebits = int(r.integers(1, 4)); emask = ((1 << ebits) - 1) << 10
        cands.append(("v2_halfbits", {"mult": mult, "mask": 0x83FF | emask, "orbits": int(r.choice([0x3800, 0x3400, 0x3C00, 0x3000])) & ~emask & 0x7C00}))
        cands.append(("v2_pop", {"mult": mult}))
    t0 = time.time()
    with ProcessPoolExecutor(10) as ex:
        res = [x for x in ex.map(score, cands, [Ks] * len(cands)) if x]
    res.sort(key=lambda x: x[f"mse_K{Ks[0]}"])
    best = {f: next((x for x in res if x["family"] == f), None) for f in ("v2_random", "v2_halfbits", "v2_pop")}
    json.dump({"seconds": round(time.time() - t0), "lloyd": {k: LLOYD_MAX[k] for k in Ks}, "best": best, "top": res[:10]}, open(a.out, "w"), indent=1)
    for f, x in best.items():
        if x: print(f, x["ops_per_weight"], {k: round(v, 4) for k, v in x.items() if k.startswith("mse_")}, x["params"])


if __name__ == "__main__":
    main()
