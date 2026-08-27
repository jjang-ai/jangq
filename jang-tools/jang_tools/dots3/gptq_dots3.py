"""GPTQ error-compensated codes for dots3 routed units (the QAT pass).

DSV4-0731-proven algorithm, adapted:
- per-expert routing-weighted Hessian H = Σ w_k x xᵀ (f64, escalating
  damping 0.01→1.0 with loud fallback), U = chol(H⁻¹)ᵀ;
- ONE H factorization per expert serves BOTH gate and up (same inputs);
- blocked column sweep on the FIXED min-max f16 grid — codes-only, storage
  byte-compatible with mx.quantize (pack self-test enforced at startup);
- BRECQ sequencing: down/W2 inputs derived through the QUANTIZED gate/up;
- per-expert never-worse-than-RTN guard; rare experts (< min-samples) keep
  RTN codes;
- inputs and weights are in FOLD DOMAIN (X1/awq_s, folded weights) so codes
  land directly in the converted bundle.

Resumable: per-layer codes file codes/layer_NN.npz. Stripe with --layers.

    python -m jang_tools.dots3.gptq_dots3 <src> <capture> <plan.json> \
        <folds.npz> <codes_dir> --layers 1-12
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .config import Dots3Config
from .folds import Folds
from .fp8 import ShardIndex

CHUNK = 32
K_SAMPLES = 1024        # Hessian rank vs in_dim 5120 — more rows, less damping
MIN_SAMPLES = 24
USE_IMATRIX_GRID = True # activation-weighted grid instead of min-max (qwen36)


# ---------- packing (byte-parity with mx.quantize affine storage) ----------
def pack_rows(q: np.ndarray, bits: int) -> np.ndarray:
    rows, cols = q.shape
    total_bits = cols * bits
    assert total_bits % 32 == 0, (cols, bits)
    out_words = total_bits // 32
    packed = np.zeros((rows, out_words), dtype=np.uint32)
    chunk = 65536
    weights32 = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    bit_idx = np.arange(bits, dtype=np.uint32)
    for r0 in range(0, rows, chunk):
        qq = q[r0:r0 + chunk].astype(np.uint32)
        bits_arr = ((qq[:, :, None] >> bit_idx) & 1).astype(np.uint32)
        stream = bits_arr.reshape(qq.shape[0], total_bits)
        words = stream.reshape(qq.shape[0], out_words, 32) * weights32
        packed[r0:r0 + chunk] = words.sum(axis=2, dtype=np.uint32)
    return packed


def extract_codes(qw, s, b, gs, bits) -> np.ndarray:
    deq = mx.dequantize(qw, s, b, group_size=gs, bits=bits)
    s_exp = np.repeat(np.asarray(s, dtype=np.float32), gs, axis=-1)
    b_exp = np.repeat(np.asarray(b, dtype=np.float32), gs, axis=-1)
    d = np.asarray(deq, dtype=np.float32)
    s_safe = np.where(s_exp == 0.0, 1.0, s_exp)
    codes = np.rint((d - b_exp) / s_safe)
    return np.clip(codes, 0, (1 << bits) - 1).astype(np.uint32)


def pack_self_test() -> None:
    rng = np.random.default_rng(7)
    for bits, gs in ((2, 64), (2, 32), (3, 64), (3, 128), (4, 64), (8, 64)):
        w = mx.array(rng.standard_normal((8, 512)).astype(np.float32))
        qw, s, b = mx.quantize(w, group_size=gs, bits=bits)
        codes = extract_codes(qw, s, b, gs, bits)
        mine = pack_rows(codes, bits)
        if not np.array_equal(mine, np.asarray(qw).view(np.uint32)):
            raise SystemExit(f"PACK SELF-TEST FAILED bits={bits} gs={gs}")
    print("PACK_SELF_TEST_OK", flush=True)


# ---------- grid + sweep -----------------------------------------------------
def minmax_grid(w2d: np.ndarray, gs: int, bits: int):
    """min-max f16 grid exactly as the converter stores it."""
    _qw, s, b = mx.quantize(mx.array(w2d), group_size=gs, bits=bits)
    s16 = np.asarray(s.astype(mx.float16))
    b16 = np.asarray(b.astype(mx.float16))
    del _qw, s, b
    return s16, b16


def rtn_codes_on_grid(W: np.ndarray, s16, b16, gs: int, bits: int) -> np.ndarray:
    """RTN codes on the STORED f16 grid (the GPTQ baseline)."""
    qmax = (1 << bits) - 1
    s_exp = np.repeat(s16.astype(np.float32), gs, axis=-1)
    b_exp = np.repeat(b16.astype(np.float32), gs, axis=-1)
    s_safe = np.where(np.abs(s_exp) < 1e-12, 1.0, s_exp)
    return np.clip(np.rint((W - b_exp) / s_safe), 0, qmax).astype(np.float32)


def imatrix_grid(W: np.ndarray, importance: np.ndarray, gs: int, bits: int):
    """Activation-weighted affine grid (qwen36-proven, byte-identical ABI).

    Returns the SAME (scales f16, biases f16) layout as minmax_grid, so codes
    fit on it stay storage-compatible. Falls back to min-max on any failure —
    a bad grid must never take the build down."""
    from ..affine import quantize_imatrix_affine_numpy
    try:
        _packed, s16, b16, _err = quantize_imatrix_affine_numpy(
            W, importance, bits=bits, group_size=gs)
        if not (np.isfinite(s16).all() and np.isfinite(b16).all()):
            raise ValueError("non-finite imatrix grid")
        return np.asarray(s16), np.asarray(b16)
    except Exception as exc:                                   # noqa: BLE001
        print(f"    imatrix grid fallback -> minmax ({exc})", flush=True)
        return minmax_grid(W, gs, bits)


def pick_grid(W: np.ndarray, xin: np.ndarray, wtok: np.ndarray, gs: int,
              bits: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit both candidate grids, score their RTN codes on real activations,
    return the better one. The imatrix grid usually wins at 2-3 bits, but it
    is a FIT — never assume; measure and keep the winner."""
    s_mm, b_mm = minmax_grid(W, gs, bits)
    if not USE_IMATRIX_GRID:
        return s_mm, b_mm, "minmax"
    imp = (xin.astype(np.float64) ** 2 * wtok[:, None]).sum(0)
    imp = np.maximum(imp, 0).astype(np.float32)
    if not np.isfinite(imp).all() or imp.max() <= 0:
        return s_mm, b_mm, "minmax"
    s_im, b_im = imatrix_grid(W, imp, gs, bits)
    e_mm = wloss(W, rtn_codes_on_grid(W, s_mm, b_mm, gs, bits),
                 s_mm, b_mm, gs, xin, wtok)
    e_im = wloss(W, rtn_codes_on_grid(W, s_im, b_im, gs, bits),
                 s_im, b_im, gs, xin, wtok)
    if e_im < e_mm:
        return s_im, b_im, "imatrix"
    return s_mm, b_mm, "minmax"


def hessian_factor(xf: np.ndarray, wtok: np.ndarray, label: str):
    """xf (K,in) f32, wtok (K,) normalized weights -> U (in,in) f32 or None."""
    in_dim = xf.shape[1]
    H = (xf.T * wtok[None, :]) @ xf
    H = H.astype(np.float64)
    di = np.arange(in_dim)
    diag = H[di, di]
    H[di, di] = np.where(diag <= 0, 1.0, diag)
    mean_diag = max(float(diag.mean()), 1e-10)
    for damp in (0.01, 0.05, 0.2, 1.0):
        try:
            Hd = H.copy()
            Hd[di, di] += damp * mean_diag
            Hinv = np.linalg.inv(Hd)
            Hinv = 0.5 * (Hinv + Hinv.T)
            U = np.ascontiguousarray(np.linalg.cholesky(Hinv).T).astype(np.float32)
            if damp != 0.01:
                print(f"    [{label}] damping escalated to {damp}", flush=True)
            return U
        except np.linalg.LinAlgError:
            continue
    print(f"    [{label}] Cholesky failed at all dampings — RTN kept", flush=True)
    return None


def gptq_sweep(W: np.ndarray, s16: np.ndarray, b16: np.ndarray, gs: int,
               bits: int, U: np.ndarray) -> np.ndarray:
    """W (out,in) f32 on fixed grid; returns codes (out,in) f32."""
    out_dim, in_dim = W.shape
    qmax = (1 << bits) - 1
    s_exp = np.repeat(s16.astype(np.float32), gs, axis=-1)
    b_exp = np.repeat(b16.astype(np.float32), gs, axis=-1)
    s_safe = np.where(np.abs(s_exp) < 1e-12, 1.0, s_exp)
    wq_work = W.copy()
    codes = np.zeros_like(W)
    di = np.arange(in_dim)
    u_diag = U[di, di]
    u_diag = np.where(np.abs(u_diag) < 1e-10, 1.0, u_diag)
    blk = 128
    for c0 in range(0, in_dim, blk):
        c1 = min(c0 + blk, in_dim)
        err_blk = np.zeros((out_dim, c1 - c0), dtype=np.float32)
        for i in range(c0, c1):
            wcol = wq_work[:, i]
            q = np.clip(np.rint((wcol - b_exp[:, i]) / s_safe[:, i]), 0, qmax)
            codes[:, i] = q
            wq_col = s_exp[:, i] * q + b_exp[:, i]
            err = (wcol - wq_col) / u_diag[i]
            err_blk[:, i - c0] = err
            if i + 1 < c1:
                wq_work[:, i + 1:c1] -= err[:, None] * U[i, i + 1:c1][None, :]
        if c1 < in_dim:
            wq_work[:, c1:] -= err_blk @ U[c0:c1, c1:]
    return codes


def wloss(W: np.ndarray, codes: np.ndarray, s16, b16, gs: int,
          xf: np.ndarray, wtok: np.ndarray) -> float:
    s_exp = np.repeat(s16.astype(np.float32), gs, axis=-1)
    b_exp = np.repeat(b16.astype(np.float32), gs, axis=-1)
    E = s_exp * codes + b_exp - W
    y = xf @ E.T
    return float((y * y * wtok[:, None]).sum())


# ---------- per-layer driver -------------------------------------------------
def spec_for(plan: dict, li: int, proj: str) -> tuple[int, int, str]:
    d = dict(plan["defaults"]["routed"])
    ov = plan.get("routed_overrides", {}).get(f"{li}:{proj}")
    if ov:
        d.update(ov)
    return d["bits"], d["group_size"], d.get("mode", "affine")


def run_layer(li: int, cfg: Dots3Config, idx: ShardIndex, capture: Path,
              plan: dict, folds: Folds, out_dir: Path) -> dict:
    t0 = time.time()
    out_path = out_dir / f"layer_{li:02d}.npz"
    if out_path.exists():
        print(f"[L{li}] codes exist — skip", flush=True)
        return {}
    x1 = np.load(capture / "x1" / f"layer_{li:02d}.npy").astype(np.float32)
    if folds.awq is not None:
        x1 = x1 / folds.awq[li][None, :]
    rt = np.load(capture / "router" / f"layer_{li:02d}.npz")
    inds, weights = rt["inds"].astype(np.int32), rt["weights"].astype(np.float32)
    E = cfg.n_routed_experts
    specs = {p: spec_for(plan, li, p) for p in ("gate_proj", "up_proj",
                                                "down_proj")}
    for p, (bits, gs, mode) in specs.items():
        if mode != "affine":
            raise SystemExit(f"L{li}:{p} mode {mode}: GPTQ affine-grid only")

    # gather per-expert samples (top-K by routing weight)
    flat_e = inds.reshape(-1)
    flat_w = weights.reshape(-1)
    flat_t = np.repeat(np.arange(x1.shape[0]), inds.shape[1])
    order = np.argsort(flat_e, kind="stable")
    fe, fw, ft = flat_e[order], flat_w[order], flat_t[order]
    starts = np.searchsorted(fe, np.arange(E))
    ends = np.searchsorted(fe, np.arange(E) + 1)

    all_codes = {p: np.zeros((E, *(
        (cfg.moe_intermediate_size, cfg.hidden_size) if p != "down_proj"
        else (cfg.hidden_size, cfg.moe_intermediate_size))), np.uint16)
        for p in specs}
    all_s = {p: None for p in specs}
    all_b = {p: None for p in specs}

    def keep_grid(p, e, s16, b16):
        if all_s[p] is None:
            all_s[p] = np.zeros((E, *s16.shape), np.float16)
            all_b[p] = np.zeros((E, *b16.shape), np.float16)
        all_s[p][e] = s16
        all_b[p][e] = b16

    stats = {p: {"base": 0.0, "final": 0.0, "wins": 0, "skipped": 0}
             for p in specs}
    grid_wins = {p: {"imatrix": 0, "minmax": 0} for p in specs}

    for e0 in range(0, E, CHUNK):
        e1 = min(e0 + CHUNK, E)
        for e in range(e0, e1):
            we = fw[starts[e]:ends[e]]
            te = ft[starts[e]:ends[e]]
            n = we.size
            if n > K_SAMPLES:
                top = np.argsort(-we)[:K_SAMPLES]
                we, te = we[top], te[top]
            xf = x1[te]
            wtok = (we / max(float(we.sum()), 1e-9)).astype(np.float32)
            pfx = f"model.layers.{li}.mlp.experts.{e}."
            Wg = folds.apply(pfx + "gate_proj.weight",
                             idx.read_dequant(pfx + "gate_proj.weight"))
            Wu = folds.apply(pfx + "up_proj.weight",
                             idx.read_dequant(pfx + "up_proj.weight"))
            Wd = folds.apply(pfx + "down_proj.weight",
                             idx.read_dequant(pfx + "down_proj.weight"))
            skip = n < MIN_SAMPLES

            deq = {}
            for p, W in (("gate_proj", Wg), ("up_proj", Wu)):
                bits, gs, _ = specs[p]
                s16, b16, which = pick_grid(W, xf, wtok, gs, bits)
                grid_wins[p][which] += 1
                base = rtn_codes_on_grid(W, s16, b16, gs, bits)
                if skip:
                    codes = base
                else:
                    U = hessian_factor(xf, wtok, f"L{li}e{e}:{p}") \
                        if p == "gate_proj" else run_layer.last_U
                    run_layer.last_U = U
                    if U is None:
                        codes = base
                    else:
                        codes = gptq_sweep(W, s16, b16, gs, bits, U)
                        lb = wloss(W, base, s16, b16, gs, xf, wtok)
                        lg = wloss(W, codes, s16, b16, gs, xf, wtok)
                        stats[p]["base"] += lb
                        if lg >= lb:
                            codes = base
                            stats[p]["final"] += lb
                        else:
                            stats[p]["final"] += lg
                            stats[p]["wins"] += 1
                all_codes[p][e] = codes.astype(np.uint16)
                keep_grid(p, e, s16, b16)
                s_exp = np.repeat(s16.astype(np.float32), gs, axis=-1)
                b_exp = np.repeat(b16.astype(np.float32), gs, axis=-1)
                deq[p] = s_exp * codes + b_exp
            # BRECQ: derive down inputs through the QUANTIZED gate/up
            g = xf @ deq["gate_proj"].T
            x2 = (g / (1 + np.exp(-np.clip(g, -30, 30)))) * (xf @ deq["up_proj"].T)
            bits, gs, _ = specs["down_proj"]
            s16, b16, which = pick_grid(Wd, x2, wtok, gs, bits)
            grid_wins["down_proj"][which] += 1
            base = rtn_codes_on_grid(Wd, s16, b16, gs, bits)
            if skip:
                codes = base
                stats["down_proj"]["skipped"] += 1
            else:
                U2 = hessian_factor(x2, wtok, f"L{li}e{e}:down")
                if U2 is None:
                    codes = base
                else:
                    codes = gptq_sweep(Wd, s16, b16, gs, bits, U2)
                    lb = wloss(Wd, base, s16, b16, gs, x2, wtok)
                    lg = wloss(Wd, codes, s16, b16, gs, x2, wtok)
                    stats["down_proj"]["base"] += lb
                    if lg >= lb:
                        codes = base
                        stats["down_proj"]["final"] += lb
                    else:
                        stats["down_proj"]["final"] += lg
                        stats["down_proj"]["wins"] += 1
            all_codes["down_proj"][e] = codes.astype(np.uint16)
            keep_grid("down_proj", e, s16, b16)
            if skip:
                stats["gate_proj"]["skipped"] += 1
                stats["up_proj"]["skipped"] += 1
            del Wg, Wu, Wd, deq
        gc.collect()
        mx.clear_cache()
        print(f"  [L{li}] experts {e0}-{e1-1} done "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # NB: np.savez appends .npz when missing — temp name must end with it.
    tmp = out_dir / f"layer_{li:02d}.tmp.npz"
    np.savez_compressed(
        tmp, **{f"{p}_codes": all_codes[p] for p in all_codes},
        **{f"{p}_scales": all_s[p] for p in all_s},
        **{f"{p}_biases": all_b[p] for p in all_b},
        specs=json.dumps({p: specs[p] for p in specs}))
    tmp.rename(out_path)
    summary = {p: {
        "improvement": 0.0 if s_["base"] == 0 else 1 - s_["final"] / s_["base"],
        "wins": s_["wins"], "skipped": s_["skipped"],
        "grid": grid_wins[p]}
        for p, s_ in stats.items()}
    print(f"[L{li}] DONE {json.dumps(summary)} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return summary


run_layer.last_U = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("capture", type=Path)
    ap.add_argument("plan", type=Path)
    ap.add_argument("folds", type=Path)
    ap.add_argument("codes_dir", type=Path)
    ap.add_argument("--layers", required=True, help="e.g. 1-12 or 3,5,9")
    a = ap.parse_args()

    mx.set_memory_limit(int(20 * 1024**3))
    pack_self_test()
    cfg = Dots3Config.load(a.src)
    idx = ShardIndex(a.src)
    plan = json.loads(a.plan.read_text())
    folds = Folds.load(a.folds)
    a.codes_dir.mkdir(parents=True, exist_ok=True)
    if "-" in a.layers:
        lo, hi = a.layers.split("-")
        layers = range(int(lo), int(hi) + 1)
    else:
        layers = [int(x) for x in a.layers.split(",")]
    report = {}
    for li in layers:
        report[li] = run_layer(li, cfg, idx, a.capture, plan, folds,
                               a.codes_dir)
        (a.codes_dir / f"report_{a.layers.replace(',', '_')}.json"
         ).write_text(json.dumps(report, indent=1))
    print("ALL_REQUESTED_LAYERS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
