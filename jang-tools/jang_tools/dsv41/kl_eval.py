"""KL / top-1 quant-quality harness for DSV4.1 JANG bundles.

Two passes, so two models never need to co-reside (each ~211 GB; a full-precision reference
needs a >256 GB host). On this 256 GB Studio the full-precision source (286 GB fp8) and a
complete RTN reference do not fit, so the reference pass is run elsewhere; this harness makes
KL "ready" — the quant-side dump runs here, and `--mode kl` diffs two dumps offline.

  # pass 1 (reference, on a host where it fits):
  python -m jang_tools.dsv41.kl_eval --mode dump --bundle <REF_BUNDLE> --rows calib/... --out ref.npz
  # pass 2 (this quant, here):
  python -m jang_tools.dsv41.kl_eval --mode dump --bundle <IMATRIX> --rows calib/... --out imatrix.npz
  # KL (anywhere):
  python -m jang_tools.dsv41.kl_eval --mode kl --ref ref.npz --cand imatrix.npz

`dump` saves, per scored next-token position, the full log-probs (fp16) + the true token, over the
SAME deterministic held-out rows (seed-stable), so the two dumps align position-for-position.
`kl` computes per-position KL(ref||cand), top-1 agreement (ref argmax == cand argmax), and reports
mean/median/p95/p99.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path.home() / "jang/jang-tools"))


def _rows(rows_path, n, seed):
    from jang_tools.dsv41.calib_generate import DEFAULT_DOMAINS
    import random
    rows = [json.loads(l) for l in open(rows_path)]
    rows = [r for r in rows if r["domain"] not in DEFAULT_DOMAINS]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def dump(a):
    import copy
    from jang_tools.dsv41.model import DSV41
    from jang_tools.dsv41.generate import load_encoding
    enc = load_encoding(Path(a.bundle).expanduser())
    m = DSV41(Path(a.bundle).expanduser(), a.engram_dir)
    rows = _rows(a.rows, a.n, a.seed)
    logps, trues, row_lens = [], [], []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = copy.deepcopy(r["messages"])
        for mm in msgs:
            if mm["role"] == "assistant":
                mm.setdefault("reasoning_content", "")
        ids = m.tokenizer.encode(enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=r["effort"]),
                                 add_special_tokens=False).ids[: a.max_tokens]
        if len(ids) < 2:
            continue
        sess = m.new_session()
        logits = m.forward(ids, sess, all_logits=True)
        lp = logits[:-1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)   # log-softmax [T-1, V]
        mx.eval(lp)
        logps.append(np.asarray(lp.astype(mx.float16)))      # fp16 to bound disk
        trues.append(np.asarray(ids[1:], dtype=np.int32))
        row_lens.append(len(ids) - 1)
        del logits, lp, sess; mx.clear_cache()
        if (i + 1) % 4 == 0:
            print(f"  dumped {i+1}/{len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)
    LP = np.concatenate(logps, axis=0)       # [P, V] fp16
    TR = np.concatenate(trues, axis=0)       # [P]
    np.savez(a.out, logprobs=LP, true=TR, lens=np.asarray(row_lens), bundle=Path(a.bundle).name)
    print(f"wrote {a.out}: {LP.shape[0]} positions x {LP.shape[1]} vocab (fp16), bundle={Path(a.bundle).name}", flush=True)


def _pct(x, p):
    return float(np.percentile(x, p))


def kl(a):
    ref = np.load(a.ref); cand = np.load(a.cand)
    rlp = ref["logprobs"].astype(np.float32); clp = cand["logprobs"].astype(np.float32)
    assert rlp.shape == clp.shape, f"shape mismatch {rlp.shape} vs {clp.shape} (dumps must use same rows/seed)"
    rp = np.exp(rlp)
    kl_pos = np.sum(rp * (rlp - clp), axis=-1)               # KL(ref||cand) per position [P]
    r_arg = rlp.argmax(-1); c_arg = clp.argmax(-1)
    top1_agree = float((r_arg == c_arg).mean())
    true = ref["true"]
    ref_top1_acc = float((r_arg == true).mean()); cand_top1_acc = float((c_arg == true).mean())
    out = {
        "ref": str(ref["bundle"]), "cand": str(cand["bundle"]), "positions": int(kl_pos.shape[0]),
        "kl_ref_cand": {"mean": float(kl_pos.mean()), "median": _pct(kl_pos, 50),
                         "p95": _pct(kl_pos, 95), "p99": _pct(kl_pos, 99)},
        "top1_agreement_ref_vs_cand": top1_agree,
        "ref_top1_acc_vs_true": ref_top1_acc, "cand_top1_acc_vs_true": cand_top1_acc,
    }
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "kl"], required=True)
    ap.add_argument("--bundle"); ap.add_argument("--engram-dir"); ap.add_argument("--rows")
    ap.add_argument("--ref"); ap.add_argument("--cand")
    ap.add_argument("--out"); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1536); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    (dump if a.mode == "dump" else kl)(a)


if __name__ == "__main__":
    main()
