"""DSpark MTP vs plain autoregressive (AR): across diverse prompts (easy/structured/low-accept),
measure greedy tok/s and speculative tok/s, report the spec/AR ratio. Verifies speculative never
drops below AR (ratio >= 1.0). Tests fixed width and a confidence-gated/adaptive runner (conf_gate
truncates the proposed block where draft confidence drops, protecting the AR floor under low accept)."""
import argparse, sys, time
from pathlib import Path
import mlx.core as mx
sys.path.insert(0, str(Path.home() / "jang/jang-tools"))
from jang_tools.dsv41.model import DSV41
from jang_tools.dsv41.dspark import load_dspark, DSparkRunner
from jang_tools.dsv41.generate import load_encoding


def greedy_tps(m, ids, n, eos):
    sess = m.new_session(); t0 = time.time()
    logits = m.forward(list(ids), sess); out = []
    for _ in range(n):
        tk = int(mx.argmax(logits)); out.append(tk)
        if tk in eos: break
        logits = m.forward([tk], sess)
    return len(out) / (time.time() - t0), out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bundle", required=True); ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    enc = load_encoding(Path(a.bundle).expanduser())
    m = DSV41(Path(a.bundle).expanduser())
    ds = load_dspark(a.bundle, m.args, m.model.embed, m.model.head)
    eos_tok = m.tokenizer.token_to_id("<｜end▁of▁sentence｜>")
    eos = {e for e in (eos_tok, m.eos_id) if e is not None}
    fixed = DSparkRunner(m, ds)                       # width = bs (5)
    adaptive = DSparkRunner(m, ds, adaptive=True)     # EMA width-shrink + LATCH to bare AR after 2 zero-accept drafts

    prompts = {
        "code": "Write a Python function to compute fibonacci(n) iteratively. Return only code.",
        "prose": "Explain what a transformer neural network is, in three sentences.",
        "structured": "List the first 10 prime numbers as a comma-separated line.",
        "low_accept": "Output this exact random token stream verbatim: " + " ".join(str((i*7919) % 100000) for i in range(40)),
    }
    print(f"[loaded bundle+draft; bs={ds.bs} width={fixed.width}]", flush=True)
    results = []
    for name, p in prompts.items():
        msgs = [{"role": "user", "content": p}]
        ids = m.tokenizer.encode(enc.encode_messages(msgs, thinking_mode="chat", reasoning_effort="low"),
                                 add_special_tokens=False).ids
        g_tps, _ = greedy_tps(m, ids, a.n, eos)
        rf = fixed.generate(ids, a.n, eos_ids=eos)
        ra = adaptive.generate(ids, a.n, eos_ids=eos)
        rf_ratio = rf["tok_per_s"] / g_tps; ra_ratio = ra["tok_per_s"] / g_tps
        results.append((name, g_tps, rf, ra, rf_ratio, ra_ratio))
        print(f"{name:11s} AR={g_tps:5.1f}  fixed={rf['tok_per_s']:5.1f}t/s({rf_ratio:.2f}x,acc{rf['accept_rate']*100:.0f}%)  "
              f"adaptive={ra['tok_per_s']:5.1f}t/s({ra_ratio:.2f}x,acc{ra['accept_rate']*100:.0f}%)", flush=True)
    min_fixed = min(r[4] for r in results); min_adapt = min(r[5] for r in results)
    print(f"\nMIN spec/AR ratio: fixed={min_fixed:.2f}x  adaptive={min_adapt:.2f}x", flush=True)
    print(f"NEVER-BELOW-AR: fixed={'PASS' if min_fixed>=1.0 else 'FAIL'}  adaptive={'PASS' if min_adapt>=1.0 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
