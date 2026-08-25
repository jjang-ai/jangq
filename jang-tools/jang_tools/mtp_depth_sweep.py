"""MTP acceptance by DEPTH — is best_depth 1, 2 or 3?

Qwen3.8 ships ONE trained MTP layer, so depth >1 means recursively re-driving
that same head: feed its own output hidden plus its own drafted token back in.
`mtp_forward(..., return_hidden=True)` returns exactly what that needs.

    step 1:  (trunk_hidden[t], tok[t+1])   -> d1[t]  predicts tok[t+2]
    step 2:  (mtp_hidden1[t],  d1[t])      -> d2[t]  predicts tok[t+3]
    step 3:  (mtp_hidden2[t],  d2[t])      -> d3[t]  predicts tok[t+4]

Two numbers matter and they are NOT the same:

  per-position acceptance  p_k = P(d_k == target)         — diagnostic
  CHAINED acceptance       P(d_1..d_k all correct)        — what decoding gets

Speculative decoding stops at the first rejection, so the expected tokens per
target forward is the chained sum:

    E[tokens/cycle] = 1 + c_1 + c_2 + ... + c_d      where c_k = chained@k

Depth only pays if the marginal chained term exceeds the extra draft cost. The
head is ~1.4% of target bytes per draft, so the bar is low — but a term that
adds ~0 still costs a forward.

    python -m jang_tools.mtp_depth_sweep <bundle> <heldout.json> [n_prompts]
"""
import json
import sys
import types
from pathlib import Path

import mlx.core as mx


def _stub():
    pkg = types.ModuleType("vmlx_engine")
    pat = types.ModuleType("vmlx_engine.patches")
    mlm = types.ModuleType("vmlx_engine.patches.mlx_lm_mtp")
    mlm.is_mtp_active = lambda: True
    pkg.patches = pat
    pat.mlx_lm_mtp = mlm
    sys.modules.update({"vmlx_engine": pkg, "vmlx_engine.patches": pat,
                        "vmlx_engine.patches.mlx_lm_mtp": mlm})


def main():
    bundle = Path(sys.argv[1])
    prompts = json.loads(Path(sys.argv[2]).read_text())["prompts"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    prompts = prompts[:n]
    MAXD = 3

    _stub()
    sys.path.insert(0, str(Path(__file__).parent))
    from vmlx_mtp import qwen35_vl
    qwen35_vl.apply()
    from mlx_vlm import load

    model, proc = load(str(bundle))
    lm = model.language_model
    if not hasattr(lm, "mtp"):
        print("INVALID: no .mtp on the patched model")
        return 3

    hit = [0] * (MAXD + 1)      # per-position correct
    chain = [0] * (MAXD + 1)    # correct AND all shallower steps correct
    tot = 0

    for p in prompts:
        text = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True)
        toks = list(proc.tokenizer.encode(text))
        ctx = len(toks)
        for _ in range(32):                      # its own greedy continuation
            lm._position_ids = None
            lm._rope_deltas = None
            o = lm(mx.array([toks]))
            lg = o.logits if hasattr(o, "logits") else o
            toks.append(int(mx.argmax(lg[0, -1]).item()))
            del o, lg
            mx.clear_cache()

        lm._position_ids = None
        lm._rope_deltas = None
        logits, hidden = lm(mx.array([toks]), return_hidden=True)
        tgt = mx.argmax(logits, -1)[0]
        mx.eval(tgt, hidden)
        T = hidden.shape[1]
        lo = max(0, ctx - 1)

        cur_h = hidden[:, : T - 1, :]
        cur_ids = mx.array([toks[1:T]])
        alive = None
        for k in range(1, MAXD + 1):
            cache = lm.make_mtp_cache()
            out = lm.mtp_forward(cur_h, cur_ids, cache, return_hidden=True)
            dlogits, dhidden = out
            d = mx.argmax(dlogits, -1)[0]
            mx.eval(d, dhidden)
            # d[t] predicts token t+1+k ; target's own prediction is tgt[t+k]
            hi = T - 1 - k
            if hi <= lo:
                break
            ok = (d[lo:hi] == tgt[lo + k: hi + k])
            mx.eval(ok)
            alive = ok if alive is None else (alive[: ok.shape[0]] & ok)
            hit[k] += int(ok.sum().item())
            chain[k] += int(alive.sum().item())
            if k == 1:
                tot += int(ok.shape[0])
            cur_h, cur_ids = dhidden, mx.expand_dims(d, 0)
            del dlogits, cache
            mx.clear_cache()
        del logits, hidden, tgt
        mx.clear_cache()

    print(f"  bundle : {bundle.name}")
    print(f"  scored : {tot} positions (generated span)\n")
    print(f"  {'depth':>6} {'per-position':>13} {'CHAINED':>9}   tokens/cycle")
    exp = 1.0
    rows = []
    for k in range(1, MAXD + 1):
        pp = 100.0 * hit[k] / tot
        cc = 100.0 * chain[k] / tot
        exp += cc / 100.0
        rows.append({"depth": k, "per_position_pct": pp, "chained_pct": cc,
                     "expected_tokens_per_cycle": exp})
        print(f"  {k:>6} {pp:>12.2f}% {cc:>8.2f}%   {exp:.3f}")

    best = max(rows, key=lambda r: r["expected_tokens_per_cycle"])
    gains = [rows[0]["expected_tokens_per_cycle"] - 1.0] + [
        rows[i]["expected_tokens_per_cycle"] - rows[i - 1]["expected_tokens_per_cycle"]
        for i in range(1, len(rows))]
    print(f"\n  marginal tokens added per depth: "
          + ", ".join(f"d{i+1} +{g:.3f}" for i, g in enumerate(gains)))
    # a depth step costs one extra head forward (~1.4% of target bytes);
    # keep it only if it adds meaningfully more than that
    keep = max([r["depth"] for r, g in zip(rows, gains) if g > 0.02] or [1])
    print(f"  recommended best_depth = {keep}  "
          f"(deepest step still adding > 0.02 tokens/cycle)")
    Path(f"/Users/eric/models/Logs/q38v2/mtp-depth-{bundle.name}.json").write_text(
        json.dumps({"bundle": bundle.name, "positions": tot, "rows": rows,
                    "recommended_best_depth": keep}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
