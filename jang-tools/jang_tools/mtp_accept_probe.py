"""MTP draft acceptance, driven by vMLX's OWN MTP head implementation.

Reimplementing the qwen3_5 MTP forward from the weights failed: even with the
contract read off vMLX's source (pre-norm hidden, concat([embedding, hidden]),
plain 1-D RoPE) a hand-rolled version scored an exact 0.00% over 1679 positions.
So don't reimplement it -- apply vMLX's patch and call its `mtp_forward`.

The patch (`vmlx_engine/patches/mlx_vlm_mtp/qwen35_vl.py`) is copied verbatim
into the scratchpad; the vmlx checkout is not touched. It only needs one thing
from the wider engine -- `is_mtp_active()` -- which gates whether `self.mtp` is
constructed at all, so that single function is stubbed True.

ACCEPTANCE, not accuracy: a draft is accepted when it matches what the TARGET
would have emitted, not the corpus token. Greedy:

    accept[t] = argmax(draft_logits[t]) == argmax(target_logits[t+1])

Teacher-forced on the model's own greedy continuation, so no decode loop and no
GDN state rollback is needed for depth 1.

    python mtp_accept_vmlx.py <bundle> <heldout.json> [n_prompts]
"""
import json
import sys
import types
from pathlib import Path

import mlx.core as mx


def _install_stub():
    """vMLX gates MTPModule construction on is_mtp_active(); make it True."""
    pkg = types.ModuleType("vmlx_engine")
    patches = types.ModuleType("vmlx_engine.patches")
    mlm = types.ModuleType("vmlx_engine.patches.mlx_lm_mtp")
    mlm.is_mtp_active = lambda: True
    pkg.patches = patches
    patches.mlx_lm_mtp = mlm
    sys.modules["vmlx_engine"] = pkg
    sys.modules["vmlx_engine.patches"] = patches
    sys.modules["vmlx_engine.patches.mlx_lm_mtp"] = mlm


def main():
    bundle = Path(sys.argv[1])
    prompts = json.loads(Path(sys.argv[2]).read_text())["prompts"]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    prompts = prompts[:n]

    _install_stub()
    sys.path.insert(0, str(Path(__file__).parent))
    from vmlx_mtp import qwen35_vl
    ok = qwen35_vl.apply()
    print(f"  vMLX MTP patch applied: {ok}")

    from mlx_vlm import load
    model, proc = load(str(bundle))
    lm = model.language_model
    has = hasattr(lm, "mtp")
    print(f"  bundle          : {bundle.name}")
    print(f"  lm.mtp present  : {has}")
    if not has:
        print("INVALID: the patched LanguageModel has no .mtp -- the head was "
              "not constructed, so nothing below would measure the real head.")
        return 3
    print(f"  mtp layers      : {len(lm.mtp.layers)}")

    tot = acc = 0
    per = []
    for p in prompts:
        text = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True)
        toks = list(proc.tokenizer.encode(text))
        ctx = len(toks)
        # Teacher-force on the model's OWN greedy continuation and score only
        # that span. Scoring the prompt too measures drafting of arbitrary
        # corpus text, which speculative decoding never does -- it drafts the
        # model's continuation. Including the prompt dominated the average and
        # understated acceptance by ~40 points.
        for _ in range(32):
            lm._position_ids = None
            lm._rope_deltas = None
            _l = lm(mx.array([toks]))
            _l = _l.logits if hasattr(_l, "logits") else _l
            toks.append(int(mx.argmax(_l[0, -1]).item()))
            del _l
            mx.clear_cache()
        X = mx.array([toks])
        lm._position_ids = None
        lm._rope_deltas = None
        out = lm(X, return_hidden=True)
        logits, hidden = out if isinstance(out, tuple) else (out.logits, None)
        if hidden is None:
            print("INVALID: return_hidden did not yield hidden states")
            return 3
        tgt = mx.argmax(logits, -1)[0]
        mx.eval(tgt, hidden)
        T = hidden.shape[1]

        cache = lm.make_mtp_cache()
        draft_logits = lm.mtp_forward(
            hidden[:, : T - 1, :], mx.array([toks[1:T]]), cache)
        d = mx.argmax(draft_logits, -1)[0]
        mx.eval(d)
        k = T - 2
        lo = max(0, ctx - 1)          # generated span only
        a = int((d[lo:k] == tgt[lo + 1: k + 1]).sum().item())
        n = max(1, k - lo)
        acc += a
        tot += n
        per.append(100.0 * a / n)
        del logits, hidden, draft_logits, d, cache
        mx.clear_cache()

    rate = 100.0 * acc / tot
    print(f"\n  depth-1 draft acceptance : {rate:.2f} %   ({acc}/{tot} positions)")
    print(f"  per-prompt min/median/max: {min(per):.1f} / "
          f"{sorted(per)[len(per)//2]:.1f} / {max(per):.1f} %")
    if rate < 1.0:
        print("\nINVALID: ~0% acceptance means the draft path is broken, not "
              "that the head is bad -- even a mis-wired head hits common tokens.")
        return 3
    Path(f"/Users/eric/models/Logs/q38v2/accept-{bundle.name}.json").write_text(
        json.dumps({"bundle": bundle.name, "positions": tot,
                    "acceptance_pct": rate, "per_prompt": per}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
