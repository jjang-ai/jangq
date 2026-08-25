"""MTP draft acceptance across a VARIATING turn shape, not a single shape.

A single-shape acceptance number is not proof. Acceptance is content-dependent
by construction -- the head drafts easily through boilerplate and badly at
high-entropy decision points -- so one prompt shape measures one slice. Worse,
the shapes that break things are the ones that CHANGE: reasoning toggling,
tools appearing mid-conversation, and (the usual killer) media and tools in the
SAME turn.

This walks ONE growing conversation and crosses the axes, measuring depth-1
acceptance at every transition against the same model:

    reasoning : on -> off -> on
    tools     : none -> offered -> called -> none
    growth    : context grows monotonically across turns
    replay    : turn 1 re-sent verbatim at the end

Each turn is rendered through the real chat template, teacher-forced on the
model's own greedy continuation, and scored as

    accept[t] = argmax(draft_logits[t]) == argmax(target_logits[t+1])

🚨 NOT COVERED: image turns, and the image+tools-in-one-turn case. This probe
feeds token ids only; the VL path needs pixel_values threaded through the outer
model. Any acceptance claim for VL usage is UNMEASURED and must not be inferred
from these numbers.

    python mtp_accept_variating.py <bundle> [gen_tokens]
"""
import json
import sys
import types
from pathlib import Path

import mlx.core as mx


def _install_stub():
    pkg = types.ModuleType("vmlx_engine")
    patches = types.ModuleType("vmlx_engine.patches")
    mlm = types.ModuleType("vmlx_engine.patches.mlx_lm_mtp")
    mlm.is_mtp_active = lambda: True
    pkg.patches = patches
    patches.mlx_lm_mtp = mlm
    sys.modules["vmlx_engine"] = pkg
    sys.modules["vmlx_engine.patches"] = patches
    sys.modules["vmlx_engine.patches.mlx_lm_mtp"] = mlm


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def main():
    bundle = Path(sys.argv[1])
    _install_stub()
    sys.path.insert(0, str(Path(__file__).parent))
    from vmlx_mtp import qwen35_vl
    qwen35_vl.apply()
    from mlx_vlm import load

    model, proc = load(str(bundle))
    lm = model.language_model
    if not hasattr(lm, "mtp"):
        print("INVALID: no .mtp on the patched model")
        return 3
    tok = proc.tokenizer

    # One conversation, grown turn by turn. `think` and `tools` vary.
    msgs = []
    turns = [
        ("T1 text, think=ON,  no tools",
         "Explain what a B-tree index does.", True, None),
        ("T2 text, think=OFF, no tools",
         "Now summarise that in one sentence.", False, None),
        ("T3 text, think=OFF, tools OFFERED",
         "What is the weather in Osaka?", False, TOOLS),
        ("T4 after tool result, tools OFFERED",
         "Given that, should I bring an umbrella?", False, TOOLS),
        ("T5 text, think=ON,  tools withdrawn",
         "Prove that the square root of 2 is irrational.", True, None),
        ("T6 REPLAY of T1 verbatim (think=ON, no tools)",
         "Explain what a B-tree index does.", True, None),
    ]
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    print(f"  bundle : {bundle.name}")
    print(f"  gen    : {n_gen} teacher-forced tokens per turn\n")
    print(f"  {'turn':<44} {'ctx':>6} {'accept':>8}")
    rows = []
    for label, content, think, tools in turns:
        msgs.append({"role": "user", "content": content})
        kw = {"tools": tools} if tools else {}
        try:
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=think, **kw)
        except TypeError:
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=think)
        ids = list(tok.encode(text))
        ctx = len(ids)

        # teacher-force on the model's own greedy continuation
        for _ in range(n_gen):
            lm._position_ids = None
            lm._rope_deltas = None
            lg = lm(mx.array([ids]))
            lg = lg.logits if hasattr(lg, "logits") else lg
            ids.append(int(mx.argmax(lg[0, -1]).item()))
            del lg
            mx.clear_cache()

        lm._position_ids = None
        lm._rope_deltas = None
        out = lm(mx.array([ids]), return_hidden=True)
        logits, hidden = out
        tgt = mx.argmax(logits, -1)[0]
        mx.eval(tgt, hidden)
        T = hidden.shape[1]
        cache = lm.make_mtp_cache()
        d = mx.argmax(lm.mtp_forward(hidden[:, : T - 1, :],
                                     mx.array([ids[1:T]]), cache), -1)[0]
        mx.eval(d)
        # score ONLY the generated span, so growth doesn't dilute the signal
        lo = max(0, ctx - 1)
        k = T - 2
        a = int((d[lo:k] == tgt[lo + 1: k + 1]).sum().item())
        n = max(1, k - lo)
        rate = 100.0 * a / n
        rows.append({"turn": label, "ctx": ctx, "scored": n,
                     "acceptance_pct": rate})
        print(f"  {label:<44} {ctx:>6} {rate:>7.1f}%")

        # append the model's own reply, plus a tool result after T3
        msgs.append({"role": "assistant",
                     "content": tok.decode(ids[ctx:]).strip()[:400]})
        if tools and label.startswith("T3"):
            msgs.append({"role": "tool", "name": "get_weather",
                         "content": '{"city":"Osaka","temp_c":19,"rain":true}'})
        del logits, hidden, tgt, d, cache
        mx.clear_cache()

    vals = [r["acceptance_pct"] for r in rows]
    print(f"\n  spread across shapes: min {min(vals):.1f}%  max {max(vals):.1f}%  "
          f"range {max(vals)-min(vals):.1f} pts")
    t1, t6 = rows[0]["acceptance_pct"], rows[-1]["acceptance_pct"]
    print(f"  replay check (T1 vs T6, same text, longer context): "
          f"{t1:.1f}% -> {t6:.1f}%  delta {t6-t1:+.1f} pts")
    print("\n  🚨 image turns and image+tools-in-one-turn are NOT covered here.")
    Path(f"/Users/eric/models/Logs/q38v2/accept-variating-{bundle.name}.json"
         ).write_text(json.dumps({"bundle": bundle.name, "turns": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
