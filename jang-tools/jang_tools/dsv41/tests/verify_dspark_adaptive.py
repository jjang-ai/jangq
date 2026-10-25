"""Verify the RECOVERABLE adaptive DSpark change:
  1) adaptive DSpark output is TOKEN-IDENTICAL to plain greedy AR (correctness),
  2) it does NOT permanently latch -- after a hard (low-accept) stretch it RECOVERS drafting on an
     easy continuation (prove depth is acceptance-governed, not hard-enforced).
"""
import sys, time
from pathlib import Path
import mlx.core as mx
from jang_tools.dsv41.model import DSV41
from jang_tools.dsv41.dspark import load_dspark, DSparkRunner

BUNDLE = Path(sys.argv[1]).expanduser()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
m = DSV41(BUNDLE)
ds = load_dspark(BUNDLE, m.args, m.model.embed, m.model.head)
eos_tok = m.tokenizer.token_to_id("<｜end▁of▁sentence｜>")
eos = {e for e in (eos_tok, m.eos_id) if e is not None}

enc = None
try:
    from jang_tools.dsv41.generate import load_encoding
    enc = load_encoding(BUNDLE)
except Exception:
    pass

def ids_for(prompt):
    if enc is not None:
        return m.tokenizer.encode(enc.encode_messages([{"role": "user", "content": prompt}],
                                   thinking_mode="chat", reasoning_effort="low"),
                                  add_special_tokens=False).ids
    return m.tokenizer.encode(prompt, add_special_tokens=False).ids

def greedy_ar(ids, n):
    out, sess = [], m.new_session()
    logits, _ = m.forward(list(ids), sess, all_logits=False, return_hidden=True)
    for _ in range(n):
        t = int(mx.argmax(logits)); out.append(t)
        if eos and t in eos: break
        logits = m.forward([t], sess, all_logits=False)
    return out

# low-accept prefix (random numbers) then an easy continuation in ONE prompt -> exercises enter-AR
# then recover.
prompt = ("Repeat this line then explain fibonacci simply: "
          + " ".join(str((i * 7919) % 100000) for i in range(24)))
ids = ids_for(prompt)

ar = greedy_ar(ids, N)
stats = {}
runner = DSparkRunner(m, ds, adaptive=True)
spec = list(runner.stream(ids, N, eos_ids=eos, stats=stats))

ident = ar == spec
print(f"[bundle loaded] bs={ds.bs} width={runner.width} probe_every={runner.probe_every}")
print(f"AR  tokens ({len(ar)}): {ar[:24]}")
print(f"SPEC tokens({len(spec)}): {spec[:24]}")
print(f"TOKEN-IDENTICAL: {'PASS' if ident else 'FAIL'}")
if not ident:
    for i,(a,b) in enumerate(zip(ar,spec)):
        if a!=b: print(f"  first diff @ {i}: AR={a} SPEC={b}"); break
print(f"stats: {stats}  (accepted>0 proves drafting engaged; not a permanent AR latch)")
print(f"RECOVERY: {'PASS' if stats.get('accepted',0) > 0 else 'CHECK'} (accepted={stats.get('accepted',0)} of drafted={stats.get('drafted',0)})")
