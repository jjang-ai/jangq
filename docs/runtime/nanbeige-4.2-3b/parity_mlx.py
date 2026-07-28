import numpy as np, mlx.core as mx
from jang_tools.nanbeige import mlx_register  # noqa: F401
from mlx_lm import load
from mlx_lm.models.cache import KVCache

SRC="/Users/eric/models/Nanbeige/Nanbeige4.2-3B"
OUT="/private/tmp/claude-501/-Users-eric-jang/85ab95b4-d966-4eb6-b700-255f21436c4b/scratchpad/nanbeige"
model, tok = load(SRC)
model.set_dtype(mx.float32)

ids = np.load(f"{OUT}/hf_ids.npy")
hf  = np.load(f"{OUT}/hf_logits.npy")
x = mx.array(ids)

# 1) full prefill, no cache
lg = np.array(model(x)[0, -1].astype(mx.float32))
def cmp(name, a, b):
    d = np.abs(a-b); den = np.abs(b).max()
    print(f"{name}: max|Δ|={d.max():.4f} rel={d.max()/den:.2e} "
          f"argmax {int(a.argmax())} vs {int(b.argmax())} "
          f"top5 {'OK' if (np.argsort(-a)[:5]==np.argsort(-b)[:5]).all() else 'MISMATCH'}")
    return d.max()/den
cmp("MLX vs HF (full prefill, fp32)", lg, hf)

# 2) incremental: prefill n-1 with 44-slot cache, then step the last token
c = model.make_cache()
print("cache slots:", len(c))
_ = model(x[:, :-1], cache=c)
lg_inc = np.array(model(x[:, -1:], cache=c)[0, -1].astype(mx.float32))
cmp("MLX incremental (44-slot) vs HF", lg_inc, hf)

# 3) NEGATIVE CONTROL: reuse one cache per layer across BOTH loops (22 slots
#    duplicated) — what a naive port does. Must diverge.
class Dup(list):
    pass
c22 = [KVCache() for _ in range(22)]
bad = Dup(c22 + c22)          # loop 1 re-reads/writes loop 0's slots
_ = model(x[:, :-1], cache=bad)
lg_bad = np.array(model(x[:, -1:], cache=bad)[0, -1].astype(mx.float32))
r = cmp("NEGATIVE CONTROL shared-cache vs HF", lg_bad, hf)
print("negative control diverges:", r > 1e-2)
